from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from groq import Groq

from src.security.sanitizer import detect_prompt_injection, sanitize_input

load_dotenv()


def load_settings() -> dict[str, Any]:
    with open("configs/settings.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


_SETTINGS = load_settings()
_CLIENT: Groq | None = None


def get_groq_client() -> Groq:
    global _CLIENT
    if _CLIENT is None:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY environment variable is missing.")
        _CLIENT = Groq(api_key=api_key)
    return _CLIENT


def get_cache_path(payload: dict[str, Any]) -> Path:
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()
    digest = hashlib.sha256(raw).hexdigest()
    path = Path("data/cache/llm") / f"{digest}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def call_llm(
    system_prompt: str,
    user_prompt: str,
    model_role: str = "query_router",
    temperature: float = 0.0,
    max_tokens: int = 1024,
    json_mode: bool = False,
    timeout_seconds: float = 20.0,
) -> tuple[str, dict[str, Any]]:
    """Execute LLM call with response caching, retry handling, and fallback support."""
    client = get_groq_client()
    cfg = _SETTINGS

    # Security check on user prompt
    if detect_prompt_injection(user_prompt):
        return (
            "Potential prompt injection detected. Request blocked by security policy.",
            {"cached": False, "tokens": 0, "model": "security_filter", "error": "prompt_injection"},
        )

    clean_user = sanitize_input(user_prompt)
    model_cfg = cfg["models"].get(model_role, cfg["models"]["query_router"])
    model_id = model_cfg["id"]

    cache_payload = {
        "model_id": model_id,
        "system_prompt": system_prompt,
        "user_prompt": clean_user,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "json_mode": json_mode,
    }
    c_path = get_cache_path(cache_payload)
    if c_path.exists():
        data = json.loads(c_path.read_text(encoding="utf-8"))
        return data["content"], {**data.get("meta", {}), "cached": True}

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": clean_user},
    ]

    last_error: Exception | None = None
    target_models = [model_id, cfg["models"]["fallback"]["id"]]

    for target_model in target_models:
        for attempt in range(cfg["resilience"]["retry"]["max_attempts"]):
            try:
                kwargs: dict[str, Any] = {
                    "model": target_model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "timeout": timeout_seconds,
                    # Never pass a `tools` list -- prevents Groq from activating
                    # its native function-calling registry.
                }
                if json_mode:
                    kwargs["response_format"] = {"type": "json_object"}

                t0 = time.time()
                response = client.chat.completions.create(**kwargs)
                duration_ms = round((time.time() - t0) * 1000, 2)

                # Handle both text and potential tool-call content gracefully
                choice = response.choices[0]
                msg = choice.message
                content = getattr(msg, "content", None) or ""

                # If the model still emitted a tool_call (shouldn't happen), extract text
                if not content and hasattr(msg, "tool_calls") and msg.tool_calls:
                    # Pull the raw arguments string as a best-effort fallback text
                    content = str(msg.tool_calls[0].function.arguments)

                usage = getattr(response, "usage", None)
                tokens = getattr(usage, "total_tokens", 0) if usage else 0

                meta = {
                    "cached": False,
                    "duration_ms": duration_ms,
                    "model": target_model,
                    "tokens": tokens,
                    "fallback_model_used": target_model != model_id,
                }

                # Save to cache
                c_path.write_text(
                    json.dumps({"content": content, "meta": meta}, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                return content, meta

            except Exception as exc:
                last_error = exc
                err_str = str(exc).lower()
                # Exponential backoff for rate limits; linear delay for transient errors
                if "rate_limit" in err_str or "429" in err_str:
                    time.sleep(min(2 ** (attempt + 1), cfg["resilience"]["retry"]["max_delay_seconds"]))
                elif "tool_use_failed" in err_str or "tool choice" in err_str:
                    # Non-retryable Groq tool-call error — skip remaining attempts for this model
                    break
                else:
                    time.sleep(cfg["resilience"]["retry"]["initial_delay_seconds"])

    # If all attempts and fallbacks fail, return graceful error dict
    return (
        f"LLM call failed after retries: {last_error}",
        {"cached": False, "tokens": 0, "model": model_id, "error": str(last_error)},
    )
