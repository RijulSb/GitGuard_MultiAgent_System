from __future__ import annotations

import re
from typing import Any

INJECTION_PATTERNS = [
    r"(?i)ignore\s+(all\s+)?(previous|prior|above)\s+instructions?",
    r"(?i)disregard\s+(all\s+)?(previous|prior|system)\s+instructions?",
    r"(?i)reveal\s+(the\s+)?(system\s+prompt|hidden\s+prompt|secret|instructions?)",
    r"(?i)output\s+(the\s+)?(system\s+prompt|developer\s+mode)",
    r"(?i)you\s+are\s+now\s+(in\s+)?(dan|developer\s+mode|unrestricted)",
    r"(?i)act\s+as\s+(an?\s+)?(unrestricted|jailbroken|evil)",
    r"(?i)<\s*\|\s*im_start\s*\|>",
    r"(?i)\[\s*INST\s*\]",
    r"(?i)(?:^|\n)\s*system\s*:",
]

SAFE_COMMAND_PATTERN = re.compile(r"^[a-zA-Z0-9_\-\.\s\:]+$")


def detect_prompt_injection(text: str) -> bool:
    """Detect common prompt injection and system jailbreak patterns."""
    if not text:
        return False
    return any(re.search(pattern, text) for pattern in INJECTION_PATTERNS)


def sanitize_input(text: str, max_length: int = 1000) -> str:
    """Sanitize user input: strip injection markers, control chars, and truncate length."""
    if not text:
        return ""
    # Strip null bytes and control chars except standard whitespace
    sanitized = "".join(c for c in text if c.isprintable() or c in ("\n", "\r", "\t"))
    # Neutralize injection attempts by escaping potential system prompt boundary tags
    for pattern in INJECTION_PATTERNS:
        sanitized = re.sub(pattern, "[BLOCKED_INJECTION_PATTERN]", sanitized)
    return sanitized.strip()[:max_length]


def sanitize_command_query(command: str) -> str:
    """Ensure command queries sent to git tools are free of shell injection metacharacters."""
    # Strip dangerous shell metacharacters (; & | ` $ > < \ " ' ! { })
    cleaned = re.sub(r"[;&|`$><\"'\\!{}]", "", command).strip()
    return cleaned[:100]
