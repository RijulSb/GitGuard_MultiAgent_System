"""Batch 3: Critique, Arbitration, and Safety Guard (Agents 8, 9, 10).

Architecture overview
─────────────────────
Agent 8  – Critic          : Reviews each draft claim-by-claim against the context pack.
                             Uses openai/gpt-oss-120b (critic_and_judge model) for best accuracy.
Agent 9  – Arbiter         : Runs the debate loop (max 2 revision rounds), merges supported
                             claims from both drafts, decides final answer or abstention.
Agent 10 – Safety Guard    : Labels the final answer LOW / MEDIUM / HIGH risk based on
                             dangerous git commands; adds safer alternatives with citations.

Error-handling philosophy
─────────────────────────
• Tool/LLM failures inside Batch 3 are caught and surfaced as partial results rather
  than pipeline crashes.  Only the Arbiter can cause an abstention.
• If the Critic JSON parse fails we fall back to full-context acceptance (0-claim critique)
  so the Arbiter can still produce a best-effort answer.
• The Safety Guard never blocks a response — it annotates it.
"""
from __future__ import annotations

import json
import re
import time
from typing import Any

import yaml

from src.agents.llm_client import call_llm
from src.graph.state import AgentState, AgentTraceEntry

# ─── shared config ────────────────────────────────────────────────────────────
with open("configs/settings.yaml", encoding="utf-8") as _f:
    _CFG = yaml.safe_load(_f)

_AGENTS_CFG = _CFG["agents"]
_SAFETY_CFG = _CFG["safety"]
_MIN_RATIO: float = float(_AGENTS_CFG["min_supported_claim_ratio"])
_MIN_CONTEXT_SCORE: float = float(_CFG["retrieval"]["score_threshold"])
_MAX_REVISION_ROUNDS: int = int(_AGENTS_CFG["max_revision_rounds"])
_ABSTENTION_MSG: str = _AGENTS_CFG["abstention_message"]

# Pre-built command sets for O(1) lookup
_HIGH_RISK: frozenset[str] = frozenset(
    cmd.strip().lower() for cmd in _SAFETY_CFG.get("high_risk_commands", [])
)
_MEDIUM_RISK: frozenset[str] = frozenset(
    cmd.strip().lower() for cmd in _SAFETY_CFG.get("medium_risk_commands", [])
)


# ─── Agent 8: Critic ──────────────────────────────────────────────────────────

def critic_agent(state: AgentState) -> AgentState:
    """Agent 8: Critic — Evaluates every claim in both drafts against the context pack.

    For each draft the model lists each claim as a sentence and marks it
    supported / unsupported with a one-line reason referencing a section_id.

    Graceful degradation
    ────────────────────
    • If the LLM call fails, we accept both drafts unconditionally (empty critique
      lists) so downstream Arbiter can still produce an answer.
    • JSON parse failures produce a partial/empty critique with an error flag.
    """
    if not state.get("is_safe", True) or state.get("is_out_of_scope", False):
        return state

    # Skip if both drafts are missing (upstream failure)
    draft_a = state.get("draft_a", "")
    draft_b = state.get("draft_b", "")
    if not draft_a and not draft_b:
        return state

    t0 = time.time()
    context_pack = state.get("context_pack", [])
    context_text = "\n\n".join(
        f"[{c['section_id']}] {c.get('heading', '')}: {c.get('text', '')[:400]}"
        for c in context_pack
    )

    def _critique_draft(draft: str, label: str) -> list[dict[str, Any]]:
        """Call the critic model and parse the JSON list of claim evaluations."""
        if not draft or draft.startswith("Drafter error:"):
            return []

        system_prompt = (
            "You are the GitGuard Critic. Your job is to evaluate EVERY claim in the "
            "provided draft against the retrieved context excerpts.\n\n"
            "INSTRUCTIONS:\n"
            "1. Split the draft into individual factual claims (one per sentence).\n"
            "2. For each claim decide if it is directly SUPPORTED or UNSUPPORTED by the "
            "context excerpts below.\n"
            "3. Output a JSON array only — no prose, no markdown fences.\n"
            "   Format: [{\"claim\": \"...\", \"supported\": true/false, "
            "\"reason\": \"short reason citing [section_id] if supported\"}]\n"
            "4. If the draft is empty or says 'Not found', return [].\n\n"
            f"CONTEXT EXCERPTS:\n{context_text}"
        )
        user_prompt = f"DRAFT {label}:\n{draft}"

        content, meta = call_llm(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model_role="critic_and_judge",
            temperature=0.0,
            max_tokens=2048,
            json_mode=True,
            timeout_seconds=25.0,
        )

        if meta.get("error"):
            return []  # Graceful: accept draft if critic LLM fails

        # Strip markdown fences if present despite json_mode=True
        cleaned = re.sub(r"```(?:json)?", "", content).strip().strip("` ")
        try:
            parsed = json.loads(cleaned)
            if isinstance(parsed, list):
                return [
                    {
                        "claim": str(item.get("claim", "")),
                        "supported": bool(item.get("supported", False)),
                        "reason": str(item.get("reason", "")),
                    }
                    for item in parsed
                    if isinstance(item, dict)
                ]
        except (json.JSONDecodeError, TypeError):
            pass
        return []

    critique_a = _critique_draft(draft_a, "A (Practical)")
    critique_b = _critique_draft(draft_b, "B (Strict)")

    trace: AgentTraceEntry = {
        "agent": "critic",
        "action": "critique_drafts",
        "duration_ms": round((time.time() - t0) * 1000, 2),
        "status": "success",
        "details": {
            "claims_a": len(critique_a),
            "supported_a": sum(1 for c in critique_a if c["supported"]),
            "claims_b": len(critique_b),
            "supported_b": sum(1 for c in critique_b if c["supported"]),
        },
    }

    return {
        **state,
        "critique_a": critique_a,
        "critique_b": critique_b,
        "traces": state.get("traces", []) + [trace],
    }


# ─── Agent 9: Arbiter ─────────────────────────────────────────────────────────

def arbiter_agent(state: AgentState) -> AgentState:
    """Agent 9: Arbiter — Runs debate loop, merges supported claims, decides abstention.

    Debate loop (max _MAX_REVISION_ROUNDS):
    ├─ Collect unsupported claims from both critiques
    ├─ If any exist and revision budget remains, send them back to the LLM for revision
    │  (a lightweight re-draft that tries to find support for the flagged claims)
    └─ After loop: merge all supported claims into a unified answer.

    Abstention conditions (either triggers abstention):
    • Supported-claim ratio < _MIN_RATIO (0.80)
    • Context score < _MIN_CONTEXT_SCORE (0.35)
    • Both drafts contain no supported claims at all
    """
    if not state.get("is_safe", True) or state.get("is_out_of_scope", False):
        # Out-of-scope path: emit clean abstention message
        return {
            **state,
            "verified_answer": _ABSTENTION_MSG,
            "abstained": True,
            "final_response": _ABSTENTION_MSG,
            "supported_claim_ratio": 0.0,
        }

    t0 = time.time()
    draft_a: str = state.get("draft_a", "")
    draft_b: str = state.get("draft_b", "")
    critique_a: list[dict] = state.get("critique_a", [])
    critique_b: list[dict] = state.get("critique_b", [])
    context_score: float = float(state.get("context_score", 0.0))
    context_pack: list[dict] = state.get("context_pack", [])
    revision_round: int = int(state.get("revision_round", 0))

    # ── Helper: compute supported ratio across both critiques ──────────────
    def _ratio(ca: list[dict], cb: list[dict]) -> float:
        total = len(ca) + len(cb)
        if total == 0:
            return 1.0  # No claims → vacuously supported (pass-through)
        supported = sum(1 for c in ca + cb if c.get("supported"))
        return round(supported / total, 4)

    def _unsupported(ca: list[dict], cb: list[dict]) -> list[str]:
        return [c["claim"] for c in ca + cb if not c.get("supported")]

    current_ratio = _ratio(critique_a, critique_b)
    unsupported_claims = _unsupported(critique_a, critique_b)

    # ── Debate loop — revise unsupported claims ────────────────────────────
    while unsupported_claims and revision_round < _MAX_REVISION_ROUNDS:
        revision_round += 1
        context_text = "\n\n".join(
            f"[{c['section_id']}] {c.get('text', '')[:300]}" for c in context_pack
        )

        revise_system = (
            "You are the GitGuard Reviser. The following claims from a draft answer "
            "were NOT supported by the retrieved context. Your task:\n"
            "1. Re-examine each claim against the context excerpts below.\n"
            "2. If a claim is genuinely unsupported by the context, replace it with a "
            "   supported statement from the context (with [section_id] citation).\n"
            "3. If the claim cannot be supported at all, remove it from the answer.\n"
            "4. Output only the revised answer text — no JSON, no preamble.\n\n"
            f"CONTEXT EXCERPTS:\n{context_text}"
        )
        claims_text = "\n".join(f"- {c}" for c in unsupported_claims[:10])
        revise_user = (
            f"UNSUPPORTED CLAIMS (round {revision_round}):\n{claims_text}\n\n"
            f"CURRENT DRAFT A:\n{draft_a}\n\nCURRENT DRAFT B:\n{draft_b}"
        )

        revised_content, revised_meta = call_llm(
            system_prompt=revise_system,
            user_prompt=revise_user,
            model_role="critic_and_judge",
            temperature=0.1,
            max_tokens=1536,
            timeout_seconds=25.0,
        )

        if not revised_meta.get("error") and revised_content.strip():
            # Use the revised text as the new draft_a for next round assessment
            draft_a = revised_content.strip()

        # Re-run a quick single-draft critique on the revised text
        if context_pack and draft_a:
            quick_system = (
                "You are the GitGuard Critic. Evaluate each claim in the draft below "
                "against the context. Return a JSON array: "
                "[{\"claim\": str, \"supported\": bool, \"reason\": str}]"
                f"\n\nCONTEXT:\n{context_text}"
            )
            qc_content, qc_meta = call_llm(
                system_prompt=quick_system,
                user_prompt=f"DRAFT:\n{draft_a}",
                model_role="critic_and_judge",
                temperature=0.0,
                max_tokens=1024,
                json_mode=True,
                timeout_seconds=20.0,
            )
            if not qc_meta.get("error"):
                try:
                    cleaned = re.sub(r"```(?:json)?", "", qc_content).strip().strip("` ")
                    parsed = json.loads(cleaned)
                    if isinstance(parsed, list):
                        critique_a = [
                            {
                                "claim": str(i.get("claim", "")),
                                "supported": bool(i.get("supported", False)),
                                "reason": str(i.get("reason", "")),
                            }
                            for i in parsed if isinstance(i, dict)
                        ]
                except (json.JSONDecodeError, TypeError):
                    pass

        current_ratio = _ratio(critique_a, critique_b)
        unsupported_claims = _unsupported(critique_a, critique_b)

    # ── Abstention check ──────────────────────────────────────────────────
    should_abstain = (
        context_score < _MIN_CONTEXT_SCORE
        or current_ratio < _MIN_RATIO
    )

    if should_abstain:
        trace: AgentTraceEntry = {
            "agent": "arbiter",
            "action": "abstain",
            "duration_ms": round((time.time() - t0) * 1000, 2),
            "status": "abstained",
            "details": {
                "reason": "low_context_score" if context_score < _MIN_CONTEXT_SCORE else "low_claim_ratio",
                "context_score": context_score,
                "supported_claim_ratio": current_ratio,
                "revision_rounds": revision_round,
            },
        }
        return {
            **state,
            "verified_answer": _ABSTENTION_MSG,
            "abstained": True,
            "final_response": _ABSTENTION_MSG,
            "supported_claim_ratio": current_ratio,
            "revision_round": revision_round,
            "traces": state.get("traces", []) + [trace],
        }

    # ── Merge supported claims from both drafts ────────────────────────────
    supported_claims_a = [c["claim"] for c in critique_a if c.get("supported")]
    supported_claims_b = [c["claim"] for c in critique_b if c.get("supported")]

    # Use the LLM to synthesize a clean unified answer from both supported claim sets
    context_text = "\n\n".join(
        f"[{c['section_id']}] {c.get('text', '')[:400]}" for c in context_pack
    )
    merge_system = (
        "You are the GitGuard Arbiter. Synthesize a clear, concise, final answer "
        "by merging the supported claims below. Rules:\n"
        "1. Preserve every [section_id] citation.\n"
        "2. De-duplicate overlapping claims (keep the more detailed one).\n"
        "3. Use markdown for git commands (backticks).\n"
        "4. Do NOT add any new information not present in the claims.\n"
        "5. Output only the final answer text — no preamble.\n\n"
        f"CONTEXT FOR REFERENCE:\n{context_text}"
    )
    claims_block = (
        "SUPPORTED CLAIMS FROM DRAFT A:\n"
        + ("\n".join(f"- {c}" for c in supported_claims_a) or "(none)")
        + "\n\nSUPPORTED CLAIMS FROM DRAFT B:\n"
        + ("\n".join(f"- {c}" for c in supported_claims_b) or "(none)")
        + f"\n\nAlso, here is Draft A (revised) for reference:\n{draft_a}"
    )

    merged_content, merged_meta = call_llm(
        system_prompt=merge_system,
        user_prompt=claims_block,
        model_role="critic_and_judge",
        temperature=0.1,
        max_tokens=2048,
        timeout_seconds=25.0,
    )

    # If merge LLM call fails, fall back to draft_a (best single source)
    verified = (
        merged_content.strip()
        if not merged_meta.get("error") and merged_content.strip()
        else (draft_a or draft_b or _ABSTENTION_MSG)
    )

    is_abstention = (
        verified.strip() == _ABSTENTION_MSG
        or not (supported_claims_a or supported_claims_b)
        or verified.strip() == "Not found in the book."
    )

    trace: AgentTraceEntry = {
        "agent": "arbiter",
        "action": "merge_and_verify" if not is_abstention else "abstain",
        "duration_ms": round((time.time() - t0) * 1000, 2),
        "status": "success" if not is_abstention else "abstained",
        "details": {
            "supported_claim_ratio": current_ratio,
            "revision_rounds": revision_round,
            "context_score": context_score,
            "verified_length": len(verified),
            "abstained": is_abstention,
        },
    }

    return {
        **state,
        "verified_answer": verified,
        "abstained": is_abstention,
        "supported_claim_ratio": current_ratio,
        "revision_round": revision_round,
        "traces": state.get("traces", []) + [trace],
    }


# ─── Agent 10: Safety Guard ───────────────────────────────────────────────────

def safety_guard_agent(state: AgentState) -> AgentState:
    """Agent 10: Safety Guard — Classifies risk level and adds safer alternatives.

    Risk classification
    ───────────────────
    • HIGH   : Verified answer contains a destructive git command (e.g. git reset --hard,
               git push --force, git filter-branch).  A safer alternative IS required.
    • MEDIUM : Answer references a reversible-but-cautious command (e.g. git rebase,
               git checkout, git restore).  A note is added.
    • LOW    : No risky commands detected.

    The Safety Guard NEVER suppresses or modifies the verified answer.
    It annotates state with risk_level, safer_alternative, and final_response.
    """
    if not state.get("is_safe", True):
        return state

    t0 = time.time()
    verified = state.get("verified_answer", "")
    abstained = state.get("abstained", False)

    # For abstentions just pass through with LOW risk
    if abstained:
        trace: AgentTraceEntry = {
            "agent": "safety_guard",
            "action": "risk_classify",
            "duration_ms": round((time.time() - t0) * 1000, 2),
            "status": "pass_through",
            "details": {"risk_level": "LOW", "abstained": True},
        }
        return {
            **state,
            "risk_level": "LOW",
            "safer_alternative": None,
            "final_response": verified,
            "traces": state.get("traces", []) + [trace],
        }

    from src.security.safety_rules import evaluate_text_safety, format_safety_annotated_response

    # Evaluate text safety using canonical safety rules
    safety_eval = evaluate_text_safety(verified)
    risk_level = safety_eval.risk_level
    detected_high = [r.canonical_command for r in safety_eval.triggered_rules if r.risk_level == "HIGH"]
    detected_medium = [r.canonical_command for r in safety_eval.triggered_rules if r.risk_level == "MEDIUM"]

    # ── Build safer alternative (HIGH risk only) ──────────────────────────
    safer_alternative: str | None = None
    if risk_level == "HIGH":
        # First check canonical rule safer alternative
        if safety_eval.safer_alternatives:
            safer_alternative = " ".join(safety_eval.safer_alternatives)
        else:
            context_text = "\n\n".join(
                f"[{c['section_id']}] {c.get('text', '')[:300]}"
                for c in state.get("context_pack", [])
            )
            safe_system = (
                "You are a Git safety expert. The following answer contains a destructive "
                "git command. Suggest a safer Git workflow that achieves the same goal without data loss. "
                "Cite [section_id] from context if applicable. Keep suggestion concise.\n\n"
                f"CONTEXT:\n{context_text}"
            )
            safe_user = f"RISKY COMMANDS: {', '.join(detected_high)}\nANSWER:\n{verified}"
            safe_content, safe_meta = call_llm(
                system_prompt=safe_system,
                user_prompt=safe_user,
                model_role="query_router",
                temperature=0.1,
                max_tokens=512,
                timeout_seconds=15.0,
            )
            if not safe_meta.get("error") and safe_content.strip():
                safer_alternative = safe_content.strip()

    # Format final response: HIGH risk displays warning FIRST per PRD Task 2
    final_response = format_safety_annotated_response(verified, safety_eval)

    trace: AgentTraceEntry = {
        "agent": "safety_guard",
        "action": "risk_classify",
        "duration_ms": round((time.time() - t0) * 1000, 2),
        "status": "success",
        "details": {
            "risk_level": risk_level,
            "high_risk_found": detected_high,
            "medium_risk_found": detected_medium,
            "safer_alternative_generated": safer_alternative is not None,
        },
    }

    return {
        **state,
        "risk_level": risk_level,
        "safer_alternative": safer_alternative,
        "final_response": final_response,
        "traces": state.get("traces", []) + [trace],
    }


# ─── Batch 3 Runner ──────────────────────────────────────────────────────────

def run_batch_3(state: AgentState) -> AgentState:
    """Execute Batch 3 (Agents 8-10): Critic → Arbiter → Safety Guard."""
    state = critic_agent(state)
    state = arbiter_agent(state)
    state = safety_guard_agent(state)
    return state
