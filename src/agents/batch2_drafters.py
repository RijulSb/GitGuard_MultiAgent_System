from __future__ import annotations

import re
import time
from typing import Any

from src.agents.llm_client import call_llm
from src.graph.state import AgentState, AgentTraceEntry
from src.tools.git_checker import check_git_command
from src.tools.section_reader import fetch_section


def reranker_agent(state: AgentState) -> AgentState:
    """Agent 5: Reranker - Merges dense & keyword results into top-5 context pack."""
    if not state.get("is_safe", True) or state.get("is_out_of_scope", False):
        return state

    t0 = time.time()
    dense = state.get("dense_chunks", [])
    keyword = state.get("keyword_chunks", [])

    dense_map = {c["chunk_id"]: c for c in dense}
    keyword_map = {c["chunk_id"]: c for c in keyword}
    all_chunks = {**dense_map, **keyword_map}

    dense_ids = [c["chunk_id"] for c in dense]
    keyword_ids = [c["chunk_id"] for c in keyword]
    merged_ids = list(dict.fromkeys(dense_ids + keyword_ids))

    def fusion_score(cid: str) -> float:
        score = 0.0
        if cid in dense_ids:
            score += 0.60 / (dense_ids.index(cid) + 1)
        if cid in keyword_ids:
            score += 0.40 / (keyword_ids.index(cid) + 1)
        return score

    scored = [(cid, fusion_score(cid)) for cid in merged_ids]
    scored.sort(key=lambda x: x[1], reverse=True)

    top_5_ids = scored[:5]
    context_pack = [{**all_chunks[cid], "fusion_score": round(score, 4)} for cid, score in top_5_ids]
    max_score = top_5_ids[0][1] if top_5_ids else 0.0

    trace: AgentTraceEntry = {
        "agent": "reranker",
        "action": "fusion_rerank",
        "duration_ms": round((time.time() - t0) * 1000, 2),
        "status": "success",
        "details": {"context_count": len(context_pack), "top_score": round(max_score, 4)},
    }

    return {
        **state,
        "context_pack": context_pack,
        "context_score": round(max_score, 4),
        "traces": state.get("traces", []) + [trace],
    }


def _execute_react_loop(
    question: str,
    context_pack: list[dict[str, Any]],
    model_role: str,
    drafter_name: str,
    style_instruction: str,
    max_tool_calls: int = 3,
) -> tuple[str, list[dict[str, Any]], int]:
    """Execute ReAct tool loop (Think -> Tool Call -> Observe -> Final Answer).

    Uses TOOL_CALL: prefix syntax (not JSON) to avoid triggering Groq's native
    function-calling registry which causes 400 'tool_use_failed' errors.
    The regex parser handles the TOOL_CALL: text format on our side.
    """
    context_text = "\n\n".join(
        f"[{c['section_id']}] {c.get('heading', '')}\n{c.get('text', '')}" for c in context_pack
    )

    # IMPORTANT: Use TOOL_CALL: prefix (not "Action:") and plain text args (not JSON/bracket syntax).
    # The Groq qwen model's native function-calling heuristic is triggered by Action+bracket patterns.
    # TOOL_CALL: is a custom text-only protocol parsed entirely on our side.
    system_prompt = f"""You are {drafter_name} in the GitGuard pair-programming system.
{style_instruction}

CONTEXT EXCERPTS (use ONLY these, do not invent facts):
{context_text}

STRICT RULES:
1. Every factual sentence MUST end with [section_id] citing the exact section used.
2. To read a section in full, write on its own line:
   TOOL_CALL: fetch_section section_id_here
3. To verify a git command or flag syntax, write on its own line:
   TOOL_CALL: check_git_command git-command-here
4. After using tools or if you have enough information, write your answer starting with:
   FINAL ANSWER:
5. DO NOT output JSON, code blocks with curly braces, or structured tool schemas.
6. Do not use knowledge outside the provided context or tool observations.
7. If the context does not contain the answer, write: FINAL ANSWER: Not found in the book."""

    user_query = f"Question: {question}"
    conversation_history = user_query
    tool_trace: list[dict[str, Any]] = []
    tokens_accumulated = 0

    # TOOL_CALL: <tool_name> <arg> (plain-text, no brackets, no JSON)
    _TOOL_RE = re.compile(r"TOOL_CALL:\s*(\w+)\s+(.+?)(?:\n|$)", re.IGNORECASE)
    # Also handle legacy "Action:" patterns in case the model uses old format despite instructions
    _ACTION_RE = re.compile(r"Action:\s*([a-zA-Z_]+)\[[\"\']?([^\"\'\\]]+)[\"\']?\]", re.IGNORECASE)

    for step in range(max_tool_calls + 1):
        response_text, meta = call_llm(
            system_prompt=system_prompt,
            user_prompt=conversation_history,
            model_role=model_role,
            temperature=0.2 if "Practical" in drafter_name else 0.1,
            max_tokens=1024,
        )
        tokens_accumulated += meta.get("tokens", 0)

        # LLM call itself failed — surface the error without stalling the pipeline
        if meta.get("error"):
            return f"Drafter error: {meta['error']}", tool_trace, tokens_accumulated

        # Check for TOOL_CALL: pattern (preferred)
        tool_match = _TOOL_RE.search(response_text)
        # Fallback: check for legacy Action: pattern
        if not tool_match:
            legacy = _ACTION_RE.search(response_text)
            if legacy:
                tool_name_raw = legacy.group(1).strip()
                tool_arg_raw = legacy.group(2).strip()
                tool_match = None  # handled below via named vars
            else:
                tool_name_raw = tool_arg_raw = None
        else:
            tool_name_raw = tool_match.group(1).strip()
            tool_arg_raw = tool_match.group(2).strip()

        if tool_name_raw and step < max_tool_calls:
            tool_name = tool_name_raw.lower()
            tool_arg = tool_arg_raw.strip().strip('"\'')

            observation = ""
            fallback_used = False
            try:
                if tool_name == "fetch_section":
                    res = fetch_section(tool_arg, timeout_seconds=3.0)
                    observation = res.data[:600] if res.success else f"Error: {res.error}"
                    fallback_used = res.source == "fallback"
                elif tool_name in ("check_git_command", "check_git"):
                    res = check_git_command(tool_arg, timeout_seconds=3.0)
                    observation = f"Command valid: {res.success}. Details: {res.data[:400]}"
                    fallback_used = res.source == "fallback"
                else:
                    observation = f"Unknown tool '{tool_name}'. Available: fetch_section, check_git_command."
            except Exception as exc:  # noqa: BLE001
                # Tool call failed — log it but do not stall; continue with what we have
                observation = f"Tool execution error: {exc}"

            tool_trace.append({
                "step": step + 1,
                "tool": tool_name,
                "arg": tool_arg,
                "fallback_used": fallback_used,
                "observation_preview": observation[:120],
            })

            conversation_history += (
                f"\n\nThought/Action (step {step + 1}):\n{response_text}"
                f"\nObservation:\n{observation}\n"
            )
            continue

        # Parse FINAL ANSWER — strip the marker
        final_answer = response_text
        for marker in ("FINAL ANSWER:", "Final Answer:", "FINAL_ANSWER:"):
            if marker in final_answer:
                final_answer = final_answer.split(marker, 1)[1].strip()
                break
        return final_answer, tool_trace, tokens_accumulated

    # Max iterations reached — return whatever we have
    return response_text, tool_trace, tokens_accumulated


def drafter_a_agent(state: AgentState) -> AgentState:
    """Agent 6: Drafter A (Practical Specialist) - Step-by-step answer with exact commands."""
    if not state.get("is_safe", True) or state.get("is_out_of_scope", False):
        return state

    t0 = time.time()
    style = """Provide a clear, practical, step-by-step troubleshooting response with concrete commands.
Cite exact sections like [section_id] for every step."""

    draft, tool_calls, tokens = _execute_react_loop(
        question=state["sanitized_question"],
        context_pack=state.get("context_pack", []),
        model_role="drafter_a",
        drafter_name="Drafter A (Practical)",
        style_instruction=style,
        max_tool_calls=3,
    )

    trace: AgentTraceEntry = {
        "agent": "drafter_a",
        "action": "draft_practical",
        "duration_ms": round((time.time() - t0) * 1000, 2),
        "status": "success",
        "details": {"tool_calls_count": len(tool_calls), "draft_len": len(draft)},
    }

    return {
        **state,
        "draft_a": draft,
        "draft_a_tool_calls": tool_calls,
        "tokens_used": state.get("tokens_used", 0) + tokens,
        "traces": state.get("traces", []) + [trace],
    }


def drafter_b_agent(state: AgentState) -> AgentState:
    """Agent 7: Drafter B (Strict / Source-Only Specialist) - Concise, exact source-derived answer."""
    if not state.get("is_safe", True) or state.get("is_out_of_scope", False):
        return state

    t0 = time.time()
    style = """Provide a strictly grounded, concise, source-only answer.
Do not assume or generalize. Cite the exact section [section_id] for every claim."""

    draft, tool_calls, tokens = _execute_react_loop(
        question=state["sanitized_question"],
        context_pack=state.get("context_pack", []),
        model_role="drafter_b",
        drafter_name="Drafter B (Strict)",
        style_instruction=style,
        max_tool_calls=3,
    )

    trace: AgentTraceEntry = {
        "agent": "drafter_b",
        "action": "draft_strict",
        "duration_ms": round((time.time() - t0) * 1000, 2),
        "status": "success",
        "details": {"tool_calls_count": len(tool_calls), "draft_len": len(draft)},
    }

    return {
        **state,
        "draft_b": draft,
        "draft_b_tool_calls": tool_calls,
        "tokens_used": state.get("tokens_used", 0) + tokens,
        "traces": state.get("traces", []) + [trace],
    }


def run_batch_2(state: AgentState) -> AgentState:
    """Execute Batch 2 (Agents 5-7): Reranker, Drafter A, and Drafter B."""
    state = reranker_agent(state)
    state = drafter_a_agent(state)
    state = drafter_b_agent(state)
    return state
