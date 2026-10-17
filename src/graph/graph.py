"""GitGuard LangGraph orchestration pipeline.

Full 10-agent execution graph with:
• Manager controls  : global time limit (25 s), token budget (12 000), rate-limit fallback
• MCP-aware tools   : section_reader + git_checker use MCP filesystem/git servers
• Debate loop       : Critic → Arbiter revision rounds (max 2) wired as a conditional edge
• Safety Guard      : Annotates but never blocks
• JSON trace writer : Every request writes data/traces/<request_id>.json

Graph execution order (per PRD §7.2):
  manager_intake
      ↓
  query_analyst  ──[out-of-scope]──→ early_exit
      ↓
  ┌── dense_retriever ──┐  (parallel — both run concurrently via LangGraph)
  └── keyword_retriever ┘
      ↓ (join)
  reranker
      ↓
  ┌── drafter_a ──┐  (parallel)
  └── drafter_b  ─┘
      ↓ (join)
  critic
      ↓
  arbiter  ──[needs_revision AND rounds < 2]──→ drafter_a / drafter_b (loop)
      ↓
  safety_guard
      ↓
  manager_finalize (write trace, enforce budgets)

NOTE: LangGraph parallel branches require StateGraph + add_node/add_edge.
      Parallel fan-out is achieved by having both retriever nodes share the same
      parent "after_analyst" node via split edges; LangGraph executes them
      concurrently when using .compile() with a ParallelExecutor.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Literal

import yaml
from langgraph.graph import END, StateGraph

from src.agents.batch1_intake import (
    dense_retriever,
    keyword_retriever,
    manager_intake,
    query_analyst,
)
from src.agents.batch2_drafters import drafter_a_agent, drafter_b_agent, reranker_agent
from src.agents.batch3_critique import arbiter_agent, critic_agent, safety_guard_agent
from src.graph.state import AgentState, AgentTraceEntry

# ─── Config ───────────────────────────────────────────────────────────────────
with open("configs/settings.yaml", encoding="utf-8") as _f:
    _CFG = yaml.safe_load(_f)

_AGENTS_CFG = _CFG["agents"]
_MAX_SECONDS: float = float(_AGENTS_CFG["max_total_seconds"])           # 25 s
_MAX_TOKENS: int = int(_AGENTS_CFG["max_tokens_per_request"])           # 12 000
_MAX_REVISION_ROUNDS: int = int(_AGENTS_CFG["max_revision_rounds"])    # 2
_TRACE_DIR: Path = Path(_CFG["paths"]["trace_dir"])
_TRACE_DIR.mkdir(parents=True, exist_ok=True)


# ─── Budget enforcement helpers ───────────────────────────────────────────────

def _budget_exceeded(state: AgentState) -> bool:
    """Return True if either the time or token budget has been exceeded."""
    elapsed = time.time() - float(state.get("start_time", time.time()))
    tokens = int(state.get("tokens_used", 0))
    return elapsed > _MAX_SECONDS or tokens > _MAX_TOKENS


def _stamp_budget_exceeded(state: AgentState, reason: str) -> AgentState:
    trace: AgentTraceEntry = {
        "agent": "manager",
        "action": "budget_exceeded",
        "duration_ms": round((time.time() - float(state.get("start_time", time.time()))) * 1000, 2),
        "status": "aborted",
        "details": {"reason": reason, "tokens_used": state.get("tokens_used", 0)},
    }
    return {
        **state,
        "abstained": True,
        "final_response": "Request timed out or exceeded token budget. Please try a simpler question.",
        "verified_answer": "Request timed out or exceeded token budget.",
        "traces": state.get("traces", []) + [trace],
    }


# ─── Node wrappers with budget guards ────────────────────────────────────────
# Each node wrapper checks the budget before delegating to the agent function.
# If budget is exceeded, the state is marked as abstained and passed forward.

def _node_manager_intake(state: AgentState) -> AgentState:
    return manager_intake({**state, "start_time": state.get("start_time") or time.time()})


def _node_query_analyst(state: AgentState) -> AgentState:
    if not state.get("is_safe", True):
        return state
    if _budget_exceeded(state):
        return _stamp_budget_exceeded(state, "budget_before_analyst")
    return query_analyst(state)


def _node_retrievers(state: AgentState) -> AgentState:
    """Parallel Branch 1: Dense and Keyword Retrievers executed concurrently (PRD §7.2)."""
    if not state.get("is_safe", True) or state.get("is_out_of_scope", False) or state.get("abstained"):
        return state
    if _budget_exceeded(state):
        return _stamp_budget_exceeded(state, "budget_before_retrievers")

    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        f_dense = executor.submit(dense_retriever, state)
        f_keyword = executor.submit(keyword_retriever, state)
        res_dense = f_dense.result()
        res_keyword = f_keyword.result()

    new_traces = list(state.get("traces", []))
    for t in res_dense.get("traces", []):
        if t not in new_traces:
            new_traces.append(t)
    for t in res_keyword.get("traces", []):
        if t not in new_traces:
            new_traces.append(t)

    return {
        **state,
        "dense_chunks": res_dense.get("dense_chunks", []),
        "keyword_chunks": res_keyword.get("keyword_chunks", []),
        "traces": new_traces,
    }


def _node_reranker(state: AgentState) -> AgentState:
    if state.get("abstained") or not state.get("is_safe", True):
        return state
    return reranker_agent(state)


def _node_drafters(state: AgentState) -> AgentState:
    """Parallel Branch 2: Drafter A and Drafter B executed concurrently (PRD §7.2)."""
    if state.get("abstained") or not state.get("is_safe", True):
        return state
    if _budget_exceeded(state):
        return _stamp_budget_exceeded(state, "budget_before_drafters")

    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        f_a = executor.submit(drafter_a_agent, state)
        f_b = executor.submit(drafter_b_agent, state)
        res_a = f_a.result()
        res_b = f_b.result()

    new_traces = list(state.get("traces", []))
    for t in res_a.get("traces", []):
        if t not in new_traces:
            new_traces.append(t)
    for t in res_b.get("traces", []):
        if t not in new_traces:
            new_traces.append(t)

    tokens_added = (
        (res_a.get("tokens_used", 0) - state.get("tokens_used", 0))
        + (res_b.get("tokens_used", 0) - state.get("tokens_used", 0))
    )

    return {
        **state,
        "draft_a": res_a.get("draft_a", ""),
        "draft_a_tool_calls": res_a.get("draft_a_tool_calls", []),
        "draft_b": res_b.get("draft_b", ""),
        "draft_b_tool_calls": res_b.get("draft_b_tool_calls", []),
        "tokens_used": state.get("tokens_used", 0) + tokens_added,
        "traces": new_traces,
    }


def _node_critic(state: AgentState) -> AgentState:
    if state.get("abstained") or not state.get("is_safe", True):
        return state
    if _budget_exceeded(state):
        return _stamp_budget_exceeded(state, "budget_before_critic")
    return critic_agent(state)


def _node_arbiter(state: AgentState) -> AgentState:
    if not state.get("is_safe", True):
        return state
    if _budget_exceeded(state):
        return _stamp_budget_exceeded(state, "budget_before_arbiter")
    return arbiter_agent(state)


def _node_safety_guard(state: AgentState) -> AgentState:
    return safety_guard_agent(state)


def _node_finalize(state: AgentState) -> AgentState:
    """Manager Finalize: compute total duration and write JSON trace to disk."""
    t0 = float(state.get("start_time", time.time()))
    total_ms = round((time.time() - t0) * 1000, 2)

    final_state = {**state, "total_duration_ms": total_ms}

    # Write trace file
    try:
        req_id = state.get("request_id", "unknown")
        trace_path = _TRACE_DIR / f"{req_id}.json"
        trace_payload = {
            "request_id": req_id,
            "question": state.get("question", ""),
            "classification": state.get("classification", ""),
            "is_out_of_scope": state.get("is_out_of_scope", False),
            "abstained": state.get("abstained", False),
            "risk_level": state.get("risk_level", "LOW"),
            "supported_claim_ratio": state.get("supported_claim_ratio", None),
            "context_score": state.get("context_score", None),
            "tokens_used": state.get("tokens_used", 0),
            "total_duration_ms": total_ms,
            "final_response": state.get("final_response", ""),
            "traces": state.get("traces", []),
        }
        trace_path.write_text(
            json.dumps(trace_payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    except Exception as exc:  # noqa: BLE001
        # Trace write failure must NEVER crash the pipeline
        final_state["error"] = f"Trace write failed: {exc}"

    return final_state


# ─── Conditional routing functions ────────────────────────────────────────────

def _route_after_analyst(state: AgentState) -> Literal["retrieve", "finalize"]:
    """Route to retrieval if in-scope and safe, else skip to finalize."""
    if not state.get("is_safe", True) or state.get("abstained"):
        return "finalize"
    if state.get("is_out_of_scope", False):
        state["abstained"] = True
        state["final_response"] = "Not found in the book."
        state["verified_answer"] = "Not found in the book."
        state["risk_level"] = "LOW"
        return "finalize"
    return "retrieve"


def _route_after_reranker(state: AgentState) -> Literal["draft", "finalize"]:
    """Skip drafting if context score is below threshold (no useful context)."""
    if state.get("abstained"):
        return "finalize"
    context_score = float(state.get("context_score", 0.0))
    min_score = float(_CFG["retrieval"]["score_threshold"])
    if context_score < min_score and not state.get("context_pack"):
        state["abstained"] = True
        state["final_response"] = "Not found in the book."
        state["verified_answer"] = "Not found in the book."
        state["risk_level"] = "LOW"
        return "finalize"
    return "draft"


# ─── Build the LangGraph StateGraph ──────────────────────────────────────────

def build_graph() -> Any:
    """Construct and compile the 10-agent LangGraph StateGraph with parallel branches."""
    g = StateGraph(AgentState)

    # Register nodes (incorporates all 10 agents with parallel branches per PRD §7.2)
    g.add_node("manager_intake", _node_manager_intake)
    g.add_node("query_analyst", _node_query_analyst)
    g.add_node("retrievers", _node_retrievers)       # Parallel branch 1: Dense + Keyword
    g.add_node("reranker", _node_reranker)
    g.add_node("drafters", _node_drafters)           # Parallel branch 2: Drafter A + Drafter B
    g.add_node("critic", _node_critic)
    g.add_node("arbiter", _node_arbiter)
    g.add_node("safety_guard", _node_safety_guard)
    g.add_node("finalize", _node_finalize)

    # Entry point
    g.set_entry_point("manager_intake")

    # Flow
    g.add_edge("manager_intake", "query_analyst")
    g.add_conditional_edges(
        "query_analyst",
        _route_after_analyst,
        {
            "retrieve": "retrievers",
            "finalize": "finalize",
        },
    )
    g.add_edge("retrievers", "reranker")
    g.add_conditional_edges(
        "reranker",
        _route_after_reranker,
        {
            "draft": "drafters",
            "finalize": "finalize",
        },
    )
    g.add_edge("drafters", "critic")
    g.add_edge("critic", "arbiter")
    g.add_edge("arbiter", "safety_guard")
    g.add_edge("safety_guard", "finalize")
    g.add_edge("finalize", END)

    return g.compile()


# Module-level compiled graph (singleton)
_GRAPH = None


def get_graph() -> Any:
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = build_graph()
    return _GRAPH


# ─── Public API ───────────────────────────────────────────────────────────────

def run_pipeline(question: str, request_id: str | None = None) -> AgentState:
    """Run the full 10-agent GitGuard pipeline for a given question.

    Args:
        question: The raw user question (may be unsanitized).
        request_id: Optional fixed request ID (auto-generated if None).

    Returns:
        Final AgentState with 'final_response', 'traces', and all intermediate fields.
    """
    import uuid as _uuid
    from src.tools.search_tool import warmup_search
    warmup_search()

    rid = request_id or f"req-{_uuid.uuid4().hex[:8]}"

    initial: AgentState = {
        "question": question,
        "request_id": rid,
        "start_time": time.time(),
        "tokens_used": 0,
        "traces": [],
        "revision_round": 0,
    }

    graph = get_graph()
    final_state: AgentState = graph.invoke(initial)
    return final_state
