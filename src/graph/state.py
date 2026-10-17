from __future__ import annotations

from typing import Any, TypedDict


class AgentTraceEntry(TypedDict, total=False):
    agent: str
    action: str
    duration_ms: float
    model_id: str | None
    fallback_used: bool
    status: str
    details: Any


class AgentState(TypedDict, total=False):
    # User Request
    question: str
    sanitized_question: str
    is_safe: bool

    # Batch 1 Data: Intake & Retrieval
    classification: str  # concept, how-to, troubleshooting, out-of-scope
    search_queries: list[str]
    is_out_of_scope: bool
    dense_chunks: list[dict[str, Any]]
    keyword_chunks: list[dict[str, Any]]

    # Batch 2 Data: Context Pack & ReAct Drafting
    context_pack: list[dict[str, Any]]
    context_score: float
    draft_a: str
    draft_b: str
    revision_round: int
    draft_a_tool_calls: list[dict[str, Any]]
    draft_b_tool_calls: list[dict[str, Any]]

    # Batch 3 Data: Critique, Arbitration & Safety
    critique_a: list[dict[str, Any]]  # [{"claim": str, "supported": bool, "reason": str}]
    critique_b: list[dict[str, Any]]
    supported_claim_ratio: float
    verified_answer: str
    abstained: bool
    risk_level: str  # LOW, MEDIUM, HIGH
    safer_alternative: str | None
    final_response: str

    # Global Observability & Trace
    request_id: str
    start_time: float
    total_duration_ms: float
    tokens_used: int
    traces: list[AgentTraceEntry]
    error: str | None
