from __future__ import annotations

import json
import re
import time
import uuid
from typing import Any

from src.agents.llm_client import call_llm
from src.graph.state import AgentState, AgentTraceEntry
from src.security.sanitizer import detect_prompt_injection, sanitize_input
from src.tools.search_tool import _get_bm25, _get_chunks


def manager_intake(state: AgentState) -> AgentState:
    """Agent 1: Manager - Initializes the request, validates security, and starts trace."""
    t0 = time.time()
    raw_question = state.get("question", "")
    request_id = state.get("request_id") or f"req-{uuid.uuid4().hex[:8]}"

    # Security check for prompt injection
    if detect_prompt_injection(raw_question):
        trace: AgentTraceEntry = {
            "agent": "manager",
            "action": "security_check",
            "duration_ms": round((time.time() - t0) * 1000, 2),
            "status": "blocked",
            "details": "Prompt injection pattern detected",
        }
        return {
            **state,
            "request_id": request_id,
            "is_safe": False,
            "final_response": "Request blocked by GitGuard security policy: detected disallowed prompt injection instructions.",
            "abstained": True,
            "traces": state.get("traces", []) + [trace],
        }

    clean_q = sanitize_input(raw_question)
    trace: AgentTraceEntry = {
        "agent": "manager",
        "action": "intake_initialized",
        "duration_ms": round((time.time() - t0) * 1000, 2),
        "status": "success",
        "details": {"clean_length": len(clean_q)},
    }
    return {
        **state,
        "request_id": request_id,
        "sanitized_question": clean_q,
        "is_safe": True,
        "start_time": state.get("start_time") or t0,
        "traces": state.get("traces", []) + [trace],
    }


def query_analyst(state: AgentState) -> AgentState:
    """Agent 2: Query Analyst - Classifies intent and produces 1-3 targeted search queries."""
    if not state.get("is_safe", True):
        return state

    t0 = time.time()
    question = state["sanitized_question"]

    system_prompt = """You are the GitGuard Query Analyst.
Analyze the user's Git query.
Output JSON only with keys:
- "classification": one of ["concept", "how-to", "troubleshooting", "out-of-scope"]
- "is_out_of_scope": boolean (true if the query is unrelated to Git or the Pro Git book, e.g. general Python coding, recipes, cloud billing, non-Git topics)
- "search_queries": list of 1 to 3 search queries optimized for finding relevant sections in the Pro Git book. Include exact Git command flags if present."""

    user_prompt = f"User Question: {question}"

    content, meta = call_llm(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        model_role="query_router",
        temperature=0.0,
        max_tokens=256,
        json_mode=True,
    )

    try:
        parsed = json.loads(content)
        classification = parsed.get("classification", "concept")
        is_out_of_scope = bool(parsed.get("is_out_of_scope", False))
        queries = parsed.get("search_queries") or [question]
    except Exception:
        classification = "concept"
        is_out_of_scope = False
        queries = [question]

    trace: AgentTraceEntry = {
        "agent": "query_analyst",
        "action": "classified_query",
        "duration_ms": round((time.time() - t0) * 1000, 2),
        "model_id": meta.get("model"),
        "status": "success",
        "details": {
            "classification": classification,
            "is_out_of_scope": is_out_of_scope,
            "queries": queries,
        },
    }

    res_state = {
        **state,
        "classification": classification,
        "is_out_of_scope": is_out_of_scope,
        "search_queries": queries,
        "tokens_used": state.get("tokens_used", 0) + meta.get("tokens", 0),
        "traces": state.get("traces", []) + [trace],
    }
    if is_out_of_scope:
        res_state["abstained"] = True
        res_state["final_response"] = "Not found in the book."
        res_state["verified_answer"] = "Not found in the book."
        res_state["risk_level"] = "LOW"

    return res_state


def dense_retriever(state: AgentState) -> AgentState:
    """Agent 3: Dense Retriever - Multi-query meaning-based search using ChromaDB (top 20).

    Runs each analyst-generated query independently, then deduplicates by chunk_id and
    returns the union top-20 ranked by best-scoring appearance. This significantly improves
    recall when a single query misses the relevant section embeddings.
    """
    if not state.get("is_safe", True) or state.get("is_out_of_scope", False):
        return state

    t0 = time.time()
    queries = state.get("search_queries", []) or [state.get("sanitized_question", "")]

    from src.tools.search_tool import search_index

    # Collect results across all queries; deduplicate by chunk_id keeping best position
    seen: dict[str, dict] = {}      # chunk_id -> chunk dict
    position_map: dict[str, int] = {}  # chunk_id -> best position (lower = better)
    fallback_used = False
    global_pos = 0

    for query in queries:
        res = search_index(query, k=20)
        if res.source == "fallback":
            fallback_used = True
        if not res.success:
            continue
        for rank, chunk in enumerate(res.data):
            cid = chunk.get("chunk_id", "")
            if not cid:
                continue
            if cid not in seen or (global_pos + rank) < position_map[cid]:
                seen[cid] = chunk
                position_map[cid] = global_pos + rank
        global_pos += len(res.data)

    # Sort by best-position score; cap at 20
    sorted_chunks = sorted(seen.values(), key=lambda c: position_map[c["chunk_id"]])[:20]

    trace: AgentTraceEntry = {
        "agent": "dense_retriever",
        "action": "multi_query_dense_search",
        "duration_ms": round((time.time() - t0) * 1000, 2),
        "fallback_used": fallback_used,
        "status": "success",
        "details": {
            "queries_used": len(queries),
            "chunks_retrieved": len(sorted_chunks),
            "source": "fallback" if fallback_used else "chromadb",
        },
    }

    return {
        **state,
        "dense_chunks": sorted_chunks,
        "traces": state.get("traces", []) + [trace],
    }


def keyword_retriever(state: AgentState) -> AgentState:
    """Agent 4: Keyword Retriever - BM25 lexical search for exact commands and flags (top 20)."""
    if not state.get("is_safe", True) or state.get("is_out_of_scope", False):
        return state

    t0 = time.time()
    query = state.get("sanitized_question", "")

    chunks = _get_chunks()
    bm25_data = _get_bm25()

    scores = bm25_data["bm25"].get_scores(re.findall(r"\w+", query.lower()))
    order = scores.argsort()[::-1][:20]
    keyword_chunks = [{**chunks[bm25_data["ids"][i]], "bm25_score": float(scores[i])} for i in order if scores[i] > 0]

    trace: AgentTraceEntry = {
        "agent": "keyword_retriever",
        "action": "bm25_search",
        "duration_ms": round((time.time() - t0) * 1000, 2),
        "status": "success",
        "details": {"chunks_retrieved": len(keyword_chunks)},
    }

    return {
        **state,
        "keyword_chunks": keyword_chunks,
        "traces": state.get("traces", []) + [trace],
    }


def run_batch_1(question: str) -> AgentState:
    """Execute Batch 1 (Agents 1-4) sequentially and return the verified context."""
    state: AgentState = {"question": question}
    state = manager_intake(state)
    state = query_analyst(state)
    state = dense_retriever(state)
    state = keyword_retriever(state)
    return state
