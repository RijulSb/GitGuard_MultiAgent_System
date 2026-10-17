"""FastAPI service for GitGuard Multi-Agent RAG (PRD Day 5 Task 1).

Exposes:
- GET  /health          : Service health check and readiness probe
- POST /api/v1/query    : Execute GitGuard 10-agent pipeline on user query
"""
from __future__ import annotations

import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional

import yaml
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Ensure project root in sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.eval.judge import extract_citations
from src.graph.graph import run_pipeline
from src.security.sanitizer import detect_prompt_injection, sanitize_input
from src.tools.search_tool import warmup_search

# Load settings
with open("configs/settings.yaml", encoding="utf-8") as f:
    _CFG = yaml.safe_load(f)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: warmup embedding models and vector store
    try:
        warmup_search()
    except Exception as exc:
        print(f"Warning during search warmup: {exc}")
    yield


app = FastAPI(
    title="GitGuard Multi-Agent RAG API",
    description="High-precision multi-agent RAG system over the Pro Git book with safety guards.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Pydantic Models ─────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000, description="The Git question to answer.")
    request_id: Optional[str] = Field(None, description="Optional custom request tracking ID.")


class QueryResponse(BaseModel):
    request_id: str
    question: str
    answer: str
    citations: list[str]
    risk_level: str
    abstained: bool
    latency_ms: float
    tokens_used: int
    traces: list[dict[str, Any]]


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str
    uptime_seconds: float
    indexes_ready: bool
    timestamp: float


_START_TIME = time.time()


# ─── Endpoints ───────────────────────────────────────────────────────────────

@app.get("/", tags=["General"])
async def root():
    return {
        "service": "GitGuard Multi-Agent RAG API",
        "version": "1.0.0",
        "documentation": "/docs",
        "health": "/health",
        "query_endpoint": "/api/v1/query",
    }


@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
async def health_check():
    indexes_exist = (
        Path(_CFG["retrieval"]["index_dir"]).exists()
        and Path(_CFG["paths"]["sections_file"]).exists()
    )
    return HealthResponse(
        status="healthy" if indexes_exist else "degraded",
        service="gitguard-rag",
        version="1.0.0",
        uptime_seconds=round(time.time() - _START_TIME, 2),
        indexes_ready=indexes_exist,
        timestamp=time.time(),
    )


@app.post("/api/v1/query", response_model=QueryResponse, tags=["Query"])
async def query_gitguard(req: QueryRequest):
    raw_question = req.question.strip()
    if not raw_question:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Question cannot be empty.")

    # Prompt injection check
    if detect_prompt_injection(raw_question):
        return QueryResponse(
            request_id=req.request_id or f"req-sec-{int(time.time())}",
            question=raw_question,
            answer="Potential prompt injection detected. Request blocked by security policy.",
            citations=[],
            risk_level="HIGH",
            abstained=True,
            latency_ms=0.0,
            tokens_used=0,
            traces=[{
                "agent": "security_sanitizer",
                "action": "block_prompt_injection",
                "duration_ms": 0.0,
                "status": "blocked",
                "details": {"reason": "prompt_injection_pattern_detected"},
            }],
        )

    clean_question = sanitize_input(raw_question)
    t0 = time.time()

    try:
        final_state = run_pipeline(clean_question, request_id=req.request_id)
        latency_ms = round((time.time() - t0) * 1000, 2)

        ans_text = final_state.get("final_response") or final_state.get("verified_answer") or ""
        citations = extract_citations(ans_text)
        risk_level = final_state.get("risk_level", "LOW")
        abstained = bool(final_state.get("abstained", False))
        tokens_used = int(final_state.get("tokens_used", 0))
        traces = list(final_state.get("traces", []))
        rid = str(final_state.get("request_id", req.request_id or "req-unknown"))

        return QueryResponse(
            request_id=rid,
            question=clean_question,
            answer=ans_text,
            citations=citations,
            risk_level=risk_level,
            abstained=abstained,
            latency_ms=latency_ms,
            tokens_used=tokens_used,
            traces=traces,
        )

    except Exception as exc:
        latency_ms = round((time.time() - t0) * 1000, 2)
        # Graceful error response without crashing
        return QueryResponse(
            request_id=req.request_id or f"req-err-{int(time.time())}",
            question=clean_question,
            answer=f"An internal error occurred while processing the request: {exc}",
            citations=[],
            risk_level="LOW",
            abstained=True,
            latency_ms=latency_ms,
            tokens_used=0,
            traces=[{
                "agent": "manager",
                "action": "internal_error",
                "duration_ms": latency_ms,
                "status": "error",
                "details": {"error": str(exc)},
            }],
        )


def main():
    import uvicorn
    host = _CFG.get("service", {}).get("api_host", "0.0.0.0")
    port = int(_CFG.get("service", {}).get("api_port", 8000))
    uvicorn.run("src.service.api:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    main()
