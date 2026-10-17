"""GitGuard Streamlit User Interface (PRD Day 5 Task 2).

Features:
- Live Q&A with Question Box & Quick Presets
- Color-coded Risk Badges (LOW, MEDIUM, HIGH)
- HIGH risk answers display the safety warning banner first
- Clickable / Interactive Book Citations showing full Pro Git excerpts
- Expandable Multi-Agent Execution Trace with agent latency and tokens
- Evaluation Dashboard: Experiments 1-5 metrics, comparison table, charts, failure analysis
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

# Ensure project root in sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.eval.judge import extract_citations
from src.graph.graph import run_pipeline
from src.security.sanitizer import detect_prompt_injection

st.set_page_config(
    page_title="GitGuard — Multi-Agent Pro Git RAG",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data
def load_sections_lookup() -> dict[str, dict[str, Any]]:
    path = Path("data/processed/sections.jsonl")
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as f:
        return {r["section_id"]: r for line in f if (r := json.loads(line))}


@st.cache_data
def load_comparison_data() -> tuple[pd.DataFrame | None, dict | None]:
    csv_path = Path("data/eval/experiment5_comparison.csv")
    json_path = Path("data/eval/experiment5_comparison.json")
    df = pd.read_csv(csv_path) if csv_path.exists() else None
    meta = json.loads(json_path.read_text(encoding="utf-8")) if json_path.exists() else None
    return df, meta


@st.cache_data
def load_failure_analysis() -> str:
    path = Path("data/eval/failure_analysis.md")
    return path.read_text(encoding="utf-8") if path.exists() else "No failure analysis available."


@st.cache_data
def load_retrieval_results() -> pd.DataFrame | None:
    path = Path("data/eval/retrieval_results.csv")
    return pd.read_csv(path) if path.exists() else None


sections_lookup = load_sections_lookup()

# ─── Sidebar Navigation ───────────────────────────────────────────────────────
st.sidebar.image("https://git-scm.com/images/logos/downloads/Git-Icon-1788C.png", width=64)
st.sidebar.title("GitGuard")
st.sidebar.caption("Autonomous 10-Agent RAG with Safety Rails")
st.sidebar.markdown("---")

app_mode = st.sidebar.radio(
    "Navigation",
    ["🛡️ Live Assistant", "📊 Benchmark & Evaluation Dashboard"],
    index=0,
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    **Architecture Highlights**:
    - 10 Collaborative Agents (LangGraph)
    - Groq LLM API (Zero Cost)
    - MCP Filesystem & Git Tooling
    - Verified Citations & Safety Guard
    """
)

# ─── PAGE 1: LIVE ASSISTANT ──────────────────────────────────────────────────
if app_mode == "🛡️ Live Assistant":
    st.title("🛡️ GitGuard: Pro Git Assistant")
    st.caption("Answers Git questions strictly grounded in the Pro Git book with automated safety enforcement.")

    # Preset Questions
    col_pre1, col_pre2, col_pre3 = st.columns(3)
    preset_q = None
    if col_pre1.button("📘 Normal: Post-Rewrite Hook"):
        preset_q = "Which Git command triggers the post-rewrite hook, but git filter-branch does not?"
    if col_pre2.button("⚠️ Risky: Force Push"):
        preset_q = "How do I force push my branch to the remote repository?"
    if col_pre3.button("🚫 Out-of-Scope: Copilot"):
        preset_q = "What is the GitHub Copilot monthly price?"

    # Input Form
    with st.form("query_form"):
        user_question = st.text_input(
            "Ask a Git question:",
            value=preset_q or "",
            placeholder="e.g. How does git restore differ from git reset?",
        )
        submitted = st.form_submit_button("Ask GitGuard", type="primary")

    if submitted and user_question.strip():
        q_text = user_question.strip()

        # Security pre-check
        if detect_prompt_injection(q_text):
            st.error("🚨 Potential prompt injection detected. Request blocked by GitGuard security policy.")
            st.stop()

        with st.spinner("Executing 10-agent reasoning & retrieval pipeline..."):
            t0 = time.time()
            state = run_pipeline(q_text)
            latency_ms = round((time.time() - t0) * 1000, 2)

        ans_text = state.get("final_response") or state.get("verified_answer") or ""
        risk_level = state.get("risk_level", "LOW")
        abstained = state.get("abstained", False)
        tokens_used = state.get("tokens_used", 0)
        traces = state.get("traces", [])
        citations = extract_citations(ans_text)

        st.markdown("### Answer")

        # ── Color-Coded Risk Badge ──
        if risk_level == "HIGH":
            st.error("🚨 **RISK LEVEL: HIGH** — This answer contains potentially destructive commands.")
        elif risk_level == "MEDIUM":
            st.warning("⚠️ **RISK LEVEL: MEDIUM** — Command rewrites history or modifies working tree.")
        else:
            st.success("✅ **RISK LEVEL: LOW** — Safe Git command.")

        # ── Answer Display ──
        if abstained and "Not found in the book" in ans_text:
            st.info(f"ℹ️ **Abstention**: {ans_text}")
        else:
            st.markdown(ans_text)

        # Performance summary pill
        st.caption(f"⏱️ **Latency**: {latency_ms:,.0f} ms | 🪙 **Tokens**: {tokens_used} | 🤖 **Agents Run**: {len(traces)}")

        # ── Interactive Citations Section ──
        if citations:
            st.markdown("---")
            st.subheader("📚 Verified Pro Git Citations")
            st.write(f"This answer cites {len(citations)} section(s) from the Pro Git book:")

            for cid in citations:
                sec_data = sections_lookup.get(cid)
                with st.expander(f"📖 Section: `{cid}`"):
                    if sec_data:
                        st.markdown(f"**Chapter**: {sec_data.get('chapter', 'N/A')}")
                        st.markdown(f"**Heading**: {sec_data.get('heading', 'N/A')} (Level {sec_data.get('heading_level', 'N/A')})")
                        st.markdown(f"**Source File**: `{sec_data.get('source_file', 'N/A')}`")
                        st.markdown("---")
                        st.text_area("Book Excerpt Text", sec_data.get("text", ""), height=180, key=f"sec_{cid}")
                    else:
                        st.warning(f"Section ID `{cid}` not found in the local book index.")

        # ── Multi-Agent Execution Trace ──
        st.markdown("---")
        with st.expander("🔍 Multi-Agent Execution Trace & Timeline"):
            st.write(f"**Request ID**: `{state.get('request_id', 'unknown')}`")
            trace_rows = []
            for t in traces:
                trace_rows.append({
                    "Agent": t.get("agent", ""),
                    "Action": t.get("action", ""),
                    "Status": t.get("status", ""),
                    "Duration (ms)": t.get("duration_ms", 0),
                    "Model": t.get("model_id") or "N/A",
                    "Details": str(t.get("details", {})),
                })
            df_trace = pd.DataFrame(trace_rows)
            st.dataframe(df_trace, use_container_width=True)

# ─── PAGE 2: BENCHMARK & EVALUATION DASHBOARD ─────────────────────────────────
else:
    st.title("📊 Benchmark & Evaluation Dashboard")
    st.caption("Complete results across all 5 benchmark experiments from the PRD.")

    df_comp, meta_comp = load_comparison_data()
    df_retrieval = load_retrieval_results()

    # Tabbed Dashboard
    tab_exp5, tab_retrieval, tab_failures = st.tabs([
        "🏆 Experiment 5: Baseline vs. Pipeline",
        "🔍 Experiments 1-4: Retrieval Benchmarks",
        "⚠️ Failure Review & Root Causes",
    ])

    with tab_exp5:
        st.subheader("Experiment 5: Single-Pass Baseline vs. 10-Agent Pipeline")
        st.write("Evaluated across the locked 72 benchmark questions (60 answerable, 12 unanswerable):")

        if df_comp is not None:
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Pipeline Faithfulness", "1.000", "0.0% diff")
            col2.metric("Pipeline Refusal Rate (Unanswerables)", "100%", "0.0% diff")
            col3.metric("Baseline Correctness", "0.958", "")
            col4.metric("Pipeline Correctness", "0.438", "-54.4%")

            st.dataframe(df_comp, use_container_width=True)

            # Chart Display
            html_chart_path = Path("data/eval/experiment5_comparison.html")
            if html_chart_path.exists():
                st.components.v1.html(html_chart_path.read_text(encoding="utf-8"), height=500)
        else:
            st.warning("Experiment 5 comparison data not found.")

    with tab_retrieval:
        st.subheader("Experiments 1-4: Chunking, Embeddings, Hybrid Modes, & Reranking")
        if df_retrieval is not None:
            st.dataframe(df_retrieval, use_container_width=True)
        else:
            st.warning("Retrieval results not found.")

    with tab_failures:
        st.subheader("PRD Task 8: Failure Review & Remediation Strategy")
        st.markdown(load_failure_analysis())
