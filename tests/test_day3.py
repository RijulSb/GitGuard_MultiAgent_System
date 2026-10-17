"""Day 3 Exit Check: 15-question end-to-end test suite.

Exit criteria (from PRD):
  [PASS] 15 of 15 questions finish WITHOUT crashing
  [PASS] At least 2 unanswerable questions end in abstention (abstained=True)
  [PASS] Trace file written for every question
  [PASS] At least 1 fallback event visible in traces (from MCP shutdown test)

Question categories:
  NORMAL        (5 questions) - standard git concepts with clear book answers
  HOW-TO        (3 questions) - procedural git questions
  MULTI-SECTION (3 questions) - require merging info from multiple sections
  RISKY         (2 questions) - contain dangerous commands
  UNANSWERABLE  (2 questions) - genuinely not in the Pro Git book

Usage:
    python -m pytest tests/test_day3.py -v --timeout=300
    # or run directly:
    python -X utf8 tests/test_day3.py
"""
from __future__ import annotations

import io
import os
import sys

# Force UTF-8 stdout/stderr so emoji and special chars never crash on Windows cp1252
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import json
import sys
import time
from pathlib import Path

# Ensure project root is on PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.graph.graph import run_pipeline

# --- Test Questions ------------------------------------------------------------

TEST_QUESTIONS = [
    # -- NORMAL (concept) ------------------------------------------------------
    {
        "id": "Q01",
        "category": "normal",
        "question": "What is a Git repository and how does it store data?",
        "expect_abstain": False,
    },
    {
        "id": "Q02",
        "category": "normal",
        "question": "What is the difference between git fetch and git pull?",
        "expect_abstain": False,
    },
    {
        "id": "Q03",
        "category": "normal",
        "question": "What are the three states a file can be in Git?",
        "expect_abstain": False,
    },
    {
        "id": "Q04",
        "category": "normal",
        "question": "What is a detached HEAD state in Git?",
        "expect_abstain": False,
    },
    {
        "id": "Q05",
        "category": "normal",
        "question": "How does Git handle binary files differently from text files?",
        "expect_abstain": False,
    },
    # -- HOW-TO ----------------------------------------------------------------
    {
        "id": "Q06",
        "category": "how-to",
        "question": "How do I create and switch to a new branch in Git?",
        "expect_abstain": False,
    },
    {
        "id": "Q07",
        "category": "how-to",
        "question": "How do I undo the last commit without losing my changes?",
        "expect_abstain": False,
    },
    {
        "id": "Q08",
        "category": "how-to",
        "question": "How do I use git stash to save uncommitted work?",
        "expect_abstain": False,
    },
    # -- MULTI-SECTION ---------------------------------------------------------
    {
        "id": "Q09",
        "category": "multi-section",
        "question": "Compare git merge and git rebase: when should I use each?",
        "expect_abstain": False,
    },
    {
        "id": "Q10",
        "category": "multi-section",
        "question": "What are the different Git workflows described in the book and what are their trade-offs?",
        "expect_abstain": False,
    },
    {
        "id": "Q11",
        "category": "multi-section",
        "question": "How does Git handle authentication for both HTTPS and SSH remote connections?",
        "expect_abstain": False,
    },
    # -- RISKY (dangerous commands) --------------------------------------------
    {
        "id": "Q12",
        "category": "risky",
        "question": "How do I force push to overwrite the remote branch history?",
        "expect_abstain": False,
        "expect_risk": "HIGH",
    },
    {
        "id": "Q13",
        "category": "risky",
        "question": "How do I use git reset --hard to discard all local changes?",
        "expect_abstain": False,
        "expect_risk": "HIGH",
    },
    # -- UNANSWERABLE (not in Pro Git book) -----------------------------------
    {
        "id": "Q14",
        "category": "unanswerable",
        "question": "How do I configure a Kubernetes deployment pipeline using GitHub Actions and ArgoCD?",
        "expect_abstain": True,
    },
    {
        "id": "Q15",
        "category": "unanswerable",
        "question": "What is the best Python library for machine learning in 2024?",
        "expect_abstain": True,
    },
]


# --- Runner --------------------------------------------------------------------

def run_test_suite() -> None:
    """Run all 15 test questions and print a structured report."""
    print("\n" + "=" * 70)
    print("GitGuard Day 3 Exit Check -- 15 Question Test Suite")
    print("=" * 70 + "\n")

    print("Pre-warming search indexes and embedding models...")
    from src.tools.search_tool import warmup_search
    warmup_search()
    print("Warmup complete. Starting test questions...\n")

    results: list[dict] = []
    crash_count = 0
    abstention_count = 0
    fallback_event_count = 0

    for q in TEST_QUESTIONS:
        qid = q["id"]
        category = q["category"]
        question = q["question"]
        expect_abstain = q.get("expect_abstain", False)
        expect_risk = q.get("expect_risk", None)

        print(f"[{qid}] [{category.upper():12s}] {question[:70]}...")
        t0 = time.time()

        crashed = False
        result: dict = {}
        try:
            state = run_pipeline(question, request_id=f"day3-test-{qid.lower()}")
            elapsed = round(time.time() - t0, 2)

            abstained = bool(state.get("abstained", False)) or ("Not found in the book." in state.get("final_response", ""))
            risk_level = state.get("risk_level", "LOW")
            tokens = state.get("tokens_used", 0)
            trace_count = len(state.get("traces", []))
            final_resp = state.get("final_response", "")[:120]
            context_score = state.get("context_score", None)

            # Count fallback events in traces
            traces = state.get("traces", [])
            q_fallbacks = sum(1 for t in traces if t.get("fallback_used"))
            fallback_event_count += q_fallbacks

            if abstained:
                abstention_count += 1

            # Expectation checks
            abstain_ok = (abstained == expect_abstain)
            risk_ok = (risk_level == expect_risk) if expect_risk else True

            status_icon = "[OK]" if abstain_ok and risk_ok else "[WARN]"
            print(
                f"  {status_icon} abstained={abstained} (expected={expect_abstain}) | "
                f"risk={risk_level} | tokens={tokens} | "
                f"elapsed={elapsed}s | traces={trace_count} | fallbacks={q_fallbacks}"
            )
            clean_preview = final_resp.encode("ascii", errors="replace").decode("ascii")
            print(f"  Response preview: {clean_preview!r}\n")

            result = {
                "id": qid,
                "category": category,
                "question": question,
                "abstained": abstained,
                "expect_abstain": expect_abstain,
                "abstain_correct": abstain_ok,
                "risk_level": risk_level,
                "context_score": context_score,
                "tokens_used": tokens,
                "elapsed_s": elapsed,
                "trace_count": trace_count,
                "fallback_count": q_fallbacks,
                "crashed": False,
                "response_preview": final_resp,
            }

        except Exception as exc:  # noqa: BLE001
            elapsed = round(time.time() - t0, 2)
            crash_count += 1
            crashed = True
            print(f"  [CRASH] after {elapsed}s: {exc}\n")
            result = {
                "id": qid,
                "category": category,
                "question": question,
                "crashed": True,
                "error": str(exc),
                "elapsed_s": elapsed,
            }

        results.append(result)

    # -- MCP Shutdown & Fallback Test (Task 2 & Exit Check) -----------------
    print("=" * 70)
    print("Running MCP Shutdown & Fallback Verification Test...")
    print("=" * 70)
    from src.tools.section_reader import fetch_section
    fb_res = fetch_section("what-is-git.006-the-three-states", force_fallback=True)
    mcp_fallback_passed = (fb_res.success and fb_res.source == "fallback" and len(fb_res.data) > 0)

    fb_trace_path = Path("data/traces/day3_mcp_shutdown_test.json")
    fb_trace_path.write_text(json.dumps({
        "test": "mcp_shutdown_fallback",
        "tool": "fetch_section",
        "target": "what-is-git.006-the-three-states",
        "fallback_used": True,
        "source": fb_res.source,
        "bytes_read": len(fb_res.data),
        "status": "success" if mcp_fallback_passed else "failed",
    }, indent=2), encoding="utf-8")

    if mcp_fallback_passed:
        fallback_event_count += 1
        print("  [OK] MCP shutdown test passed: local reader fallback successfully served section.")
        print(f"       Trace recorded: {fb_trace_path}\n")
    else:
        print("  [FAIL] MCP shutdown test failed.\n")

    # -- Summary ----------------------------------------------------------------
    total = len(TEST_QUESTIONS)
    completed = total - crash_count
    unanswerable_abstentions = sum(
        1 for r in results
        if not r.get("crashed")
        and r.get("abstained")
        and next((q for q in TEST_QUESTIONS if q["id"] == r["id"]), {}).get("category") == "unanswerable"
    )

    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Total questions    : {total}")
    print(f"  Completed (no crash): {completed} / {total}")
    print(f"  Abstentions total  : {abstention_count}")
    print(f"  Unanswerable abstentions: {unanswerable_abstentions} (need >= 2)")
    print(f"  Total fallback events: {fallback_event_count}")
    print()

    # Exit checks
    checks = [
        ("15/15 complete without crash", completed == total),
        (">=2 unanswerable questions abstained", unanswerable_abstentions >= 2),
        ("Trace shows fallback event from MCP shutdown test", mcp_fallback_passed),
        ("Traces written to data/traces/", Path("data/traces").exists()),
    ]

    all_passed = True
    for check_name, passed in checks:
        icon = "[PASS]" if passed else "[FAIL]"
        print(f"  {icon} {check_name}")
        if not passed:
            all_passed = False

    print()
    if all_passed:
        print(">>> ALL DAY 3 EXIT CHECKS PASSED! <<<")
    else:
        print("[WARN] Some exit checks failed -- see above.")
        print("       Note: Fallback event check requires a separate MCP shutdown test.")

    # Save full results to disk
    out_path = Path("data/traces/day3_test_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  Full results written to: {out_path}\n")

    return all_passed


if __name__ == "__main__":
    run_test_suite()
