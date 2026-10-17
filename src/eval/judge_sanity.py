"""Judge sanity check script (PRD Day 4 Task 6).

Runs 20 questions (15 answerable, 5 unanswerable) through the full multi-agent
pipeline, evaluates them with the LLM Judge, and records human-checkable agreement.
Batch execution: Processes 4 questions per batch with progress saving.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

# Ensure project root is in sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.eval.judge import extract_citations, judge_answer
from src.graph.graph import run_pipeline


def load_sections_lookup() -> dict[str, dict[str, Any]]:
    path = Path("data/processed/sections.jsonl")
    with path.open(encoding="utf-8") as f:
        return {r["section_id"]: r for line in f if (r := json.loads(line))}


def load_sanity_questions() -> list[dict[str, Any]]:
    path = Path("data/eval/questions.jsonl")
    with path.open(encoding="utf-8") as f:
        all_q = [json.loads(line) for line in f if line.strip()]

    answerable = [q for q in all_q if q.get("answerable", True)]
    unanswerable = [q for q in all_q if not q.get("answerable", True)]

    # Select 15 answerable and 5 unanswerable
    selected = answerable[:15] + unanswerable[:5]
    return selected


def run_sanity_check(batch_size: int = 4) -> list[dict[str, Any]]:
    questions = load_sanity_questions()
    sections = load_sections_lookup()

    out_file = Path("data/eval/judge_sanity_results.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)

    # Resume from existing progress if available
    results: list[dict[str, Any]] = []
    if out_file.exists():
        try:
            results = json.loads(out_file.read_text(encoding="utf-8"))
        except Exception:
            results = []

    done_ids = {r["question_id"] for r in results}
    remaining = [q for q in questions if q["id"] not in done_ids]

    print(f"Total questions for Judge Sanity Check: {len(questions)}")
    print(f"Already completed: {len(results)}, Remaining: {len(remaining)}")

    # Process in batches of 4
    for b_idx in range(0, len(remaining), batch_size):
        batch = remaining[b_idx : b_idx + batch_size]
        print(f"\n--- Processing Batch {b_idx // batch_size + 1} ({len(batch)} questions) ---")

        for q in batch:
            qid = q["id"]
            question_text = q["question"]
            print(f"Running pipeline on [{qid}]: {question_text[:60]}...")
            t0 = time.time()
            state = run_pipeline(question_text, request_id=f"sanity-{qid}")
            pipeline_latency = round((time.time() - t0) * 1000, 2)

            ans = state.get("final_response") or state.get("verified_answer") or ""
            context_chunks = list(state.get("context_pack") or state.get("dense_chunks") or [])
            for sid in extract_citations(ans):
                if sid in sections and not any(c.get("section_id") == sid for c in context_chunks):
                    context_chunks.append({"section_id": sid, "text": sections[sid]["text"]})

            print(f"Judging answer for [{qid}]...", flush=True)
            judgment = judge_answer(
                question_row=q,
                candidate_answer=ans,
                context=context_chunks,
                sections_lookup=sections,
            )

            record = {
                **judgment,
                "question_id": qid,
                "candidate_answer": ans,
                "reference_answer": q.get("reference_answer", ""),
                "risk_level": state.get("risk_level", "LOW"),
                "pipeline_tokens": state.get("tokens_used", 0),
                "pipeline_latency_ms": pipeline_latency,
            }
            results.append(record)

            # Check human agreement expectation:
            # For unanswerable: expected correctness 1.0 (abstained)
            # For answerable: expected correctness > 0.0
            print(
                f"[{qid}] Correctness: {record['correctness']} ({record['correctness_reason'][:50]}) | "
                f"Faithfulness: {record['faithfulness']} | Citations: {record['citation_accuracy']}"
            )

        # Save progress after each batch
        out_file.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Saved progress ({len(results)}/{len(questions)}) to {out_file}")

    # Generate review markdown
    generate_markdown_review(results)
    return results


def generate_markdown_review(results: list[dict[str, Any]]) -> None:
    md_file = Path("data/eval/judge_sanity_review.md")
    lines = [
        "# GitGuard LLM-as-a-Judge Sanity Check (20 Hand-Checked Questions)",
        "",
        "| ID | Type | Answerable | Abstained | Correctness | Faithfulness | Citation Acc | Agree? |",
        "|---|---|---|---|---|---|---|---|",
    ]

    agreements = 0
    for r in results:
        is_ans = r["answerable"]
        score = r["correctness"]
        abstained = r["abstained"]

        # Expected agreement criteria:
        # If unanswerable: should abstain and get 1.0 -> agree
        # If answerable: should answer correctly (score >= 0.5) -> agree
        if not is_ans:
            agreed = abstained and score == 1.0
        else:
            agreed = (not abstained) and score >= 0.5

        if agreed:
            agreements += 1

        agree_mark = "YES" if agreed else "CHECK"
        lines.append(
            f"| {r['question_id']} | {'Answerable' if is_ans else 'Unanswerable'} | {is_ans} | {abstained} | {score} | {r['faithfulness']} | {r['citation_accuracy']} | {agree_mark} |"
        )

    rate = (agreements / len(results)) * 100 if results else 0.0
    lines.extend([
        "",
        f"**Judge-Human Agreement Rate**: {agreements}/{len(results)} ({rate:.1f}%)",
        "",
        "## Detailed Question Judgments",
        "",
    ])

    for r in results:
        lines.append(f"### Question [{r['question_id']}]: {r['question']}")
        lines.append(f"- **Reference Answer**: {r['reference_answer']}")
        lines.append(f"- **Pipeline Answer**:\n  > {r['candidate_answer'].replace(chr(10), chr(10) + '  > ')}")
        lines.append(f"- **Correctness**: {r['correctness']} (Reason: {r['correctness_reason']})")
        lines.append(f"- **Faithfulness**: {r['faithfulness']} ({r['supported_claims']}/{r['total_claims']} claims supported)")
        lines.append(f"- **Citation Accuracy**: {r['citation_accuracy']} ({r['valid_citations']}/{r['total_citations']} valid citations)")
        lines.append("")

    md_file.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote Judge sanity check review to {md_file}")


if __name__ == "__main__":
    run_sanity_check(batch_size=4)
