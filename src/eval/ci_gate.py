"""CI Retrieval Regression Gate (PRD Day 5 Task 3).

Evaluates Recall@5 on the 30-question CI benchmark subset without making any LLM calls.
Zero API cost, deterministic, and fast (< 10 seconds).
Fails (exit code 1) if Recall@5 drops below the configured threshold (0.85).
Supports --deliberate-break flag to test failure behavior.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

# Ensure project root in sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.eval.experiments import dense_many, keyword_many, load_chunks
from src.eval.metrics import retrieval_metrics


def run_ci_gate(deliberate_break: bool = False) -> float:
    with open("configs/settings.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    ci_cfg = cfg["evaluation"]["retrieval_regression"]
    min_threshold = float(ci_cfg.get("minimum_score", 0.85))
    dataset_path = Path(cfg["evaluation"].get("ci_subset_path", "data/eval/ci_30_questions.jsonl"))

    if not dataset_path.exists():
        print(f"Error: CI dataset not found at {dataset_path}", file=sys.stderr)
        sys.exit(1)

    with dataset_path.open(encoding="utf-8") as f:
        data = [json.loads(line) for line in f if line.strip()]

    print(f"============================================================")
    print(f"GitGuard CI Retrieval Regression Gate")
    print(f"Dataset: {dataset_path} ({len(data)} questions)")
    print(f"Threshold: Recall@5 >= {min_threshold}")
    print(f"Mode: {'DELIBERATE REGRESSION TEST (--deliberate-break)' if deliberate_break else 'NORMAL GATE'}")
    print(f"============================================================")

    strategy = cfg["chunking"]["default_strategy"]  # parent_linked
    model_key = cfg["retrieval"]["default_embedding"]  # bge_small_en_v1_5
    lookup = load_chunks(cfg, strategy)

    queries = [x["question"] for x in data]

    if deliberate_break:
        # Deliberately degraded retrieval: returns dummy/unrelated chunk IDs
        print("Injecting deliberate retrieval degradation for regression test...")
        dense_results = [["dummy_unrelated_chunk_000"] for _ in queries]
        keyword_results = [["dummy_unrelated_chunk_000"] for _ in queries]
    else:
        dense_results = dense_many(queries, cfg, strategy, model_key, cfg["retrieval"]["dense_top_k"])
        keyword_results = keyword_many(queries, cfg, strategy, cfg["retrieval"]["keyword_top_k"])

    evaluated = []
    for idx, item in enumerate(data):
        d = dense_results[idx]
        k = keyword_results[idx]
        merged = list(dict.fromkeys(d + k))
        # Reciprocal rank hybrid merge (dense 0.6, keyword 0.4)
        retrieved = sorted(
            merged,
            key=lambda x: -(0.6 / (d.index(x) + 1) if x in d else 0) - (0.4 / (k.index(x) + 1) if x in k else 0),
        )
        retrieved_section_ids = [lookup[cid]["section_id"] for cid in retrieved if cid in lookup]
        evaluated.append({
            "retrieved_section_ids": retrieved_section_ids[:10],
            "ground_truth_section_ids": item["ground_truth_section_ids"],
        })

    metrics = retrieval_metrics(evaluated)
    recall_at_5 = metrics["recall_at_5"]
    mrr = metrics["mrr"]

    print(f"CI Retrieval Results:")
    print(f"  Recall@1 : {metrics['recall_at_1']:.4f}")
    print(f"  Recall@3 : {metrics['recall_at_3']:.4f}")
    print(f"  Recall@5 : {recall_at_5:.4f}")
    print(f"  Recall@10: {metrics['recall_at_10']:.4f}")
    print(f"  MRR      : {mrr:.4f}")
    print(f"============================================================")

    if recall_at_5 >= min_threshold:
        print(f"PASS: Recall@5 ({recall_at_5:.4f}) meets minimum threshold ({min_threshold:.4f}).")
        return recall_at_5
    else:
        print(
            f"FAIL: Recall@5 ({recall_at_5:.4f}) dropped below minimum threshold ({min_threshold:.4f})!",
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GitGuard CI Retrieval Gate")
    parser.add_argument("--deliberate-break", action="store_true", help="Simulate a broken retrieval system")
    args = parser.parse_args()

    run_ci_gate(deliberate_break=args.deliberate_break)
