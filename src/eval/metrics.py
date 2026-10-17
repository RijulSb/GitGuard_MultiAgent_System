from __future__ import annotations

from collections.abc import Iterable


def hit_at(retrieved: list[str], truth: set[str], k: int) -> bool:
    return bool(set(retrieved[:k]) & truth)


def reciprocal_rank(retrieved: list[str], truth: set[str]) -> float:
    for rank, item in enumerate(retrieved, 1):
        if item in truth:
            return 1.0 / rank
    return 0.0


def retrieval_metrics(rows: Iterable[dict], ks: tuple[int, ...] = (1, 3, 5, 10)) -> dict[str, float]:
    rows = list(rows)
    if not rows:
        return {f"recall_at_{k}": 0.0 for k in ks} | {"mrr": 0.0}
    result = {f"recall_at_{k}": sum(hit_at(r["retrieved_section_ids"], set(r["ground_truth_section_ids"]), k) for r in rows) / len(rows) for k in ks}
    result["mrr"] = sum(reciprocal_rank(r["retrieved_section_ids"], set(r["ground_truth_section_ids"])) for r in rows) / len(rows)
    return result


def abstention_metrics(rows: Iterable[dict]) -> dict[str, float]:
    """Compute abstention metrics:
    - unanswerable_refusal_rate: % of unanswerable questions correctly refused
    - false_refusal_rate: % of answerable questions incorrectly refused
    - abstention_accuracy: overall accuracy of abstention decisions
    """
    rows_list = list(rows)
    if not rows_list:
        return {
            "unanswerable_refusal_rate": 0.0,
            "false_refusal_rate": 0.0,
            "abstention_accuracy": 0.0,
        }

    unanswerables = [r for r in rows_list if not r.get("answerable", True)]
    answerables = [r for r in rows_list if r.get("answerable", True)]

    # Correct refusal: question is unanswerable and agent abstained
    correct_refusals = sum(1 for r in unanswerables if r.get("abstained", False))
    # False refusal: question is answerable but agent abstained
    false_refusals = sum(1 for r in answerables if r.get("abstained", False))

    unans_rate = (correct_refusals / len(unanswerables)) if unanswerables else 1.0
    false_rate = (false_refusals / len(answerables)) if answerables else 0.0

    # Total correct decisions = correct refusals on unans + correct attempts on answerable
    correct_attempts = len(answerables) - false_refusals
    overall_acc = (correct_refusals + correct_attempts) / len(rows_list)

    return {
        "unanswerable_refusal_rate": round(unans_rate, 4),
        "false_refusal_rate": round(false_rate, 4),
        "abstention_accuracy": round(overall_acc, 4),
    }


def aggregate_eval_metrics(results: list[dict]) -> dict[str, float]:
    """Compute overall summary metrics for an evaluation run."""
    if not results:
        return {}

    n = len(results)
    mean_faith = sum(r.get("faithfulness", 0.0) for r in results) / n
    mean_corr = sum(r.get("correctness", 0.0) for r in results) / n
    mean_cit = sum(r.get("citation_accuracy", 0.0) for r in results) / n
    mean_lat = sum(r.get("latency_ms", 0.0) for r in results) / n
    mean_tok = sum(r.get("tokens", 0) for r in results) / n

    abs_metrics = abstention_metrics(results)

    return {
        "count": n,
        "faithfulness": round(mean_faith, 4),
        "correctness": round(mean_corr, 4),
        "citation_accuracy": round(mean_cit, 4),
        **abs_metrics,
        "latency_ms": round(mean_lat, 2),
        "tokens": round(mean_tok, 1),
    }

