"""Unit tests for GitGuard Judge evaluation module."""
import json
import pytest
from src.eval.judge import (
    is_abstention,
    extract_citations,
    evaluate_citations,
    evaluate_correctness,
    evaluate_faithfulness,
)
from src.eval.metrics import abstention_metrics, aggregate_eval_metrics


def test_is_abstention():
    assert is_abstention("Not found in the book.") is True
    assert is_abstention("This topic is not covered in the Pro Git book.") is True
    assert is_abstention("Request timed out or exceeded token budget.") is True
    assert is_abstention("git commit -m 'Initial commit'") is False
    assert is_abstention("") is True


def test_extract_citations():
    text = "Run git restore [undoing.004-undoing-things-with-git-restore] or git reset [reset.001]."
    cits = extract_citations(text)
    assert cits == ["undoing.004-undoing-things-with-git-restore", "reset.001"]

    # Filter out non-section tags
    text_with_warning = "[HIGH RISK COMMAND WARNING] Don't do this. See [undoing.004]."
    cits2 = extract_citations(text_with_warning)
    assert cits2 == ["undoing.004"]


def test_evaluate_citations():
    sections = {
        "sec.001": {"text": "dummy"},
        "sec.002": {"text": "dummy"},
    }

    # Abstention -> 1.0
    res = evaluate_citations("Not found in the book.", sections)
    assert res["citation_accuracy"] == 1.0

    # No citations in factual answer -> 0.0
    res = evaluate_citations("Run git reset --hard HEAD", sections)
    assert res["citation_accuracy"] == 0.0

    # Valid citation -> 1.0
    res = evaluate_citations("Use git rebase [sec.001]", sections)
    assert res["citation_accuracy"] == 1.0
    assert res["valid_citations"] == 1

    # Mixed valid and invalid citation -> 0.5
    res = evaluate_citations("Check [sec.001] and [sec.999]", sections)
    assert res["citation_accuracy"] == 0.5
    assert res["valid_citations"] == 1
    assert res["total_citations"] == 2


def test_evaluate_correctness_unanswerable():
    # Unanswerable + correctly abstained -> 1.0
    res = evaluate_correctness("How do I fix billing?", "Not found in the book.", "Not found in the book.", is_answerable=False)
    assert res["score"] == 1.0

    # Unanswerable + hallucinated answer -> 0.0
    res = evaluate_correctness("How do I fix billing?", "Go to settings > billing.", "Not found in the book.", is_answerable=False)
    assert res["score"] == 0.0


def test_evaluate_correctness_answerable():
    # Answerable + false refusal -> 0.0
    res = evaluate_correctness("How do I commit?", "Not found in the book.", "Use git commit.", is_answerable=True)
    assert res["score"] == 0.0


def test_abstention_metrics():
    rows = [
        # Answerables
        {"answerable": True, "abstained": False},  # Correct attempt
        {"answerable": True, "abstained": False},  # Correct attempt
        {"answerable": True, "abstained": True},   # False refusal
        # Unanswerables
        {"answerable": False, "abstained": True},  # Correct refusal
        {"answerable": False, "abstained": False}, # False attempt (hallucination)
    ]
    m = abstention_metrics(rows)
    # unanswerable_refusal_rate: 1/2 = 0.5
    assert m["unanswerable_refusal_rate"] == 0.5
    # false_refusal_rate: 1/3 = 0.3333
    assert m["false_refusal_rate"] == 0.3333
    # abstention_accuracy: (1 + 2) / 5 = 3/5 = 0.6
    assert m["abstention_accuracy"] == 0.6
