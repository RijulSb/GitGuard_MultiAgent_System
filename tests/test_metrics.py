from src.eval.metrics import retrieval_metrics


def test_recall_and_mrr():
    rows = [
        {"retrieved_section_ids": ["a", "b", "c"], "ground_truth_section_ids": ["a"]},
        {"retrieved_section_ids": ["x", "b", "c"], "ground_truth_section_ids": ["b"]},
        {"retrieved_section_ids": ["x", "y", "z"], "ground_truth_section_ids": ["q"]},
    ]
    result = retrieval_metrics(rows)
    assert result["recall_at_1"] == 1 / 3
    assert result["recall_at_3"] == 2 / 3
    assert result["recall_at_5"] == 2 / 3
    assert round(result["mrr"], 6) == round((1 + 0.5) / 3, 6)
