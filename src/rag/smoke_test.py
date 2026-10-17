from __future__ import annotations

import json
import re
from pathlib import Path

from .baseline import answer, load_config

QUESTIONS = [
    "What is version control?",
    "How do I create a new Git repository?",
    "How do I stage changes before committing?",
    "How do I see the history of a repository?",
    "How do I create and switch to a new branch?",
    "What is the difference between merging and rebasing?",
    "How do I rename a Git branch?",
    "How do I add a remote repository?",
    "What does git reset --hard do?",
    "How can I undo the last commit while keeping my changes?",
]


def run() -> None:
    cfg = load_config()
    report = []
    for question in QUESTIONS:
        result = answer(question)
        available = set(result["citations"])
        cited = set(re.findall(r"\[([^\]]+)\]", result["answer"]))
        valid = bool(cited) and cited.issubset(available)
        report.append({"question": question, "valid_citations": valid, "cited": sorted(cited), "retrieved": sorted(available), "cached": result["cached"]})
        print(f"{'PASS' if valid else 'FAIL'}: {question}")

    path = Path(cfg["paths"]["evaluation_data"]) / "smoke_test.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    passed = sum(row["valid_citations"] for row in report)
    print(f"Citation smoke test: {passed}/{len(report)} passed")
    if passed != len(report):
        raise SystemExit(1)


if __name__ == "__main__":
    run()
