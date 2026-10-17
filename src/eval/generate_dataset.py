from __future__ import annotations

import hashlib
import json
import os
import random
import re
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

UNANSWERABLE = [
    "How do I resolve a GitHub Actions billing failure?",
    "How do I configure Git LFS storage limits on GitHub?",
    "What is the GitHub Copilot monthly price?",
    "How do I create a GitLab merge request approval rule?",
    "How do I recover a deleted GitHub repository?",
    "What is the latest Git version's new feature?",
    "How do I configure Bitbucket branch permissions?",
    "How do I use GitHub Codespaces billing controls?",
    "How do I rotate a GitHub personal access token using its web UI?",
    "How do I configure GitLab CI protected variables?",
    "How do I migrate a repository to Azure DevOps?",
    "How do I enable GitHub Advanced Security?",
]


def cfg() -> dict[str, Any]:
    with open("configs/settings.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def sections() -> list[dict[str, Any]]:
    with open("data/processed/sections.jsonl", encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip() and len(json.loads(x)["text"].strip()) > 80]


def ask(client: Groq, model: str, section: dict[str, Any], cache: Path) -> dict[str, Any]:
    prompt = f"""Create one answerable evaluation item from this Pro Git section.
Return JSON only with keys: question, reference_answer, type.
Type must be one of factual, how-to, troubleshooting, multi-section.
The answer must be supported only by the supplied text and must be concise.
section_id: {section['section_id']}
heading: {section['heading']}
text: {section['text'][:6000]}"""
    key = hashlib.sha256(prompt.encode()).hexdigest()
    path = cache / f"{key}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    response = client.chat.completions.create(
        model=model,
        temperature=0.1,
        max_tokens=400,
        response_format={"type": "json_object"},
        messages=[{"role": "system", "content": "You generate rigorous RAG evaluation data."}, {"role": "user", "content": prompt}],
    )
    data = json.loads(response.choices[0].message.content)
    if data.get("type") not in {"factual", "how-to", "troubleshooting", "multi-section"}:
        data["type"] = "factual"
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return data


def main() -> None:
    config = cfg()
    out = Path(config["evaluation"]["dataset_path"])
    cache = Path(config["cache"]["directory"]) / "dataset_generation"
    cache.mkdir(parents=True, exist_ok=True)
    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    model = config["models"]["drafter_a"]["id"]
    rng = random.Random(42)
    source = sections()
    chosen_50 = rng.sample(source, 50)
    remaining = [s for s in source if s not in chosen_50]
    chosen_10 = rng.sample(remaining, 10)
    chosen = chosen_50 + chosen_10
    rows = []
    for i, section in enumerate(chosen, 1):
        item = ask(client, model, section, cache)
        rows.append({"id": f"q{i:03d}", "question": item["question"].strip(), "reference_answer": item["reference_answer"].strip(), "type": item["type"], "answerable": True, "ground_truth_section_ids": [section["section_id"]], "reviewed": False})
        print(f"generated {i}/60: {item['question'][:60]}")
    for i, question in enumerate(UNANSWERABLE, 61):
        rows.append({"id": f"q{i:03d}", "question": question, "reference_answer": "Not found in the book.", "type": "unanswerable", "answerable": False, "ground_truth_section_ids": [], "reviewed": True})
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    meta = {"status": "candidate_quick_test", "answerable_candidates": 60, "unanswerable": 12, "total": 72, "seed": 42, "deviation": "Configured for 60 candidate answerable questions for fast execution per user instruction; manual review required before locking."}
    Path("data/eval/dataset_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"wrote {len(rows)} candidate records to {out}")


if __name__ == "__main__":
    main()
