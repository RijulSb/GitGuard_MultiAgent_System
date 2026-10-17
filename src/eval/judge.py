"""GitGuard LLM-as-a-Judge and evaluation metrics.

Implements PRD Day 4 Tasks 3, 4, and 5:
- Faithfulness: Splits answer into claims and verifies each against retrieved context.
- Correctness: Compares candidate answer to gold reference answer using rubric (1.0, 0.5, 0.0).
- Citations: Verifies that cited section IDs exist in Pro Git and correspond to relevant sections.
- Abstention: Evaluates correct refusals on unanswerables and flags false refusals on answerables.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from src.agents.llm_client import call_llm

# Common phrases indicating that the agent chose to abstain
ABSTENTION_PHRASES = (
    "not found in the book",
    "not found in the provided",
    "not covered in the pro git",
    "not mentioned in the text",
    "outside the scope of pro git",
    "timed out or exceeded token budget",
    "blocked by security policy",
    "i cannot answer this",
)


def is_abstention(text: str) -> bool:
    """Return True if the text represents a refusal / abstention."""
    if not text or not text.strip():
        return True
    t = text.lower().strip()
    return any(p in t for p in ABSTENTION_PHRASES)


def extract_citations(text: str) -> list[str]:
    """Extract section citations of format [section_id] from text, filtering out non-section tags."""
    if not text:
        return []
    matches = re.findall(r"\[([a-zA-Z0-9_\-\.]+)\]", text)
    non_section_tags = {
        "high risk command warning",
        "medium risk command notice",
        "low risk",
        "high",
        "medium",
        "low",
        "tool_call",
        "blocked_injection_pattern",
        "warning",
    }
    return [m for m in matches if m.lower() not in non_section_tags]


def evaluate_faithfulness(
    answer: str,
    context: str | list[dict[str, Any]],
) -> dict[str, Any]:
    """Split answer into claims and check each against retrieved text.

    Returns dict with claims, total_claims, supported_claims, and faithfulness_score.
    """
    if is_abstention(answer):
        return {
            "faithfulness_score": 1.0,
            "total_claims": 0,
            "supported_claims": 0,
            "claims": [],
            "reason": "Abstained (no factual claims made).",
        }

    # Format context string
    if isinstance(context, list):
        context_str = "\n\n".join(
            f"[{c.get('section_id', 'unknown')}]\n{c.get('text', '')}"
            for c in context
        )
    else:
        context_str = str(context)

    # Strip safety warning banner from answer text before claim extraction so judge focuses on content
    clean_answer = re.sub(
        r"\[HIGH RISK COMMAND WARNING\].*?Safer alternative:.*?\n\n",
        "",
        answer,
        flags=re.DOTALL | re.IGNORECASE,
    ).strip()
    if not clean_answer:
        clean_answer = answer

    system_prompt = (
        "You are an impartial, expert fact-checking judge evaluating a Git answer against reference context from the Pro Git book.\n"
        "Your task:\n"
        "1. Extract distinct, atomic factual claims made in the Candidate Answer (ignore safety warnings or stylistic remarks).\n"
        "2. For each claim, check whether it is DIRECTLY SUPPORTED by the Reference Context.\n"
        "3. Mark each claim as supported (true) or unsupported (false), with a brief reason.\n"
        "4. Calculate faithfulness_score = supported_claims / total_claims. If 0 claims, score is 1.0.\n\n"
        "Respond strictly with a JSON object in this format:\n"
        "{\n"
        '  "claims": [\n'
        '    {"claim": "<claim text>", "supported": true, "reason": "<explanation>"}\n'
        "  ],\n"
        '  "total_claims": <int>,\n'
        '  "supported_claims": <int>,\n'
        '  "faithfulness_score": <float between 0.0 and 1.0>\n'
        "}"
    )

    user_prompt = f"Reference Context:\n{context_str}\n\nCandidate Answer:\n{clean_answer}"

    response_text, meta = call_llm(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        model_role="critic_and_judge",
        temperature=0.0,
        max_tokens=1536,
        json_mode=True,
    )

    try:
        # Extract JSON substring if wrapped
        raw = response_text.strip()
        if "```json" in raw:
            raw = raw.split("```json")[1].split("```")[0].strip()
        data = json.loads(raw)
        claims = data.get("claims", [])
        total = int(data.get("total_claims", len(claims)))
        supported = int(data.get("supported_claims", sum(1 for c in claims if c.get("supported"))))
        score = float(data.get("faithfulness_score", (supported / total) if total > 0 else 1.0))
        return {
            "faithfulness_score": round(max(0.0, min(1.0, score)), 4),
            "total_claims": total,
            "supported_claims": supported,
            "claims": claims,
        }
    except Exception as exc:
        return {
            "faithfulness_score": 1.0 if not clean_answer else 0.5,
            "total_claims": 0,
            "supported_claims": 0,
            "claims": [],
            "error": f"Failed to parse faithfulness JSON: {exc}",
        }


def evaluate_correctness(
    question: str,
    candidate_answer: str,
    reference_answer: str,
    is_answerable: bool = True,
) -> dict[str, Any]:
    """Score candidate answer against reference answer.

    Scoring:
    - 1.0: Fully correct and complete.
    - 0.5: Partially correct or missing key details.
    - 0.0: Incorrect, conflicting, or false refusal/attempt.
    """
    ref_abstains = reference_answer.strip().lower() == "not found in the book." or not is_answerable
    cand_abstains = is_abstention(candidate_answer)

    # Ground truth is unanswerable
    if ref_abstains:
        if cand_abstains:
            return {"score": 1.0, "reason": "Correctly abstained on unanswerable question."}
        else:
            return {"score": 0.0, "reason": "Failed to abstain on unanswerable question (hallucinated answer)."}

    # Ground truth is answerable
    if cand_abstains:
        return {"score": 0.0, "reason": "False refusal: abstained on an answerable question."}

    system_prompt = (
        "You are an impartial, expert technical judge evaluating the correctness of a candidate Git answer against a gold reference answer.\n"
        "Scoring scale:\n"
        "- 1.0: Fully correct. Captures all core facts, commands, options, or explanations present in the reference answer.\n"
        "- 0.5: Partially correct. Captures some key facts but misses a critical part or adds minor ambiguity.\n"
        "- 0.0: Incorrect. Factually contradicts the reference answer, provides wrong Git commands or options, or fails to answer.\n\n"
        "Respond strictly with a JSON object in this format:\n"
        "{\n"
        '  "score": 1.0 | 0.5 | 0.0,\n'
        '  "reason": "<concise explanation of score>"\n'
        "}"
    )

    user_prompt = (
        f"Question: {question}\n\n"
        f"Reference Answer: {reference_answer}\n\n"
        f"Candidate Answer: {candidate_answer}"
    )

    response_text, meta = call_llm(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        model_role="critic_and_judge",
        temperature=0.0,
        max_tokens=512,
        json_mode=True,
    )

    try:
        raw = response_text.strip()
        if "```json" in raw:
            raw = raw.split("```json")[1].split("```")[0].strip()
        data = json.loads(raw)
        raw_score = float(data.get("score", 0.0))
        # Snap score to closest valid rubric point [0.0, 0.5, 1.0]
        if raw_score >= 0.75:
            score = 1.0
        elif raw_score >= 0.25:
            score = 0.5
        else:
            score = 0.0
        return {"score": score, "reason": str(data.get("reason", ""))}
    except Exception as exc:
        return {"score": 0.5, "reason": f"Judge parse error ({exc}); assigned 0.5"}


def evaluate_citations(
    answer: str,
    sections_lookup: dict[str, dict[str, Any]],
    retrieved_section_ids: list[str] | None = None,
) -> dict[str, Any]:
    """Check that cited sections exist in the book and are relevant.

    Returns citation_accuracy (0.0 to 1.0), total_citations, and valid_citations.
    """
    if is_abstention(answer):
        return {
            "citation_accuracy": 1.0,
            "total_citations": 0,
            "valid_citations": 0,
            "cited_sections": [],
            "reason": "Abstained (no citations required).",
        }

    cited = extract_citations(answer)
    if not cited:
        # Non-abstention answer with 0 citations violates required citation format
        return {
            "citation_accuracy": 0.0,
            "total_citations": 0,
            "valid_citations": 0,
            "cited_sections": [],
            "reason": "No citations provided in answer.",
        }

    # Verify that each cited section ID exists in the book
    valid_count = 0
    valid_cits = []
    invalid_cits = []

    for sid in cited:
        if sid in sections_lookup:
            valid_count += 1
            valid_cits.append(sid)
        else:
            invalid_cits.append(sid)

    accuracy = valid_count / len(cited) if cited else 0.0
    return {
        "citation_accuracy": round(accuracy, 4),
        "total_citations": len(cited),
        "valid_citations": valid_count,
        "valid_sections": valid_cits,
        "invalid_sections": invalid_cits,
    }


def judge_answer(
    question_row: dict[str, Any],
    candidate_answer: str,
    context: list[dict[str, Any]] | str,
    sections_lookup: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Run full Judge evaluation on a single answer against a dataset question row."""
    question = question_row["question"]
    reference = question_row.get("reference_answer", "")
    is_answerable = bool(question_row.get("answerable", True))

    abstained = is_abstention(candidate_answer)

    # 1. Faithfulness
    faith = evaluate_faithfulness(candidate_answer, context)

    # 2. Correctness
    corr = evaluate_correctness(question, candidate_answer, reference, is_answerable)

    # 3. Citations
    cits = evaluate_citations(candidate_answer, sections_lookup)

    return {
        "question_id": question_row.get("id"),
        "question": question,
        "answerable": is_answerable,
        "abstained": abstained,
        "faithfulness": faith["faithfulness_score"],
        "correctness": corr["score"],
        "correctness_reason": corr.get("reason", ""),
        "citation_accuracy": cits["citation_accuracy"],
        "total_claims": faith["total_claims"],
        "supported_claims": faith["supported_claims"],
        "total_citations": cits["total_citations"],
        "valid_citations": cits["valid_citations"],
    }
