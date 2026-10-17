"""Unit tests for GitGuard Safety Rules & Safety Guard Agent (Day 4 Tasks 1 & 2).

Verifies:
1. Every risky command is properly caught (reset --hard, push --force, clean -fd,
   branch -D, checkout ., filter-branch, rebase).
2. Variations with flags/arguments are correctly caught.
3. Every rule provides:
   - A non-empty explanation of the danger
   - A safer alternative
   - A section citation from the Pro Git book ([section_id])
4. Harmless commands (status, log, diff, add, branch, clone, commit) are NOT flagged (LOW risk).
5. HIGH risk answers display the warning FIRST (at the top of the final response).
6. MEDIUM risk answers display a note at the bottom.
"""
from __future__ import annotations

import pytest

from src.security.safety_rules import (
    SAFETY_RULES,
    evaluate_text_safety,
    format_safety_annotated_response,
)


# ─── 1. Canonical Risky Commands ──────────────────────────────────────────────

@pytest.mark.parametrize(
    ("command_text", "expected_risk", "expected_rule_id"),
    [
        ("Run `git reset --hard` to discard everything.", "HIGH", "RESET_HARD"),
        ("Execute `git reset --hard HEAD~1` to revert commit.", "HIGH", "RESET_HARD"),
        ("You can run `git push --force origin main`.", "HIGH", "PUSH_FORCE"),
        ("Run `git push -f` to overwrite remote.", "HIGH", "PUSH_FORCE"),
        ("Use `git push --force-with-lease`.", "HIGH", "PUSH_FORCE"),
        ("Run `git clean -fd` to remove all untracked files.", "HIGH", "CLEAN_FORCE"),
        ("Use `git clean -f` to clean up.", "HIGH", "CLEAN_FORCE"),
        ("Delete the branch with `git branch -D feature`.", "HIGH", "BRANCH_FORCE_DELETE"),
        ("Run `git checkout .` to revert working directory.", "HIGH", "CHECKOUT_DOT"),
        ("Use `git filter-branch --tree-filter 'rm -f passwords.txt' HEAD`.", "HIGH", "FILTER_BRANCH"),
        ("Run `git rebase main` on your branch.", "MEDIUM", "REBASE_SHARED"),
        ("Run `git restore app.py` to discard changes.", "MEDIUM", "RESTORE_OR_CHECKOUT_FILE"),
        ("Run `git checkout src/main.py` to restore file.", "MEDIUM", "RESTORE_OR_CHECKOUT_FILE"),
    ],
)
def test_risky_commands_detected(command_text: str, expected_risk: str, expected_rule_id: str) -> None:
    eval_res = evaluate_text_safety(command_text)
    assert eval_res.risk_level == expected_risk, f"Expected {expected_risk} for '{command_text}', got {eval_res.risk_level}"
    rule_ids = [r.rule_id for r in eval_res.triggered_rules]
    assert expected_rule_id in rule_ids, f"Expected rule {expected_rule_id} in {rule_ids}"


# ─── 2. Harmless Commands NOT Flagged ─────────────────────────────────────────

@pytest.mark.parametrize(
    "harmless_text",
    [
        "Run `git status` to see what files are staged or modified.",
        "Check your commit history with `git log --oneline --graph`.",
        "Inspect changes using `git diff HEAD~1`.",
        "Stage your files with `git add .` before committing.",
        "Commit your changes with `git commit -m 'Initial commit'`.",
        "Clone the repository with `git clone https://github.com/progit/progit2.git`.",
        "List all local branches using `git branch`.",
        "Create a branch safely with `git branch feature-x`.",
        "Switch branches using `git switch main`.",
        "Fetch updates from the remote with `git fetch origin`.",
        "Pull remote changes using `git pull origin main`.",
        "Show commit details with `git show HEAD`.",
    ],
)
def test_harmless_commands_not_flagged(harmless_text: str) -> None:
    eval_res = evaluate_text_safety(harmless_text)
    assert eval_res.risk_level == "LOW", f"Harmless text was incorrectly flagged: {eval_res.risk_level} ({eval_res.detected_commands})"
    assert len(eval_res.triggered_rules) == 0


# ─── 3. Rules Metadata Integrity ─────────────────────────────────────────────

def test_all_safety_rules_have_explanations_and_citations() -> None:
    """Verify that every rule has an explanation, safer alternative, and valid [section_id]."""
    for rule in SAFETY_RULES:
        assert rule.explanation, f"Rule {rule.rule_id} is missing explanation"
        assert rule.safer_alternative, f"Rule {rule.rule_id} is missing safer_alternative"
        assert rule.section_citation.startswith("[") and rule.section_citation.endswith("]"), (
            f"Rule {rule.rule_id} has invalid citation format: {rule.section_citation}"
        )


# ─── 4. Warning Placement (HIGH Risk First) ───────────────────────────────────

def test_high_risk_warning_displayed_first() -> None:
    """PRD Task 2: HIGH risk answers show the warning first."""
    answer = "To discard all local changes immediately, run `git reset --hard HEAD` [undoing.001-undoing-things]."
    eval_res = evaluate_text_safety(answer)
    assert eval_res.risk_level == "HIGH"

    annotated = format_safety_annotated_response(answer, eval_res)
    # The warning MUST start with [SAFETY WARNING] at index 0
    assert annotated.startswith("[SAFETY WARNING] (HIGH RISK):"), (
        f"HIGH risk warning did not appear first. Output starts with:\n{annotated[:100]}"
    )
    # The original answer must still be present
    assert answer in annotated
    # The safer alternative must be present with citation
    assert "git stash" in annotated
    assert "[undoing.001-undoing-things]" in annotated


def test_medium_risk_note_displayed_at_bottom() -> None:
    """MEDIUM risk answers show a note at the bottom."""
    answer = "To incorporate the latest changes, run `git rebase main` on your local branch."
    eval_res = evaluate_text_safety(answer)
    assert eval_res.risk_level == "MEDIUM"

    annotated = format_safety_annotated_response(answer, eval_res)
    # Should start with the original answer
    assert annotated.startswith(answer)
    # Note should appear at the bottom
    assert "[NOTE] (MEDIUM RISK):" in annotated
    assert "[git-branching.006-rebasing]" in annotated


def test_low_risk_answer_unchanged() -> None:
    """LOW risk answers should not be modified."""
    answer = "To see commit history, run `git log --oneline` [basics.004-viewing-the-commit-history]."
    eval_res = evaluate_text_safety(answer)
    assert eval_res.risk_level == "LOW"

    annotated = format_safety_annotated_response(answer, eval_res)
    assert annotated == answer
