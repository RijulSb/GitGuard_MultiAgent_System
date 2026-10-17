"""GitGuard Safety Rules & Risk Registry.

Provides canonical classification, rationale, and safer alternatives
pointing to Pro Git book sections for dangerous/destructive Git commands.

PRD Task 1 & 2 requirements:
- Risky command list: reset --hard, push --force, clean -fd, branch -D, checkout .,
  rebase on shared branches, filter-branch and similar.
- Each rule provides:
  1. Risk level (HIGH, MEDIUM, LOW)
  2. Clear explanation of why the command is dangerous
  3. Safer alternative pointing to exact Pro Git section ID
- Unit-testable: Harmless commands are never flagged; risky commands are caught.
- HIGH risk answers display the warning FIRST (at top).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

RiskLevel = Literal["LOW", "MEDIUM", "HIGH"]


@dataclass(frozen=True)
class SafetyRule:
    rule_id: str
    command_pattern: re.Pattern
    canonical_command: str
    risk_level: RiskLevel
    explanation: str
    safer_alternative: str
    section_citation: str


# ─── Canonical Safety Rules Registry ──────────────────────────────────────────

SAFETY_RULES: list[SafetyRule] = [
    SafetyRule(
        rule_id="RESET_HARD",
        command_pattern=re.compile(r"\bgit\s+reset\s+--hard\b", re.IGNORECASE),
        canonical_command="git reset --hard",
        risk_level="HIGH",
        explanation="Discards all uncommitted changes in the working tree and index, moving the branch pointer. Overwritten uncommitted files cannot be recovered.",
        safer_alternative="Use `git stash` to preserve uncommitted work or `git reset --soft HEAD~1` to keep working tree modifications staged.",
        section_citation="[undoing.001-undoing-things]",
    ),
    SafetyRule(
        rule_id="PUSH_FORCE",
        command_pattern=re.compile(r"\bgit\s+push\s+(?:-[a-zA-Z]*f\b|--force\b|--force-with-lease\b)", re.IGNORECASE),
        canonical_command="git push --force",
        risk_level="HIGH",
        explanation="Overwrites remote repository history without checking for divergent commits, potentially deleting colleagues' pushed commits on shared branches.",
        safer_alternative="Use `git push --force-with-lease` which only pushes if your local tracking branch matches the remote ref, preventing accidental overwrites of others' work.",
        section_citation="[git-branching.005-remote-branches]",
    ),
    SafetyRule(
        rule_id="CLEAN_FORCE",
        command_pattern=re.compile(r"\bgit\s+clean\s+-[a-zA-Z]*f[a-zA-Z]*\b", re.IGNORECASE),
        canonical_command="git clean -fd",
        risk_level="HIGH",
        explanation="Permanently removes untracked files and directories from the working tree without moving them to a trash bin.",
        safer_alternative="Run `git clean -nd` first to dry-run and preview files before deletion, or use `git stash -u` to archive untracked files.",
        section_citation="[stashing-cleaning.003-cleaning-your-working-directory]",
    ),
    SafetyRule(
        rule_id="BRANCH_FORCE_DELETE",
        command_pattern=re.compile(r"\bgit\s+branch\s+-[a-zA-Z]*D\b", re.IGNORECASE),
        canonical_command="git branch -D",
        risk_level="HIGH",
        explanation="Forces branch deletion even if it contains unmerged commits, which can orphan and lose commit history.",
        safer_alternative="Use `git branch -d` (lowercase) which checks if the branch has been fully merged into upstream before deleting.",
        section_citation="[git-branching.003-basic-branching-and-merging]",
    ),
    SafetyRule(
        rule_id="CHECKOUT_DOT",
        command_pattern=re.compile(r"\bgit\s+checkout\s+\.(?=[^\w]|$)", re.IGNORECASE),
        canonical_command="git checkout .",
        risk_level="HIGH",
        explanation="Overwrites all unstaged changes in the current working directory to match the index. Discarded modifications are permanently lost.",
        safer_alternative="Use `git stash` to preserve modifications, or inspect individual files with `git diff` before selectively reverting with `git restore <file>`.",
        section_citation="[undoing.001-undoing-things]",
    ),
    SafetyRule(
        rule_id="FILTER_BRANCH",
        command_pattern=re.compile(r"\bgit\s+filter-branch\b", re.IGNORECASE),
        canonical_command="git filter-branch",
        risk_level="HIGH",
        explanation="Rewrites commit history across branches by recalculating commit hashes, breaking repository consistency for everyone collaborating on the project.",
        safer_alternative="Use interactive rebase (`git rebase -i`) on local unpushed branches, or use `git-filter-repo`.",
        section_citation="[rewriting-history.005-filter-branch]",
    ),
    SafetyRule(
        rule_id="REBASE_SHARED",
        command_pattern=re.compile(r"\bgit\s+rebase\b", re.IGNORECASE),
        canonical_command="git rebase",
        risk_level="MEDIUM",
        explanation="Rebasing rewrites commit hashes. If applied to commits already pushed to a shared remote, it creates divergent duplicate histories for collaborators.",
        safer_alternative="Use `git merge` when integrating changes from or into shared public branches. Follow the Golden Rule of Rebasing: never rebase public commits.",
        section_citation="[git-branching.006-rebasing]",
    ),
    SafetyRule(
        rule_id="RESTORE_OR_CHECKOUT_FILE",
        command_pattern=re.compile(r"\bgit\s+(?:restore|checkout)\s+[^-][^\n]*", re.IGNORECASE),
        canonical_command="git restore / git checkout",
        risk_level="MEDIUM",
        explanation="Replaces working tree files with index or commit versions, discarding unstaged modifications in those specific files.",
        safer_alternative="Check `git status` and `git diff` to inspect changes before running `git restore <file>` or `git checkout -- <file>`.",
        section_citation="[undoing.001-undoing-things]",
    ),
]


@dataclass
class SafetyEvaluation:
    risk_level: RiskLevel = "LOW"
    triggered_rules: list[SafetyRule] = field(default_factory=list)
    detected_commands: list[str] = field(default_factory=list)
    explanations: list[str] = field(default_factory=list)
    safer_alternatives: list[str] = field(default_factory=list)


def evaluate_text_safety(text: str) -> SafetyEvaluation:
    """Scan arbitrary answer text for dangerous Git commands.

    Returns a SafetyEvaluation object with the highest risk level detected,
    all triggered rules, explanations, and safer alternatives.
    """
    if not text or not text.strip():
        return SafetyEvaluation(risk_level="LOW")

    triggered: list[SafetyRule] = []
    seen_ids: set[str] = set()

    # Match against canonical rules
    for rule in SAFETY_RULES:
        if rule.command_pattern.search(text) and rule.rule_id not in seen_ids:
            triggered.append(rule)
            seen_ids.add(rule.rule_id)

    if not triggered:
        return SafetyEvaluation(risk_level="LOW")

    # Determine highest risk level: HIGH > MEDIUM > LOW
    has_high = any(r.risk_level == "HIGH" for r in triggered)
    highest_risk: RiskLevel = "HIGH" if has_high else "MEDIUM"

    # Filter to highest-risk rules for clean messaging
    primary_rules = [r for r in triggered if r.risk_level == highest_risk]

    return SafetyEvaluation(
        risk_level=highest_risk,
        triggered_rules=triggered,
        detected_commands=[r.canonical_command for r in primary_rules],
        explanations=[f"{r.canonical_command}: {r.explanation}" for r in primary_rules],
        safer_alternatives=[
            f"{r.safer_alternative} {r.section_citation}" for r in primary_rules
        ],
    )


def format_safety_annotated_response(raw_answer: str, eval_result: SafetyEvaluation) -> str:
    """Format final answer with safety annotations per PRD rules:

    - HIGH risk: The warning and safer alternative appear FIRST (at the top).
    - MEDIUM risk: The cautionary note appears at the bottom.
    - LOW risk: Answer is returned unchanged.
    """
    if eval_result.risk_level == "LOW" or not eval_result.triggered_rules:
        return raw_answer

    if eval_result.risk_level == "HIGH":
        cmds = ", ".join(f"`{c}`" for c in eval_result.detected_commands)
        explanations = " ".join(eval_result.explanations)
        alternatives = " ".join(eval_result.safer_alternatives)

        warning_header = (
            f"[SAFETY WARNING] (HIGH RISK): The answer below contains destructive command(s) ({cmds}).\n"
            f"**Risk**: {explanations}\n"
            f"**Safer alternative**: {alternatives}\n\n"
            f"---\n\n"
        )
        return warning_header + raw_answer

    # MEDIUM risk: note appended at the bottom
    cmds = ", ".join(f"`{c}`" for c in eval_result.detected_commands)
    alternatives = " ".join(eval_result.safer_alternatives)
    note_footer = (
        f"\n\n---\n"
        f"[NOTE] (MEDIUM RISK): Command(s) {cmds} can rewrite commit history or alter working tree files.\n"
        f"**Best practice**: {alternatives}"
    )
    return raw_answer + note_footer
