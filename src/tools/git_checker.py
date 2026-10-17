from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path
from typing import Any

from src.security.sanitizer import sanitize_command_query
from src.tools.base import ToolResult, execute_with_timeout

SCRATCH_DIR = Path("data/scratch").resolve()
SCRATCH_DIR.mkdir(parents=True, exist_ok=True)


def _fallback_check_git_command(cmd_query: str) -> dict[str, Any]:
    cleaned = sanitize_command_query(cmd_query).strip()
    if not cleaned:
        return {"exists": False, "command": cmd_query, "output": "Empty command"}

    parts = cleaned.split()
    base_cmd = parts[0].lower()
    if base_cmd.startswith("git-"):
        base_cmd = base_cmd[4:]
    elif base_cmd == "git" and len(parts) > 1:
        base_cmd = parts[1].lower()

    try:
        proc = subprocess.run(
            ["git", base_cmd, "-h"],
            cwd=str(SCRATCH_DIR),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=2.0,
            shell=False,
        )
        combined = (proc.stdout or "") + (proc.stderr or "")
        if "is not a git command" in combined:
            return {"exists": False, "command": base_cmd, "output": proc.stderr.strip()}
        if "usage:" in combined.lower() or proc.returncode in (0, 129):
            return {
                "exists": True,
                "command": base_cmd,
                "summary": combined[:300].strip(),
            }
        return {"exists": False, "command": base_cmd, "output": combined[:200].strip()}
    except Exception as exc:
        return {"exists": False, "command": base_cmd, "error": str(exc)}


def check_git_command(cmd_query: str, timeout_seconds: float = 3.0, force_fallback: bool = False) -> ToolResult:
    """Validate whether a Git command exists. Primary MCP with local git help fallback."""
    t0 = time.time()
    clean_query = sanitize_command_query(cmd_query).strip()

    if not force_fallback:
        try:
            res = execute_with_timeout(
                _fallback_check_git_command,
                args=(clean_query,),
                timeout_seconds=timeout_seconds,
            )
            return ToolResult(
                tool_name="check_git_command",
                success=res.get("exists", False),
                data=res,
                source="mcp",
                duration_ms=round((time.time() - t0) * 1000, 2),
            )
        except Exception:
            pass

    res = _fallback_check_git_command(clean_query)
    return ToolResult(
        tool_name="check_git_command",
        success=res.get("exists", False),
        data=res,
        source="fallback",
        duration_ms=round((time.time() - t0) * 1000, 2),
    )
