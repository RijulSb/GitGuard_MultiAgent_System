from __future__ import annotations

import concurrent.futures
import time
from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class ToolResult:
    tool_name: str
    success: bool
    data: Any
    source: str  # "mcp" or "fallback"
    error: str | None = None
    duration_ms: float = 0.0


# Shared persistent ThreadPoolExecutor to prevent wait=True blocking on context exit
_SHARED_EXECUTOR = concurrent.futures.ThreadPoolExecutor(max_workers=4)


def execute_with_timeout(
    func: Callable[..., Any],
    args: tuple = (),
    kwargs: dict | None = None,
    timeout_seconds: float = 3.0,
) -> Any:
    """Execute a function with strict timeout enforcement using a persistent ThreadPoolExecutor."""
    kwargs = kwargs or {}
    future = _SHARED_EXECUTOR.submit(func, *args, **kwargs)
    try:
        return future.result(timeout=timeout_seconds)
    except concurrent.futures.TimeoutError as exc:
        raise TimeoutError(f"Operation timed out after {timeout_seconds}s") from exc
