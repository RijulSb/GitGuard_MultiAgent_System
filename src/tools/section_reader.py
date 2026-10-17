from __future__ import annotations

import asyncio
import os
import shutil
import time
from pathlib import Path

from src.security.sanitizer import sanitize_input
from src.tools.base import ToolResult, execute_with_timeout

SECTIONS_DIR = Path("data/processed/sections").resolve()


def _fallback_read_section(section_id: str) -> str:
    clean_id = sanitize_input(section_id).strip()
    target_file = SECTIONS_DIR / f"{clean_id}.txt"
    if not target_file.exists():
        raise FileNotFoundError(f"Section not found: {clean_id}")
    return target_file.read_text(encoding="utf-8")


async def _mcp_read_section_async(section_id: str) -> str:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    npx_path = shutil.which("npx") or "npx"
    server_params = StdioServerParameters(
        command=npx_path,
        args=["-y", "@modelcontextprotocol/server-filesystem", str(SECTIONS_DIR)],
        env=dict(os.environ),
    )
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            clean_id = sanitize_input(section_id).strip()
            filepath = str(SECTIONS_DIR / f"{clean_id}.txt")
            result = await session.call_tool("read_file", arguments={"path": filepath})
            content_blocks = getattr(result, "content", [])
            if content_blocks and hasattr(content_blocks[0], "text"):
                return content_blocks[0].text
            raise RuntimeError(f"Unexpected MCP response: {result}")


def _try_mcp_read(section_id: str) -> str:
    return asyncio.run(_mcp_read_section_async(section_id))


def fetch_section(section_id: str, timeout_seconds: float = 3.0, force_fallback: bool = False) -> ToolResult:
    """Fetch section content. Tries MCP filesystem server first; falls back to local reader."""
    t0 = time.time()
    clean_id = sanitize_input(section_id).strip()

    if not force_fallback:
        try:
            content = execute_with_timeout(
                _try_mcp_read,
                args=(clean_id,),
                timeout_seconds=timeout_seconds,
            )
            return ToolResult(
                tool_name="fetch_section",
                success=True,
                data=content,
                source="mcp",
                duration_ms=round((time.time() - t0) * 1000, 2),
            )
        except Exception as exc:
            # Fallback path on any MCP error or timeout
            pass

    # Local fallback
    try:
        content = _fallback_read_section(clean_id)
        return ToolResult(
            tool_name="fetch_section",
            success=True,
            data=content,
            source="fallback",
            error=None,
            duration_ms=round((time.time() - t0) * 1000, 2),
        )
    except Exception as exc:
        return ToolResult(
            tool_name="fetch_section",
            success=False,
            data="",
            source="fallback",
            error=str(exc),
            duration_ms=round((time.time() - t0) * 1000, 2),
        )
