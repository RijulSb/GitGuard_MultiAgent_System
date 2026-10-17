"""Unit tests for tool fallback mechanisms (PRD Day 5 Task 4).

Tests that tools try primary MCP/ChromaDB and fall back gracefully to local
readers, git help, and FAISS+BM25 without stalling or crashing.
"""
import pytest
from src.tools.git_checker import check_git_command
from src.tools.search_tool import search_index
from src.tools.section_reader import fetch_section


def test_fetch_section_fallback_success():
    # Test valid section reading via fallback
    res = fetch_section("hooks.006-other-client-hooks", force_fallback=True)
    assert res.success is True
    assert res.source == "fallback"
    assert "post-rewrite" in str(res.data)
    assert res.duration_ms >= 0.0


def test_fetch_section_fallback_missing():
    # Test non-existent section handling
    res = fetch_section("non_existent_section_99999", force_fallback=True)
    assert res.success is False
    assert res.source == "fallback"
    assert res.error is not None


def test_check_git_command_fallback_valid():
    # Test valid command verification via local git help
    res = check_git_command("git rebase", force_fallback=True)
    assert res.success is True
    assert res.source == "fallback"
    assert isinstance(res.data, dict)
    assert res.data.get("exists") is True


def test_check_git_command_fallback_invalid():
    # Test invalid command verification
    res = check_git_command("git invalidsubcommandxyz123", force_fallback=True)
    assert res.success is False
    assert res.source == "fallback"
    assert res.data.get("exists") is False


def test_search_index_fallback_faiss_bm25():
    # Test hybrid search index via fallback (FAISS + BM25)
    res = search_index("rebase branch", k=3, force_fallback=True)
    assert res.success is True
    assert res.source == "fallback"
    assert len(res.data) > 0
    assert any("rebase" in c.get("text", "").lower() for c in res.data)
