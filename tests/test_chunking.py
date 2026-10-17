"""Unit tests for GitGuard chunking strategies (PRD Day 5 Task 4)."""
import pytest
from src.rag.indexing import clean_text, load_config, make_chunks, split_tokens, tokenizer


@pytest.fixture
def sample_sections():
    return [
        {
            "section_id": "test.001-sample-heading",
            "chapter": "01-introduction",
            "heading": "Sample Heading",
            "heading_level": 2,
            "heading_path": ["Introduction", "Sample Heading"],
            "source_file": "01-intro.adoc",
            "text": "Git is a fast, scalable, distributed revision control system.\n" * 15,
        },
        {
            "section_id": "test.002-second-heading",
            "chapter": "01-introduction",
            "heading": "Second Heading",
            "heading_level": 2,
            "heading_path": ["Introduction", "Second Heading"],
            "source_file": "01-intro.adoc",
            "text": "Every Git clone is a full repository with complete history and tracking.\n" * 10,
        },
    ]


@pytest.fixture
def config():
    return load_config()


def test_clean_text():
    dirty = "(((index term)))\n.Table Title\nSome content here.\n\n\n\nMore text."
    cleaned = clean_text(dirty)
    assert "(((index term)))" not in cleaned
    assert ".Table Title" not in cleaned
    assert "\n\n\n" not in cleaned
    assert "Some content here." in cleaned


def test_split_tokens_without_tokenizer():
    text = "word " * 100
    chunks = split_tokens(text, size=20, overlap=5, enc=None)
    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk.split()) <= 20


def test_make_chunks_all_strategies(sample_sections, config):
    chunks = make_chunks(sample_sections, config)

    assert "fixed" in chunks
    assert "heading_aware" in chunks
    assert "parent_linked" in chunks

    # Check parent_linked chunks
    parent_chunks = chunks["parent_linked"]
    assert len(parent_chunks) > 0
    for c in parent_chunks:
        assert "section_id" in c
        assert "parent_section_id" in c
        assert c["parent_section_id"] == c["section_id"]
        assert "chunk_id" in c
        assert "::parent_linked:" in c["chunk_id"]
        assert "heading" in c
        assert "text" in c
        assert len(c["text"]) > 0


def test_chunk_id_continuity(sample_sections, config):
    chunks = make_chunks(sample_sections, config)
    for strategy in ("fixed", "heading_aware", "parent_linked"):
        c_list = chunks[strategy]
        assert len(c_list) > 0
        for idx, item in enumerate(c_list):
            assert "::" in item["chunk_id"]
            assert item["source_file"] == "01-intro.adoc"
