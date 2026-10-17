from __future__ import annotations

import glob
import json
import os
import pickle
import re
import time
from pathlib import Path
from typing import Any

import numpy as np

from src.security.sanitizer import sanitize_input
from src.tools.base import ToolResult, execute_with_timeout

_MODELS: dict[str, Any] = {}
_CHUNKS: dict[str, Any] = {}
_FAISS_INDEX: Any = None
_FAISS_IDS: list[str] = []
_BM25_DATA: Any = None
_CHROMA_CLIENT: Any = None
_CHROMA_COLLECTION: Any = None


def _get_chroma_collection() -> Any:
    global _CHROMA_CLIENT, _CHROMA_COLLECTION
    if _CHROMA_COLLECTION is None:
        import chromadb

        _CHROMA_CLIENT = chromadb.PersistentClient(path="data/processed/indexes/chroma/bge_small_en_v1_5")
        _CHROMA_COLLECTION = _CHROMA_CLIENT.get_collection("parent_linked_bge_small_en_v1_5")
    return _CHROMA_COLLECTION


def warmup_search() -> None:
    """Pre-warm in-memory chunks, embedder model, and Chroma collection.

    Calling this at startup avoids cold-load delays (loading PyTorch tensors,
    Chroma SQLite) during active agent requests, allowing search_index queries
    to complete in <0.5 seconds.
    """
    _get_chunks()
    _get_embedder()
    _get_chroma_collection()


def get_local_model_path(repo_id: str) -> str:
    """Resolve local HuggingFace cache snapshot path to bypass all remote checks."""
    safe_name = "models--" + repo_id.replace("/", "--")
    cache_root = Path.home() / ".cache" / "huggingface" / "hub" / safe_name / "snapshots"
    snapshots = list(cache_root.glob("*"))
    if snapshots:
        return str(snapshots[0])
    return repo_id


def _get_embedder(model_key: str = "bge_small_en_v1_5") -> Any:
    global _MODELS
    from sentence_transformers import SentenceTransformer

    if model_key not in _MODELS:
        repo_id = "BAAI/bge-small-en-v1.5" if model_key == "bge_small_en_v1_5" else "sentence-transformers/all-MiniLM-L6-v2"
        local_path = get_local_model_path(repo_id)
        _MODELS[model_key] = SentenceTransformer(local_path, device="cpu")
    return _MODELS[model_key]


def _get_chunks() -> dict[str, Any]:
    global _CHUNKS
    if not _CHUNKS:
        chunks_path = Path("data/processed/chunks/parent_linked.jsonl")
        with chunks_path.open(encoding="utf-8") as f:
            _CHUNKS = {r["chunk_id"]: r for line in f if (r := json.loads(line))}
    return _CHUNKS


def _get_bm25() -> Any:
    global _BM25_DATA
    if _BM25_DATA is None:
        bm25_path = Path("data/processed/indexes/parent_linked.bm25.pkl")
        with bm25_path.open("rb") as f:
            _BM25_DATA = pickle.load(f)
    return _BM25_DATA


def _get_faiss() -> tuple[Any, list[str]]:
    global _FAISS_INDEX, _FAISS_IDS
    import faiss

    chunks = _get_chunks()
    if _FAISS_INDEX is None:
        emb_files = list(Path("data/cache/embeddings").glob("parent_linked_bge_small_en_v1_5_*.npy"))
        if emb_files:
            embeddings = np.load(emb_files[0]).astype("float32")
            faiss.normalize_L2(embeddings)
            index = faiss.IndexFlatIP(embeddings.shape[1])
            index.add(embeddings)
            _FAISS_INDEX = index
            _FAISS_IDS = list(chunks.keys())
    return _FAISS_INDEX, _FAISS_IDS


def _primary_chroma_search(query: str, k: int) -> list[dict[str, Any]]:
    collection = _get_chroma_collection()
    model = _get_embedder("bge_small_en_v1_5")
    chunks = _get_chunks()
    vector = model.encode([query], normalize_embeddings=True)[0].tolist()
    res = collection.query(query_embeddings=[vector], n_results=k)
    ids = res["ids"][0]
    return [{**chunks[cid], "retrieval_mode": "chromadb_primary"} for cid in ids if cid in chunks]


def _fallback_faiss_bm25_search(query: str, k: int) -> list[dict[str, Any]]:
    model = _get_embedder("bge_small_en_v1_5")
    chunks = _get_chunks()
    bm25_data = _get_bm25()
    faiss_index, faiss_ids = _get_faiss()

    # 1. FAISS dense search
    vector = model.encode([query], normalize_embeddings=True).astype("float32")
    distances, indices = faiss_index.search(vector, k)
    dense_ids = [faiss_ids[i] for i in indices[0] if i < len(faiss_ids)]

    # 2. BM25 keyword search
    scores = bm25_data["bm25"].get_scores(re.findall(r"\w+", query.lower()))
    keyword_order = scores.argsort()[::-1][:k]
    keyword_ids = [bm25_data["ids"][i] for i in keyword_order if scores[i] > 0]

    # 3. Hybrid reciprocal rank fusion
    merged = list(dict.fromkeys(dense_ids + keyword_ids))
    ranked = sorted(
        merged,
        key=lambda x: -(0.6 / (dense_ids.index(x) + 1) if x in dense_ids else 0.0)
        - (0.4 / (keyword_ids.index(x) + 1) if x in keyword_ids else 0.0),
    )
    return [{**chunks[cid], "retrieval_mode": "faiss_bm25_fallback"} for cid in ranked[:k] if cid in chunks]


def search_index(
    query: str,
    k: int = 5,
    timeout_seconds: float = 5.0,
    force_fallback: bool = False,
) -> ToolResult:
    """Search index. Primary ChromaDB; Fallback FAISS + BM25 with strict timeout."""
    t0 = time.time()
    sanitized_q = sanitize_input(query).strip()

    if not force_fallback:
        try:
            results = execute_with_timeout(
                _primary_chroma_search,
                args=(sanitized_q, k),
                timeout_seconds=timeout_seconds,
            )
            return ToolResult(
                tool_name="search_index",
                success=True,
                data=results,
                source="mcp",
                duration_ms=round((time.time() - t0) * 1000, 2),
            )
        except Exception:
            pass

    # Local FAISS + BM25 fallback
    try:
        results = _fallback_faiss_bm25_search(sanitized_q, k)
        return ToolResult(
            tool_name="search_index",
            success=True,
            data=results,
            source="fallback",
            duration_ms=round((time.time() - t0) * 1000, 2),
        )
    except Exception as exc:
        return ToolResult(
            tool_name="search_index",
            success=False,
            data=[],
            source="fallback",
            error=str(exc),
            duration_ms=round((time.time() - t0) * 1000, 2),
        )
