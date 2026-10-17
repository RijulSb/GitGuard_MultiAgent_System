from __future__ import annotations

import hashlib
import json
import os
import pickle
import re
import time
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

from .indexing import load_config

load_dotenv()


def _cache_path(cfg: dict[str, Any], payload: dict[str, Any]) -> Path:
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()
    key = hashlib.sha256(raw).hexdigest()
    path = Path(cfg["cache"]["directory"]) / "llm" / f"{key}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


_EMBED_MODEL = None
_CHROMA_CLIENT = None
_COLLECTIONS: dict[str, Any] = {}
_BM25_DATA: dict[str, Any] = {}
_ROWS_CACHE: dict[str, dict[str, Any]] = {}


def _load_rows(cfg: dict[str, Any], strategy: str) -> dict[str, dict[str, Any]]:
    if strategy not in _ROWS_CACHE:
        path = Path(cfg["paths"]["processed_data"]) / "chunks" / f"{strategy}.jsonl"
        with path.open(encoding="utf-8") as f:
            _ROWS_CACHE[strategy] = {r["chunk_id"]: r for line in f if (r := json.loads(line))}
    return _ROWS_CACHE[strategy]


def _dense(query: str, cfg: dict[str, Any], strategy: str, k: int) -> list[tuple[str, float]]:
    global _EMBED_MODEL, _CHROMA_CLIENT
    import chromadb
    from sentence_transformers import SentenceTransformer

    model_spec = next(x for x in cfg["models"]["embeddings"] if x["key"] == cfg["retrieval"]["default_embedding"])
    if _EMBED_MODEL is None:
        _EMBED_MODEL = SentenceTransformer(model_spec["id"], device="cpu")
    embedding = _EMBED_MODEL.encode([query], normalize_embeddings=True)[0].tolist()

    coll_key = f"{strategy}_{model_spec['key']}"
    if coll_key not in _COLLECTIONS:
        if _CHROMA_CLIENT is None:
            _CHROMA_CLIENT = chromadb.PersistentClient(path=str(Path(cfg["retrieval"]["index_dir"]) / "chroma" / model_spec["key"]))
        _COLLECTIONS[coll_key] = _CHROMA_CLIENT.get_collection(coll_key)

    collection = _COLLECTIONS[coll_key]
    result = collection.query(query_embeddings=[embedding], n_results=k)
    return list(zip(result["ids"][0], [1 - x for x in result["distances"][0]]))


def _keyword(query: str, cfg: dict[str, Any], strategy: str, k: int) -> list[tuple[str, float]]:
    global _BM25_DATA
    if strategy not in _BM25_DATA:
        path = Path(cfg["retrieval"]["index_dir"]) / f"{strategy}.bm25.pkl"
        with path.open("rb") as f:
            _BM25_DATA[strategy] = pickle.load(f)
    data = _BM25_DATA[strategy]
    scores = data["bm25"].get_scores(re.findall(r"\w+", query.lower()))
    order = scores.argsort()[::-1][:k]
    return [(data["ids"][i], float(scores[i])) for i in order if scores[i] > 0]


def retrieve(query: str, cfg: dict[str, Any], k: int = 5) -> list[dict[str, Any]]:
    strategy = cfg["chunking"]["default_strategy"]
    rows = _load_rows(cfg, strategy)
    dense = _dense(query, cfg, strategy, cfg["retrieval"]["dense_top_k"])
    keyword = _keyword(query, cfg, strategy, cfg["retrieval"]["keyword_top_k"])
    scores: dict[str, float] = {}
    for rank, (chunk_id, _) in enumerate(dense, 1):
        scores[chunk_id] = scores.get(chunk_id, 0.0) + cfg["retrieval"]["hybrid"]["dense_weight"] / rank
    for rank, (chunk_id, _) in enumerate(keyword, 1):
        scores[chunk_id] = scores.get(chunk_id, 0.0) + cfg["retrieval"]["hybrid"]["keyword_weight"] / rank
    return [{**rows[cid], "retrieval_score": score} for cid, score in sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]]


def _groq_answer(question: str, context: list[dict[str, Any]], cfg: dict[str, Any]) -> tuple[str, int]:
    from groq import Groq

    model = cfg["models"]["drafter_a"]
    sources = "\n\n".join(f"[{r['section_id']}]\n{r['text']}" for r in context)
    system = "Answer only from the supplied Pro Git excerpts. Every factual sentence must end with one or more exact citations like [section_id]. If unsupported, say: Not found in the book. Do not use web knowledge."
    user = f"Question: {question}\n\nSources:\n{sources}"
    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    last_error = None
    for attempt in range(cfg["resilience"]["retry"]["max_attempts"]):
        try:
            response = client.chat.completions.create(
                model=model["id"],
                temperature=model["temperature"],
                max_tokens=model["max_tokens"],
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                timeout=cfg["resilience"]["request_timeout_seconds"],
            )
            text = response.choices[0].message.content.strip()
            usage = getattr(response, "usage", None)
            tokens = getattr(usage, "total_tokens", 0) if usage else 0
            return text, tokens
        except Exception as exc:
            last_error = exc
            if attempt + 1 == cfg["resilience"]["retry"]["max_attempts"]:
                raise
            time.sleep(2 ** attempt)
    raise last_error  # pragma: no cover


def answer(question: str, config_path: str = "configs/settings.yaml") -> dict[str, Any]:
    t0 = time.time()
    cfg = load_config(config_path)
    context = retrieve(question, cfg, cfg["retrieval"]["rerank_top_k"])
    payload = {"model": cfg["models"]["drafter_a"]["id"], "question": question, "context": [r["chunk_id"] for r in context]}
    cache = _cache_path(cfg, payload)
    if cache.exists():
        result = json.loads(cache.read_text(encoding="utf-8"))
        result["cached"] = True
        result.setdefault("latency_ms", round((time.time() - t0) * 1000, 2))
        result.setdefault("tokens", 500)
        return result
    ans_text, tokens = _groq_answer(question, context, cfg)
    latency_ms = round((time.time() - t0) * 1000, 2)
    result = {
        "question": question,
        "answer": ans_text,
        "citations": [r["section_id"] for r in context],
        "context": context,
        "cached": False,
        "latency_ms": latency_ms,
        "tokens": tokens,
    }
    cache.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result
