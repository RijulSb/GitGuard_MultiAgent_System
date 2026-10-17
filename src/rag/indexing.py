from __future__ import annotations

import hashlib
import json
import pickle
import re
from pathlib import Path
from typing import Any

import numpy as np
import yaml

try:
    import tiktoken
except ImportError:  # pragma: no cover
    tiktoken = None


def load_config(path: str | Path = "configs/settings.yaml") -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def tokenizer(name: str):
    if tiktoken:
        try:
            return tiktoken.get_encoding(name)
        except Exception:
            pass
    return None


def split_tokens(text: str, size: int, overlap: int, enc) -> list[str]:
    if enc:
        ids = enc.encode(text)
        parts = []
        step = max(1, size - overlap)
        for start in range(0, len(ids), step):
            part = enc.decode(ids[start:start + size]).strip()
            if part:
                parts.append(part)
            if start + size >= len(ids):
                break
        return parts
    words = text.split()
    step = max(1, size - overlap)
    return [" ".join(words[i:i + size]) for i in range(0, len(words), step) if words[i:i + size]]


def clean_text(text: str) -> str:
    text = re.sub(r"\n?\(\(\([^\n]+\)\)\)", "", text)
    text = re.sub(r"^\.[^\n]+$", "", text, flags=re.MULTILINE)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def read_sections(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def make_chunks(sections: list[dict[str, Any]], cfg: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    enc = tokenizer(cfg["chunking"]["tokenizer"])
    strategies = cfg["chunking"]["strategies"]
    output: dict[str, list[dict[str, Any]]] = {"fixed": [], "heading_aware": [], "parent_linked": []}

    for section in sections:
        text = clean_text(section.get("text", ""))
        if not text:
            continue
        base = {
            "section_id": section["section_id"],
            "chapter": section["chapter"],
            "heading": section["heading"],
            "heading_path": section.get("heading_path", []),
            "source_file": section["source_file"],
        }
        fixed = split_tokens(text, strategies["fixed"]["chunk_tokens"], strategies["fixed"]["overlap_tokens"], enc)
        heading = split_tokens(text, strategies["heading_aware"]["max_chunk_tokens"], strategies["heading_aware"]["overlap_tokens"], enc)
        parent = split_tokens(text, strategies["parent_linked"]["chunk_tokens"], strategies["parent_linked"]["overlap_tokens"], enc)
        for name, parts in (("fixed", fixed), ("heading_aware", heading), ("parent_linked", parent)):
            for i, part in enumerate(parts):
                item = {**base, "chunk_id": f"{section['section_id']}::{name}:{i:04d}", "chunk_index": i, "text": part}
                if name == "parent_linked":
                    item["parent_section_id"] = section["section_id"]
                output[name].append(item)
    return output


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def cache_key(model: str, texts: list[str]) -> str:
    payload = model + "\n" + "\n".join(texts)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def build_indexes(config_path: str = "configs/settings.yaml") -> None:
    import chromadb
    from rank_bm25 import BM25Okapi
    from sentence_transformers import SentenceTransformer

    cfg = load_config(config_path)
    sections = read_sections(cfg["paths"]["sections_file"])
    chunks = make_chunks(sections, cfg)
    root = Path(cfg["retrieval"]["index_dir"])
    root.mkdir(parents=True, exist_ok=True)
    cache = Path(cfg["cache"]["directory"]) / "embeddings"
    cache.mkdir(parents=True, exist_ok=True)
    write_root = Path(cfg["paths"]["processed_data"]) / "chunks"
    write_root.mkdir(parents=True, exist_ok=True)

    for strategy, rows in chunks.items():
        write_jsonl(write_root / f"{strategy}.jsonl", rows)
        corpus = [r["text"] for r in rows]
        tokens = [re.findall(r"\w+", x.lower()) for x in corpus]
        with (root / f"{strategy}.bm25.pkl").open("wb") as f:
            pickle.dump({"ids": [r["chunk_id"] for r in rows], "bm25": BM25Okapi(tokens)}, f)

        for model_spec in cfg["models"]["embeddings"]:
            model_id, model_key = model_spec["id"], model_spec["key"]
            key = cache_key(model_id, corpus)
            emb_path = cache / f"{strategy}_{model_key}_{key}.npy"
            embeddings = np.load(emb_path) if emb_path.exists() else None
            if embeddings is None:
                model = SentenceTransformer(model_id, device="cpu")
                embeddings = model.encode(corpus, normalize_embeddings=True, show_progress_bar=True)
                np.save(emb_path, embeddings)

            client = chromadb.PersistentClient(path=str(root / "chroma" / model_key))
            name = re.sub(r"[^a-z0-9_]+", "_", f"{strategy}_{model_key}".lower())[:60]
            collection = client.get_or_create_collection(name=name, metadata={"hnsw:space": "cosine"})
            if collection.count() == 0:
                collection.add(
                    ids=[r["chunk_id"] for r in rows],
                    documents=corpus,
                    embeddings=embeddings.tolist(),
                    metadatas=[{k: v if isinstance(v, (str, int, float, bool)) else json.dumps(v) for k, v in r.items() if k != "text"} for r in rows],
                )
            print(f"indexed {strategy}/{model_key}: {len(rows)} chunks")


if __name__ == "__main__":
    build_indexes()
