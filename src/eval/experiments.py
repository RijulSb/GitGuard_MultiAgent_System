from __future__ import annotations

import argparse
import json
import pickle
import re
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from src.eval.metrics import retrieval_metrics

_MODELS: dict[str, Any] = {}
_RERANKER: Any = None


def load_cfg() -> dict[str, Any]:
    with open("configs/settings.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_dataset(cfg: dict[str, Any], limit: int) -> list[dict[str, Any]]:
    with open(cfg["evaluation"]["dataset_path"], encoding="utf-8") as f:
        rows = [json.loads(x) for x in f if x.strip()]
    return [x for x in rows if x["answerable"]][:limit or None]


def load_chunks(cfg: dict[str, Any], strategy: str) -> dict[str, dict[str, Any]]:
    path = Path(cfg["paths"]["processed_data"]) / "chunks" / f"{strategy}.jsonl"
    with path.open(encoding="utf-8") as f:
        return {r["chunk_id"]: r for x in f if (r := json.loads(x))}


def keyword_many(queries: list[str], cfg: dict[str, Any], strategy: str, top: int) -> list[list[str]]:
    path = Path(cfg["retrieval"]["index_dir"]) / f"{strategy}.bm25.pkl"
    with path.open("rb") as f:
        data = pickle.load(f)
    output = []
    for query in queries:
        scores = data["bm25"].get_scores(re.findall(r"\w+", query.lower()))
        output.append([data["ids"][i] for i in scores.argsort()[::-1][:top] if scores[i] > 0])
    return output


def dense_many(queries: list[str], cfg: dict[str, Any], strategy: str, model_key: str, top: int) -> list[list[str]]:
    import chromadb
    from sentence_transformers import SentenceTransformer

    spec = next(x for x in cfg["models"]["embeddings"] if x["key"] == model_key)
    if model_key not in _MODELS:
        _MODELS[model_key] = SentenceTransformer(spec["id"], device="cpu")
    vectors = _MODELS[model_key].encode(queries, normalize_embeddings=True, batch_size=32, show_progress_bar=True)
    client = chromadb.PersistentClient(path=str(Path(cfg["retrieval"]["index_dir"]) / "chroma" / model_key))
    collection = client.get_collection(f"{strategy}_{model_key}")
    result = collection.query(query_embeddings=vectors.tolist(), n_results=top)
    return result["ids"]


def rerank(query: str, ids: list[str], lookup: dict[str, dict[str, Any]], enabled: bool) -> list[str]:
    global _RERANKER
    if not enabled:
        return ids
    try:
        if _RERANKER is None:
            try:
                from sentence_transformers import CrossEncoder
                _RERANKER = CrossEncoder("BAAI/bge-reranker-base", max_length=512, local_files_only=True)
            except Exception as exc:
                print(f"reranker local weights not available ({exc}); using merged order fallback")
                _RERANKER = False
        if _RERANKER is False:
            return ids
        scores = _RERANKER.predict([(query, lookup[x]["text"]) for x in ids])
        return [x for _, x in sorted(zip(scores, ids), reverse=True)]
    except Exception as exc:
        print(f"reranker unavailable; using merged order: {exc}")
        return ids


def evaluate(cfg: dict[str, Any], data: list[dict[str, Any]], strategy: str, model_key: str, mode: str, use_reranker: bool) -> dict[str, float]:
    lookup = load_chunks(cfg, strategy)
    queries = [x["question"] for x in data]
    dense_results = dense_many(queries, cfg, strategy, model_key, cfg["retrieval"]["dense_top_k"]) if mode in {"dense", "hybrid"} else [[] for _ in queries]
    keyword_results = keyword_many(queries, cfg, strategy, cfg["retrieval"]["keyword_top_k"]) if mode in {"keyword", "hybrid"} else [[] for _ in queries]
    evaluated = []
    for index, item in enumerate(data, 1):
        d, k = dense_results[index - 1], keyword_results[index - 1]
        if mode == "dense":
            retrieved = d
        elif mode == "keyword":
            retrieved = k
        else:
            merged = list(dict.fromkeys(d + k))
            retrieved = sorted(merged, key=lambda x: -(1 / (d.index(x) + 1) if x in d else 0) - (1 / (k.index(x) + 1) if x in k else 0))
        retrieved = rerank(item["question"], retrieved, lookup, use_reranker)
        evaluated.append({"retrieved_section_ids": [lookup[x]["section_id"] for x in retrieved[:10]], "ground_truth_section_ids": item["ground_truth_section_ids"]})
    return retrieval_metrics(evaluated)


def save_outputs(results: list[dict[str, Any]], winner: dict[str, Any] | None = None) -> None:
    out = Path("data/eval")
    out.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(results)
    frame.to_csv(out / "retrieval_results.csv", index=False)
    frame.to_json(out / "retrieval_results.json", orient="records", indent=2)
    try:
        import plotly.express as px
        px.bar(frame, x="experiment", y="recall_at_5", color="strategy", hover_data=["embedding", "mode", "reranker"]).write_html(out / "retrieval_results.html")
    except Exception as exc:
        (out / "chart_error.txt").write_text(str(exc), encoding="utf-8")
    if winner:
        Path("configs/day2_selected_retrieval.yaml").write_text(yaml.safe_dump({k: winner[k] for k in ("strategy", "embedding", "mode", "reranker", "recall_at_5", "mrr")}, sort_keys=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    cfg = load_cfg()
    data = load_dataset(cfg, args.limit)
    progress_path = Path("data/eval/retrieval_progress.json")
    results = json.loads(progress_path.read_text(encoding="utf-8")) if progress_path.exists() else []
    done = {(x["experiment"], x["strategy"], x["embedding"], x["mode"], x["reranker"]) for x in results}

    def run(experiment: str, strategy: str, embedding: str, mode: str, reranker: bool) -> None:
        key = (experiment, strategy, embedding, mode, reranker)
        if key in done:
            print(f"skip cached: {key}")
            return
        print(f"start: {key}")
        result = {"experiment": experiment, "strategy": strategy, "embedding": embedding, "mode": mode, "reranker": reranker, **evaluate(cfg, data, strategy, embedding, mode, reranker)}
        results.append(result)
        progress_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
        save_outputs(results, max(results, key=lambda x: (x["recall_at_5"], x["mrr"])))
        print(f"done: {key} Recall@5={result['recall_at_5']:.3f} MRR={result['mrr']:.3f}")

    default_model = cfg["retrieval"]["default_embedding"]
    for strategy in ("fixed", "heading_aware", "parent_linked"):
        run("chunking", strategy, default_model, "dense", False)
    best_strategy = max((x for x in results if x["experiment"] == "chunking"), key=lambda x: (x["recall_at_5"], x["mrr"]))["strategy"]
    for model in [x["key"] for x in cfg["models"]["embeddings"]]:
        run("embeddings", best_strategy, model, "dense", False)
    best_model = max((x for x in results if x["experiment"] == "embeddings"), key=lambda x: (x["recall_at_5"], x["mrr"]))["embedding"]
    for mode in ("dense", "keyword", "hybrid"):
        run("retrieval", best_strategy, best_model, mode, False)
    best_mode = max((x for x in results if x["experiment"] == "retrieval"), key=lambda x: (x["recall_at_5"], x["mrr"]))["mode"]
    run("reranking", best_strategy, best_model, best_mode, True)
    winner = max(results, key=lambda x: (x["recall_at_5"], x["mrr"]))
    save_outputs(results, winner)
    print(pd.DataFrame(results).to_string(index=False))
    print(f"winner: {winner}")


if __name__ == "__main__":
    main()
