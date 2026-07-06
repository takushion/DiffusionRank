import argparse
import json
import os
import time
from transformers.utils import logging

import numpy as np
import pytrec_eval
from tqdm import tqdm
from datasets import load_dataset

from eval_utils import LladaForEval, PermutationListwiseWrapper

logging.set_verbosity_error()


def rerank_sliding_window(
    query: str,
    docs: list[str],
    permutation_fn,
    window_size: int = 20,
    step: int = 10,
):
    n = len(docs)
    if n == 0:
        return []

    scores = np.zeros(n, dtype=np.float64)

    for start in range(0, n, step):
        end = min(start + window_size, n)
        window_docs = docs[start:end]
        perm, _ = permutation_fn(query, window_docs)
        for rank, idx_in_window in enumerate(perm):
            global_idx = start + idx_in_window
            scores[global_idx] += window_size - rank

    ranked_indices = np.argsort(-scores)
    return ranked_indices.tolist()


def load_beir_dataset(dataset_name: str):
    queries = load_dataset(f"BeIR/{dataset_name}", "queries", split="queries")
    qrels_ds = load_dataset(f"BeIR/{dataset_name}", "qrels", split="qrels")

    queries_map = {}
    for row in queries:
        queries_map[str(row["_id"])] = row.get("text", "")

    qrels_map = {}
    for row in qrels_ds:
        qid = str(row["query-id"])
        did = str(row["corpus-id"])
        score = int(row["score"])
        qrels_map.setdefault(qid, {})[did] = score

    needed_dids = set()
    for qid, docs in qrels_map.items():
        needed_dids.update(docs.keys())

    corpus_ds = load_dataset(f"BeIR/{dataset_name}", "corpus", split="corpus")
    corpus_map = {}
    for row in corpus_ds:
        did = str(row["_id"])
        if did in needed_dids:
            text = row.get("text", "")
            title = row.get("title", "")
            corpus_map[did] = f"{title} {text}".strip()

    return corpus_map, queries_map, qrels_map


def evaluate_dataset(
    dataset_name: str,
    corpus_map: dict,
    queries_map: dict,
    qrels_map: dict,
    permutation_fn,
    topk: int = 100,
    window_size: int = 20,
    step: int = 10,
):
    run = {}
    skipped = 0

    for qid in tqdm(queries_map, desc=f"Evaluating {dataset_name}"):
        if qid not in qrels_map:
            skipped += 1
            continue

        candidate_dids = list(qrels_map[qid].keys())[:topk]
        doc_texts = [corpus_map.get(did, "") for did in candidate_dids]
        doc_texts = [t for t in doc_texts if t]

        if not doc_texts:
            skipped += 1
            continue

        ranked_indices = rerank_sliding_window(
            queries_map[qid], doc_texts, permutation_fn,
            window_size=window_size, step=step,
        )

        run[qid] = {}
        for rank, idx in enumerate(ranked_indices):
            did = candidate_dids[idx]
            run[qid][did] = float(len(ranked_indices) - rank)

    if not run:
        return {}, {}

    evaluator = pytrec_eval.RelevanceEvaluator(
        qrels_map, {"ndcg_cut.10", "ndcg_cut.20", "ndcg_cut.100", "map_cut.100"}
    )
    per_query = evaluator.evaluate(run)

    avg = {}
    for metric in ["ndcg_cut_10", "ndcg_cut_20", "ndcg_cut_100", "map_cut_100"]:
        values = [per_query[qid].get(metric, 0.0) for qid in per_query]
        avg[metric] = float(np.mean(values)) if values else 0.0

    return avg, per_query


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="GSAI-ML/LLaDA-1.5")
    parser.add_argument("--rope-scaling-factor", type=float, default=1.0)
    parser.add_argument("--model-args", type=str, default="")
    parser.add_argument("--reranking-args", type=str, default="")
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--topk", type=int, default=100)
    parser.add_argument("--window-size", type=int, default=20)
    parser.add_argument("--step", type=int, default=10)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    assert args.step <= args.window_size, "step must be <= window_size"

    model = LladaForEval(
        model_path=args.model,
        rope_scaling_factor=args.rope_scaling_factor,
        device=args.device,
    )

    perm_kwargs = eval(f"dict({args.model_args})") if args.model_args else {}
    permutation_fn = PermutationListwiseWrapper(model, **perm_kwargs)

    results = {}
    for dataset_name in args.datasets:
        print(f"\nLoading dataset: {dataset_name}")
        corpus_map, queries_map, qrels_map = load_beir_dataset(dataset_name)
        n_qrels = sum(len(v) for v in qrels_map.values())
        print(f"  Corpus: {len(corpus_map)} docs, Queries: {len(queries_map)} with qrels, Total judgments: {n_qrels}")

        avg_metrics, per_query_metrics = evaluate_dataset(
            dataset_name, corpus_map, queries_map, qrels_map,
            permutation_fn,
            topk=args.topk,
            window_size=args.window_size,
            step=args.step,
        )
        results[dataset_name] = avg_metrics
        print(f"  Results: {json.dumps(avg_metrics, indent=4)}")

    output = {
        "args": vars(args),
        "results": results,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    print(f"\n{'='*60}")
    print(f"Summary:")
    print(json.dumps(output, indent=2))

    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
        out_path = os.path.join(
            args.output_dir,
            f"{time.strftime('%Y%m%d_%H%M%S')}.json"
        )
        with open(out_path, "w") as f:
            json.dump(output, f, indent=2, default=str)
        print(f"\nResults saved to: {out_path}")
