"""BM25 parameter tuning — grid search k1/b on BEIR datasets.

Searches for the best (k1, b) pair on NFCorpus, then validates cross-domain
on SciFact and FiQA. Uses the same metrics as beir_eval.py (NDCG@10).

Usage::

    # Quick tune on NFCorpus only
    uv run python tests/benchmarks/bm25_tune.py --dataset nfcorpus

    # Full cross-domain tune (NFCorpus → validate on scifact + fiqa)
    uv run python tests/benchmarks/bm25_tune.py --all

    # Custom grid
    uv run python tests/benchmarks/bm25_tune.py --k1 0.9,1.2,1.5 --b 0.5,0.75,0.9

    # Save results
    uv run python tests/benchmarks/bm25_tune.py --all --json results.json
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

_DEFAULT_K1_GRID = [0.5, 0.9, 1.2, 1.5, 2.0]
_DEFAULT_B_GRID = [0.3, 0.5, 0.75, 0.9]
_BEIR_DATASETS = ("nfcorpus", "scifact", "fiqa")


@dataclass(frozen=True, slots=True)
class TuneResult:
    """Result for one (k1, b) configuration on one dataset."""

    dataset: str
    k1: float
    b: float
    ndcg_at_10: float
    map_at_100: float
    avg_latency_s: float
    num_queries: int


def _compute_ndcg(ranked_ids: list[str], qrel_scores: dict[str, int], k: int) -> float:
    """Compute NDCG@k."""
    dcg = 0.0
    for i, doc_id in enumerate(ranked_ids[:k]):
        rel = qrel_scores.get(doc_id, 0)
        dcg += (2**rel - 1) / math.log2(i + 2)
    ideal_rels = sorted(qrel_scores.values(), reverse=True)[:k]
    idcg = sum((2**r - 1) / math.log2(i + 2) for i, r in enumerate(ideal_rels))
    return dcg / idcg if idcg > 0 else 0.0


def _compute_ap(ranked_ids: list[str], relevant: set[str], k: int) -> float:
    """Compute Average Precision@k."""
    hits = 0
    sum_precision = 0.0
    for i, doc_id in enumerate(ranked_ids[:k]):
        if doc_id in relevant:
            hits += 1
            sum_precision += hits / (i + 1)
    return sum_precision / len(relevant) if relevant else 0.0


def load_beir_dataset(dataset_name: str) -> tuple:
    """Load a BEIR dataset."""
    from collections import defaultdict

    from datasets import load_dataset

    corpus_ds = load_dataset(f"BeIR/{dataset_name}", "corpus", split="corpus")
    queries_ds = load_dataset(f"BeIR/{dataset_name}", "queries", split="queries")
    qrels_ds = load_dataset(f"BeIR/{dataset_name}-qrels", split="test")

    corpus = {doc["_id"]: f"{doc['title']}. {doc['text']}" for doc in corpus_ds}
    queries = {q["_id"]: q["text"] for q in queries_ds}

    qrels: dict[str, dict[str, int]] = defaultdict(dict)
    for qr in qrels_ds:
        qrels[str(qr["query-id"])][str(qr["corpus-id"])] = qr["score"]

    test_qids = [qid for qid in qrels if qid in queries]
    return corpus, queries, dict(qrels), test_qids


def run_bm25_with_params(
    corpus: dict[str, str],
    queries: dict[str, str],
    qrels: dict[str, dict[str, int]],
    test_qids: list[str],
    dataset_name: str,
    *,
    k1: float = 1.5,
    b: float = 0.75,
    top_k: int = 100,
) -> TuneResult:
    """Run BM25 with specific k1/b and compute NDCG@10."""
    from kaos_nlp_core.search import Searcher

    # Build index
    records = [{"id": i, "text": text} for i, (_, text) in enumerate(corpus.items())]
    id_to_uri = dict(enumerate(corpus.keys()))
    searcher = Searcher.from_documents(records, bm25_k1=k1, bm25_b=b)

    ndcg_scores: list[float] = []
    ap_scores: list[float] = []
    latencies: list[float] = []

    for qid in test_qids:
        query_text = queries[qid]
        qrel_scores = qrels.get(qid, {})
        relevant = {did for did, score in qrel_scores.items() if score > 0}

        t0 = time.perf_counter()
        hits = searcher.search(query_text, top_k=top_k)
        t1 = time.perf_counter()

        ranked = [id_to_uri[int(h.doc_id)] for h in hits if int(h.doc_id) in id_to_uri]

        ndcg_scores.append(_compute_ndcg(ranked, qrel_scores, 10))
        ap_scores.append(_compute_ap(ranked, relevant, 100))
        latencies.append(t1 - t0)

    n = len(ndcg_scores) or 1
    return TuneResult(
        dataset=dataset_name,
        k1=k1,
        b=b,
        ndcg_at_10=sum(ndcg_scores) / n,
        map_at_100=sum(ap_scores) / n,
        avg_latency_s=sum(latencies) / n,
        num_queries=len(ndcg_scores),
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="BM25 k1/b parameter tuning")
    parser.add_argument(
        "--dataset",
        choices=_BEIR_DATASETS,
        default=None,
        help="Tune on specific dataset (default: nfcorpus)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Tune on NFCorpus, validate on SciFact + FiQA",
    )
    parser.add_argument(
        "--k1",
        default=None,
        help=f"Comma-separated k1 values (default: {_DEFAULT_K1_GRID})",
    )
    parser.add_argument(
        "--b",
        default=None,
        help=f"Comma-separated b values (default: {_DEFAULT_B_GRID})",
    )
    parser.add_argument("--json", type=str, default=None)
    args = parser.parse_args(argv)

    k1_grid = [float(x) for x in args.k1.split(",")] if args.k1 else _DEFAULT_K1_GRID
    b_grid = [float(x) for x in args.b.split(",")] if args.b else _DEFAULT_B_GRID

    # Tune dataset is nfcorpus by default (most queries, most challenging)
    tune_dataset = args.dataset or "nfcorpus"
    validate_datasets = ["scifact", "fiqa"] if args.all else []

    # Load tune dataset
    sys.stdout.write(f"Loading {tune_dataset}...\n")
    sys.stdout.flush()
    t0 = time.perf_counter()
    corpus, queries, qrels, test_qids = load_beir_dataset(tune_dataset)
    t1 = time.perf_counter()
    sys.stdout.write(f"  {len(corpus)} docs, {len(test_qids)} queries ({t1 - t0:.1f}s)\n\n")
    sys.stdout.flush()

    # Grid search
    grid = list(itertools.product(k1_grid, b_grid))
    sys.stdout.write(f"Grid search: {len(grid)} combinations on {tune_dataset}\n")
    sys.stdout.write(f"{'k1':>6} {'b':>6} {'NDCG@10':>8} {'MAP@100':>8} {'Latency':>8}\n")
    sys.stdout.write(f"{'-' * 42}\n")
    sys.stdout.flush()

    tune_results: list[TuneResult] = []
    for k1, b in grid:
        result = run_bm25_with_params(
            corpus, queries, qrels, test_qids, tune_dataset, k1=k1, b=b
        )
        tune_results.append(result)
        sys.stdout.write(
            f"{k1:>6.2f} {b:>6.2f} {result.ndcg_at_10:>8.4f} "
            f"{result.map_at_100:>8.4f} {result.avg_latency_s:>7.4f}s\n"
        )
        sys.stdout.flush()

    # Find best
    best = max(tune_results, key=lambda r: r.ndcg_at_10)
    default = next((r for r in tune_results if r.k1 == 1.5 and r.b == 0.75), None)

    sys.stdout.write(f"\nBest on {tune_dataset}: k1={best.k1}, b={best.b}, NDCG@10={best.ndcg_at_10:.4f}\n")
    if default:
        delta = best.ndcg_at_10 - default.ndcg_at_10
        sys.stdout.write(
            f"Default (k1=1.5, b=0.75): NDCG@10={default.ndcg_at_10:.4f} "
            f"(delta={delta:+.4f})\n"
        )
    sys.stdout.flush()

    # Cross-domain validation
    validation_results: dict[str, list[TuneResult]] = {}
    for vd in validate_datasets:
        sys.stdout.write(f"\nValidating on {vd}...\n")
        sys.stdout.flush()
        v_corpus, v_queries, v_qrels, v_qids = load_beir_dataset(vd)
        sys.stdout.write(f"  {len(v_corpus)} docs, {len(v_qids)} queries\n")

        # Run best and default on validation set
        v_best = run_bm25_with_params(
            v_corpus, v_queries, v_qrels, v_qids, vd, k1=best.k1, b=best.b
        )
        v_default = run_bm25_with_params(
            v_corpus, v_queries, v_qrels, v_qids, vd, k1=1.5, b=0.75
        )
        validation_results[vd] = [v_best, v_default]

        delta = v_best.ndcg_at_10 - v_default.ndcg_at_10
        sys.stdout.write(
            f"  Best (k1={best.k1}, b={best.b}): NDCG@10={v_best.ndcg_at_10:.4f}\n"
            f"  Default (k1=1.5, b=0.75):    NDCG@10={v_default.ndcg_at_10:.4f} "
            f"(delta={delta:+.4f})\n"
        )
        sys.stdout.flush()

    # Summary
    sys.stdout.write(f"\n{'=' * 60}\n")
    sys.stdout.write("PARAMETER TUNING SUMMARY\n")
    sys.stdout.write(f"{'=' * 60}\n")
    sys.stdout.write(f"Best params: k1={best.k1}, b={best.b}\n")
    sys.stdout.write(f"Tune set ({tune_dataset}): NDCG@10={best.ndcg_at_10:.4f}\n")
    for vd, vr in validation_results.items():
        sys.stdout.write(f"Validation ({vd}): NDCG@10={vr[0].ndcg_at_10:.4f} "
                        f"(default={vr[1].ndcg_at_10:.4f})\n")

    # Recommendation
    if default:
        improvement = best.ndcg_at_10 - default.ndcg_at_10
        if improvement > 0.02:
            sys.stdout.write(f"\nRECOMMENDATION: Update defaults to k1={best.k1}, b={best.b} "
                           f"(+{improvement:.4f} NDCG@10)\n")
        else:
            sys.stdout.write(f"\nRECOMMENDATION: Keep defaults (k1=1.5, b=0.75). "
                           f"Improvement only {improvement:+.4f} (<0.02 threshold)\n")
    sys.stdout.flush()

    # Save results
    if args.json:
        output = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "tune_dataset": tune_dataset,
            "k1_grid": k1_grid,
            "b_grid": b_grid,
            "best": asdict(best),
            "default": asdict(default) if default else None,
            "tune_results": [asdict(r) for r in tune_results],
            "validation": {
                k: [asdict(r) for r in v] for k, v in validation_results.items()
            },
        }
        out_path = Path(args.json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(output, indent=2))
        sys.stdout.write(f"\nSaved to {args.json}\n")


if __name__ == "__main__":
    main()
