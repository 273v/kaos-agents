"""BEIR-standard IR evaluation harness.

Runs the KAOS retrieval pipeline against standard BEIR benchmark datasets
with proper IR metrics (NDCG@10, MAP@100, Recall@10/50/100).

Compares:
- BM25-only baseline (plain search_memory, no expansion)
- KAOS adaptive pipeline (BM25 + Lexicon + PRF + HyDE + reflection)

All test queries are run. No truncation, no caps, no cherry-picking.

Usage::

    # Run BM25 baseline on NFCorpus (all 323 test queries)
    uv run python tests/benchmarks/beir_eval.py --dataset nfcorpus --method bm25

    # Run adaptive pipeline on NFCorpus
    uv run python tests/benchmarks/beir_eval.py --dataset nfcorpus --method adaptive

    # Run both and compare
    uv run python tests/benchmarks/beir_eval.py --dataset nfcorpus --method both

    # Save results
    uv run python tests/benchmarks/beir_eval.py --dataset nfcorpus --method both --json results.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

from tests.benchmarks.metrics import compute_ap, compute_ndcg, compute_recall

_BEIR_DATASETS = ("nfcorpus", "scifact", "fiqa")
_EVAL_KS = (10, 50, 100)


@dataclass(frozen=True, slots=True)
class QueryMetrics:
    """IR metrics for a single query."""

    query_id: str
    ndcg_at_10: float
    ap_at_100: float
    recall_at_10: float


@dataclass(frozen=True, slots=True)
class Metrics:
    """Standard IR metrics for one run."""

    ndcg_at_10: float
    map_at_100: float
    recall_at_10: float
    recall_at_50: float
    recall_at_100: float
    num_queries: int
    avg_latency_s: float


@dataclass(slots=True)  # Mutable: accumulated
class EvalResult:
    """Full evaluation result."""

    dataset: str = ""
    method: str = ""
    corpus_size: int = 0
    metrics: Metrics | None = None
    per_query: list[dict] = field(default_factory=list)


def load_beir_dataset(dataset_name: str) -> tuple:
    """Load a BEIR dataset: corpus, queries, qrels."""
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


def build_memory(corpus: dict[str, str]):
    """Load corpus into SessionMemory."""
    from kaos_agents.memory.session import SessionMemory
    from kaos_agents.types.memory import MemoryType

    memory = SessionMemory("beir-eval")
    uri_to_item: dict[str, str] = {}
    for doc_id, text in corpus.items():
        item = memory.add(MemoryType.DOCUMENTS, text, metadata={"uri": doc_id})
        uri_to_item[doc_id] = item.id
    return memory, uri_to_item


def run_bm25_baseline(
    memory, queries: dict[str, str], test_qids: list[str]
) -> tuple[dict[str, list[str]], dict[str, float]]:
    """Run plain BM25 (no expansion) and return ranked doc IDs per query."""
    from kaos_agents.memory.search import search_memory
    from kaos_agents.types.memory import MemoryType

    results: dict[str, list[str]] = {}
    latencies: dict[str, float] = {}

    for i, qid in enumerate(test_qids):
        query_text = queries[qid]
        t0 = time.perf_counter()
        hits = search_memory(
            memory,
            query_text,
            sections=[MemoryType.DOCUMENTS],
            top_k=100,
            expand_relations=[],
        )
        t1 = time.perf_counter()

        ranked = []
        for h in hits:
            items = memory.get_by_ids(MemoryType.DOCUMENTS, {h.item_id})
            if items:
                ranked.append((items[0].metadata or {}).get("uri", ""))
        results[qid] = ranked
        latencies[qid] = t1 - t0

        if (i + 1) % 50 == 0:
            sys.stdout.write(f"  BM25: {i + 1}/{len(test_qids)} queries\n")
            sys.stdout.flush()

    return results, latencies


def run_adaptive_pipeline(
    memory, queries: dict[str, str], test_qids: list[str], *, max_rounds: int = 2
) -> tuple[dict[str, list[str]], dict[str, float]]:
    """Run adaptive retrieval pipeline and return ranked doc IDs per query."""
    from kaos_agents.context.retrieval import adaptive_retrieve
    from kaos_agents.types.memory import MemoryType

    results: dict[str, list[str]] = {}
    latencies: dict[str, float] = {}

    for i, qid in enumerate(test_qids):
        query_text = queries[qid]
        t0 = time.perf_counter()
        ar = adaptive_retrieve(memory, query_text, max_rounds=max_rounds, top_k=100)
        t1 = time.perf_counter()

        ranked = []
        for r in ar.results:
            items = memory.get_by_ids(MemoryType.DOCUMENTS, {r.item_id})
            if items:
                ranked.append((items[0].metadata or {}).get("uri", ""))
        results[qid] = ranked
        latencies[qid] = t1 - t0

        if (i + 1) % 50 == 0:
            sys.stdout.write(f"  Adaptive: {i + 1}/{len(test_qids)} queries\n")
            sys.stdout.flush()

    return results, latencies


def evaluate(
    results: dict[str, list[str]],
    qrels: dict[str, dict[str, int]],
    latencies: dict[str, float],
    test_qids: list[str],
) -> tuple[Metrics, list[QueryMetrics]]:
    """Compute standard IR metrics from ranked results.

    Returns aggregated metrics and per-query breakdowns.
    """
    ndcg_scores = []
    ap_scores = []
    recall_at: dict[int, list[float]] = {k: [] for k in _EVAL_KS}
    per_query: list[QueryMetrics] = []

    for qid in test_qids:
        if qid not in results:
            continue
        ranked = results[qid]
        qrel_scores = qrels.get(qid, {})
        relevant = {did for did, score in qrel_scores.items() if score > 0}

        q_ndcg = compute_ndcg(ranked, qrel_scores, 10)
        q_ap = compute_ap(ranked, relevant, 100)
        q_recall_10 = compute_recall(ranked, relevant, 10)

        ndcg_scores.append(q_ndcg)
        ap_scores.append(q_ap)
        recall_at[10].append(q_recall_10)
        for k in _EVAL_KS:
            if k != 10:
                recall_at[k].append(compute_recall(ranked, relevant, k))

        per_query.append(
            QueryMetrics(
                query_id=qid,
                ndcg_at_10=q_ndcg,
                ap_at_100=q_ap,
                recall_at_10=q_recall_10,
            )
        )

    n = len(ndcg_scores) or 1
    avg_lat = sum(latencies.values()) / len(latencies) if latencies else 0.0

    metrics = Metrics(
        ndcg_at_10=sum(ndcg_scores) / n,
        map_at_100=sum(ap_scores) / n,
        recall_at_10=sum(recall_at[10]) / n,
        recall_at_50=sum(recall_at[50]) / n,
        recall_at_100=sum(recall_at[100]) / n,
        num_queries=len(ndcg_scores),
        avg_latency_s=avg_lat,
    )
    return metrics, per_query


# Published BEIR BM25 baselines for comparison (from BEIR paper, Thakur et al. 2021)
_BEIR_BM25_BASELINES = {
    "nfcorpus": {"ndcg_at_10": 0.325},
    "scifact": {"ndcg_at_10": 0.665},
    "fiqa": {"ndcg_at_10": 0.236},
}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="BEIR-standard IR evaluation")
    parser.add_argument("--dataset", choices=_BEIR_DATASETS, required=True)
    parser.add_argument("--method", choices=("bm25", "adaptive", "both"), default="both")
    parser.add_argument("--max-rounds", type=int, default=2, help="Max adaptive retrieval rounds")
    parser.add_argument("--json", type=str, default=None)
    args = parser.parse_args(argv)

    sys.stdout.write(f"Loading {args.dataset}...\n")
    sys.stdout.flush()
    t0 = time.perf_counter()
    corpus, queries, qrels, test_qids = load_beir_dataset(args.dataset)
    t1 = time.perf_counter()
    sys.stdout.write(
        f"  Corpus: {len(corpus)} docs, Queries: {len(test_qids)} test, {t1 - t0:.1f}s\n"
    )
    sys.stdout.flush()

    sys.stdout.write("Building memory index...\n")
    sys.stdout.flush()
    t0 = time.perf_counter()
    memory, _uri_to_item = build_memory(corpus)
    t1 = time.perf_counter()
    sys.stdout.write(f"  Indexed {len(corpus)} docs in {t1 - t0:.1f}s\n")
    sys.stdout.flush()

    all_results: dict[str, Metrics] = {}
    all_per_query: dict[str, list[QueryMetrics]] = {}

    if args.method in ("bm25", "both"):
        sys.stdout.write(f"\nRunning BM25 baseline ({len(test_qids)} queries)...\n")
        sys.stdout.flush()
        bm25_results, bm25_latencies = run_bm25_baseline(memory, queries, test_qids)
        bm25_metrics, bm25_pq = evaluate(bm25_results, qrels, bm25_latencies, test_qids)
        all_results["bm25"] = bm25_metrics
        all_per_query["bm25"] = bm25_pq

    if args.method in ("adaptive", "both"):
        sys.stdout.write(
            f"\nRunning adaptive pipeline ({len(test_qids)} queries, "
            f"max_rounds={args.max_rounds})...\n"
        )
        sys.stdout.flush()
        adap_results, adap_latencies = run_adaptive_pipeline(
            memory, queries, test_qids, max_rounds=args.max_rounds
        )
        adap_metrics, adap_pq = evaluate(adap_results, qrels, adap_latencies, test_qids)
        all_results["adaptive"] = adap_metrics
        all_per_query["adaptive"] = adap_pq

    # Print results
    published = _BEIR_BM25_BASELINES.get(args.dataset, {})
    sys.stdout.write(f"\n{'=' * 70}\n")
    sys.stdout.write(f"RESULTS: {args.dataset} ({len(corpus)} docs, {len(test_qids)} queries)\n")
    sys.stdout.write(f"{'=' * 70}\n")
    sys.stdout.write(
        f"{'Method':<15} {'NDCG@10':>8} {'MAP@100':>8} {'R@10':>6} "
        f"{'R@50':>6} {'R@100':>6} {'Latency':>8}\n"
    )
    sys.stdout.write(f"{'-' * 70}\n")

    if published:
        sys.stdout.write(f"{'Published BM25':<15} {published.get('ndcg_at_10', 0):.3f}\n")

    for method, m in all_results.items():
        sys.stdout.write(
            f"{method:<15} {m.ndcg_at_10:>8.3f} {m.map_at_100:>8.3f} "
            f"{m.recall_at_10:>6.1%} {m.recall_at_50:>6.1%} {m.recall_at_100:>6.1%} "
            f"{m.avg_latency_s:>7.2f}s\n"
        )
    sys.stdout.flush()

    # Determine JSON output path
    json_path: str | None = args.json
    if json_path is None:
        today = time.strftime("%Y-%m-%d")
        benchmarks_dir = Path(__file__).resolve().parent.parent.parent / "docs" / "benchmarks"
        json_path = str(benchmarks_dir / f"{args.dataset}-{args.method}-{today}.json")
        sys.stderr.write(f"WARNING: --json not specified; results will be saved to {json_path}\n")

    out_path = Path(json_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    output = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "dataset": args.dataset,
        "corpus_size": len(corpus),
        "num_queries": len(test_qids),
        "published_baseline": published,
        "results": {k: asdict(v) for k, v in all_results.items()},
        "per_query": {k: [asdict(q) for q in qs] for k, qs in all_per_query.items()},
    }
    out_path.write_text(json.dumps(output, indent=2))
    sys.stdout.write(f"\nSaved to {json_path}\n")


if __name__ == "__main__":
    main()
