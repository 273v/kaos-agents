"""Cross-domain retrieval benchmark suite.

.. deprecated::
    This benchmark uses custom metrics and limited queries (default 10).
    Use ``tests/benchmarks/beir_eval.py`` (standard IR metrics on BEIR datasets) or
    ``tests/benchmarks/agent_eval.py`` (full agent evaluation) instead.

Tests the adaptive retrieval pipeline across multiple domains to measure
generalization vs. overfitting. Each benchmark loads a standard dataset,
runs queries through the pipeline, and measures recall against ground truth.

Datasets:
- EDGAR (legal): 997 SEC contract exhibits — needle-in-haystack
- NFCorpus (medical): 3,633 PubMed nutrition abstracts — vocabulary gap
- SciFact (scientific): 5,183 paper abstracts — claim verification

Usage::

    # Run all benchmarks (retrieval-only, no LLM)
    uv run python tests/benchmarks/cross_domain_benchmark.py

    # Run specific dataset
    uv run python tests/benchmarks/cross_domain_benchmark.py --dataset nfcorpus

    # Run with LLM rounds (HyDE + multi-query)
    uv run python tests/benchmarks/cross_domain_benchmark.py --with-llm

    # Save results to JSON
    uv run python tests/benchmarks/cross_domain_benchmark.py --json results.json

    # Run full agent (not just retrieval)
    uv run python tests/benchmarks/cross_domain_benchmark.py --agent
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

_DATASETS = ("edgar", "nfcorpus", "scifact", "fiqa")

_MAX_QUERIES = 10
_MAX_CHARS_PER_DOC = 5000
_MAX_CORPUS_SIZE = 5000  # Cap corpus for very large datasets (FiQA has 57K)


@dataclass(frozen=True, slots=True)
class QueryResult:
    """Result for one query."""

    query_id: str
    query_text: str
    ground_truth_count: int
    true_positives: int
    recall: float
    unique_items: int
    total_rounds: int
    latency_s: float
    strategies: list[str]


@dataclass(slots=True)  # Mutable: accumulated during benchmark
class DatasetResult:
    """Result for one dataset."""

    dataset: str = ""
    corpus_size: int = 0
    num_queries: int = 0
    avg_recall: float = 0.0
    avg_latency_s: float = 0.0
    queries: list[QueryResult] = field(default_factory=list)


def _load_edgar() -> tuple[Any, dict[str, set[str]], dict[str, str]]:
    """Load EDGAR agreements and compute ground truth."""
    from tests.benchmarks.kl3m_loader import load_edgar_agreements
    from tests.benchmarks.needle_haystack_benchmark import (
        compute_ground_truth_coc_acceleration,
        compute_ground_truth_noncompete,
    )

    docs = load_edgar_agreements(max_chars_per_doc=50_000)
    gt_nc = compute_ground_truth_noncompete(docs)
    gt_coc = compute_ground_truth_coc_acceleration(docs)

    from kaos_agents.memory.session import SessionMemory
    from kaos_agents.types.memory import MemoryType

    memory = SessionMemory("bench-edgar")
    for identifier, text in docs:
        sid = identifier.rstrip("/").split("/")[-1]
        memory.add(MemoryType.DOCUMENTS, text, metadata={"uri": f"edgar:{sid}"})

    queries = {
        "nc-1": "non-compete non-competition restrictive covenant duration years",
        "coc-1": "change of control acceleration vesting equity options",
    }
    qrels: dict[str, set[str]] = {
        "nc-1": {f"edgar:{g.short_id}" for g in gt_nc},
        "coc-1": {f"edgar:{g.short_id}" for g in gt_coc},
    }
    return memory, qrels, queries


def _load_beir(
    dataset_name: str, *, max_queries: int = _MAX_QUERIES
) -> tuple[Any, dict[str, set[str]], dict[str, str]]:
    """Load a BeIR dataset (NFCorpus or SciFact)."""
    from datasets import load_dataset

    from kaos_agents.memory.session import SessionMemory
    from kaos_agents.types.memory import MemoryType

    corpus_ds = load_dataset(f"BeIR/{dataset_name}", "corpus", split="corpus")
    queries_ds = load_dataset(f"BeIR/{dataset_name}", "queries", split="queries")
    qrels_ds = load_dataset(f"BeIR/{dataset_name}-qrels", split="test")

    # Build corpus ID set from qrels first so we can subsample large corpora
    # while keeping all relevant documents
    relevant_corpus_ids: set[str] = set()
    for qr in qrels_ds:
        if qr["score"] > 0:
            relevant_corpus_ids.add(str(qr["corpus-id"]))

    memory = SessionMemory(f"bench-{dataset_name}")
    loaded = 0
    for doc in corpus_ds:
        if loaded >= _MAX_CORPUS_SIZE and doc["_id"] not in relevant_corpus_ids:
            continue
        text = f"{doc['title']}. {doc['text']}"
        if len(text) > _MAX_CHARS_PER_DOC:
            text = text[:_MAX_CHARS_PER_DOC]
        memory.add(MemoryType.DOCUMENTS, text, metadata={"uri": doc["_id"]})
        loaded += 1

    qrels: dict[str, set[str]] = defaultdict(set)
    for qr in qrels_ds:
        if qr["score"] > 0:
            qrels[str(qr["query-id"])].add(str(qr["corpus-id"]))

    query_map = {q["_id"]: q["text"] for q in queries_ds}
    queries = {qid: query_map[qid] for qid in list(qrels.keys())[:max_queries] if qid in query_map}

    return memory, dict(qrels), queries


def run_retrieval_benchmark(
    memory: Any,
    qrels: dict[str, set[str]],
    queries: dict[str, str],
    dataset_name: str,
    *,
    max_rounds: int = 2,
    top_k: int = 50,
    model: str | None = None,
) -> DatasetResult:
    """Run adaptive retrieval on all queries and measure recall."""
    from kaos_agents.context.retrieval import adaptive_retrieve
    from kaos_agents.types.memory import MemoryType

    result = DatasetResult(
        dataset=dataset_name,
        corpus_size=memory.section_item_count(MemoryType.DOCUMENTS),
    )

    for qid, query_text in queries.items():
        gt_ids = qrels.get(qid, set())
        if not gt_ids:
            continue

        t0 = time.perf_counter()
        ar = adaptive_retrieve(memory, query_text, max_rounds=max_rounds, top_k=top_k, model=model)
        t1 = time.perf_counter()

        found_ids: set[str] = set()
        for r in ar.results:
            items = memory.get_by_ids(MemoryType.DOCUMENTS, {r.item_id})
            if items:
                found_ids.add((items[0].metadata or {}).get("uri", ""))

        tp = len(found_ids & gt_ids)
        recall = tp / len(gt_ids) if gt_ids else 0.0
        strategies = sorted({rd.strategy for rd in ar.rounds})

        result.queries.append(
            QueryResult(
                query_id=qid,
                query_text=query_text[:100],
                ground_truth_count=len(gt_ids),
                true_positives=tp,
                recall=recall,
                unique_items=ar.unique_items,
                total_rounds=ar.total_rounds,
                latency_s=t1 - t0,
                strategies=strategies,
            )
        )

    result.num_queries = len(result.queries)
    if result.queries:
        result.avg_recall = sum(q.recall for q in result.queries) / len(result.queries)
        result.avg_latency_s = sum(q.latency_s for q in result.queries) / len(result.queries)

    return result


def main(argv: list[str] | None = None) -> None:
    warnings.warn(
        "cross_domain_benchmark is deprecated. "
        "Use tests/benchmarks/beir_eval.py or tests/benchmarks/agent_eval.py instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    print(
        "\u26a0 DEPRECATED: This benchmark uses custom metrics and limited queries (default 10).\n"
        "Use tests/benchmarks/beir_eval.py (standard IR metrics on BEIR datasets) or\n"
        "tests/benchmarks/agent_eval.py (full agent evaluation) instead.\n",
        file=sys.stderr,
    )

    parser = argparse.ArgumentParser(description="Cross-domain retrieval benchmark")
    parser.add_argument(
        "--dataset",
        choices=_DATASETS,
        default=None,
        help="Run specific dataset (default: all)",
    )
    parser.add_argument(
        "--with-llm",
        action="store_true",
        help="Include LLM rounds (HyDE + multi-query). Requires API key.",
    )
    parser.add_argument(
        "--max-queries",
        type=int,
        default=_MAX_QUERIES,
        help=f"Max queries per dataset (default: {_MAX_QUERIES})",
    )
    parser.add_argument("--json", type=str, default=None, help="Save results to JSON file")
    parser.add_argument("--top-k", type=int, default=50, help="Top-K per retrieval round")
    args = parser.parse_args(argv)

    max_queries = args.max_queries

    datasets = [args.dataset] if args.dataset else list(_DATASETS)
    max_rounds = 3 if args.with_llm else 2
    from kaos_agents.settings import DEFAULT_MODEL

    model = DEFAULT_MODEL if args.with_llm else None

    all_results: list[DatasetResult] = []

    for ds_name in datasets:
        sys.stdout.write(f"\n{'=' * 60}\n")
        sys.stdout.write(f"BENCHMARK: {ds_name}\n")
        sys.stdout.write(f"{'=' * 60}\n")
        sys.stdout.flush()

        t0 = time.perf_counter()
        if ds_name == "edgar":
            memory, qrels, queries = _load_edgar()
        else:
            memory, qrels, queries = _load_beir(ds_name, max_queries=max_queries)
        t1 = time.perf_counter()

        from kaos_agents.types.memory import MemoryType

        n_docs = memory.section_item_count(MemoryType.DOCUMENTS)
        sys.stdout.write(f"Loaded {n_docs} docs in {t1 - t0:.1f}s\n")
        sys.stdout.write(f"Queries: {len(queries)} (max_rounds={max_rounds})\n")
        sys.stdout.flush()

        result = run_retrieval_benchmark(
            memory, qrels, queries, ds_name, max_rounds=max_rounds, top_k=args.top_k, model=model
        )
        all_results.append(result)

        sys.stdout.write(f"\nResults for {ds_name}:\n")
        for q in result.queries:
            sys.stdout.write(
                f"  [{q.query_id}] recall={q.recall:.1%} "
                f"({q.true_positives}/{q.ground_truth_count})"
                f" unique={q.unique_items} rounds={q.total_rounds} {q.latency_s:.1f}s\n"
            )
        sys.stdout.write(
            f"  AVG recall={result.avg_recall:.1%}, latency={result.avg_latency_s:.1f}s\n"
        )
        sys.stdout.flush()

    # Summary table
    sys.stdout.write(f"\n{'=' * 60}\n")
    sys.stdout.write("CROSS-DOMAIN SUMMARY\n")
    sys.stdout.write(f"{'=' * 60}\n")
    sys.stdout.write(
        f"{'Dataset':<12} {'Corpus':>6} {'Queries':>7} {'Avg Recall':>10} {'Avg Latency':>11}\n"
    )
    for r in all_results:
        sys.stdout.write(
            f"{r.dataset:<12} {r.corpus_size:>6} {r.num_queries:>7} "
            f"{r.avg_recall:>9.1%} {r.avg_latency_s:>10.1f}s\n"
        )
    sys.stdout.flush()

    if args.json:
        output = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "config": {
                "max_rounds": max_rounds,
                "with_llm": args.with_llm,
                "top_k": args.top_k,
                "max_queries": args.max_queries,
            },
            "results": [asdict(r) for r in all_results],
        }
        Path(args.json).write_text(json.dumps(output, indent=2, default=str))
        sys.stdout.write(f"\nResults saved to {args.json}\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
