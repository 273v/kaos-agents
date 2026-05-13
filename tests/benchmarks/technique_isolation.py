"""Measure each retrieval technique in isolation against the BM25 baseline.

Tests individual techniques on BEIR datasets to determine which help and
which hurt. Any technique that degrades NDCG@10 on ANY dataset is flagged.

Techniques (no LLM required):
- bm25: Plain BM25 baseline
- lexicon: BM25 with OpenGloss synonym/inflection expansion
- prf: BM25 + Pseudo-Relevance Feedback (top-term re-query)

Techniques (LLM required, use --with-llm):
- hyde: BM25 with HyDE pseudo-document generation

Usage::

    # Run cheap techniques on NFCorpus
    uv run python tests/benchmarks/technique_isolation.py --dataset nfcorpus

    # Run all techniques including LLM-based
    uv run python tests/benchmarks/technique_isolation.py --dataset nfcorpus --with-llm

    # Run on all 3 datasets
    uv run python tests/benchmarks/technique_isolation.py --all

    # Save results
    uv run python tests/benchmarks/technique_isolation.py --all --json results.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

from tests.benchmarks.metrics import compute_ap, compute_ndcg, compute_recall

_BEIR_DATASETS = ("nfcorpus", "scifact", "fiqa")
_CHEAP_TECHNIQUES = ("bm25", "lexicon", "prf")
_RERANK_TECHNIQUES = ("bm25_rerank",)
_LLM_TECHNIQUES = ("hyde",)


@dataclass(frozen=True, slots=True)
class TechniqueResult:
    """Result for one technique on one dataset."""

    dataset: str
    technique: str
    ndcg_at_10: float
    map_at_100: float
    recall_at_10: float
    recall_at_100: float
    avg_latency_s: float
    num_queries: int


def _evaluate_ranked(
    all_ranked: dict[str, list[str]],
    qrels: dict[str, dict[str, int]],
    test_qids: list[str],
    latencies: dict[str, float],
) -> tuple[float, float, float, float, float]:
    """Compute NDCG@10, MAP@100, R@10, R@100, avg_latency."""
    ndcg_scores, ap_scores, r10_scores, r100_scores = [], [], [], []
    for qid in test_qids:
        if qid not in all_ranked:
            continue
        ranked = all_ranked[qid]
        qrel_scores = qrels.get(qid, {})
        relevant = {did for did, s in qrel_scores.items() if s > 0}
        ndcg_scores.append(compute_ndcg(ranked, qrel_scores, 10))
        ap_scores.append(compute_ap(ranked, relevant, 100))
        r10_scores.append(compute_recall(ranked, relevant, 10))
        r100_scores.append(compute_recall(ranked, relevant, 100))
    n = len(ndcg_scores) or 1
    avg_lat = sum(latencies.values()) / len(latencies) if latencies else 0.0
    return (
        sum(ndcg_scores) / n,
        sum(ap_scores) / n,
        sum(r10_scores) / n,
        sum(r100_scores) / n,
        avg_lat,
    )


def _resolve_uri(memory, hit) -> str:
    """Get the URI from a search result."""
    from kaos_agents.types.memory import MemoryType

    items = memory.get_by_ids(MemoryType.DOCUMENTS, {hit.item_id})
    if items:
        return (items[0].metadata or {}).get("uri", "")
    return ""


def run_bm25(memory, queries, test_qids):
    """Plain BM25 — no expansion."""
    from kaos_agents.memory.search import search_memory
    from kaos_agents.types.memory import MemoryType

    all_ranked: dict[str, list[str]] = {}
    latencies: dict[str, float] = {}
    for i, qid in enumerate(test_qids):
        t0 = time.perf_counter()
        hits = search_memory(
            memory,
            queries[qid],
            sections=[MemoryType.DOCUMENTS],
            top_k=100,
            expand_relations=[],
        )
        latencies[qid] = time.perf_counter() - t0
        all_ranked[qid] = [_resolve_uri(memory, h) for h in hits]
        if (i + 1) % 100 == 0:
            sys.stdout.write(f"    bm25: {i + 1}/{len(test_qids)}\n")
            sys.stdout.flush()
    return all_ranked, latencies


def run_lexicon(memory, queries, test_qids):
    """BM25 with Lexicon synonym + inflection expansion."""
    from kaos_agents.memory.search import search_memory
    from kaos_agents.types.memory import MemoryType

    all_ranked: dict[str, list[str]] = {}
    latencies: dict[str, float] = {}
    for i, qid in enumerate(test_qids):
        t0 = time.perf_counter()
        hits = search_memory(
            memory,
            queries[qid],
            sections=[MemoryType.DOCUMENTS],
            top_k=100,
            expand_relations=["synonym", "inflection"],
        )
        latencies[qid] = time.perf_counter() - t0
        all_ranked[qid] = [_resolve_uri(memory, h) for h in hits]
        if (i + 1) % 100 == 0:
            sys.stdout.write(f"    lexicon: {i + 1}/{len(test_qids)}\n")
            sys.stdout.flush()
    return all_ranked, latencies


def run_prf(memory, queries, test_qids):
    """BM25 + Pseudo-Relevance Feedback (re-query with top terms from initial results)."""
    from kaos_agents.memory.search import search_memory
    from kaos_agents.types.memory import MemoryType

    all_ranked: dict[str, list[str]] = {}
    latencies: dict[str, float] = {}

    for i, qid in enumerate(test_qids):
        t0 = time.perf_counter()

        # Initial BM25 pass
        initial_hits = search_memory(
            memory,
            queries[qid],
            sections=[MemoryType.DOCUMENTS],
            top_k=10,
            expand_relations=[],
        )

        # Extract top terms from initial results (simple TF)
        if initial_hits:
            word_counts: Counter[str] = Counter()
            for h in initial_hits[:5]:
                words = h.content.lower().split()
                word_counts.update(w for w in words if len(w) >= 4)
            # Remove query terms and stopwords
            query_terms = set(queries[qid].lower().split())
            top_terms = [w for w, _ in word_counts.most_common(10) if w not in query_terms][:5]
            expanded_query = queries[qid] + " " + " ".join(top_terms)
        else:
            expanded_query = queries[qid]

        # Re-query with expanded terms
        hits = search_memory(
            memory,
            expanded_query,
            sections=[MemoryType.DOCUMENTS],
            top_k=100,
            expand_relations=[],
        )
        latencies[qid] = time.perf_counter() - t0
        all_ranked[qid] = [_resolve_uri(memory, h) for h in hits]

        if (i + 1) % 100 == 0:
            sys.stdout.write(f"    prf: {i + 1}/{len(test_qids)}\n")
            sys.stdout.flush()

    return all_ranked, latencies


def run_hyde(memory, queries, test_qids):
    """BM25 with HyDE pseudo-document generation (requires LLM)."""
    from kaos_agents.context.retrieval import _generate_pseudo_document
    from kaos_agents.memory.search import search_memory
    from kaos_agents.types.memory import MemoryType

    all_ranked: dict[str, list[str]] = {}
    latencies: dict[str, float] = {}

    for i, qid in enumerate(test_qids):
        t0 = time.perf_counter()
        pseudo_doc = _generate_pseudo_document(queries[qid])
        search_query = pseudo_doc if pseudo_doc else queries[qid]
        hits = search_memory(
            memory,
            search_query,
            sections=[MemoryType.DOCUMENTS],
            top_k=100,
            expand_relations=[],
        )
        latencies[qid] = time.perf_counter() - t0
        all_ranked[qid] = [_resolve_uri(memory, h) for h in hits]

        if (i + 1) % 10 == 0:
            sys.stdout.write(f"    hyde: {i + 1}/{len(test_qids)}\n")
            sys.stdout.flush()

    return all_ranked, latencies


def run_bm25_rerank(memory, queries, test_qids):
    """BM25 + cross-encoder reranking (requires sentence-transformers)."""
    import asyncio

    from kaos_agents.memory.search import search_memory
    from kaos_agents.types.memory import MemoryType

    try:
        from kaos_nlp_core.retrieval.protocol import RetrievalResult
        from kaos_nlp_transformers.reranker import CrossEncoderReranker

        reranker = CrossEncoderReranker.load()
    except ImportError as exc:
        sys.stderr.write(f"    bm25_rerank: skipped — {exc}\n")
        return {}, {}

    _PASSAGE_TRUNCATE = 2000

    all_ranked: dict[str, list[str]] = {}
    latencies: dict[str, float] = {}

    for i, qid in enumerate(test_qids):
        t0 = time.perf_counter()
        hits = search_memory(
            memory,
            queries[qid],
            sections=[MemoryType.DOCUMENTS],
            top_k=100,
            expand_relations=[],
        )

        if len(hits) >= 5:
            retrieval_results = [
                RetrievalResult(
                    text=h.content[:_PASSAGE_TRUNCATE],
                    score=h.score,
                    doc_id=h.item_id,
                )
                for h in hits
            ]
            ranked = asyncio.run(reranker.rerank(queries[qid], retrieval_results, top_k=100))
            # Map back to URIs
            item_id_to_uri: dict[str, str] = {}
            for h in hits:
                uri = _resolve_uri(memory, h)
                item_id_to_uri[h.item_id] = uri
            all_ranked[qid] = [item_id_to_uri.get(r.result.doc_id, "") for r in ranked]
        else:
            all_ranked[qid] = [_resolve_uri(memory, h) for h in hits]

        latencies[qid] = time.perf_counter() - t0

        if (i + 1) % 50 == 0:
            sys.stdout.write(f"    bm25_rerank: {i + 1}/{len(test_qids)}\n")
            sys.stdout.flush()

    return all_ranked, latencies


_RUNNERS = {
    "bm25": run_bm25,
    "lexicon": run_lexicon,
    "prf": run_prf,
    "bm25_rerank": run_bm25_rerank,
    "hyde": run_hyde,
}


def load_and_index(dataset_name: str):
    """Load BEIR dataset and build memory index."""
    from tests.benchmarks.beir_eval import build_memory, load_beir_dataset

    corpus, queries, qrels, test_qids = load_beir_dataset(dataset_name)
    memory, _ = build_memory(corpus)
    return memory, queries, qrels, test_qids, len(corpus)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Technique isolation benchmark")
    parser.add_argument("--dataset", choices=_BEIR_DATASETS, default=None)
    parser.add_argument("--all", action="store_true", help="Run on all 3 datasets")
    parser.add_argument(
        "--with-rerank", action="store_true", help="Include cross-encoder reranking"
    )
    parser.add_argument("--with-llm", action="store_true", help="Include LLM techniques (hyde)")
    parser.add_argument("--json", type=str, default=None)
    args = parser.parse_args(argv)

    datasets = list(_BEIR_DATASETS) if args.all else [args.dataset or "nfcorpus"]
    techniques = list(_CHEAP_TECHNIQUES)
    if args.with_rerank:
        techniques.extend(_RERANK_TECHNIQUES)
    if args.with_llm:
        techniques.extend(_LLM_TECHNIQUES)

    all_results: list[TechniqueResult] = []

    for ds in datasets:
        sys.stdout.write(f"\n{'=' * 60}\n{ds}\n{'=' * 60}\n")
        sys.stdout.flush()

        memory, queries, qrels, test_qids, corpus_size = load_and_index(ds)
        sys.stdout.write(f"  {corpus_size} docs, {len(test_qids)} queries\n\n")
        sys.stdout.flush()

        for tech in techniques:
            sys.stdout.write(f"  Running {tech}...\n")
            sys.stdout.flush()

            runner = _RUNNERS[tech]
            ranked, latencies = runner(memory, queries, test_qids)
            ndcg, mapp, r10, r100, avg_lat = _evaluate_ranked(ranked, qrels, test_qids, latencies)

            result = TechniqueResult(
                dataset=ds,
                technique=tech,
                ndcg_at_10=ndcg,
                map_at_100=mapp,
                recall_at_10=r10,
                recall_at_100=r100,
                avg_latency_s=avg_lat,
                num_queries=len(test_qids),
            )
            all_results.append(result)

            sys.stdout.write(
                f"    NDCG@10={ndcg:.4f}  MAP@100={mapp:.4f}  "
                f"R@10={r10:.1%}  R@100={r100:.1%}  "
                f"latency={avg_lat:.3f}s\n"
            )
            sys.stdout.flush()

    # Summary table
    sys.stdout.write(f"\n{'=' * 70}\nSUMMARY\n{'=' * 70}\n")
    sys.stdout.write(
        f"{'Dataset':<12} {'Technique':<10} {'NDCG@10':>8} {'MAP@100':>8} "
        f"{'R@10':>6} {'R@100':>6} {'Delta':>7}\n"
    )
    sys.stdout.write(f"{'-' * 70}\n")

    baselines: dict[str, float] = {}
    for r in all_results:
        if r.technique == "bm25":
            baselines[r.dataset] = r.ndcg_at_10

    for r in all_results:
        base = baselines.get(r.dataset, 0.0)
        delta = r.ndcg_at_10 - base
        flag = " HURTS" if delta < -0.005 and r.technique != "bm25" else ""
        sys.stdout.write(
            f"{r.dataset:<12} {r.technique:<10} {r.ndcg_at_10:>8.4f} "
            f"{r.map_at_100:>8.4f} {r.recall_at_10:>5.1%} {r.recall_at_100:>5.1%} "
            f"{delta:>+7.4f}{flag}\n"
        )
    sys.stdout.flush()

    if args.json:
        output = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "datasets": datasets,
            "techniques": techniques,
            "results": [asdict(r) for r in all_results],
            "baselines": baselines,
        }
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(output, indent=2))
        sys.stdout.write(f"\nSaved to {args.json}\n")


if __name__ == "__main__":
    main()
