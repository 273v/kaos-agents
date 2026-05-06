"""Needle-in-haystack benchmark: multi-phase agent planning on 997 EDGAR agreements.

.. deprecated::
    This benchmark uses cherry-picked queries with regex ground truth.
    Use ``tests/benchmarks/beir_eval.py`` (standard IR metrics on BEIR datasets) or
    ``tests/benchmarks/agent_eval.py`` (full agent evaluation) instead.

Tests the kaos-agents planning layer on realistic legal research tasks
that require triage (finding needles), extraction, cross-referencing,
and synthesis across a ~1000-document corpus.

Ground truth is pre-computed from regex patterns in the raw text so we
can measure recall/precision of the agent's BM25 triage and LLM analysis.

Usage::

    # Run the triage-only benchmark (no LLM, tests BM25 retrieval)
    uv run python tests/benchmarks/needle_haystack_benchmark.py --triage-only

    # Run full agent benchmark (requires API keys)
    uv run python tests/benchmarks/needle_haystack_benchmark.py --scenario 1

    # Run all scenarios
    uv run python tests/benchmarks/needle_haystack_benchmark.py --all
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import warnings
from dataclasses import dataclass

from kaos_agents.context.assemble import assemble_context
from kaos_agents.context.triage import triage_corpus
from kaos_agents.memory.session import SessionMemory
from kaos_agents.types.memory import MemoryType


@dataclass(frozen=True, slots=True)
class GroundTruthDoc:
    """A document known to contain a needle."""

    short_id: str
    identifier: str
    reason: str
    detail: str


@dataclass(slots=True)  # Mutable: metrics accumulated during benchmark
class BenchmarkResult:
    """Results from one benchmark scenario."""

    scenario: str = ""
    total_docs: int = 0
    ground_truth_count: int = 0

    # Triage metrics
    triage_selected: int = 0
    triage_true_positives: int = 0
    triage_recall: float = 0.0
    triage_precision: float = 0.0
    triage_latency_ms: float = 0.0

    # Assemble metrics
    assemble_selected: int = 0
    assemble_true_positives: int = 0
    assemble_latency_ms: float = 0.0

    # Agent metrics (only populated for full runs)
    agent_plan_steps: int = 0
    agent_cost_usd: float = 0.0
    agent_tokens: int = 0
    agent_latency_s: float = 0.0
    agent_found_needles: int = 0

    def to_dict(self) -> dict:
        from dataclasses import asdict

        return asdict(self)


def _short_id(identifier: str) -> str:
    parts = identifier.rstrip("/").split("/")
    return parts[-1] if parts else identifier


def compute_ground_truth_noncompete(
    docs: list[tuple[str, str]],
) -> list[GroundTruthDoc]:
    """Find documents that contain non-compete clauses with extractable durations."""
    results: list[GroundTruthDoc] = []
    seen: set[str] = set()

    for identifier, text in docs:
        lower = text.lower()
        has_nc = any(
            t in lower
            for t in [
                "non-compete",
                "non-competition",
                "noncompete",
                "noncompetition",
            ]
        )
        if not has_nc:
            continue

        for match in re.finditer(r"non[- ]?compet\w*", lower):
            window_start = max(0, match.start() - 100)
            window_end = min(len(text), match.end() + 500)
            window = text[window_start:window_end]

            dur_match = re.search(r"(\w+)\s*\((\d+)\)\s*(month|year)s?", window, re.IGNORECASE)
            if dur_match:
                num = int(dur_match.group(2))
                unit = dur_match.group(3).lower()
                months = num * 12 if "year" in unit else num
                sid = _short_id(identifier)
                if sid not in seen:
                    seen.add(sid)
                    results.append(
                        GroundTruthDoc(
                            short_id=sid,
                            identifier=identifier,
                            reason="non-compete with duration",
                            detail=f"{months} months ({dur_match.group(0)})",
                        )
                    )
                break

    return results


def compute_ground_truth_coc_acceleration(
    docs: list[tuple[str, str]],
) -> list[GroundTruthDoc]:
    """Find documents with Change of Control + acceleration/vesting language."""
    results: list[GroundTruthDoc] = []

    for identifier, text in docs:
        lower = text.lower()
        has_coc = "change of control" in lower or "change in control" in lower
        has_accel = "acceler" in lower and ("vest" in lower or "option" in lower)
        if has_coc and has_accel:
            sid = _short_id(identifier)
            results.append(
                GroundTruthDoc(
                    short_id=sid,
                    identifier=identifier,
                    reason="CoC with acceleration",
                    detail="change of control + acceleration/vesting",
                )
            )

    return results


def load_corpus_into_memory(
    docs: list[tuple[str, str]],
) -> SessionMemory:
    """Load documents into a SessionMemory DOCUMENTS section."""
    memory = SessionMemory("benchmark-needle")
    for identifier, text in docs:
        sid = _short_id(identifier)
        memory.add(
            MemoryType.DOCUMENTS,
            text,
            metadata={"uri": f"edgar:{sid}", "identifier": identifier},
        )
    return memory


def run_triage_benchmark(
    memory: SessionMemory,
    query: str,
    ground_truth: list[GroundTruthDoc],
    *,
    max_selected: int = 50,
) -> BenchmarkResult:
    """Run BM25 triage and measure recall/precision against ground truth."""
    gt_ids = {gt.short_id for gt in ground_truth}

    t0 = time.perf_counter()
    triage = triage_corpus(memory, query, threshold=20, max_selected=max_selected)
    t1 = time.perf_counter()

    if triage is None:
        return BenchmarkResult(
            scenario=f"triage: {query[:50]}",
            total_docs=memory.section_item_count(MemoryType.DOCUMENTS),
            ground_truth_count=len(ground_truth),
        )

    selected_ids: set[str] = set()
    for uri in triage.selected_uris:
        sid = uri.removeprefix("edgar:")
        selected_ids.add(sid)

    true_positives = len(selected_ids & gt_ids)
    recall = true_positives / len(gt_ids) if gt_ids else 0.0
    precision = true_positives / len(selected_ids) if selected_ids else 0.0

    return BenchmarkResult(
        scenario=f"triage: {query[:50]}",
        total_docs=triage.total_documents,
        ground_truth_count=len(ground_truth),
        triage_selected=triage.selected_count,
        triage_true_positives=true_positives,
        triage_recall=recall,
        triage_precision=precision,
        triage_latency_ms=(t1 - t0) * 1000,
    )


def run_assemble_benchmark(
    memory: SessionMemory,
    query: str,
    ground_truth: list[GroundTruthDoc],
) -> BenchmarkResult:
    """Run assemble_context and measure recall against ground truth."""
    gt_ids = {gt.short_id for gt in ground_truth}

    t0 = time.perf_counter()
    result = assemble_context(
        memory,
        query,
        sections=[MemoryType.DOCUMENTS],
        total_budget_tokens=100_000,
        retrieval_threshold=20,
        search_top_k=50,
    )
    t1 = time.perf_counter()

    docs = result.get(MemoryType.DOCUMENTS, [])
    selected_ids: set[str] = set()
    for item in docs:
        uri = (item.metadata or {}).get("uri", "")
        sid = uri.removeprefix("edgar:")
        selected_ids.add(sid)

    true_positives = len(selected_ids & gt_ids)

    return BenchmarkResult(
        scenario=f"assemble: {query[:50]}",
        total_docs=memory.section_item_count(MemoryType.DOCUMENTS),
        ground_truth_count=len(ground_truth),
        assemble_selected=len(docs),
        assemble_true_positives=true_positives,
        assemble_latency_ms=(t1 - t0) * 1000,
    )


def main(argv: list[str] | None = None) -> None:
    warnings.warn(
        "needle_haystack_benchmark is deprecated. "
        "Use tests/benchmarks/beir_eval.py or tests/benchmarks/agent_eval.py instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    print(
        "\u26a0 DEPRECATED: This benchmark uses cherry-picked queries with regex ground truth.\n"
        "Use tests/benchmarks/beir_eval.py (standard IR metrics on BEIR datasets) or\n"
        "tests/benchmarks/agent_eval.py (full agent evaluation) instead.\n",
        file=sys.stderr,
    )

    parser = argparse.ArgumentParser(description="Needle-in-haystack benchmark")
    parser.add_argument(
        "--triage-only",
        action="store_true",
        help="Run triage-only benchmarks (no LLM calls)",
    )
    parser.add_argument("--scenario", type=int, help="Run specific scenario (1, 2, or 3)")
    parser.add_argument("--all", action="store_true", help="Run all scenarios")
    parser.add_argument("--max-docs", type=int, default=None, help="Cap on documents to load")
    parser.add_argument("--json", action="store_true", help="Output JSON")
    args = parser.parse_args(argv)

    from tests.benchmarks.kl3m_loader import load_edgar_agreements

    print("Loading EDGAR agreements corpus...")
    t0 = time.perf_counter()
    docs = load_edgar_agreements(max_docs=args.max_docs, max_chars_per_doc=50_000)
    t1 = time.perf_counter()
    print(f"Loaded {len(docs)} documents in {t1 - t0:.1f}s")

    print("Computing ground truth...")
    gt_noncomp = compute_ground_truth_noncompete(docs)
    gt_coc = compute_ground_truth_coc_acceleration(docs)
    print(f"  Non-compete with durations: {len(gt_noncomp)} documents")
    print(f"  CoC + acceleration: {len(gt_coc)} documents")

    print("Loading into SessionMemory...")
    memory = load_corpus_into_memory(docs)
    total = memory.section_item_count(MemoryType.DOCUMENTS)
    print(f"  {total} documents in DOCUMENTS section")
    print()

    results: list[BenchmarkResult] = []

    if args.triage_only or args.all or args.scenario is None:
        print("=" * 60)
        print("TRIAGE BENCHMARKS (BM25 only, no LLM)")
        print("=" * 60)

        # Scenario 1: Non-compete triage
        queries_s1 = [
            "non-compete non-competition restrictive covenant duration",
            "employee non-compete agreement years months",
            "covenant not to compete period",
        ]
        for query in queries_s1:
            r = run_triage_benchmark(memory, query, gt_noncomp, max_selected=50)
            results.append(r)
            print(
                f"  [{r.scenario}]\n"
                f"    Selected: {r.triage_selected}/{r.total_docs} | "
                f"TP: {r.triage_true_positives}/{r.ground_truth_count} | "
                f"Recall: {r.triage_recall:.1%} | Precision: {r.triage_precision:.1%} | "
                f"{r.triage_latency_ms:.1f}ms"
            )

        print()

        # Scenario 2: CoC acceleration triage
        queries_s2 = [
            "change of control acceleration vesting",
            "change in control equity option accelerate",
            "termination change control vesting acceleration provisions",
        ]
        for query in queries_s2:
            r = run_triage_benchmark(memory, query, gt_coc, max_selected=50)
            results.append(r)
            print(
                f"  [{r.scenario}]\n"
                f"    Selected: {r.triage_selected}/{r.total_docs} | "
                f"TP: {r.triage_true_positives}/{r.ground_truth_count} | "
                f"Recall: {r.triage_recall:.1%} | Precision: {r.triage_precision:.1%} | "
                f"{r.triage_latency_ms:.1f}ms"
            )

        print()

        # Assemble benchmarks
        print("ASSEMBLE CONTEXT BENCHMARKS")
        print("-" * 40)
        r = run_assemble_benchmark(memory, "non-compete duration years", gt_noncomp)
        results.append(r)
        print(
            f"  [{r.scenario}]\n"
            f"    Selected: {r.assemble_selected} | TP: {r.assemble_true_positives}/"
            f"{r.ground_truth_count} | {r.assemble_latency_ms:.1f}ms"
        )

        r = run_assemble_benchmark(memory, "change of control acceleration", gt_coc)
        results.append(r)
        print(
            f"  [{r.scenario}]\n"
            f"    Selected: {r.assemble_selected} | TP: {r.assemble_true_positives}/"
            f"{r.ground_truth_count} | {r.assemble_latency_ms:.1f}ms"
        )

    if args.json:
        print()
        print(json.dumps([r.to_dict() for r in results], indent=2))


if __name__ == "__main__":
    main()
