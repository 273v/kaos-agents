"""Benchmark: does lexicon expansion or multi-query rewriting improve
tool retrieval over plain BM25?

The G1 ``ToolRetrieval`` plumbing already accepts a ``lexicon=`` kwarg
and a single ``query`` string. This benchmark answers two open
questions before we decide whether to wire either into
``bridge_runtime_tools`` by default:

1. **Does lexicon expansion help on the tool catalog?** The platform-
   wide cross-domain BEIR finding (see
   ``feedback_benchmark_first.md``) was that lexicon expansion *hurts*
   on long documents. Tool descriptions are short (~50 tokens) and
   sparse — the docstring on ``ToolRetrieval`` already speculates that
   they're a special case where expansion may help. This is the
   empirical check.

2. **Does multi-query rewriting help?** Have an LLM propose N
   paraphrases of the query, run BM25 on each, fuse with Reciprocal
   Rank Fusion. This is the standard query-expansion baseline from
   the IR literature (Wang et al. 2023 *Query2doc*; Gao et al. 2023
   *HyDE* — both use a generative model to rewrite, then retrieve).

Conditions tested (full factorial: 2 x 2 = 4):
  A: plain BM25                           — baseline
  B: BM25 + lexicon synonyms              — synonym expansion at index
  C: BM25 + multi-query (RRF-fused)       — LLM rewrites
  D: BM25 + lexicon + multi-query         — both

Metrics:
  * Hit@K (K=1, 3, 5): fraction of queries with ≥1 relevant tool in top-K
  * MRR@10: mean reciprocal rank of first relevant hit, capped at K=10

Sliced by category (direct / synonym / conceptual) and overall.

To run:
  uv run --no-sync python tests/benchmarks/tool_retrieval_bench.py

Saves a JSON report alongside this file for downstream comparison.
Costs ~25 cheap Anthropic calls when multi-query is enabled
(``--with-multi-query``). Skip the LLM tier with ``--no-multi-query``.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from kaos_core.registry.container import KaosRuntime
from kaos_nlp_core.lexicon import default_opengloss_lexicon

from kaos_agents.runtime.tool_retrieval import ToolRetrieval
from tests.benchmarks.tool_retrieval_queries import (
    BENCHMARK_QUERIES,
    LabeledQuery,
    queries_by_category,
)

DEFAULT_REWRITE_MODEL = "anthropic:claude-haiku-4-5"
DEFAULT_NUM_REWRITES = 3
DEFAULT_RRF_K = 60  # standard RRF constant per Cormack et al. 2009


# ---------------------------------------------------------------------------
# Catalog
# ---------------------------------------------------------------------------


def make_real_catalog_runtime() -> KaosRuntime:
    """Same catalog as the integration tests — pdf + web + tabular + office."""
    runtime = KaosRuntime()
    loaded = 0
    for mod_name in (
        "kaos_pdf.tools",
        "kaos_web.tools",
        "kaos_tabular.tools",
        "kaos_office.tools",
    ):
        try:
            mod = importlib.import_module(mod_name)
        except ImportError:
            continue
        for attr in dir(mod):
            if attr.startswith("register_") and "tool" in attr:
                getattr(mod, attr)(runtime)
                loaded += 1
                break
    if loaded < 2:
        msg = f"Need at least 2 catalog modules; only {loaded} available"
        raise RuntimeError(msg)
    return runtime


# ---------------------------------------------------------------------------
# Multi-query rewriting (LLM-driven)
# ---------------------------------------------------------------------------


async def rewrite_query(query: str, *, n: int, model: str) -> list[str]:
    """Ask an LLM to propose N paraphrases of ``query``.

    Returns a list of length ``n`` (or fewer if the model abstains).
    The original query is NOT included — the caller should union the
    original with these rewrites.
    """
    from kaos_llm_core.programs.call import Call
    from kaos_llm_core.signatures.fields import InputField, OutputField
    from kaos_llm_core.signatures.signature import Signature

    class _RewriteSig(Signature):
        """Generate paraphrases of a tool-search query.

        The output paraphrases should preserve the user's intent but
        vary the vocabulary — use synonyms, hypernyms, alternative
        phrasings. The goal is to surface tools whose descriptions
        use different words for the same concept.
        """

        query: str = InputField(description="The original user query.")
        n: int = InputField(description="Number of paraphrases to generate.")
        rewrites: list[str] = OutputField(
            description=(
                "List of N paraphrases. Each should be a distinct "
                "rewording — not the same query with cosmetic changes. "
                "Use synonyms, alternate verbs, and different noun "
                "phrasings."
            ),
        )

    call = Call(_RewriteSig, model=model)
    result = await call(query=query, n=n)
    rewrites = list(result.rewrites)[:n]
    # Defensive: drop empties and exact dupes of the original.
    return [r for r in rewrites if r and r.strip() and r.strip() != query.strip()]


# ---------------------------------------------------------------------------
# Reciprocal Rank Fusion
# ---------------------------------------------------------------------------


def rrf_fuse(rankings: list[list[str]], *, k: int = DEFAULT_RRF_K) -> list[str]:
    """Standard RRF over multiple ranked lists.

    Cormack et al. 2009. Score for item i across rankings r_1..r_n is
    Σ 1/(k + rank_j(i)), where missing items contribute 0.
    Returns items sorted by fused score descending.
    """
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank)
    return sorted(scores, key=lambda x: scores[x], reverse=True)


# ---------------------------------------------------------------------------
# Single-condition evaluation
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class QueryResult:
    """Outcome of one query under one condition."""

    query: str
    category: str
    relevant: list[str]
    ranked: list[str]  # top-10 by default
    first_relevant_rank: int | None  # 1-based; None if no relevant in top-K
    hit_at_1: bool
    hit_at_3: bool
    hit_at_5: bool


def evaluate_ranking(
    q: LabeledQuery,
    ranked_tool_names: list[str],
) -> QueryResult:
    """Score one ranked list against the labeled relevant set."""
    first_rank: int | None = None
    for i, name in enumerate(ranked_tool_names, start=1):
        if name in q.relevant:
            first_rank = i
            break
    top1 = set(ranked_tool_names[:1])
    top3 = set(ranked_tool_names[:3])
    top5 = set(ranked_tool_names[:5])
    return QueryResult(
        query=q.query,
        category=q.category,
        relevant=sorted(q.relevant),
        ranked=ranked_tool_names[:10],
        first_relevant_rank=first_rank,
        hit_at_1=bool(top1 & q.relevant),
        hit_at_3=bool(top3 & q.relevant),
        hit_at_5=bool(top5 & q.relevant),
    )


def run_single_condition(
    retrieval: ToolRetrieval,
    query: str,
    *,
    top_k: int = 10,
) -> list[str]:
    """Plain single-query retrieval — returns ranked tool names."""
    hits = retrieval.search(query, top_k=top_k)
    return [h.tool.metadata.name for h in hits]


def run_multi_query_condition(
    retrieval: ToolRetrieval,
    queries: list[str],
    *,
    top_k: int = 10,
    rrf_k: int = DEFAULT_RRF_K,
) -> list[str]:
    """Run retrieval for each query, fuse via RRF."""
    rankings: list[list[str]] = []
    for q in queries:
        hits = retrieval.search(q, top_k=top_k)
        rankings.append([h.tool.metadata.name for h in hits])
    fused = rrf_fuse(rankings, k=rrf_k)
    return fused[:top_k]


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ConditionScores:
    """Aggregated scores for one (condition, slice) pair."""

    n: int
    hit_at_1: float
    hit_at_3: float
    hit_at_5: float
    mrr_at_10: float


def aggregate(results: list[QueryResult]) -> ConditionScores:
    n = len(results)
    if n == 0:
        return ConditionScores(0, 0.0, 0.0, 0.0, 0.0)
    h1 = sum(1 for r in results if r.hit_at_1) / n
    h3 = sum(1 for r in results if r.hit_at_3) / n
    h5 = sum(1 for r in results if r.hit_at_5) / n
    rr_sum = sum(
        1.0 / r.first_relevant_rank
        for r in results
        if r.first_relevant_rank is not None and r.first_relevant_rank <= 10
    )
    mrr = rr_sum / n
    return ConditionScores(n=n, hit_at_1=h1, hit_at_3=h3, hit_at_5=h5, mrr_at_10=mrr)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


async def main(*, with_multi_query: bool, num_rewrites: int, save_path: Path) -> None:
    runtime = make_real_catalog_runtime()
    catalog_size = len(list(runtime.tools.list_tool_objects()))
    print(f"Loaded catalog: {catalog_size} tools.")

    print("Loading lexicon...")
    t0 = time.perf_counter()
    lexicon = default_opengloss_lexicon()
    print(f"  loaded in {time.perf_counter() - t0:.2f}s.")

    retrieval_plain = ToolRetrieval.from_runtime(runtime)
    retrieval_lex = ToolRetrieval.from_runtime(runtime, lexicon=lexicon)

    # Pre-fetch rewrites once so we score the same paraphrases across
    # condition C (no lexicon) and condition D (with lexicon).
    rewrites_by_query: dict[str, list[str]] = {}
    if with_multi_query:
        print(
            f"Generating {num_rewrites} rewrites per query "
            f"({len(BENCHMARK_QUERIES)} queries) via {DEFAULT_REWRITE_MODEL}..."
        )
        t0 = time.perf_counter()
        for q in BENCHMARK_QUERIES:
            rewrites = await rewrite_query(
                q.query,
                n=num_rewrites,
                model=DEFAULT_REWRITE_MODEL,
            )
            rewrites_by_query[q.query] = rewrites
            print(f"  [{q.category}] {q.query!r}")
            for r in rewrites:
                print(f"     → {r!r}")
        print(f"  done in {time.perf_counter() - t0:.1f}s.")

    # Run all four conditions.
    print("\nEvaluating conditions...")
    results: dict[str, list[QueryResult]] = {"A": [], "B": [], "C": [], "D": []}
    for q in BENCHMARK_QUERIES:
        # Condition A: plain
        ranked_a = run_single_condition(retrieval_plain, q.query, top_k=10)
        results["A"].append(evaluate_ranking(q, ranked_a))

        # Condition B: plain + lexicon
        ranked_b = run_single_condition(retrieval_lex, q.query, top_k=10)
        results["B"].append(evaluate_ranking(q, ranked_b))

        if with_multi_query:
            rewrites = rewrites_by_query.get(q.query, [])
            all_queries = [q.query, *rewrites]
            # Condition C: multi-query, no lexicon
            ranked_c = run_multi_query_condition(
                retrieval_plain,
                all_queries,
                top_k=10,
            )
            results["C"].append(evaluate_ranking(q, ranked_c))
            # Condition D: multi-query + lexicon
            ranked_d = run_multi_query_condition(
                retrieval_lex,
                all_queries,
                top_k=10,
            )
            results["D"].append(evaluate_ranking(q, ranked_d))

    # Aggregate overall and per-category.
    by_category = queries_by_category()
    rows: list[dict] = []
    condition_names = {
        "A": "plain BM25",
        "B": "BM25 + lexicon",
        "C": "BM25 + multi-query",
        "D": "BM25 + lex + multi-query",
    }

    print("\n=== OVERALL ===")
    print(f"{'condition':<28} {'n':>3}  {'H@1':>6}  {'H@3':>6}  {'H@5':>6}  {'MRR@10':>7}")
    for cond in ("A", "B", "C", "D"):
        if not results[cond]:
            continue
        scores = aggregate(results[cond])
        rows.append(
            {
                "condition": condition_names[cond],
                "slice": "overall",
                **asdict(scores),
            }
        )
        print(
            f"{condition_names[cond]:<28} {scores.n:>3}  "
            f"{scores.hit_at_1:>6.1%}  {scores.hit_at_3:>6.1%}  "
            f"{scores.hit_at_5:>6.1%}  {scores.mrr_at_10:>7.3f}"
        )

    for cat in ("direct", "synonym", "conceptual"):
        if cat not in by_category:
            continue
        cat_queries = {lq.query for lq in by_category[cat]}
        print(f"\n=== {cat.upper()} (n={len(cat_queries)}) ===")
        print(f"{'condition':<28} {'n':>3}  {'H@1':>6}  {'H@3':>6}  {'H@5':>6}  {'MRR@10':>7}")
        for cond in ("A", "B", "C", "D"):
            if not results[cond]:
                continue
            cat_results = [r for r in results[cond] if r.query in cat_queries]
            scores = aggregate(cat_results)
            rows.append(
                {
                    "condition": condition_names[cond],
                    "slice": cat,
                    **asdict(scores),
                }
            )
            print(
                f"{condition_names[cond]:<28} {scores.n:>3}  "
                f"{scores.hit_at_1:>6.1%}  {scores.hit_at_3:>6.1%}  "
                f"{scores.hit_at_5:>6.1%}  {scores.mrr_at_10:>7.3f}"
            )

    # Per-query details for diagnostic review.
    detail: list[dict] = []
    for cond in ("A", "B", "C", "D"):
        for r in results[cond]:
            detail.append(
                {
                    "condition": cond,
                    "condition_name": condition_names[cond],
                    "query": r.query,
                    "category": r.category,
                    "relevant": r.relevant,
                    "ranked_top_10": r.ranked,
                    "first_relevant_rank": r.first_relevant_rank,
                    "hit_at_1": r.hit_at_1,
                    "hit_at_3": r.hit_at_3,
                    "hit_at_5": r.hit_at_5,
                }
            )

    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_path.write_text(
        json.dumps(
            {
                "catalog_size": catalog_size,
                "rewrite_model": DEFAULT_REWRITE_MODEL if with_multi_query else None,
                "num_rewrites_per_query": num_rewrites if with_multi_query else 0,
                "rrf_k": DEFAULT_RRF_K,
                "rewrites_by_query": rewrites_by_query,
                "summary": rows,
                "detail": detail,
            },
            indent=2,
        )
    )
    print(f"\nSaved full results to {save_path}")


def cli() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--with-multi-query",
        action="store_true",
        help="Run conditions C and D (requires ANTHROPIC_API_KEY).",
    )
    parser.add_argument(
        "--no-multi-query",
        dest="with_multi_query",
        action="store_false",
        help="Skip LLM rewrites; only run A and B (free).",
    )
    parser.add_argument(
        "--num-rewrites",
        type=int,
        default=DEFAULT_NUM_REWRITES,
        help=f"Paraphrases per query (default {DEFAULT_NUM_REWRITES}).",
    )
    parser.add_argument(
        "--save",
        type=Path,
        default=Path(__file__).parent / "data" / "tool_retrieval_results.json",
        help="Path to save full results JSON.",
    )
    parser.set_defaults(with_multi_query=True)
    args = parser.parse_args()

    if args.with_multi_query and "ANTHROPIC_API_KEY" not in os.environ:
        raise SystemExit(
            "ANTHROPIC_API_KEY missing — re-run with --no-multi-query, or export ANTHROPIC_API_KEY."
        )

    asyncio.run(
        main(
            with_multi_query=args.with_multi_query,
            num_rewrites=args.num_rewrites,
            save_path=args.save,
        )
    )


if __name__ == "__main__":
    cli()
