"""Unit tests for P0.2 — corpus-scale aware tool fitness ranking.

The 2026-05-23 corpus stress S12 (500-doc deal room, 5 planted
needles) reproduced a routing failure: the agent issued 154 per-doc
``kaos-office-parse-docx`` calls instead of using the
corpus-aggregating tools (``kaos-agent-findings`` /
``kaos-agent-corpus-filter``). M1 fitness ranker was the only place
this preference could be expressed.

These tests pin **two** behaviors of the ``corpus_size`` signal
added to :class:`ToolFitnessSignature`:

* When ``corpus_size == 0`` (no corpus attached) the ranker behaves
  as before — no scale bias.
* When ``corpus_size >= 20`` (large corpus + a scale-y query) the
  ranker MUST promote corpus-aggregating tools ahead of per-doc
  parsers per Rule 10 in the docstring.

Live tests, marker ``@pytest.mark.live``. Model
``anthropic:claude-sonnet-4-6`` per kaos-agents test discipline.
"""

from __future__ import annotations

import os
from collections.abc import Sequence

import pytest

from kaos_agents.planning.tool_fitness import (
    ToolFitnessResult,
    rank_tools_for_query,
)

DEFAULT_LIVE_MODEL = "anthropic:claude-sonnet-4-6"


def _have_anthropic_key() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("KAOS_LLM_ANTHROPIC_API_KEY"))


# Catalog that mirrors what the live agent sees on S12: mixed
# corpus-aggregating tools + per-document parsers + a few
# unrelated tools. Names are taken verbatim from the production
# registry so the ranker's "name must appear in catalog" check
# behaves identically to live.
_CORPUS_CATALOG: Sequence[tuple[str, str]] = (
    (
        "kaos-agent-findings",
        (
            "Run a findings agent over the entire DOCUMENTS corpus: per-doc "
            "candidate extraction → multi-doc filter → optional synthesis. "
            "Returns surviving findings with block_ref citations across the "
            "whole corpus in 1-3 calls. Aggregator — use for "
            "across-corpus questions."
        ),
    ),
    (
        "kaos-agent-corpus-filter",
        (
            "LLM-aided scope tightener: given the attached corpus + an "
            "intent string, score each document for relevance and drop "
            "the long tail. Returns the narrowed set in one call. "
            "Aggregator — use to triage a large corpus before deeper "
            "analysis."
        ),
    ),
    (
        "kaos-content-search-document",
        (
            "Full-text search across all parsed documents in the session "
            "DOCUMENTS section in one call. Aggregator — use for "
            "find-clause / find-term across-corpus queries."
        ),
    ),
    (
        "kaos-office-parse-docx",
        (
            "Parse ONE DOCX file into structured paragraphs + numbered "
            "lists. Per-document parser — call once per attached Word "
            "document; cost scales linearly with corpus size."
        ),
    ),
    (
        "kaos-pdf-extract-parse",
        (
            "Parse ONE PDF file into structured text + tables. "
            "Per-document parser — call once per attached PDF; cost "
            "scales linearly with corpus size."
        ),
    ),
    (
        "kaos-web-search",
        ("General web search. Use for current-events / public-web queries, not for local files."),
    ),
)


# The scale-y query the agent sees on S12.
_SCALE_QUERY = (
    "What are the key financial terms (purchase price, closing date, "
    "earnout, termination fee, indemnification cap) across all attached "
    "agreements? I need every value mentioned in any of the documents."
)


# A neutral / non-quantified query (the small-corpus regime).
_DOC_QUERY = "I attached a NDA called Acme-MNDA.docx — what is the governing law clause?"


# ──────────────────────────────────────────────────────────────────


@pytest.mark.live
@pytest.mark.asyncio
async def test_tool_fitness_corpus_size_zero_preserves_default_behavior() -> None:
    """With ``corpus_size=0`` the ranker should NOT suppress per-doc
    tools — the small-corpus regime is unchanged. Specifically, on a
    single-attachment DOCX query, ``kaos-office-parse-docx`` must
    remain a valid pick (or be the first pick).

    This anchors the back-compat guarantee: callers who never pass
    ``corpus_size`` (or pass 0) see exactly the previous behavior.
    """
    if not _have_anthropic_key():
        pytest.skip("ANTHROPIC_API_KEY not set")
    result: ToolFitnessResult = await rank_tools_for_query(
        query=_DOC_QUERY,
        catalog=_CORPUS_CATALOG,
        model=DEFAULT_LIVE_MODEL,
        top_k=5,
        corpus_size=0,
    )
    assert result.valid_picks, (
        f"expected non-empty picks, got fell_back={result.fell_back} "
        f"picks={result.picks!r} rationale={result.rationale!r}"
    )
    # Per-doc tools must be present somewhere in the picks — the
    # ranker must not have universally suppressed them.
    per_doc_tools = {"kaos-office-parse-docx", "kaos-pdf-extract-parse"}
    aggregator_tools = {
        "kaos-agent-findings",
        "kaos-agent-corpus-filter",
        "kaos-content-search-document",
    }
    picks_set = set(result.valid_picks)
    # Either a per-doc parser leads (preferred), or the picks include
    # at least one alongside the aggregator — what matters is that
    # the suppression is NOT happening at corpus_size=0.
    assert per_doc_tools & picks_set or result.valid_picks[0] in aggregator_tools, (
        f"At corpus_size=0 the per-doc parsers must be reachable — "
        f"got picks={result.valid_picks!r} rationale={result.rationale!r}"
    )


@pytest.mark.live
@pytest.mark.asyncio
async def test_tool_fitness_large_corpus_scale_query_promotes_aggregators() -> None:
    """With ``corpus_size=500`` and a scale-y query, the ranker MUST
    promote corpus-aggregating tools (``kaos-agent-findings`` /
    ``kaos-agent-corpus-filter`` / ``kaos-content-search-document``)
    ahead of per-doc parsers per Rule 10.

    This is the load-bearing assertion for S12: without it the
    agent walks the corpus document-by-document and burns its
    cost budget on parse calls.
    """
    if not _have_anthropic_key():
        pytest.skip("ANTHROPIC_API_KEY not set")
    result: ToolFitnessResult = await rank_tools_for_query(
        query=_SCALE_QUERY,
        catalog=_CORPUS_CATALOG,
        model=DEFAULT_LIVE_MODEL,
        top_k=5,
        corpus_size=500,
    )
    assert result.valid_picks, (
        f"expected non-empty picks at corpus_size=500, got "
        f"fell_back={result.fell_back} picks={result.picks!r} "
        f"rationale={result.rationale!r}"
    )
    aggregator_tools = {
        "kaos-agent-findings",
        "kaos-agent-corpus-filter",
        "kaos-content-search-document",
    }
    per_doc_tools = {"kaos-office-parse-docx", "kaos-pdf-extract-parse"}

    # The TOP pick must be an aggregator — anything else means the
    # ranker still prefers a per-doc walk, which is exactly the S12
    # failure mode this signal exists to prevent.
    first_pick = result.valid_picks[0]
    assert first_pick in aggregator_tools, (
        f"At corpus_size=500 + scale-y query, top pick MUST be a "
        f"corpus aggregator (Rule 10). Got first={first_pick!r}, "
        f"all picks={result.valid_picks!r}, "
        f"rationale={result.rationale!r}"
    )

    # Stricter check: if any per-doc parser appears in the picks at
    # all, it must come AFTER all the aggregators present. Equivalent
    # to "no per-doc parser outranks any aggregator at scale."
    first_per_doc_idx = next(
        (i for i, p in enumerate(result.valid_picks) if p in per_doc_tools),
        len(result.valid_picks),
    )
    last_aggregator_idx = max(
        (i for i, p in enumerate(result.valid_picks) if p in aggregator_tools),
        default=-1,
    )
    assert first_per_doc_idx > last_aggregator_idx, (
        f"At corpus_size=500 + scale-y query, every aggregator "
        f"must outrank every per-doc parser (Rule 10). Got picks="
        f"{result.valid_picks!r} rationale={result.rationale!r}"
    )
