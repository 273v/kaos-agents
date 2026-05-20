"""Live evals for :class:`~kaos_agents.planning.tool_fitness.ToolFitnessSignature`.

M1 mitigation from
``kaos-modules/docs/plans/2026-05-19-agentic-failure-taxonomy.md``.

The 2026-05-19 kaos-agents-bench baseline showed that the canonical
KFM-B05 failure (senator-EDGAR mis-routing) reproduces on
``gpt-5.4-mini`` even with the full tool catalog visible: the
classifier correctly routes the query to ``tool_use`` and yet the
worker LLM emits text without dispatching any tool span. The
mitigation is a pre-ReAct fitness ranker that consumes the catalog
+ user query and emits top-K picks; the integration layer then
narrows the catalog presented to the worker.

These tests assert the **ranker primitive** behaves correctly on
both default-matrix models WITHOUT touching the integration layer:

* Given a catalog containing a web-search tool + several
  domain-specialised source tools + an irrelevant tool, the ranker
  picks the web-search tool first for a current-officeholder
  question.
* Given a catalog containing an office-document parser + several
  web tools, the ranker picks the parser first for an
  "attached DOCX" question.
* Given a catalog where no tool fits, the ranker honours the
  small-talk exit clause and returns an empty ``picks`` tuple
  (rule 7 in the Signature docstring).

Required env vars: ``OPENAI_API_KEY`` (for ``gpt-5.4-mini``) and
``ANTHROPIC_API_KEY`` (for ``claude-sonnet-4-6``). Cost budget per
case: < $0.01 each (Signature output is short).
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass

import pytest

from kaos_agents.planning.tool_fitness import (
    ToolFitnessResult,
    rank_tools_for_query,
)

MODELS: tuple[str, ...] = (
    "openai:gpt-5.4-mini",
    "anthropic:claude-sonnet-4-6",
)


def _have_key_for(model: str) -> bool:
    provider = model.split(":", 1)[0]
    env = {
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "google": "GOOGLE_API_KEY",
    }.get(provider)
    return bool(env and os.environ.get(env))


# Catalog snapshots — small, hand-curated. Each test only needs a
# handful of plausible tools; the integration layer will pass the
# full registered catalog at runtime.

_SENATOR_CATALOG: tuple[tuple[str, str], ...] = (
    (
        "kaos-web-search",
        (
            "General web search via SerpAPI / Exa / Brave. Use for "
            "current-events / current-officeholder / current-news queries."
        ),
    ),
    (
        "kaos-web-get-markdown",
        (
            "Fetch a URL and return readable markdown. Use for retrieving "
            "specific web pages once you know the URL."
        ),
    ),
    (
        "kaos-source-edgar-search",
        ("SEC EDGAR filings index. Use for public-company financial filings (10-K, 10-Q, 8-K)."),
    ),
    (
        "kaos-source-fr-search",
        (
            "US Federal Register search. Use for federal rule and notice "
            "retrieval by agency / date / RIN."
        ),
    ),
    (
        "kaos-source-ecfr-search",
        (
            "US Code of Federal Regulations search. Use for consolidated "
            "regulatory text by CFR title and section."
        ),
    ),
    (
        "kaos-source-gleif-lei",
        (
            "GLEIF legal-entity-identifier lookup. Use for "
            "corporate-entity LEI / OFAC-class company-identity questions."
        ),
    ),
    (
        "kaos-pdf-parse",
        (
            "Parse a local PDF file into structured text. Use only on "
            "PDFs already in the session VFS."
        ),
    ),
)

_DOCX_CATALOG: tuple[tuple[str, str], ...] = (
    (
        "kaos-office-parse-docx",
        (
            "Parse a DOCX file into structured paragraphs + numbered "
            "lists. Use for attached Word documents."
        ),
    ),
    (
        "kaos-content-search-document",
        (
            "Full-text search inside an already-parsed document. Use to "
            "find clauses, terms, sections."
        ),
    ),
    (
        "kaos-web-search",
        ("General web search. Use for current-events / public-web questions, not for local files."),
    ),
    (
        "kaos-source-edgar-search",
        ("SEC EDGAR filings. Use for public-company filings, never for arbitrary documents."),
    ),
)

_SMALLTALK_CATALOG: tuple[tuple[str, str], ...] = (
    (
        "kaos-source-edgar-search",
        "SEC EDGAR filings index.",
    ),
    (
        "kaos-source-fr-search",
        "Federal Register search.",
    ),
)


@dataclass(frozen=True, slots=True)
class FitnessCase:
    case_id: str
    query: str
    catalog: Sequence[tuple[str, str]]
    must_include_first_of: tuple[str, ...]
    must_not_include: tuple[str, ...] = ()
    allow_empty_picks: bool = False


_CASES: tuple[FitnessCase, ...] = (
    FitnessCase(
        case_id="senator-web-first",
        query="who is the current us federal senator for lansing michigan",
        catalog=_SENATOR_CATALOG,
        must_include_first_of=("kaos-web-search",),
        must_not_include=(
            "kaos-source-edgar-search",
            "kaos-source-gleif-lei",
            "kaos-source-ecfr-search",
        ),
    ),
    FitnessCase(
        case_id="docx-parser-first",
        query=("I attached a NDA called Acme-MNDA.docx — what is the governing law clause?"),
        catalog=_DOCX_CATALOG,
        must_include_first_of=(
            "kaos-office-parse-docx",
            "kaos-content-search-document",
        ),
        must_not_include=("kaos-source-edgar-search",),
    ),
    FitnessCase(
        case_id="smalltalk-empty-picks",
        query="hi there, how are you?",
        catalog=_SMALLTALK_CATALOG,
        must_include_first_of=(),
        allow_empty_picks=True,
    ),
)


@pytest.mark.live
@pytest.mark.parametrize("model", MODELS, ids=lambda m: m.replace(":", "_"))
@pytest.mark.parametrize("case", _CASES, ids=lambda c: c.case_id)
async def test_tool_fitness_ranker_live(case: FitnessCase, model: str) -> None:
    """ToolFitnessSignature picks domain-correct top-K on each model.

    Asserts on ``valid_picks`` (catalog-validated) rather than the
    raw ``picks`` so hallucinated tool names don't accidentally pass.
    """
    if not _have_key_for(model):
        pytest.skip(f"no API key for {model}")

    result: ToolFitnessResult = await rank_tools_for_query(
        query=case.query,
        catalog=case.catalog,
        model=model,
        top_k=4,
    )

    if case.allow_empty_picks:
        # Small-talk exit clause: if picks are empty, that's the
        # correct answer. If they aren't, the rationale should at
        # least signal low confidence.
        if result.valid_picks:
            assert result.confidence <= 0.6, (
                f"[{case.case_id}/{model}] small-talk picks "
                f"={result.valid_picks!r} confidence={result.confidence} "
                f"rationale={result.rationale!r}"
            )
        return

    assert result.valid_picks, (
        f"[{case.case_id}/{model}] expected non-empty valid_picks, "
        f"got fell_back={result.fell_back} raw_picks={result.picks!r} "
        f"rationale={result.rationale!r}"
    )
    first_pick = result.valid_picks[0]
    assert first_pick in case.must_include_first_of, (
        f"[{case.case_id}/{model}] expected first pick to be one of "
        f"{case.must_include_first_of}, got {first_pick!r}. "
        f"All picks: {result.valid_picks!r}. "
        f"Rationale: {result.rationale!r}"
    )
    for forbidden in case.must_not_include:
        assert forbidden not in result.valid_picks, (
            f"[{case.case_id}/{model}] {forbidden!r} should not "
            f"appear in picks={result.valid_picks!r}"
        )
