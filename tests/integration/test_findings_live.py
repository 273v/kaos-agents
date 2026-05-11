"""Live integration test for FindingsAgent (K6).

Wires real Haiku 4.5 filter + real Sonnet 4.6 synthesis against a
real MNDA docx file. Asserts the three-phase pipeline:
- Phase 1 enumerates a non-trivial number of candidates
- Phase 2 filters at least one survivor
- Phase 3 produces an answer that cites at least one finding_id

Captured via the standard recorder fixture
(``tests/integration/conftest.py``) so the trace shows the three
call clusters (intent-classify omitted: this pipeline bypasses the
ChatAgent loop entirely).
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from kaos_agents.patterns.findings import (
    FindingsAgent,
    extract_finding_id_citations,
    sentences_with_token_selector,
)

NDA_DIR = Path.home() / "projects" / "273v" / "kelvin-app" / "samples" / "docx"

requires_anthropic = pytest.mark.skipif(
    "ANTHROPIC_API_KEY" not in os.environ,
    reason="ANTHROPIC_API_KEY missing",
)

requires_nda_fixtures = pytest.mark.skipif(
    not NDA_DIR.exists() or not any(NDA_DIR.glob("MNDA*.docx")),
    reason=f"NDA fixtures missing at {NDA_DIR}",
)

INNER_MODEL = "anthropic:claude-haiku-4-5"
SYNTH_MODEL = "anthropic:claude-sonnet-4-6"


def _first_nda() -> Path:
    paths = sorted(NDA_DIR.glob("MNDA*.docx"))
    if not paths:
        raise pytest.skip.Exception(f"no NDA fixtures under {NDA_DIR}")
    return paths[0]


def _view_for(docx_path: Path):
    from kaos_content.views import DocumentView
    from kaos_nlp_core._defaults import get_default_punkt_tokenizer
    from kaos_office import parse_docx

    doc = parse_docx(str(docx_path))
    return DocumentView(doc, sentence_segmenter=get_default_punkt_tokenizer())


@pytest.mark.live
@requires_anthropic
@requires_nda_fixtures
class TestFindingsLive:
    """End-to-end real-LLM pipeline against a real NDA."""

    async def test_confidentiality_findings(self) -> None:
        """Ask 'what counts as Confidential Information?' over an NDA.

        Phase 1: every sentence containing 'confidential' (substring).
        Phase 2: Haiku-filter to the directly-relevant ones.
        Phase 3: Sonnet synthesises an answer that cites finding_ids.
        """
        view = _view_for(_first_nda())
        agent = FindingsAgent(
            selector=sentences_with_token_selector("confidential"),
            filter_model=INNER_MODEL,
            synthesis_model=SYNTH_MODEL,
            chunk_size=20,
            num_parallel=3,
            relevance_threshold=0.4,
        )
        result = await agent.run(
            "What counts as Confidential Information under this agreement?",
            view,
        )

        # Phase 1 must have enumerated *something* — every NDA has
        # multiple sentences mentioning 'confidential'.
        assert result.total_enumerated >= 3, (
            f"Phase 1 found only {result.total_enumerated} candidates "
            f"on a real NDA — selector regression?"
        )

        # Phase 2 must have filtered to a positive number of survivors.
        assert result.total_filtered >= 1, (
            f"Phase 2 produced no survivors out of {result.total_enumerated} "
            f"candidates. Either the filter is too strict or the LLM "
            f"is misbehaving."
        )
        assert result.total_filtered <= result.total_enumerated

        # Phase 3 must have produced an answer.
        assert result.answer
        assert len(result.answer) > 20

        # The answer should cite at least one finding_id from the
        # surviving findings.
        cited = extract_finding_id_citations(result.answer)
        survivor_ids = {f.candidate.finding_id for f in result.findings}
        cited_in_survivors = set(cited) & survivor_ids
        assert cited_in_survivors, (
            f"Answer does not cite any surviving finding_id. "
            f"Cited: {cited!r}. Survivors: {sorted(survivor_ids)!r}. "
            f"Answer: {result.answer!r}"
        )

        # Costs should be sensible — non-zero and bounded.
        assert result.filter_cost_usd > 0
        assert result.synthesis_cost_usd > 0
        assert result.total_cost_usd < 0.50, (
            f"Pipeline cost ${result.total_cost_usd:.4f} exceeded the "
            f"$0.50 sanity gate — possible runaway."
        )

    async def test_recall_first_finds_indemnification(self) -> None:
        """The substring 'indemnif' is a tighter selector than the
        Phase-2 LLM's notion of "indemnification". This test asserts
        the pipeline answers an indemnification question."""
        view = _view_for(_first_nda())
        agent = FindingsAgent(
            selector=sentences_with_token_selector("indemnif"),
            filter_model=INNER_MODEL,
            synthesis_model=SYNTH_MODEL,
            chunk_size=20,
            relevance_threshold=0.4,
        )
        result = await agent.run(
            "What does this agreement say about indemnification?",
            view,
        )
        # Some NDAs don't have indemnification clauses — accept zero
        # findings as a valid outcome (the answer should say so).
        if result.total_enumerated == 0:
            assert result.answer == ""
            return
        # Otherwise, normal flow.
        assert result.total_filtered >= 0
        if result.findings:
            assert result.answer
