"""Unit tests for kaos_agents.patterns.findings (K6).

Deterministic — patches ``_filter_chunk`` and ``_synthesize`` to
stubs so the three-phase pipeline can be exercised without an LLM.
Live integration tests live in ``tests/integration/test_findings_live.py``.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from kaos_agents.patterns import findings as findings_mod
from kaos_agents.patterns.findings import (
    FilteredFinding,
    FindingCandidate,
    FindingsAgent,
    FindingsResult,
    every_sentence_selector,
    extract_finding_id_citations,
    sentences_with_token_selector,
)

# ---------------------------------------------------------------------------
# Fake DocumentView — duck-typed sentences/paragraphs/section_by_ref
# ---------------------------------------------------------------------------


class _FakeSentence:
    def __init__(
        self,
        text: str,
        paragraph_ref: str = "#/body/0",
        section_ref: str | None = None,
        page: int | None = None,
    ) -> None:
        self.text = text
        self.paragraph_ref = paragraph_ref
        self.section_ref = section_ref
        self.page = page


class _FakeView:
    """Stand-in for DocumentView with just the surface findings uses."""

    def __init__(self, sentences: list[_FakeSentence]) -> None:
        self.sentences = sentences

    def section_by_ref(self, _ref: str) -> None:
        return None


def _sample_view() -> _FakeView:
    return _FakeView(
        [
            _FakeSentence(
                "The cap on indemnification is $100,000 per occurrence.",
                paragraph_ref="#/body/3",
            ),
            _FakeSentence(
                "Indemnification carve-outs apply for gross negligence.",
                paragraph_ref="#/body/4",
            ),
            _FakeSentence(
                "The Term is twenty-four months from the Effective Date.",
                paragraph_ref="#/body/1",
            ),
            _FakeSentence(
                "Confidential Information includes business plans.",
                paragraph_ref="#/body/2",
            ),
        ]
    )


# ---------------------------------------------------------------------------
# Stubs for the LLM-driven helpers
# ---------------------------------------------------------------------------


async def _stub_filter_indemnif_only(
    chunk: tuple[FindingCandidate, ...],
    *,
    question: str,
    model: str,
    threshold: float,
) -> tuple[tuple[FilteredFinding, ...], float]:
    """Survivors: any candidate text containing 'indemnif'."""
    survivors: list[FilteredFinding] = []
    for cand in chunk:
        if "indemnif" in cand.text.lower():
            survivors.append(
                FilteredFinding(candidate=cand, relevance=0.9, reasoning="mentions indemnification")
            )
    return tuple(survivors), 0.001  # fake $0.001 per chunk


async def _stub_filter_keep_nothing(
    chunk: tuple[FindingCandidate, ...],
    **_kwargs: Any,
) -> tuple[tuple[FilteredFinding, ...], float]:
    return (), 0.001


async def _stub_synthesize_count(
    *,
    question: str,
    findings: tuple[FilteredFinding, ...],
    model: str,
) -> tuple[str, float]:
    """Synthesis that lists the surviving finding_ids inline."""
    cited = " ".join(f"[{f.candidate.finding_id}]" for f in findings)
    answer = f"Found {len(findings)} relevant items: {cited}"
    return answer, 0.005


# ---------------------------------------------------------------------------
# Value type tests
# ---------------------------------------------------------------------------


class TestValueTypes:
    def test_candidate_frozen(self) -> None:
        c = FindingCandidate(finding_id="abc12345", text="hello")
        with pytest.raises((AttributeError, TypeError)):
            c.text = "world"  # ty: ignore[invalid-assignment]

    def test_findings_result_total_cost_sums(self) -> None:
        r = FindingsResult(
            question="q",
            answer="a",
            findings=(),
            total_enumerated=0,
            total_filtered=0,
            filter_cost_usd=0.01,
            synthesis_cost_usd=0.02,
            filter_calls=2,
        )
        assert r.total_cost_usd == pytest.approx(0.03)
        # filter_calls + 1 synthesis call
        assert r.total_llm_calls == 3


# ---------------------------------------------------------------------------
# Selectors
# ---------------------------------------------------------------------------


class TestSelectors:
    def test_every_sentence_selector(self) -> None:
        view = _sample_view()
        cands = list(every_sentence_selector(view, "any question"))  # ty: ignore[invalid-argument-type]
        assert len(cands) == 4
        for c in cands:
            assert c.text
            assert c.finding_id
            assert len(c.finding_id) == 8

    def test_every_sentence_selector_skips_empty(self) -> None:
        view = _FakeView(
            [
                _FakeSentence("real sentence"),
                _FakeSentence("   "),
                _FakeSentence(""),
            ]
        )
        cands = list(every_sentence_selector(view, "q"))  # ty: ignore[invalid-argument-type]
        assert len(cands) == 1

    def test_sentences_with_token_selector_substring(self) -> None:
        view = _sample_view()
        sel = sentences_with_token_selector("indemnif")
        cands = list(sel(view, "q"))  # ty: ignore[invalid-argument-type]
        assert len(cands) == 2
        for c in cands:
            assert "indemnif" in c.text.lower()

    def test_sentences_with_token_selector_case_insensitive(self) -> None:
        view = _sample_view()
        sel = sentences_with_token_selector("INDEMNIF", case_sensitive=False)
        cands = list(sel(view, "q"))  # ty: ignore[invalid-argument-type]
        assert len(cands) == 2

    def test_sentences_with_token_selector_case_sensitive(self) -> None:
        view = _sample_view()
        sel = sentences_with_token_selector("INDEMNIF", case_sensitive=True)
        cands = list(sel(view, "q"))  # ty: ignore[invalid-argument-type]
        assert len(cands) == 0


# ---------------------------------------------------------------------------
# Construction guards
# ---------------------------------------------------------------------------


class TestFindingsAgentConstruction:
    def test_chunk_size_validated(self) -> None:
        with pytest.raises(ValueError, match="chunk_size"):
            FindingsAgent(selector=every_sentence_selector, chunk_size=0)

    def test_num_parallel_validated(self) -> None:
        with pytest.raises(ValueError, match="num_parallel"):
            FindingsAgent(selector=every_sentence_selector, num_parallel=0)

    def test_relevance_threshold_range(self) -> None:
        with pytest.raises(ValueError, match="relevance_threshold"):
            FindingsAgent(selector=every_sentence_selector, relevance_threshold=1.5)
        with pytest.raises(ValueError, match="relevance_threshold"):
            FindingsAgent(selector=every_sentence_selector, relevance_threshold=-0.1)


# ---------------------------------------------------------------------------
# Three-phase pipeline (stub LLM)
# ---------------------------------------------------------------------------


class TestPipeline:
    def test_end_to_end_with_indemnif_selector(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Full pipeline with the substring selector + indemnif filter stub."""
        monkeypatch.setattr(findings_mod, "_filter_chunk", _stub_filter_indemnif_only)
        monkeypatch.setattr(findings_mod, "_synthesize", _stub_synthesize_count)

        view = _sample_view()
        agent = FindingsAgent(
            selector=sentences_with_token_selector("indemnif"),
            chunk_size=5,
        )
        result = asyncio.run(agent.run("What about indemnification?", view))  # ty: ignore[invalid-argument-type]
        assert result.total_enumerated == 2
        assert result.total_filtered == 2
        assert result.filter_calls == 1
        assert "Found 2 relevant items" in result.answer
        # All citations resolved to a real finding_id
        cited = extract_finding_id_citations(result.answer)
        assert len(cited) == 2

    def test_no_candidates_skips_filter_and_synthesis(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Phase 1 returns nothing → no LLM calls at all."""
        called: list[str] = []

        async def _should_not_be_called(*args: Any, **kwargs: Any) -> Any:
            called.append("filter")
            return (), 0.0

        async def _synth_should_not_be_called(**kwargs: Any) -> tuple[str, float]:
            called.append("synthesize")
            return "", 0.0

        monkeypatch.setattr(findings_mod, "_filter_chunk", _should_not_be_called)
        monkeypatch.setattr(findings_mod, "_synthesize", _synth_should_not_be_called)

        view = _FakeView([])
        agent = FindingsAgent(selector=every_sentence_selector)
        result = asyncio.run(agent.run("q", view))  # ty: ignore[invalid-argument-type]
        assert result.total_enumerated == 0
        assert result.total_filtered == 0
        assert result.filter_calls == 0
        assert result.answer == ""
        assert called == []

    def test_all_filtered_out_skips_synthesis(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Phase 2 produces zero survivors → Phase 3 is skipped."""
        called: list[str] = []

        async def _synth_should_not_be_called(**kwargs: Any) -> tuple[str, float]:
            called.append("synthesize")
            return "", 0.0

        monkeypatch.setattr(findings_mod, "_filter_chunk", _stub_filter_keep_nothing)
        monkeypatch.setattr(findings_mod, "_synthesize", _synth_should_not_be_called)

        view = _sample_view()
        agent = FindingsAgent(selector=every_sentence_selector, chunk_size=2)
        result = asyncio.run(agent.run("q", view))  # ty: ignore[invalid-argument-type]
        assert result.total_enumerated == 4
        assert result.total_filtered == 0
        assert result.filter_calls == 2  # 4 candidates / chunk_size=2
        assert result.answer == ""
        assert called == []  # synthesis NOT called

    def test_chunking_respects_chunk_size(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """4 candidates with chunk_size=2 → 2 filter calls."""
        seen_chunk_sizes: list[int] = []

        async def _track_chunk(
            chunk: tuple[FindingCandidate, ...],
            **kwargs: Any,
        ) -> tuple[tuple[FilteredFinding, ...], float]:
            seen_chunk_sizes.append(len(chunk))
            return (), 0.001

        monkeypatch.setattr(findings_mod, "_filter_chunk", _track_chunk)
        monkeypatch.setattr(findings_mod, "_synthesize", _stub_synthesize_count)

        view = _sample_view()
        agent = FindingsAgent(selector=every_sentence_selector, chunk_size=2)
        asyncio.run(agent.run("q", view))  # ty: ignore[invalid-argument-type]
        assert seen_chunk_sizes == [2, 2]

    def test_filter_cost_accumulates(self, monkeypatch: pytest.MonkeyPatch) -> None:
        async def _stub_filter_costly(
            chunk: tuple[FindingCandidate, ...],
            **kwargs: Any,
        ) -> tuple[tuple[FilteredFinding, ...], float]:
            survivors = tuple(
                FilteredFinding(candidate=c, relevance=0.9, reasoning="ok") for c in chunk
            )
            return survivors, 0.01  # $0.01 per chunk

        monkeypatch.setattr(findings_mod, "_filter_chunk", _stub_filter_costly)
        monkeypatch.setattr(findings_mod, "_synthesize", _stub_synthesize_count)

        view = _sample_view()
        agent = FindingsAgent(selector=every_sentence_selector, chunk_size=2)
        result = asyncio.run(agent.run("q", view))  # ty: ignore[invalid-argument-type]
        # 2 chunks * $0.01 + $0.005 synthesis
        assert result.filter_cost_usd == pytest.approx(0.02)
        assert result.synthesis_cost_usd == pytest.approx(0.005)
        assert result.total_cost_usd == pytest.approx(0.025)

    def test_results_sorted_by_relevance_descending(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Surviving findings emerge in descending relevance order."""

        async def _stub_filter_varied(
            chunk: tuple[FindingCandidate, ...],
            **kwargs: Any,
        ) -> tuple[tuple[FilteredFinding, ...], float]:
            survivors = []
            for i, c in enumerate(chunk):
                # Manufacture varied relevances
                rel = (i + 1) / (len(chunk) + 1)
                survivors.append(FilteredFinding(candidate=c, relevance=rel, reasoning="x"))
            return tuple(survivors), 0.001

        monkeypatch.setattr(findings_mod, "_filter_chunk", _stub_filter_varied)
        monkeypatch.setattr(findings_mod, "_synthesize", _stub_synthesize_count)

        view = _sample_view()
        agent = FindingsAgent(selector=every_sentence_selector, chunk_size=4)
        result = asyncio.run(agent.run("q", view))  # ty: ignore[invalid-argument-type]
        relevances = [f.relevance for f in result.findings]
        assert relevances == sorted(relevances, reverse=True)


# ---------------------------------------------------------------------------
# Citation helper
# ---------------------------------------------------------------------------


class TestExtractFindingIdCitations:
    def test_extracts_one_id(self) -> None:
        ids = extract_finding_id_citations("see [abcdef12] for details")
        assert ids == ("abcdef12",)

    def test_extracts_multiple_ids(self) -> None:
        text = "consider [aaaaaaaa] and [bbbbbbbb]; also [cccccccc]"
        ids = extract_finding_id_citations(text)
        assert ids == ("aaaaaaaa", "bbbbbbbb", "cccccccc")

    def test_ignores_non_hex(self) -> None:
        ids = extract_finding_id_citations("[xyzxyzxy] [12345678]")
        assert ids == ("12345678",)

    def test_dedups_in_order(self) -> None:
        ids = extract_finding_id_citations("[aaaaaaaa] and again [aaaaaaaa]")
        assert ids == ("aaaaaaaa",)

    def test_empty_text(self) -> None:
        assert extract_finding_id_citations("") == ()
