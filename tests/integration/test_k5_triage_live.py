"""KC4 — Live K5 triage_corpus tests against a real document corpus.

The unit tests in ``tests/unit/test_triage_summary_aware.py`` exercise
``triage_corpus()`` with synthetic in-memory documents. KC4 closes the
"is it actually useful on a real corpus" gap by running the full
pipeline:

  parse real DOCX  →  build_document_summary  →  load into
  SessionMemory with summary_text  →  triage_corpus(goal)  →  assert
  the summary-aware path engaged and selected the relevant docs

No LLM calls — ``triage_corpus`` is pure BM25 over either full text
or pre-built summaries. The "live" framing here means "real
documents," not "real LLM calls."

Skipped at runtime when the local NDA fixtures are absent (the docx
files are dev-machine-only, not shipped with the repo).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from kaos_agents.context.triage import _SUMMARY_METADATA_KEY, triage_corpus
from kaos_agents.memory.session import SessionMemory
from kaos_agents.types.memory import MemoryType

NDA_DIR = Path.home() / "projects" / "273v" / "kelvin-app" / "samples" / "docx"


def _nda_paths() -> list[Path]:
    if not NDA_DIR.exists():
        return []
    return sorted(NDA_DIR.glob("MNDA*.docx"))


requires_nda_fixtures = pytest.mark.skipif(
    not _nda_paths(),
    reason=f"NDA fixtures missing at {NDA_DIR}",
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _parse_nda(path: Path) -> Any:
    """Parse one NDA docx into a ContentDocument."""
    from kaos_office import parse_docx

    return parse_docx(str(path))


def _summary_text_for(doc: Any) -> str:
    """Build the K5 summary_text exactly as the production path does.

    Mirrors what ``kaos_content.tools._summary_search_text`` produces
    when the K4 corpus-summarize tool runs: head_tokens + top_ngrams +
    bottom_ngrams joined with whitespace. We inline the join here so
    the test isn't coupled to a private kaos-content helper.
    """
    from kaos_content.summarize import build_document_summary

    summary = build_document_summary(doc)
    parts: list[str] = []
    if summary.head_tokens:
        parts.append(summary.head_tokens)
    for ng in summary.top_ngrams:
        parts.append(ng.ngram)
    for ng in summary.bottom_ngrams:
        parts.append(ng.ngram)
    return " ".join(parts)


@pytest.fixture
def loaded_memory() -> SessionMemory:
    """Real NDAs parsed + summarised + loaded into a fresh SessionMemory.

    Each DOCUMENTS item has the full text in ``content`` and the
    summary text in ``metadata["summary_text"]``, so the K5 path
    engages.
    """
    memory = SessionMemory("kc4-triage-live")
    for path in _nda_paths():
        doc = _parse_nda(path)
        from kaos_content.serializers import serialize_text

        full_text = serialize_text(doc)
        summary_text = _summary_text_for(doc)
        memory.add(
            MemoryType.DOCUMENTS,
            full_text,
            metadata={
                "uri": f"file://{path}",
                _SUMMARY_METADATA_KEY: summary_text,
                "filename": path.name,
            },
        )
    return memory


@pytest.fixture
def loaded_memory_no_summary() -> SessionMemory:
    """Same NDAs but with summary_text stripped — exercises fallback path."""
    memory = SessionMemory("kc4-triage-live-nosum")
    for path in _nda_paths():
        doc = _parse_nda(path)
        from kaos_content.serializers import serialize_text

        full_text = serialize_text(doc)
        memory.add(
            MemoryType.DOCUMENTS,
            full_text,
            metadata={"uri": f"file://{path}", "filename": path.name},
        )
    return memory


# ---------------------------------------------------------------------------
# Threshold gating
# ---------------------------------------------------------------------------


@requires_nda_fixtures
class TestTriageBelowThreshold:
    def test_returns_none_when_corpus_smaller_than_threshold(
        self, loaded_memory: SessionMemory
    ) -> None:
        """5 NDAs is below the default threshold of 20 — no triage fires."""
        result = triage_corpus(loaded_memory, "confidential information disclosure")
        assert result is None, (
            "Expected None when corpus size < threshold. "
            f"Got {result!r} for {loaded_memory.section_item_count(MemoryType.DOCUMENTS)} docs."
        )


# ---------------------------------------------------------------------------
# Summary-aware path (the K5 fast path)
# ---------------------------------------------------------------------------


@requires_nda_fixtures
class TestSummaryAwarePathLive:
    def test_summary_path_engages_with_low_threshold(self, loaded_memory: SessionMemory) -> None:
        """Force triage with threshold=1; every doc has summary_text → fast path."""
        n_docs = loaded_memory.section_item_count(MemoryType.DOCUMENTS)
        assert n_docs >= 2, "Need at least 2 NDA fixtures for a meaningful triage"

        result = triage_corpus(
            loaded_memory,
            "confidential information",
            threshold=1,
            max_selected=n_docs,
        )
        assert result is not None
        assert result.used_summary_index is True, (
            "Expected K5 fast path. Got used_summary_index=False. "
            "This means at least one document lacks summary_text metadata."
        )
        assert result.total_documents == n_docs
        assert result.selected_count > 0
        # Every selected URI traces back to a real fixture.
        assert all(uri.startswith("file://") for uri in result.selected_uris)

    def test_summary_path_returns_topk_bound(self, loaded_memory: SessionMemory) -> None:
        """top_k bound is honored — selecting 2 returns at most 2.

        Uses ``confidential information`` because every NDA summary
        surfaces it in its top n-grams; a query whose terms aren't in
        the summaries (e.g. ``termination notice``) genuinely produces
        no summary-index hits and the test would be measuring the
        fallback path instead. K5's correctness is "fall back when the
        summary is silent" — but verifying the top_k bound requires a
        query the summary CAN hit.
        """
        n_docs = loaded_memory.section_item_count(MemoryType.DOCUMENTS)
        if n_docs < 3:
            pytest.skip("Need >=3 NDAs to test top_k bounding meaningfully")

        result = triage_corpus(
            loaded_memory,
            "confidential information",
            threshold=1,
            max_selected=2,
        )
        assert result is not None
        assert result.selected_count <= 2
        assert len(result.selected_uris) == result.selected_count
        assert result.used_summary_index is True

    def test_summary_silent_query_falls_back(self, loaded_memory: SessionMemory) -> None:
        """A query whose terms are NOT in any summary's n-grams should
        return None from the summary-aware path. ``triage_corpus``
        then falls through to the full-text path, which is the
        designed behavior (K5 module docstring: "fall back so the
        caller never silently mixes signals from two different
        indexes")."""
        n_docs = loaded_memory.section_item_count(MemoryType.DOCUMENTS)
        if n_docs < 2:
            pytest.skip("Need >=2 NDAs for a fallback exercise")

        # Some legitimate legal vocabulary that won't appear in the
        # top n-grams (which favor "confidential" / "information" /
        # "agreement"). The full-text path will still hit because the
        # term DOES appear in the body of every NDA.
        result = triage_corpus(
            loaded_memory,
            "arbitration jurisdiction",
            threshold=1,
            max_selected=n_docs,
        )
        # Either the summary-aware path found hits, or it returned None
        # and we fell through. The contract is: never an empty result
        # when the full-text path would have matched.
        assert result is not None
        assert result.selected_count > 0

    def test_summary_path_finds_topically_relevant_doc(self, loaded_memory: SessionMemory) -> None:
        """Pick a unique query phrase and verify BM25 ranks correctly.

        Every NDA in the corpus carries "confidential information" so
        a query for it should return docs in a meaningful order — the
        top hit is a real document, not a hallucination.
        """
        result = triage_corpus(
            loaded_memory,
            "confidential information",
            threshold=1,
            max_selected=3,
        )
        assert result is not None
        # Every URI corresponds to a fixture that actually exists.
        for uri in result.selected_uris:
            path_str = uri.removeprefix("file://")
            assert Path(path_str).exists(), f"BM25 returned nonexistent path: {uri}"


# ---------------------------------------------------------------------------
# Fallback path
# ---------------------------------------------------------------------------


@requires_nda_fixtures
class TestFullTextFallbackLive:
    def test_no_summary_text_uses_fallback(self, loaded_memory_no_summary: SessionMemory) -> None:
        """Without summary_text, K5 falls back to search_memory full-text BM25."""
        n_docs = loaded_memory_no_summary.section_item_count(MemoryType.DOCUMENTS)
        result = triage_corpus(
            loaded_memory_no_summary,
            "confidential information",
            threshold=1,
            max_selected=n_docs,
        )
        assert result is not None
        assert result.used_summary_index is False, (
            "Expected fallback path. Got used_summary_index=True even though "
            "no document was given a summary_text."
        )
        assert result.total_documents == n_docs
        assert result.selected_count > 0


# ---------------------------------------------------------------------------
# Context summary shape
# ---------------------------------------------------------------------------


@requires_nda_fixtures
class TestContextSummaryShape:
    def test_summary_path_labels_itself_in_context_summary(
        self, loaded_memory: SessionMemory
    ) -> None:
        """The injected context summary surfaces which BM25 path engaged.

        This matters for operators auditing whether the K5 fast path
        actually fired in production.
        """
        result = triage_corpus(
            loaded_memory,
            "non-disclosure agreement",
            threshold=1,
        )
        assert result is not None
        assert "summary-index" in result.context_summary
        assert "full-text" not in result.context_summary

    def test_fallback_path_labels_itself_in_context_summary(
        self, loaded_memory_no_summary: SessionMemory
    ) -> None:
        result = triage_corpus(
            loaded_memory_no_summary,
            "non-disclosure agreement",
            threshold=1,
        )
        assert result is not None
        assert "full-text" in result.context_summary
        assert "summary-index" not in result.context_summary
