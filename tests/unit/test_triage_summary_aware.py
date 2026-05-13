"""Tests for the summary-aware BM25 path in kaos_agents.context.triage (K5).

The summary-aware path engages when every DOCUMENTS item carries a
``summary_text`` field in its metadata. These tests verify:
- Partial coverage falls back to the full-text path
- Full coverage uses the summary-aware path
- Summary-aware ranking still picks the relevant docs
- The returned TriageResult.used_summary_index flag reflects which
  path was taken (operators can verify the fast path engaged)
"""

from __future__ import annotations

from kaos_agents.context.triage import triage_corpus
from kaos_agents.memory.session import SessionMemory
from kaos_agents.types.memory import MemoryType


def _summary_text(*ngrams: str) -> str:
    """Build a fake summary_text by joining n-grams with whitespace.

    Mirrors what kaos_content.tools._summary_search_text produces — the
    concatenated head_tokens + top_ngrams + bottom_ngrams of a
    DocumentSummary. Keeping the assembly local to the test so the
    test isn't coupled to kaos-content internals.
    """
    return " ".join(ngrams)


class TestSummaryAwarePathEngagement:
    def test_partial_coverage_falls_back(self) -> None:
        """If even one item lacks summary_text, fall back to the
        full-text path. No mixing."""
        memory = SessionMemory("test")
        for i in range(25):
            metadata = {"uri": f"doc:{i}"}
            # Only half have summary_text.
            if i % 2 == 0:
                metadata["summary_text"] = _summary_text("agreement", "confidential")
            memory.add(
                MemoryType.DOCUMENTS,
                f"document {i} body about confidential information",
                metadata=metadata,
            )
        result = triage_corpus(memory, "confidential", threshold=20)
        assert result is not None
        assert result.used_summary_index is False, (
            "Expected fallback to full-text path when coverage is partial"
        )

    def test_full_coverage_uses_summary_path(self) -> None:
        """Every item has summary_text → the summary-aware path engages."""
        memory = SessionMemory("test")
        for i in range(25):
            memory.add(
                MemoryType.DOCUMENTS,
                f"document {i} full body",
                metadata={
                    "uri": f"doc:{i}",
                    "summary_text": _summary_text("agreement", "confidential", "information"),
                },
            )
        result = triage_corpus(memory, "confidential", threshold=20)
        assert result is not None
        assert result.used_summary_index is True

    def test_zero_coverage_uses_full_text_path(self) -> None:
        """No items have summary_text — existing behaviour, unchanged."""
        memory = SessionMemory("test")
        for i in range(25):
            memory.add(
                MemoryType.DOCUMENTS,
                f"document {i} body about confidential information",
                metadata={"uri": f"doc:{i}"},
            )
        result = triage_corpus(memory, "confidential", threshold=20)
        assert result is not None
        assert result.used_summary_index is False


class TestSummaryAwareRankingQuality:
    def test_summary_path_picks_relevant_doc(self) -> None:
        """When using summaries, BM25 over the summary text must
        still surface the relevant doc."""
        memory = SessionMemory("test")
        # Add docs whose summary_text matches one of two themes.
        for i in range(15):
            memory.add(
                MemoryType.DOCUMENTS,
                f"unrelated content {i}",
                metadata={
                    "uri": f"doc:nda-{i}",
                    "summary_text": _summary_text(
                        "non disclosure agreement",
                        "confidential information",
                        "trade secrets",
                    ),
                },
            )
        for i in range(15):
            memory.add(
                MemoryType.DOCUMENTS,
                f"unrelated content {i}",
                metadata={
                    "uri": f"doc:lease-{i}",
                    "summary_text": _summary_text(
                        "lease agreement",
                        "real property",
                        "rent payments",
                    ),
                },
            )
        result = triage_corpus(
            memory,
            "confidential trade secrets information",
            threshold=20,
            max_selected=10,
        )
        assert result is not None
        assert result.used_summary_index is True
        # All selected URIs should be NDAs.
        nda_count = sum(1 for u in result.selected_uris if "nda" in u)
        assert nda_count >= 8, (
            f"Expected mostly NDAs in top-10; got "
            f"{nda_count}/{len(result.selected_uris)}. URIs: {result.selected_uris}"
        )

    def test_max_selected_respected_on_summary_path(self) -> None:
        memory = SessionMemory("test")
        for i in range(50):
            memory.add(
                MemoryType.DOCUMENTS,
                f"document {i}",
                metadata={
                    "uri": f"doc:{i}",
                    "summary_text": _summary_text("agreement", f"clause-{i}"),
                },
            )
        result = triage_corpus(memory, "agreement", threshold=20, max_selected=3)
        assert result is not None
        assert result.selected_count <= 3
        assert result.used_summary_index is True


class TestSummaryAwarePathEdgeCases:
    def test_empty_summary_text_treated_as_missing(self) -> None:
        """An item whose summary_text is an empty string falls back to
        the full-text path. The full-text search must still find hits;
        we seed each item's content with the query term so the
        fallback path produces non-empty results."""
        memory = SessionMemory("test")
        for i in range(25):
            memory.add(
                MemoryType.DOCUMENTS,
                # Content carries the query term so the full-text
                # fallback path can rank these items.
                f"document {i} about the agreement and confidentiality",
                metadata={
                    "uri": f"doc:{i}",
                    "summary_text": "" if i == 0 else _summary_text("agreement"),
                },
            )
        result = triage_corpus(memory, "agreement", threshold=20)
        assert result is not None
        assert result.used_summary_index is False

    def test_non_string_summary_text_treated_as_missing(self) -> None:
        """Defensive: if a caller stashes a dict under summary_text the
        path falls back rather than crashing."""
        memory = SessionMemory("test")
        for i in range(25):
            memory.add(
                MemoryType.DOCUMENTS,
                f"document {i} about the agreement and confidentiality",
                metadata={
                    "uri": f"doc:{i}",
                    "summary_text": {"oops": True} if i == 0 else _summary_text("agreement"),
                },
            )
        result = triage_corpus(memory, "agreement", threshold=20)
        assert result is not None
        assert result.used_summary_index is False

    def test_context_summary_labels_path(self) -> None:
        """The injected planning summary names which path was used."""
        memory = SessionMemory("test")
        for i in range(25):
            memory.add(
                MemoryType.DOCUMENTS,
                f"document {i}",
                metadata={
                    "uri": f"doc:{i}",
                    "summary_text": _summary_text("agreement"),
                },
            )
        result = triage_corpus(memory, "agreement", threshold=20)
        assert result is not None
        assert "summary-index" in result.context_summary

    def test_full_text_path_labels_context(self) -> None:
        memory = SessionMemory("test")
        for i in range(25):
            memory.add(
                MemoryType.DOCUMENTS,
                f"document {i} agreement",
                metadata={"uri": f"doc:{i}"},
            )
        result = triage_corpus(memory, "agreement", threshold=20)
        assert result is not None
        assert "full-text" in result.context_summary
