"""Unit tests for kaos_agents.memory.lessons.

Deterministic-only — covers the Lesson value type, write/read
round-trip, and the recall path against a SessionMemory built
with the platform-default section configuration.
"""

from __future__ import annotations

import pytest

from kaos_agents.memory.lessons import (
    DEFAULT_RECALL_TOP_K,
    Lesson,
    _roundtrip_lesson,
    lessons_to_context_block,
    read_lessons,
    recall_lessons,
    write_lesson,
)
from kaos_agents.memory.session import SessionMemory
from kaos_agents.types.memory import DEFAULT_SECTIONS, MemoryType

# ---------------------------------------------------------------------------
# Lesson — type-level invariants
# ---------------------------------------------------------------------------


class TestLessonType:
    def test_minimal_lesson_constructs(self) -> None:
        lesson = Lesson(
            situation="user asks 'which is longest' on similar docs",
            observation="agent refused with 'insufficient evidence'",
            takeaway="aggregation questions need compute tools, not more retrieval",
        )
        assert lesson.evidence_refs == ()
        assert lesson.confidence is None
        assert lesson.tags == ()

    def test_empty_field_rejected(self) -> None:
        with pytest.raises(ValueError, match="at least 1"):
            Lesson(situation="", observation="x", takeaway="y")
        with pytest.raises(ValueError, match="at least 1"):
            Lesson(situation="x", observation="", takeaway="y")
        with pytest.raises(ValueError, match="at least 1"):
            Lesson(situation="x", observation="y", takeaway="")

    def test_confidence_range_enforced(self) -> None:
        with pytest.raises(ValueError, match="less than or equal to 1"):
            Lesson(situation="a", observation="b", takeaway="c", confidence=1.5)
        with pytest.raises(ValueError, match="greater than or equal to 0"):
            Lesson(situation="a", observation="b", takeaway="c", confidence=-0.1)

    def test_frozen(self) -> None:
        lesson = Lesson(situation="a", observation="b", takeaway="c")
        with pytest.raises(ValueError, match="frozen"):
            lesson.situation = "z"  # type: ignore[misc]

    def test_extra_fields_rejected(self) -> None:
        with pytest.raises(ValueError, match="Extra inputs are not permitted"):
            Lesson(situation="a", observation="b", takeaway="c", extra_field=1)  # ty: ignore[unknown-argument]

    def test_searchable_text_includes_all_three_channels(self) -> None:
        lesson = Lesson(
            situation="situation_token_xyz",
            observation="observation_token_uvw",
            takeaway="takeaway_token_pqr",
            tags=("tag_token_abc",),
        )
        text = lesson.to_searchable_text()
        assert "situation_token_xyz" in text
        assert "observation_token_uvw" in text
        assert "takeaway_token_pqr" in text
        assert "tag_token_abc" in text

    def test_recall_summary_includes_situation_and_takeaway(self) -> None:
        lesson = Lesson(
            situation="when you see X",
            observation="x happened",
            takeaway="do Y",
            confidence=0.8,
            evidence_refs=("event:abc",),
        )
        summary = lesson.to_recall_summary()
        assert "when you see X" in summary
        assert "do Y" in summary
        assert "0.80" in summary
        assert "event:abc" in summary


# ---------------------------------------------------------------------------
# Round-trip — agent synthesizes via Call output → JSON → reconstruct
# ---------------------------------------------------------------------------


class TestRoundTrip:
    def test_full_roundtrip_preserves_every_field(self) -> None:
        lesson = Lesson(
            situation="situation text",
            observation="observation text",
            takeaway="takeaway text",
            evidence_refs=("event:1", "block_ref:#/body/3"),
            confidence=0.75,
            tags=("pathology", "vocabulary"),
        )
        rebuilt = _roundtrip_lesson(lesson)
        assert rebuilt == lesson

    def test_minimal_lesson_roundtrips(self) -> None:
        lesson = Lesson(situation="a", observation="b", takeaway="c")
        assert _roundtrip_lesson(lesson) == lesson


# ---------------------------------------------------------------------------
# Write + read — session memory integration
# ---------------------------------------------------------------------------


def _build_memory() -> SessionMemory:
    """SessionMemory with the platform-default section set (LESSONS included)."""
    return SessionMemory(session_id="test-session", sections=DEFAULT_SECTIONS)


def _make_lesson(seq: int) -> Lesson:
    return Lesson(
        situation=f"situation {seq}",
        observation=f"observed pattern {seq}",
        takeaway=f"do thing number {seq}",
    )


class TestWriteRead:
    def test_write_then_read_preserves_lessons(self) -> None:
        memory = _build_memory()
        for i in range(3):
            write_lesson(memory, _make_lesson(i))
        recovered = read_lessons(memory)
        assert len(recovered) == 3
        for i, lesson in enumerate(recovered):
            assert lesson == _make_lesson(i)

    def test_read_lessons_on_empty_section_returns_empty_list(self) -> None:
        memory = _build_memory()
        assert read_lessons(memory) == []

    def test_write_raises_when_section_missing(self) -> None:
        # Build a memory without the LESSONS section.
        no_lessons = tuple(c for c in DEFAULT_SECTIONS if c.memory_type != MemoryType.LESSONS)
        memory = SessionMemory(session_id="no-lessons", sections=no_lessons)
        with pytest.raises(KeyError, match="LESSONS section"):
            write_lesson(memory, _make_lesson(0))

    def test_read_returns_empty_when_section_missing(self) -> None:
        no_lessons = tuple(c for c in DEFAULT_SECTIONS if c.memory_type != MemoryType.LESSONS)
        memory = SessionMemory(session_id="no-lessons", sections=no_lessons)
        # Quiet path: read returns [] rather than raising. Recall path
        # similarly returns [] when the section is absent.
        assert read_lessons(memory) == []


# ---------------------------------------------------------------------------
# recall_lessons — BM25 over situation
# ---------------------------------------------------------------------------


class TestRecallLessons:
    def test_empty_query_returns_no_lessons(self) -> None:
        memory = _build_memory()
        write_lesson(memory, _make_lesson(0))
        assert recall_lessons(memory, "", top_k=3) == []
        assert recall_lessons(memory, "   ", top_k=3) == []

    def test_recall_ranks_by_situation_overlap(self) -> None:
        memory = _build_memory()
        write_lesson(
            memory,
            Lesson(
                situation="user asks comparative questions across NDAs",
                observation="agent refused",
                takeaway="use kaos-content-stats for aggregation",
            ),
        )
        write_lesson(
            memory,
            Lesson(
                situation="user asks about PDF page counts",
                observation="agent looped",
                takeaway="use kaos-pdf-metadata",
            ),
        )
        write_lesson(
            memory,
            Lesson(
                situation="long-running federal-register research",
                observation="costs spiked",
                takeaway="cap with --max-cost",
            ),
        )
        hits = recall_lessons(memory, "compare NDAs side-by-side", top_k=3)
        assert len(hits) >= 1
        # The NDA comparison lesson should rank top.
        assert "NDA" in hits[0].situation

    def test_top_k_respected(self) -> None:
        memory = _build_memory()
        for i in range(5):
            write_lesson(memory, _make_lesson(i))
        hits = recall_lessons(memory, "situation 2", top_k=2)
        assert len(hits) <= 2

    def test_default_top_k_constant(self) -> None:
        assert DEFAULT_RECALL_TOP_K > 0
        assert DEFAULT_RECALL_TOP_K <= 10


# ---------------------------------------------------------------------------
# lessons_to_context_block — prompt rendering
# ---------------------------------------------------------------------------


class TestContextBlock:
    def test_empty_list_returns_empty_string(self) -> None:
        assert lessons_to_context_block([]) == ""

    def test_renders_each_lesson(self) -> None:
        block = lessons_to_context_block(
            [
                Lesson(situation="A", observation="o", takeaway="do A"),
                Lesson(situation="B", observation="o", takeaway="do B"),
            ]
        )
        assert "Lesson 1" in block
        assert "Lesson 2" in block
        assert "A" in block
        assert "do A" in block
        assert "do B" in block

    def test_unconditional_prepend_safe(self) -> None:
        # The empty-list contract: "" so callers can prepend without branching.
        original_prompt = "You are a helpful assistant."
        prefix = lessons_to_context_block([])
        combined = f"{prefix}{original_prompt}"
        assert combined == original_prompt
