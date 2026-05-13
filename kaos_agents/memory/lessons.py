"""Reflexion memory — cross-session distilled lessons.

The :class:`~kaos_agents.types.memory.MemoryType.LESSONS` section is
where the agent stores *what it learned* from one session for recall
in future sessions. It's the compounding primitive: every session
makes the next one cheaper / smarter, because the agent doesn't
re-discover the same gotchas forever.

Distinct from :class:`~kaos_agents.types.memory.MemoryType.REFLECTION`,
which is per-turn:

- ``REFLECTION``: scratchpad for the current turn ("I just tried X
  and it failed because Y"). Useful within the turn, fades after.
- ``LESSONS``: compact, durable "if you ever see situation X again,
  the takeaway is Y." Recalled in future sessions via BM25 over
  the ``situation`` text.

A lesson carries four fields:

- ``situation``: a short description of the conditions under which
  this lesson applies. The BM25-indexed text — phrase it the way a
  future agent would describe the situation it's facing.
- ``observation``: what the agent observed during the original session.
- ``takeaway``: the actionable insight extracted from the observation.
- ``evidence_refs``: optional pointers to source artifacts (event ids,
  block_refs, doc URIs) so the lesson is auditable.

Usage::

    # End of session: distill a lesson and write it.
    lesson = Lesson(
        situation="user asks 'which X is longest' on a corpus of similar docs",
        observation="agent retrieved all docs but refused with 'insufficient evidence'",
        takeaway="aggregation questions need compute tools, not more retrieval",
        evidence_refs=("event:turn-abc123/CitationFound",),
    )
    write_lesson(memory, lesson)

    # Start of next session: recall lessons whose situation matches.
    relevant = recall_lessons(memory, situation_query="ranking files by size", top_k=3)
    for lesson in relevant:
        # surface in the agent's context for the upcoming turn
        ...

Lesson distillation can be automated by a ``ReflexionHook`` (deferred
to a follow-up) that listens for ``Span(TURN, COMPLETE)`` events and
asks the LLM to summarize the turn into a Lesson. The hook is
optional — manual ``write_lesson`` calls are equally valid.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

from kaos_agents.types.memory import MemoryType

if TYPE_CHECKING:
    from kaos_agents.memory.session import SessionMemory


# Max lessons returned per ``recall_lessons`` call by default. Past
# ~5, the prompt-context cost outweighs the value of the next-most-
# relevant lesson; callers can raise for forensic queries.
DEFAULT_RECALL_TOP_K: int = 5


class Lesson(BaseModel):
    """A distilled cross-session lesson, written once and recalled by similarity.

    The ``situation`` field is what gets BM25-indexed for recall —
    phrase it the way a future agent (or the LLM doing situation
    matching) would describe the situation it's facing, not the way
    you'd write a postmortem header. Specific concrete vocabulary
    beats abstract framing for retrieval purposes.

    Attributes:
        situation: BM25-indexed condition description (1-2 sentences).
        observation: What was actually observed during the original
            session. Audit-grade narrative — don't compress.
        takeaway: The actionable insight. Should read as advice to a
            future agent in the same situation. Imperative voice
            preferred ("Use compute tools when..." vs "Compute tools
            help when...").
        evidence_refs: Optional anchors back to the original session's
            artifacts. Free-form strings — convention is
            ``event:<id>``, ``block_ref:<json_pointer>``, ``doc_uri:<uri>``.
        confidence: Optional 0..1 self-rating. Useful when an
            automatic distiller writes lessons and a human reviewer
            later wants to filter low-confidence ones.
        tags: Optional categorical tags for filtering / grouping.
            Free-form; common values: ``"pathology"``, ``"workflow"``,
            ``"vocabulary"``, ``"cost"``, ``"refusal"``.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    situation: str = Field(min_length=1)
    observation: str = Field(min_length=1)
    takeaway: str = Field(min_length=1)
    evidence_refs: tuple[str, ...] = Field(default_factory=tuple)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    tags: tuple[str, ...] = Field(default_factory=tuple)

    def to_searchable_text(self) -> str:
        """Build the BM25-indexed text for this lesson.

        Composition: situation as primary signal, takeaway as
        secondary, observation as a fallback for queries that match
        the observed phenomenon rather than the high-level situation
        framing.
        """
        parts = [self.situation, self.takeaway, self.observation]
        if self.tags:
            parts.append(" ".join(self.tags))
        return "\n".join(parts)

    def to_recall_summary(self) -> str:
        """Compact rendering for prompt context.

        Returns a 3-line summary suitable for injection into the
        agent's working memory at session start: situation header,
        takeaway as the actionable line, and a confidence/evidence
        footnote when available.
        """
        lines = [
            f"Situation: {self.situation}",
            f"Takeaway: {self.takeaway}",
        ]
        if self.confidence is not None:
            lines.append(f"Confidence: {self.confidence:.2f}")
        if self.evidence_refs:
            lines.append(f"Evidence: {', '.join(self.evidence_refs[:3])}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Read/write helpers
# ---------------------------------------------------------------------------


def write_lesson(memory: SessionMemory, lesson: Lesson) -> None:
    """Persist a lesson to the LESSONS section of session memory.

    The lesson is serialized as a JSON blob in the section's content
    so that BM25 search over ``situation`` + ``takeaway`` + observation
    works without a custom retrieval path. The full Pydantic round-trip
    is preserved via ``read_lessons``.

    Args:
        memory: The session memory to write into.
        lesson: The :class:`Lesson` to persist.

    Raises:
        KeyError: if the memory has no LESSONS section configured.
            That should not happen under default configuration —
            LESSONS is in the platform-default section set as of
            this writing — but the error is explicit if a custom
            section list omits it.
    """
    if not memory.has_section(MemoryType.LESSONS):
        raise KeyError(
            "SessionMemory has no LESSONS section configured. "
            "Default configuration includes it; check the SectionConfig "
            "list passed to SessionMemory if you customized it."
        )

    # Two-channel storage:
    # - ``content``: plain searchable text (situation + takeaway +
    #   observation + tags). BM25 tokenizes this directly so morphology
    #   ("compare" / "comparative") and adjacent forms behave naturally.
    # - ``metadata["lesson_json"]``: the full Pydantic round-trip
    #   payload. ``read_lessons`` reconstructs the typed value from
    #   here. Storing JSON in content would corrupt BM25 by tokenizing
    #   JSON punctuation alongside the words.
    memory.add(
        MemoryType.LESSONS,
        lesson.to_searchable_text(),
        tags=lesson.tags,
        metadata={
            "lesson": True,
            "lesson_json": lesson.model_dump_json(),
            "situation_preview": lesson.situation[:120],
        },
    )


def read_lessons(memory: SessionMemory) -> list[Lesson]:
    """Read all lessons stored in the LESSONS section.

    Returns lessons in insertion order. Use :func:`recall_lessons`
    for query-driven retrieval; this is the dump path for audit or
    explicit iteration.
    """
    if not memory.has_section(MemoryType.LESSONS):
        return []
    # ``get`` returns items in insertion order; max_tokens=None means
    # "everything in the section" (small enough that this is fine).
    items = memory.get(MemoryType.LESSONS)
    out: list[Lesson] = []
    for item in items:
        # The full Pydantic round-trip payload lives in metadata
        # under ``lesson_json``; the item's ``content`` is the
        # plain searchable text (see write_lesson rationale).
        payload = (item.metadata or {}).get("lesson_json")
        if not isinstance(payload, str):
            continue
        try:
            out.append(Lesson.model_validate_json(payload))
        except (ValueError, TypeError):
            # Malformed entry — skip silently. Should not happen if
            # write_lesson is the only writer.
            continue
    return out


def recall_lessons(
    memory: SessionMemory,
    situation_query: str,
    *,
    top_k: int = DEFAULT_RECALL_TOP_K,
) -> list[Lesson]:
    """Retrieve lessons most relevant to ``situation_query`` via BM25.

    Args:
        memory: The session memory to search.
        situation_query: Free-form description of the current
            situation. Typically the user's message or a summary of
            the upcoming turn.
        top_k: Max lessons to return. Default 5. Past ~5 the
            prompt-context cost typically outweighs the value of the
            next-most-relevant lesson.

    Returns:
        Ranked list of :class:`Lesson` (most relevant first). Empty
        when the LESSONS section is empty or unconfigured.
    """
    from kaos_agents.memory.search import search_memory

    if not situation_query or not situation_query.strip():
        return []
    if not memory.has_section(MemoryType.LESSONS):
        return []
    if memory.section_item_count(MemoryType.LESSONS) == 0:
        return []

    results = search_memory(
        memory,
        situation_query,
        top_k=top_k,
        sections=[MemoryType.LESSONS],
    )
    # Build an item_id → Lesson lookup so we can reconstruct the typed
    # values from the metadata payload (the search result carries only
    # plain content + item_id).
    items_by_id = {item.id: item for item in memory.get(MemoryType.LESSONS)}
    out: list[Lesson] = []
    for r in results:
        item = items_by_id.get(r.item_id)
        if item is None:
            continue
        payload = (item.metadata or {}).get("lesson_json")
        if not isinstance(payload, str):
            continue
        try:
            out.append(Lesson.model_validate_json(payload))
        except (ValueError, TypeError):
            continue
    return out


def lessons_to_context_block(lessons: list[Lesson]) -> str:
    """Render a list of lessons as a single prompt-friendly block.

    Use to inject recalled lessons at session start::

        recalled = recall_lessons(memory, user_message)
        if recalled:
            extra_instruction = lessons_to_context_block(recalled)
            # ... append to the agent's instructions

    Returns the empty string for an empty list so callers can
    unconditionally prepend without branching.
    """
    if not lessons:
        return ""
    blocks = [
        f"## Lessons from prior sessions ({len(lessons)})",
        "",
        "Apply these takeaways when the situation matches. They were",
        "distilled from earlier sessions; ignore any that don't fit.",
        "",
    ]
    for i, lesson in enumerate(lessons, start=1):
        blocks.append(f"### Lesson {i}")
        blocks.append(lesson.to_recall_summary())
        blocks.append("")
    return "\n".join(blocks)


__all__ = [
    "DEFAULT_RECALL_TOP_K",
    "Lesson",
    "lessons_to_context_block",
    "read_lessons",
    "recall_lessons",
    "write_lesson",
]


# Round-trip safety: serializing a Lesson must always be re-parseable.
# Exposed as a module-level callable so tests can sanity-check without
# importing the helpers individually.
def _roundtrip_lesson(lesson: Lesson) -> Lesson:
    """Internal: dump → parse round-trip. Used by unit tests."""
    return Lesson.model_validate_json(lesson.model_dump_json())


# Quiet the unused-helper warning when this module is imported by callers
# that don't need it. Keeps the helper testable.
_ = json
