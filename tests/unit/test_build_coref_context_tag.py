"""Unit tests for build_coref_context_tag (plan §Issue 8 / B1.5 wire-up).

The :func:`build_coref_context_tag` helper composes
:func:`resolve_ordinal` + :func:`format_coreference_tag` against a
SessionMemory section so the agent run loop can call a single
function after assemble_context.

This file pins:

- ``None`` when no ordinal phrase fires.
- ``None`` when the section is empty or absent.
- The confident in-range tag string when "the third NDA" resolves
  against five DOCUMENTS items.
- The clarify-ambiguity tag string when the user references an
  ordinal beyond the candidate count ("the eighth NDA" against 5).
- ``min_confidence`` gating — "the next" (confidence=0.5) returns a
  tag at default ``0.5`` but is suppressed at ``0.99``.
- The default ``label_for`` extracts the filename metadata anchor
  so the rendered referent matches the WU-G.2 corpus handle.
- A custom ``label_for`` callback overrides the default.
- Non-DOCUMENTS sections work when explicitly passed.
"""

from __future__ import annotations

import pytest

from kaos_agents.context.coreference import build_coref_context_tag
from kaos_agents.memory.session import SessionMemory
from kaos_agents.types.memory import MemoryType


def _make_memory_with_docs(*filenames: str) -> SessionMemory:
    """Build a SessionMemory with N DOCUMENTS items keyed by filename."""
    memory = SessionMemory("test-session")
    for fname in filenames:
        memory.add(
            MemoryType.DOCUMENTS,
            content=f"[document body for {fname}]",
            metadata={"filename": fname},
        )
    return memory


# ── No-op cases ─────────────────────────────────────────────────────


@pytest.mark.unit
def test_no_ordinal_returns_none() -> None:
    """Non-ordinal messages skip the tag-injection step."""
    memory = _make_memory_with_docs("nda-1.pdf", "nda-2.pdf", "nda-3.pdf")
    tag = build_coref_context_tag(memory, "Tell me about the NDAs.")
    assert tag is None


@pytest.mark.unit
def test_empty_documents_section_returns_none() -> None:
    """No candidates → no tag (the resolver can't bind anything)."""
    memory = SessionMemory("test-session")
    # Section exists but is empty.
    tag = build_coref_context_tag(memory, "Tell me about the third NDA.")
    assert tag is None


# ── Confident in-range resolution ───────────────────────────────────


@pytest.mark.unit
def test_third_nda_resolves_against_five_docs() -> None:
    """Plan acceptance fixture pattern."""
    memory = _make_memory_with_docs(
        "nda-1.pdf",
        "nda-2.pdf",
        "nda-3.pdf",
        "nda-4.pdf",
        "nda-5.pdf",
    )
    tag = build_coref_context_tag(memory, "Governing law on the third NDA?")
    assert tag is not None
    # The rendered tag references the filename metadata anchor for
    # the third candidate (1-based ordinal → 0-based index 2).
    assert "nda-3.pdf" in tag
    assert "position 3" in tag
    assert "the third NDA" in tag  # echo of the matched phrase
    # Tag is wrapped in <context>...</context> per the worker prompt
    # contract.
    assert tag.startswith("<context>")
    assert tag.rstrip().endswith("</context>")


# ── Out-of-range clarification branch ───────────────────────────────


@pytest.mark.unit
def test_out_of_range_ordinal_renders_clarify_tag() -> None:
    """Ordinal beyond the candidate count flags the ambiguity rather
    than silently binding to the last candidate."""
    memory = _make_memory_with_docs("nda-1.pdf", "nda-2.pdf", "nda-3.pdf")
    tag = build_coref_context_tag(memory, "What about the eighth NDA?")
    assert tag is not None
    assert "out of range" in tag.lower()
    assert "the eighth NDA" in tag
    # The worker is instructed to ask for clarification.
    assert "clarify" in tag.lower() or "refuse" in tag.lower()


# ── min_confidence gate ─────────────────────────────────────────────


@pytest.mark.unit
def test_the_next_renders_low_confidence_tag_at_default_threshold() -> None:
    """At ``min_confidence=0.5`` (default), the "the next" heuristic
    (confidence=0.5) surfaces a clarify-ambiguity tag."""
    memory = _make_memory_with_docs("nda-1.pdf", "nda-2.pdf")
    tag = build_coref_context_tag(memory, "Let's discuss the next document.")
    assert tag is not None
    assert "low confidence" in tag.lower()


@pytest.mark.unit
def test_the_next_suppressed_at_high_min_confidence() -> None:
    """Bumping ``min_confidence`` past 0.5 hides the heuristic."""
    memory = _make_memory_with_docs("nda-1.pdf", "nda-2.pdf")
    tag = build_coref_context_tag(
        memory,
        "Let's discuss the next document.",
        min_confidence=0.99,
    )
    assert tag is None


# ── Label-for callback ──────────────────────────────────────────────


@pytest.mark.unit
def test_default_label_uses_filename_metadata() -> None:
    """Default label_for mirrors the WU-G.2 corpus-handle anchor."""
    memory = _make_memory_with_docs("amendment-redline.docx")
    tag = build_coref_context_tag(memory, "Show me the first one.")
    assert tag is not None
    assert "amendment-redline.docx" in tag


@pytest.mark.unit
def test_default_label_falls_back_to_content_first_line() -> None:
    """A document with no filename metadata uses the first content
    line so the rendered <referent> is never empty."""
    memory = SessionMemory("test-session")
    memory.add(
        MemoryType.DOCUMENTS,
        content="MUTUAL NON-DISCLOSURE AGREEMENT\n\n1. Definitions.",
    )
    tag = build_coref_context_tag(memory, "Tell me about the first one.")
    assert tag is not None
    assert "MUTUAL NON-DISCLOSURE AGREEMENT" in tag


@pytest.mark.unit
def test_custom_label_for_overrides_default() -> None:
    """A caller can supply their own label_for to render referents
    differently (e.g. a doc-id rather than filename)."""
    memory = _make_memory_with_docs("foo.pdf", "bar.pdf", "baz.pdf")

    def label_for(item: object) -> str:
        return f"DOC#{getattr(item, 'id', '?')[:8]}"

    tag = build_coref_context_tag(
        memory,
        "Show me the second one.",
        label_for=label_for,
    )
    assert tag is not None
    assert "DOC#" in tag


# ── Section override ────────────────────────────────────────────────


@pytest.mark.unit
def test_section_kwarg_resolves_against_findings() -> None:
    """Passing ``section=FINDINGS`` lets a research agent resolve
    "the third finding" against the same primitive."""
    memory = SessionMemory("test-session")
    memory.add(MemoryType.FINDINGS, content="Finding A: foo")
    memory.add(MemoryType.FINDINGS, content="Finding B: bar")
    memory.add(MemoryType.FINDINGS, content="Finding C: baz")
    tag = build_coref_context_tag(
        memory,
        "Elaborate on the third finding.",
        section=MemoryType.FINDINGS,
    )
    assert tag is not None
    assert "position 3" in tag
    # Default label falls back to first non-empty content line.
    assert "Finding C" in tag


@pytest.mark.unit
def test_full_unpruned_section_drives_resolution_not_assembled_window() -> None:
    """ "the third NDA" must resolve against the FULL DOCUMENTS section
    (across the whole session), not the post-BM25-prune window the
    assemble_context layer hands to the worker.

    This is the core invariant the integration step preserves:
    candidates come from ``memory.get(DOCUMENTS)`` directly, NOT
    from the assembled context dict. Otherwise a turn that BM25-
    drops doc-3 from the assembled context would silently bind
    "the third NDA" to whatever happened to survive instead.
    """
    memory = _make_memory_with_docs("nda-1.pdf", "nda-2.pdf", "nda-3.pdf", "nda-4.pdf", "nda-5.pdf")
    tag = build_coref_context_tag(memory, "Governing law on the third NDA?")
    assert tag is not None
    # Always the 3rd by *full-session* order, regardless of what
    # any future BM25 prune would have chosen.
    assert "nda-3.pdf" in tag
