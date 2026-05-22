"""Unit tests for :func:`format_staleness_hint` (plan §Issue 8 / B1.4
worker-prompt integration).

Plan §Issue 8 / B1.4 acceptance row: "When surfacing past-TTL item,
mark ``needs_reverification=True`` in next thinking_note". This
file pins the worker-prompt tag formatter that produces that hint —
the analog of :func:`format_coreference_tag` for the staleness
domain.

Contract:

- empty input → ``None`` (caller skips injection entirely);
- TTL ≤ 0 → ``None`` (gate disabled);
- single-item input → tag carries the item label + TTL seconds +
  ``needs_reverification=True`` line;
- multi-item input → tag carries up to ``max_items`` labels + a
  ``"(+N more)"`` suffix when truncated;
- default ``label_for`` resolves through the corpus-handle anchor
  precedence (filename → uri → source_uri → name → source → first
  content line → ``item:<id>`` fallback);
- custom ``label_for`` callback overrides the default;
- the tag is wrapped in ``<context>...</context>`` per the worker-
  prompt contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from kaos_agents.memory.staleness import format_staleness_hint


@dataclass
class _FakeItem:
    """Mimics the SessionMemory item shape — attribute access for
    metadata + content + id."""

    id: str = "fake-1"
    metadata: dict[str, Any] | None = None
    content: str = ""


# ── No-op cases ─────────────────────────────────────────────────────


@pytest.mark.unit
def test_empty_list_returns_none() -> None:
    """Caller skips injection when there's nothing to warn about."""
    assert format_staleness_hint([], ttl_seconds=3600.0) is None


@pytest.mark.unit
def test_ttl_zero_returns_none() -> None:
    """Gate disabled by TTL=0 → no hint even with items present."""
    items = [_FakeItem(metadata={"filename": "a.pdf"})]
    assert format_staleness_hint(items, ttl_seconds=0.0) is None


@pytest.mark.unit
def test_ttl_negative_returns_none() -> None:
    """Negative TTL is meaningless; the helper treats it as disabled."""
    items = [_FakeItem(metadata={"filename": "a.pdf"})]
    assert format_staleness_hint(items, ttl_seconds=-1.0) is None


# ── Single-item rendering ───────────────────────────────────────────


@pytest.mark.unit
def test_single_stale_item_renders_full_tag() -> None:
    """The canonical single-item path produces the full
    ``<context>...needs_reverification=True...</context>`` tag."""
    items = [_FakeItem(metadata={"filename": "edgar-10k.pdf"})]
    tag = format_staleness_hint(items, ttl_seconds=3600.0)
    assert tag is not None
    assert tag.startswith("<context>")
    assert tag.rstrip().endswith("</context>")
    assert "edgar-10k.pdf" in tag
    # TTL is rendered as an integer for human readability.
    assert "3600s" in tag
    # The load-bearing flag — caller emits the structured field
    # alongside but the text-level marker is what the LLM consumes.
    assert "needs_reverification=True" in tag
    # The "1 tool-result item" count is present (singular shape).
    assert "1 tool-result item" in tag


# ── Multi-item rendering ────────────────────────────────────────────


@pytest.mark.unit
def test_multi_item_lists_each_label_up_to_max() -> None:
    """Up to ``max_items`` labels appear in the rendered tag."""
    items = [_FakeItem(id=f"i-{i}", metadata={"filename": f"fr-{i}.html"}) for i in range(3)]
    tag = format_staleness_hint(items, ttl_seconds=60.0, max_items=5)
    assert tag is not None
    for i in range(3):
        assert f"fr-{i}.html" in tag
    assert "3 tool-result item(s)" in tag


@pytest.mark.unit
def test_overflow_renders_count_suffix() -> None:
    """When the stale list exceeds ``max_items``, the tag carries
    the ``(+N more)`` suffix so the model knows the visible labels
    aren't exhaustive."""
    items = [_FakeItem(id=f"i-{i}", metadata={"filename": f"doc-{i}.pdf"}) for i in range(10)]
    tag = format_staleness_hint(items, ttl_seconds=60.0, max_items=3)
    assert tag is not None
    # Three visible labels.
    assert "doc-0.pdf" in tag
    assert "doc-1.pdf" in tag
    assert "doc-2.pdf" in tag
    # Overflow indicator (10 - 3 = 7 more).
    assert "(+ 7 more)" in tag
    assert "10 tool-result item(s)" in tag


# ── Default label_for precedence ───────────────────────────────────


@pytest.mark.unit
def test_default_label_uses_filename_metadata() -> None:
    """Default label_for mirrors the WU-G.2 corpus-handle anchor."""
    items = [_FakeItem(metadata={"filename": "contract.docx"})]
    tag = format_staleness_hint(items, ttl_seconds=3600.0)
    assert tag is not None and "contract.docx" in tag


@pytest.mark.unit
def test_default_label_falls_back_to_content_first_line() -> None:
    """No filename → first non-empty content line wins."""
    items = [_FakeItem(metadata={}, content="MUTUAL NDA\n\n1. Definitions")]
    tag = format_staleness_hint(items, ttl_seconds=3600.0)
    assert tag is not None
    assert "MUTUAL NDA" in tag


@pytest.mark.unit
def test_default_label_falls_back_to_item_id() -> None:
    """No filename + no content → synthetic ``item:<id>`` so the
    tag is never empty."""
    items = [_FakeItem(id="abc-123", metadata={}, content="")]
    tag = format_staleness_hint(items, ttl_seconds=3600.0)
    assert tag is not None
    assert "item:abc-123" in tag


# ── Custom label_for override ───────────────────────────────────────


@pytest.mark.unit
def test_custom_label_for_overrides_default() -> None:
    """A caller can supply their own label function (e.g. an
    artifact-id renderer)."""
    items = [_FakeItem(id="art-99", metadata={"filename": "ignored"})]

    def label_for(item: Any) -> str:
        return f"ART#{getattr(item, 'id', '?')}"

    tag = format_staleness_hint(items, ttl_seconds=3600.0, label_for=label_for)
    assert tag is not None
    assert "ART#art-99" in tag
    # The metadata.filename was ignored — only the custom label is used.
    assert "ignored" not in tag


# ── Dict-style items ────────────────────────────────────────────────


@pytest.mark.unit
def test_dict_style_items_work_for_jsonl_payloads() -> None:
    """Items deserialised from JSONL come as dicts, not dataclasses.
    The default label_for must handle both shapes — pin it."""
    items: list[dict[str, Any]] = [
        {"metadata": {"filename": "live-quote.json"}, "content": ""},
        {"id": "fallback-id", "metadata": {}, "content": ""},
    ]
    tag = format_staleness_hint(items, ttl_seconds=300.0)
    assert tag is not None
    assert "live-quote.json" in tag
    assert "item:fallback-id" in tag


# ── Plan-acceptance shape ───────────────────────────────────────────


@pytest.mark.unit
def test_tag_carries_reverification_marker_for_caller_field() -> None:
    """The structured ``needs_reverification=True`` field that the
    plan calls out lives in the persisted thinking_note JSON; the
    tag mirrors it in the LLM-visible text so the model has the
    same signal whether or not it sees the structured field. Pin
    that mirror."""
    items = [_FakeItem(metadata={"filename": "x"})]
    tag = format_staleness_hint(items, ttl_seconds=60.0)
    assert tag is not None
    # The marker appears verbatim (not just "True needs_reverification"
    # or any close-but-different variant).
    assert "needs_reverification=True" in tag
