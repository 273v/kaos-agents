"""Unit tests for the ``emit_memory_added`` helper.

Theme B of the 2026-05-25 A+ audit: many helpers write to ``SessionMemory``
sections (FINDINGS / REFLECTION / MESSAGES / DOCUMENTS) without emitting
the canonical ``MemoryEvent(kind=ADDED, ...)``. The helper closes that
observability gap by reading the active emitter from the ContextVar
installed by ``use_emitter`` (Theme A) and emitting on its behalf —
no-op when no emitter is in scope (back-compat with non-agent callers).
"""

from __future__ import annotations

from kaos_agents.events import (
    EventEmitter,
    MemoryEvent,
    MemoryEventKind,
    active_emitter,
    collect_events,
    emit_memory_added,
    use_emitter,
)


class TestEmitMemoryAddedHelper:
    def test_no_emitter_in_scope_is_noop(self) -> None:
        """Without ``use_emitter``, the helper returns None and emits nothing."""
        # Sanity: no emitter installed at module-load time.
        assert active_emitter() is None

        with collect_events() as coll:
            result = emit_memory_added("MESSAGES", item_count=1)

        assert result is None
        assert len(coll.events) == 0

    def test_emits_memory_added_via_active_emitter(self) -> None:
        """Inside ``use_emitter``, the helper emits a MemoryEvent into the collector."""
        emitter = EventEmitter(session_id="s1", run_id="r1")

        with collect_events() as coll, use_emitter(emitter):
            result = emit_memory_added(
                "FINDINGS",
                item_count=3,
                attributes={"verified": True},
            )

        assert result is not None
        assert isinstance(result, MemoryEvent)
        assert result.kind == MemoryEventKind.ADDED
        assert result.section == "FINDINGS"
        assert result.item_count == 3
        assert result.attributes == {"verified": True}

        # Collector saw exactly one event — the MemoryEvent we emitted.
        assert len(coll.events) == 1
        assert coll.events[0] is result

    def test_default_item_count_is_one(self) -> None:
        """item_count defaults to 1 (the most common case for append-one writes)."""
        emitter = EventEmitter(session_id="s1", run_id="r1")
        with collect_events() as coll, use_emitter(emitter):
            emit_memory_added("REFLECTION")
        evt = coll.events[0]
        assert isinstance(evt, MemoryEvent)
        assert evt.item_count == 1
        assert evt.attributes == {}
