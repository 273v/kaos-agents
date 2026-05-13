"""Integration tests — EventEmitter <-> EventCollector (Phase 0.B).

Covers the additive changes to ``EventEmitter``:

1. ``emit()`` pushes every constructed event to the active collector
   (no-op when none active — preserves backward compat for tests
   that don't open a collect_events scope).
2. ``span()`` synthesizes ``parent_span_id`` from the active
   collector's span stack on START phases when the caller didn't
   pass one explicitly. COMPLETE/ERROR phases never auto-synthesize.
3. Explicit ``parent_span_id`` always wins (synthesis is fall-back).
4. The constructed Span lands in ``collector.events`` with the
   populated parent_span_id.
"""

from __future__ import annotations

from kaos_agents.events.collector import collect_events
from kaos_agents.events.emitter import EventEmitter
from kaos_agents.events.lifecycle import IntentClassified
from kaos_agents.events.spans import Span, SpanSubject

_SID = "sess-1"
_RID = "run-1"


# ---------------------------------------------------------------------------
# emit() with no collector active — backward compat
# ---------------------------------------------------------------------------


class TestEmitterNoCollector:
    def test_no_collector_parent_stays_none(self) -> None:
        """Pre-Phase-0.B behavior: every Span has parent_span_id=None
        when no collector scope is open."""
        emitter = EventEmitter(session_id=_SID, run_id=_RID)
        span = emitter.span_start(SpanSubject.TURN, name="turn.1")
        assert span.parent_span_id is None
        # Sanity: emit() returned a Span, no error.
        assert isinstance(span, Span)


# ---------------------------------------------------------------------------
# emit() pushes to active collector
# ---------------------------------------------------------------------------


class TestEmitPushesToCollector:
    def test_span_event_lands_in_collector(self) -> None:
        emitter = EventEmitter(session_id=_SID, run_id=_RID)
        with collect_events() as coll:
            span = emitter.span_start(SpanSubject.TURN, name="turn.1")
        assert span in coll.events

    def test_value_event_also_lands_in_collector(self) -> None:
        """Non-Span events go through emit() too — they should also push."""
        emitter = EventEmitter(session_id=_SID, run_id=_RID)
        with collect_events() as coll:
            event = emitter.emit(IntentClassified, intent="tool_use", confidence=0.9, reasoning="r")
        assert event in coll.events

    def test_multiple_events_in_emission_order(self) -> None:
        emitter = EventEmitter(session_id=_SID, run_id=_RID)
        with collect_events() as coll:
            s1 = emitter.span_start(SpanSubject.TURN, name="turn.1")
            s2 = emitter.span_start(SpanSubject.STEP, name="step.1")
            s3 = emitter.span_complete(SpanSubject.STEP, span_id=s2.span_id)
            s4 = emitter.span_complete(SpanSubject.TURN, span_id=s1.span_id)
        assert coll.events == [s1, s2, s3, s4]


# ---------------------------------------------------------------------------
# parent_span_id synthesis from collector stack
# ---------------------------------------------------------------------------


class TestParentSpanSynthesis:
    def test_outer_turn_parents_inner_step(self) -> None:
        """Inside a TURN span, a STEP span_start auto-parents to TURN."""
        emitter = EventEmitter(session_id=_SID, run_id=_RID)
        with collect_events():
            turn = emitter.span_start(SpanSubject.TURN, name="turn.1")
            step = emitter.span_start(SpanSubject.STEP, name="step.s001")
        assert turn.parent_span_id is None  # no outer scope at the time
        assert step.parent_span_id == turn.span_id

    def test_three_level_nesting_threads_parents(self) -> None:
        emitter = EventEmitter(session_id=_SID, run_id=_RID)
        with collect_events():
            turn = emitter.span_start(SpanSubject.TURN, name="turn.1")
            step = emitter.span_start(SpanSubject.STEP, name="step.1")
            tool = emitter.span_start(SpanSubject.TOOL_CALL, name="tool.k1")
        assert turn.parent_span_id is None
        assert step.parent_span_id == turn.span_id
        assert tool.parent_span_id == step.span_id

    def test_seeded_collector_provides_root_parent(self) -> None:
        """``collect_events(parent_span_id=X)`` — first emitted START
        gets X as its parent. Used by sub-agent delegation: the
        Runner opens the child scope seeded with the parent
        SUBAGENT span_id so the sub-agent's TURN start parents
        cleanly to the SUBAGENT span."""
        emitter = EventEmitter(session_id=_SID, run_id=_RID)
        with collect_events(parent_span_id="parent-from-runner"):
            child_turn = emitter.span_start(SpanSubject.TURN, name="turn.1")
        assert child_turn.parent_span_id == "parent-from-runner"

    def test_explicit_parent_overrides_synthesis(self) -> None:
        """When the caller passes parent_span_id=, that value wins."""
        emitter = EventEmitter(session_id=_SID, run_id=_RID)
        with collect_events():
            turn = emitter.span_start(SpanSubject.TURN, name="turn.1")
            # Explicit parent — should NOT be overwritten by the synthesis.
            step = emitter.span_start(
                SpanSubject.STEP, name="step.1", parent_span_id="explicit-parent"
            )
        # turn was the top of the stack, but explicit value dominates.
        assert turn.parent_span_id is None
        assert step.parent_span_id == "explicit-parent"

    def test_complete_phase_does_not_auto_parent(self) -> None:
        """COMPLETE spans never synthesize. Caller already has the
        START's parent_span_id in scope if propagation is desired."""
        emitter = EventEmitter(session_id=_SID, run_id=_RID)
        with collect_events():
            turn = emitter.span_start(SpanSubject.TURN, name="turn.1")
            # Inside the TURN, a STEP completes.
            # Synthesis on COMPLETE would (wrongly) set parent_span_id
            # to turn.span_id from the stack — verify it stays None.
            step_complete = emitter.span_complete(SpanSubject.STEP, span_id="some-step-id")
        # Sanity: turn captured.
        assert turn.parent_span_id is None
        # Critical: COMPLETE has no auto-parent.
        assert step_complete.parent_span_id is None

    def test_error_phase_does_not_auto_parent(self) -> None:
        emitter = EventEmitter(session_id=_SID, run_id=_RID)
        with collect_events():
            _ = emitter.span_start(SpanSubject.TURN, name="turn.1")
            err = emitter.span_error(
                SpanSubject.STEP,
                span_id="some-step-id",
                error_type="boom",
                error_message="kaboom",
            )
        assert err.parent_span_id is None

    def test_parent_drops_to_grandparent_after_complete(self) -> None:
        """Once the STEP completes, the next START in scope parents to
        TURN again — verifies the stack pops properly via the collector."""
        emitter = EventEmitter(session_id=_SID, run_id=_RID)
        with collect_events():
            turn = emitter.span_start(SpanSubject.TURN, name="turn.1")
            step1 = emitter.span_start(SpanSubject.STEP, name="step.1")
            _ = emitter.span_complete(SpanSubject.STEP, span_id=step1.span_id)
            # After step1.COMPLETE the stack top is back to turn.
            step2 = emitter.span_start(SpanSubject.STEP, name="step.2")
        assert step1.parent_span_id == turn.span_id
        assert step2.parent_span_id == turn.span_id

    def test_no_collector_no_synthesis(self) -> None:
        """No collector scope -> no synthesis. Emitter's existing
        behavior (parent_span_id=None unless caller passes it) is
        preserved."""
        emitter = EventEmitter(session_id=_SID, run_id=_RID)
        # No collect_events block.
        span = emitter.span_start(SpanSubject.STEP, name="step.x")
        assert span.parent_span_id is None
