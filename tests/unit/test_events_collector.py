"""Tests for the contextvar-based EventCollector (Phase 0.B).

Covers the standalone collector primitives:

- ``push_event`` is a no-op when no collector is active.
- ``collect_events`` captures every emitted event in order.
- Nested collect_events scopes are independent — outer doesn't see
  inner events, inner doesn't leak into outer.
- ``EventCollector.current_parent_span_id`` reflects the live span
  stack (None empty, top of stack with one open Span, new top when
  two are nested).
- ``EventCollector`` pops on Span(COMPLETE) and tolerates out-of-order
  completions (lenient pop).
- ContextVar isolation across asyncio tasks: each task sees its own
  collector.

The behavior under test does not depend on any specific Span subject;
TURN/STEP/etc. are interchangeable for the stack discipline.
"""

from __future__ import annotations

import asyncio

from kaos_agents.events.collector import (
    EventCollector,
    active_collector,
    collect_events,
    push_event,
)
from kaos_agents.events.spans import Span, SpanPhase, SpanSubject

# ---------------------------------------------------------------------------
# Minimal Span factory — events need full base fields populated.
# ---------------------------------------------------------------------------

_TS = 1.0
_SID = "sess-1"
_RID = "run-1"


def _span(
    *,
    phase: SpanPhase,
    span_id: str,
    subject: SpanSubject = SpanSubject.TURN,
    parent_span_id: str | None = None,
    sequence: int = 0,
) -> Span:
    return Span(
        timestamp=_TS,
        sequence=sequence,
        session_id=_SID,
        run_id=_RID,
        subject=subject,
        phase=phase,
        span_id=span_id,
        parent_span_id=parent_span_id,
        name=f"{subject.value}.{span_id}",
        attributes={},
    )


# ---------------------------------------------------------------------------
# push_event when no collector is active
# ---------------------------------------------------------------------------


class TestPushEventNoCollector:
    def test_push_event_no_collector_is_noop(self) -> None:
        """push_event swallows the call when no collector has been opened."""
        # No collector active — push must not raise.
        assert active_collector() is None
        push_event(_span(phase=SpanPhase.START, span_id="s1"))
        # Still no collector after the push.
        assert active_collector() is None


# ---------------------------------------------------------------------------
# collect_events captures events in order
# ---------------------------------------------------------------------------


class TestCollectEventsCapture:
    def test_collect_events_captures_in_order(self) -> None:
        """Events appended in the order they were pushed."""
        with collect_events() as coll:
            push_event(_span(phase=SpanPhase.START, span_id="a", sequence=0))
            push_event(_span(phase=SpanPhase.START, span_id="b", sequence=1))
            push_event(_span(phase=SpanPhase.COMPLETE, span_id="b", sequence=2))
            push_event(_span(phase=SpanPhase.COMPLETE, span_id="a", sequence=3))
        assert len(coll) == 4
        assert isinstance(coll.events[0], Span)
        assert [ev.span_id for ev in coll.events if isinstance(ev, Span)] == [
            "a",
            "b",
            "b",
            "a",
        ]

    def test_collector_iter(self) -> None:
        with collect_events() as coll:
            push_event(_span(phase=SpanPhase.START, span_id="a"))
            push_event(_span(phase=SpanPhase.COMPLETE, span_id="a"))
        items = list(coll)
        assert len(items) == 2

    def test_active_collector_inside_block(self) -> None:
        with collect_events() as coll:
            assert active_collector() is coll
        assert active_collector() is None


# ---------------------------------------------------------------------------
# Nested collect_events
# ---------------------------------------------------------------------------


class TestCollectEventsNested:
    def test_nested_inner_outer_independent(self) -> None:
        """Inner captures inner-only; outer captures outer-only.

        Mirrors the ``collect_traces`` discipline in kaos-llm-core: the
        *inner* collector intercepts events while it is innermost; the
        *outer* collector resumes once the inner block exits and never
        sees the inner events.
        """
        with collect_events() as outer:
            push_event(_span(phase=SpanPhase.START, span_id="outer-1", sequence=0))
            with collect_events() as inner:
                push_event(_span(phase=SpanPhase.START, span_id="inner-1", sequence=1))
                push_event(_span(phase=SpanPhase.COMPLETE, span_id="inner-1", sequence=2))
            # After inner block exits, outer should be active again.
            push_event(_span(phase=SpanPhase.COMPLETE, span_id="outer-1", sequence=3))

        outer_ids = [ev.span_id for ev in outer.events if isinstance(ev, Span)]
        inner_ids = [ev.span_id for ev in inner.events if isinstance(ev, Span)]
        assert outer_ids == ["outer-1", "outer-1"]
        assert inner_ids == ["inner-1", "inner-1"]


# ---------------------------------------------------------------------------
# Span stack discipline
# ---------------------------------------------------------------------------


class TestSpanStack:
    def test_empty_stack_returns_none(self) -> None:
        coll = EventCollector()
        assert coll.current_parent_span_id() is None

    def test_one_start_pushes(self) -> None:
        coll = EventCollector()
        coll.append(_span(phase=SpanPhase.START, span_id="t1"))
        assert coll.current_parent_span_id() == "t1"

    def test_nested_starts_track_top(self) -> None:
        coll = EventCollector()
        coll.append(_span(phase=SpanPhase.START, span_id="t1"))
        coll.append(_span(phase=SpanPhase.START, span_id="t2", subject=SpanSubject.STEP))
        assert coll.current_parent_span_id() == "t2"

    def test_complete_pops_back_to_parent(self) -> None:
        coll = EventCollector()
        coll.append(_span(phase=SpanPhase.START, span_id="t1"))
        coll.append(_span(phase=SpanPhase.START, span_id="t2", subject=SpanSubject.STEP))
        coll.append(_span(phase=SpanPhase.COMPLETE, span_id="t2", subject=SpanSubject.STEP))
        assert coll.current_parent_span_id() == "t1"
        coll.append(_span(phase=SpanPhase.COMPLETE, span_id="t1"))
        assert coll.current_parent_span_id() is None

    def test_error_pops_like_complete(self) -> None:
        coll = EventCollector()
        coll.append(_span(phase=SpanPhase.START, span_id="t1"))
        coll.append(_span(phase=SpanPhase.ERROR, span_id="t1"))
        assert coll.current_parent_span_id() is None

    def test_lenient_pop_out_of_order(self) -> None:
        """COMPLETE for an outer span pops the outer + everything above it.

        Tolerates pathological streams where an inner span never received
        its COMPLETE — completing the outer still drains the stack so
        future siblings don't get parented to the dangling inner.
        """
        coll = EventCollector()
        coll.append(_span(phase=SpanPhase.START, span_id="outer"))
        coll.append(_span(phase=SpanPhase.START, span_id="inner", subject=SpanSubject.STEP))
        # Out-of-order: complete outer while inner is still open.
        coll.append(_span(phase=SpanPhase.COMPLETE, span_id="outer"))
        # Both should be popped — stack is now empty.
        assert coll.current_parent_span_id() is None

    def test_complete_for_unknown_span_id_is_noop(self) -> None:
        """COMPLETE for a span_id not in the stack leaves the stack alone."""
        coll = EventCollector()
        coll.append(_span(phase=SpanPhase.START, span_id="t1"))
        coll.append(_span(phase=SpanPhase.COMPLETE, span_id="ghost"))
        assert coll.current_parent_span_id() == "t1"

    def test_progress_does_not_affect_stack(self) -> None:
        """PROGRESS phases do not push or pop the stack."""
        coll = EventCollector()
        coll.append(_span(phase=SpanPhase.START, span_id="t1"))
        coll.append(_span(phase=SpanPhase.PROGRESS, span_id="t1"))
        assert coll.current_parent_span_id() == "t1"

    def test_seeded_parent_span_id(self) -> None:
        """``collect_events(parent_span_id=...)`` seeds the stack."""
        with collect_events(parent_span_id="parent-from-outer") as coll:
            assert coll.current_parent_span_id() == "parent-from-outer"
            push_event(_span(phase=SpanPhase.START, span_id="child"))
            assert coll.current_parent_span_id() == "child"


# ---------------------------------------------------------------------------
# ContextVar isolation across asyncio tasks
# ---------------------------------------------------------------------------


class TestContextVarIsolation:
    async def test_two_tasks_have_independent_collectors(self) -> None:
        """Each asyncio task sees its own active collector.

        contextvars copy on ``asyncio.create_task`` boundaries, but each
        task's ``ContextVar.set()`` is visible only within that task's
        copy of the context. Two concurrent ``collect_events`` blocks
        in two tasks must therefore not cross-contaminate.
        """

        async def worker(span_id: str, sleep_after: float) -> EventCollector:
            with collect_events() as coll:
                push_event(_span(phase=SpanPhase.START, span_id=span_id))
                await asyncio.sleep(sleep_after)
                push_event(_span(phase=SpanPhase.COMPLETE, span_id=span_id))
            return coll

        # Run two workers concurrently with interleaved sleeps so their
        # spans are emitted with the other task active in between.
        coll_a, coll_b = await asyncio.gather(
            worker("a", 0.001),
            worker("b", 0.0),
        )
        a_ids = [ev.span_id for ev in coll_a.events if isinstance(ev, Span)]
        b_ids = [ev.span_id for ev in coll_b.events if isinstance(ev, Span)]
        # Each task captures only its own events.
        assert a_ids == ["a", "a"]
        assert b_ids == ["b", "b"]

    async def test_isolated_context_does_not_see_outer_collector(self) -> None:
        """Code run in a fresh ``contextvars.Context`` doesn't see the
        outer-task collector — push_event is a no-op there."""
        import contextvars

        with collect_events() as outer:
            push_event(_span(phase=SpanPhase.START, span_id="outer-1"))

            def isolated_worker() -> None:
                # Fresh context — no collector active.
                assert active_collector() is None
                push_event(_span(phase=SpanPhase.START, span_id="lost"))

            ctx = contextvars.Context()
            ctx.run(isolated_worker)

        # Only "outer-1" reached the outer collector — "lost" had no
        # collector to push to.
        outer_ids = [ev.span_id for ev in outer.events if isinstance(ev, Span)]
        assert outer_ids == ["outer-1"]
