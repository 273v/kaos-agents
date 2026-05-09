"""Tests for kaos_agents.hooks.adapter — KaosHook → kaos-llm-core bridge.

Phase 0.C: the adapter wraps a tuple of :class:`KaosHook` instances into
a :class:`CallHooks` (per-Call boundary) and :class:`ProgramHooks`
(per-Program boundary). When kaos-llm-core fires a hook lifecycle event,
the adapter synthesizes a :class:`Span` and dispatches it through
:func:`dispatch_hook` so every registered :class:`KaosHook` observes
both the outer agent stream and the inner kaos-llm-core layer.

What we cover here:

* ``adapt_hooks(())`` returns empty ``CallHooks`` / ``ProgramHooks``
  (all four slots ``None``) — preserves the kaos-llm-core fast path.
* Each kaos-llm-core lifecycle hook (``on_call_start`` /
  ``on_call_end`` / ``on_call_error`` / ``on_validation_retry`` plus
  the four ``on_program_*`` slots) routes through the adapter and
  reaches every :class:`KaosHook` in the tuple.
* Errors raised by a :class:`KaosHook` callback do not break the
  forwarder (kaos-llm-core's ``fire_hook`` already swallows; the
  adapter must be transparent to that).
* The synthesized span's ``session_id`` / ``run_id`` flow through from
  the ``context=`` kwarg when supplied.

We intentionally do not assert exact span_id values or timestamps;
the contract is "a Span event with the right subject/phase reaches
every KaosHook in the tuple".
"""

from __future__ import annotations

import asyncio

import pytest
from kaos_llm_core.programs.hooks import CallHooks, fire_hook
from kaos_llm_core.programs.program_hooks import ProgramHooks, fire_program_hook

from kaos_agents.events import Span, SpanPhase, SpanSubject
from kaos_agents.hooks.adapter import adapt_hooks
from kaos_agents.hooks.base import KaosHook

# ----- Test doubles ------------------------------------------------------


class _RecordingHook(KaosHook):
    """KaosHook that records every event passed to ``on_event``.

    ``on_event`` is the catch-all dispatched by :func:`dispatch_hook`
    before the typed-method routing; it sees every span the adapter
    forwards regardless of subject or phase, which makes assertions
    simple and decoupled from dispatch_hook's internal routing table.
    """

    def __init__(self) -> None:
        self.events: list[object] = []

    async def on_event(self, event):  # type: ignore[override]
        self.events.append(event)


class _RaisingHook(KaosHook):
    """KaosHook that raises in ``on_event`` to exercise exception paths."""

    def __init__(self) -> None:
        self.fired = False

    async def on_event(self, event):  # type: ignore[override]
        self.fired = True
        msg = "boom from KaosHook"
        raise RuntimeError(msg)


class _StubCall:
    """Stand-in for kaos-llm-core ``Call`` — only its identity matters."""


class _StubProgram:
    """Stand-in for kaos-llm-core ``Program``."""


class _StubInvocation:
    """Stand-in for kaos-llm-core ``Invocation`` — only ``usage`` is read."""

    def __init__(self, usage=None) -> None:
        self.usage = usage


class _StubUsage:
    """Stand-in for ``InvocationUsage`` — adapter reads four named fields."""

    def __init__(
        self,
        input_tokens: int = 10,
        output_tokens: int = 20,
        total_tokens: int = 30,
        cost_usd: float = 0.001,
    ) -> None:
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.total_tokens = total_tokens
        self.cost_usd = cost_usd


class _StubContext:
    """Stand-in for ``KaosContext`` — adapter reads ``session_id`` / ``run_id``."""

    def __init__(self, session_id: str = "sess-123", run_id: str = "run-456") -> None:
        self.session_id = session_id
        self.run_id = run_id


async def _drain() -> None:
    """Yield to the event loop so fire-and-forget dispatch tasks run.

    The adapter schedules ``dispatch_hook`` via ``loop.create_task``;
    that task is queued but not yet executed when ``fire_hook`` returns.
    A single ``await asyncio.sleep(0)`` is not always sufficient because
    the dispatch coroutine has its own internal awaits (logging, etc.),
    so we drain a few times.
    """
    for _ in range(5):
        await asyncio.sleep(0)


# ----- adapt_hooks empty-tuple short-circuit -----------------------------


@pytest.mark.unit
class TestAdaptHooksEmpty:
    def test_empty_returns_empty_call_hooks(self) -> None:
        call_hooks, _program_hooks = adapt_hooks(())
        assert isinstance(call_hooks, CallHooks)
        assert call_hooks.on_call_start is None
        assert call_hooks.on_call_end is None
        assert call_hooks.on_call_error is None
        assert call_hooks.on_validation_retry is None

    def test_empty_returns_empty_program_hooks(self) -> None:
        _call_hooks, program_hooks = adapt_hooks(())
        assert isinstance(program_hooks, ProgramHooks)
        assert program_hooks.on_program_start is None
        assert program_hooks.on_iteration is None
        assert program_hooks.on_program_end is None
        assert program_hooks.on_program_error is None

    def test_non_empty_populates_all_slots(self) -> None:
        call_hooks, program_hooks = adapt_hooks((_RecordingHook(),))
        assert call_hooks.on_call_start is not None
        assert call_hooks.on_call_end is not None
        assert call_hooks.on_call_error is not None
        assert call_hooks.on_validation_retry is not None
        assert program_hooks.on_program_start is not None
        assert program_hooks.on_iteration is not None
        assert program_hooks.on_program_end is not None
        assert program_hooks.on_program_error is not None


# ----- CallHooks forwarding ---------------------------------------------


@pytest.mark.unit
class TestCallHooksForwarding:
    async def test_on_call_start_reaches_kaos_hook(self) -> None:
        hook = _RecordingHook()
        call_hooks, _ = adapt_hooks((hook,))
        call = _StubCall()

        # Use fire_hook the way kaos-llm-core's Call._execute does.
        fire_hook(call_hooks.on_call_start, call, {"x": 1}, context=_StubContext())
        await _drain()

        assert len(hook.events) == 1
        event = hook.events[0]
        assert isinstance(event, Span)
        assert event.subject == SpanSubject.LLM_CALL
        assert event.phase == SpanPhase.START
        assert event.session_id == "sess-123"
        assert event.run_id == "run-456"
        # Inputs are recorded via repr() in the attributes bag.
        assert "call_type" in event.attributes
        assert event.attributes["call_type"] == "_StubCall"

    async def test_on_call_end_reaches_kaos_hook_with_usage(self) -> None:
        hook = _RecordingHook()
        call_hooks, _ = adapt_hooks((hook,))
        call = _StubCall()
        invocation = _StubInvocation(
            usage=_StubUsage(input_tokens=5, output_tokens=7, total_tokens=12, cost_usd=0.0042)
        )

        # Realistic flow: start then end share the same span_id.
        fire_hook(call_hooks.on_call_start, call, {"y": 2})
        fire_hook(call_hooks.on_call_end, call, {"y": 2}, invocation)
        await _drain()

        assert len(hook.events) == 2
        end_event = hook.events[1]
        assert isinstance(end_event, Span)
        assert end_event.subject == SpanSubject.LLM_CALL
        assert end_event.phase == SpanPhase.COMPLETE
        # Usage fields surface in the attributes bag.
        assert end_event.attributes.get("input_tokens") == 5
        assert end_event.attributes.get("output_tokens") == 7
        assert end_event.attributes.get("total_tokens") == 12
        assert end_event.attributes.get("cost_usd") == pytest.approx(0.0042)
        # Start and end share the same span_id (correlation).
        start_event = hook.events[0]
        assert isinstance(start_event, Span)
        assert start_event.span_id == end_event.span_id

    async def test_on_call_error_reaches_kaos_hook_with_error_fields(self) -> None:
        hook = _RecordingHook()
        call_hooks, _ = adapt_hooks((hook,))
        call = _StubCall()
        exc = ValueError("upstream provider timeout")

        fire_hook(call_hooks.on_call_start, call, {})
        fire_hook(call_hooks.on_call_error, call, {}, exc)
        await _drain()

        assert len(hook.events) == 2
        error_event = hook.events[1]
        assert isinstance(error_event, Span)
        assert error_event.subject == SpanSubject.LLM_CALL
        assert error_event.phase == SpanPhase.ERROR
        assert error_event.error_type == "ValueError"
        assert error_event.error_message == "upstream provider timeout"

    async def test_on_validation_retry_reaches_kaos_hook(self) -> None:
        hook = _RecordingHook()
        call_hooks, _ = adapt_hooks((hook,))
        call = _StubCall()

        fire_hook(call_hooks.on_call_start, call, {"q": 0})
        fire_hook(
            call_hooks.on_validation_retry,
            call,
            {"q": 0},
            2,
            ValueError("decode failed"),
        )
        await _drain()

        assert len(hook.events) == 2
        retry_event = hook.events[1]
        assert isinstance(retry_event, Span)
        assert retry_event.subject == SpanSubject.LLM_CALL
        assert retry_event.phase == SpanPhase.PROGRESS
        assert retry_event.attributes.get("retry_attempt") == 2

    async def test_multiple_kaos_hooks_all_receive_event(self) -> None:
        h1 = _RecordingHook()
        h2 = _RecordingHook()
        call_hooks, _ = adapt_hooks((h1, h2))

        fire_hook(call_hooks.on_call_start, _StubCall(), {})
        await _drain()

        assert len(h1.events) == 1
        assert len(h2.events) == 1


# ----- ProgramHooks forwarding -----------------------------------------


@pytest.mark.unit
class TestProgramHooksForwarding:
    async def test_on_program_start_reaches_kaos_hook(self) -> None:
        hook = _RecordingHook()
        _, program_hooks = adapt_hooks((hook,))
        program = _StubProgram()

        fire_program_hook(
            program_hooks.on_program_start,
            program,
            {"goal": "test"},
            context=_StubContext(session_id="sx", run_id="rx"),
        )
        await _drain()

        assert len(hook.events) == 1
        event = hook.events[0]
        assert isinstance(event, Span)
        assert event.subject == SpanSubject.STEP
        assert event.phase == SpanPhase.START
        assert event.session_id == "sx"
        assert event.run_id == "rx"
        assert event.attributes.get("program_type") == "_StubProgram"

    async def test_on_iteration_reaches_kaos_hook(self) -> None:
        hook = _RecordingHook()
        _, program_hooks = adapt_hooks((hook,))
        program = _StubProgram()

        fire_program_hook(program_hooks.on_program_start, program, {})
        fire_program_hook(
            program_hooks.on_iteration,
            program,
            3,
            {"score": 0.9, "tool_call": "kaos-web-fetch"},
        )
        await _drain()

        assert len(hook.events) == 2
        iter_event = hook.events[1]
        assert isinstance(iter_event, Span)
        assert iter_event.subject == SpanSubject.STEP
        assert iter_event.phase == SpanPhase.PROGRESS
        assert iter_event.attributes.get("iteration") == 3

    async def test_on_program_end_reaches_kaos_hook_with_usage(self) -> None:
        hook = _RecordingHook()
        _, program_hooks = adapt_hooks((hook,))
        program = _StubProgram()
        invocation = _StubInvocation(usage=_StubUsage(total_tokens=42, cost_usd=0.05))

        fire_program_hook(program_hooks.on_program_start, program, {})
        fire_program_hook(program_hooks.on_program_end, program, {}, invocation)
        await _drain()

        assert len(hook.events) == 2
        end_event = hook.events[1]
        assert isinstance(end_event, Span)
        assert end_event.phase == SpanPhase.COMPLETE
        assert end_event.attributes.get("total_tokens") == 42
        assert end_event.attributes.get("cost_usd") == pytest.approx(0.05)

    async def test_on_program_error_reaches_kaos_hook(self) -> None:
        hook = _RecordingHook()
        _, program_hooks = adapt_hooks((hook,))
        program = _StubProgram()

        fire_program_hook(program_hooks.on_program_start, program, {})
        fire_program_hook(
            program_hooks.on_program_error,
            program,
            {},
            RuntimeError("budget exceeded"),
        )
        await _drain()

        assert len(hook.events) == 2
        error_event = hook.events[1]
        assert isinstance(error_event, Span)
        assert error_event.phase == SpanPhase.ERROR
        assert error_event.error_type == "RuntimeError"
        assert error_event.error_message == "budget exceeded"


# ----- Exception transparency -------------------------------------------


@pytest.mark.unit
class TestExceptionTransparency:
    async def test_raising_kaos_hook_does_not_break_call_forwarder(self) -> None:
        """A raising KaosHook must not break the adapter or kaos-llm-core.

        ``dispatch_hook`` already swallows exceptions from individual
        :class:`KaosHook` callbacks. ``fire_hook`` in kaos-llm-core also
        swallows exceptions in the adapter callback itself. So the
        composed pipeline must be transparent to a raising user hook.
        """
        bad = _RaisingHook()
        good = _RecordingHook()
        # Ordering: bad first, then good — make sure the good hook still
        # observes the event after the bad one explodes.
        call_hooks, _ = adapt_hooks((bad, good))

        # If anything propagates up, this would re-raise from fire_hook.
        fire_hook(call_hooks.on_call_start, _StubCall(), {})
        await _drain()

        assert bad.fired is True
        assert len(good.events) == 1

    async def test_raising_kaos_hook_does_not_break_program_forwarder(self) -> None:
        bad = _RaisingHook()
        good = _RecordingHook()
        _, program_hooks = adapt_hooks((bad, good))

        fire_program_hook(program_hooks.on_program_start, _StubProgram(), {})
        await _drain()

        assert bad.fired is True
        assert len(good.events) == 1


# ----- Context propagation ---------------------------------------------


@pytest.mark.unit
class TestContextPropagation:
    async def test_context_session_run_ids_flow_into_span(self) -> None:
        hook = _RecordingHook()
        call_hooks, program_hooks = adapt_hooks((hook,))

        fire_hook(
            call_hooks.on_call_start,
            _StubCall(),
            {},
            context=_StubContext(session_id="alpha", run_id="beta"),
        )
        fire_program_hook(
            program_hooks.on_program_start,
            _StubProgram(),
            {},
            context=_StubContext(session_id="gamma", run_id="delta"),
        )
        await _drain()

        assert len(hook.events) == 2
        call_event = hook.events[0]
        program_event = hook.events[1]
        assert isinstance(call_event, Span)
        assert isinstance(program_event, Span)
        assert call_event.session_id == "alpha"
        assert call_event.run_id == "beta"
        assert program_event.session_id == "gamma"
        assert program_event.run_id == "delta"

    async def test_none_context_yields_empty_ids(self) -> None:
        hook = _RecordingHook()
        call_hooks, _ = adapt_hooks((hook,))

        fire_hook(call_hooks.on_call_start, _StubCall(), {}, context=None)
        await _drain()

        assert len(hook.events) == 1
        event = hook.events[0]
        assert isinstance(event, Span)
        assert event.session_id == ""
        assert event.run_id == ""
