"""CircuitBreaker state machine."""

from __future__ import annotations

import pytest

from kaos_agents.action.circuit import CircuitBreaker, CircuitState
from kaos_agents.events.spans import Span, SpanPhase, SpanSubject
from kaos_agents.hooks.base import HookAction


def _span(tool_name: str, *, phase: SpanPhase, is_error: bool = False) -> Span:
    return Span(
        timestamp=0.0,
        sequence=0,
        session_id="s1",
        run_id="r1",
        subject=SpanSubject.TOOL_CALL,
        phase=phase,
        span_id="span-1",
        name=f"tool.{tool_name}",
        attributes={"tool_name": tool_name, "call_id": "c1", "is_error": is_error},
    )


class TestConstructorValidation:
    def test_failure_threshold_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="failure_threshold"):
            CircuitBreaker(failure_threshold=0)

    def test_reset_timeout_non_negative(self) -> None:
        with pytest.raises(ValueError, match="reset_timeout_seconds"):
            CircuitBreaker(reset_timeout_seconds=-1.0)


class TestStateMachine:
    def test_starts_closed(self) -> None:
        cb = CircuitBreaker(failure_threshold=2, reset_timeout_seconds=30.0)
        assert cb.state_for("x") == CircuitState.CLOSED
        assert cb.allow("x") is True

    def test_opens_after_threshold_failures(self) -> None:
        cb = CircuitBreaker(failure_threshold=3, reset_timeout_seconds=30.0)
        cb.record_failure("x")
        cb.record_failure("x")
        assert cb.state_for("x") == CircuitState.CLOSED
        cb.record_failure("x")
        assert cb.state_for("x") == CircuitState.OPEN
        assert cb.allow("x") is False

    def test_success_resets_consecutive_failures(self) -> None:
        cb = CircuitBreaker(failure_threshold=3, reset_timeout_seconds=30.0)
        cb.record_failure("x")
        cb.record_failure("x")
        cb.record_success("x")
        cb.record_failure("x")
        cb.record_failure("x")
        # Two failures since the success — still under threshold.
        assert cb.state_for("x") == CircuitState.CLOSED

    def test_open_to_half_open_after_timeout(self) -> None:
        cb = CircuitBreaker(failure_threshold=1, reset_timeout_seconds=10.0)
        cb.record_failure("x")
        assert cb.state_for("x") == CircuitState.OPEN
        # Force-age the breaker.
        cb._states["x"].opened_at -= 11.0
        assert cb.state_for("x") == CircuitState.HALF_OPEN

    def test_half_open_allows_one_probe(self) -> None:
        cb = CircuitBreaker(failure_threshold=1, reset_timeout_seconds=10.0)
        cb.record_failure("x")
        cb._states["x"].opened_at -= 11.0
        # Now half-open after timeout elapsed.
        assert cb.state_for("x") == CircuitState.HALF_OPEN
        assert cb.allow("x") is True
        # Second probe refused while the first is in flight.
        assert cb.allow("x") is False

    def test_half_open_success_closes(self) -> None:
        cb = CircuitBreaker(failure_threshold=1, reset_timeout_seconds=10.0)
        cb.record_failure("x")
        cb._states["x"].opened_at -= 11.0
        assert cb.state_for("x") == CircuitState.HALF_OPEN
        assert cb.allow("x") is True  # probe
        cb.record_success("x")
        assert cb.state_for("x") == CircuitState.CLOSED
        assert cb.allow("x") is True

    def test_half_open_failure_reopens(self) -> None:
        cb = CircuitBreaker(failure_threshold=1, reset_timeout_seconds=10.0)
        cb.record_failure("x")
        # Force-age the breaker so state_for transitions OPEN -> HALF_OPEN.
        cb._states["x"].opened_at -= 11.0
        assert cb.state_for("x") == CircuitState.HALF_OPEN
        assert cb.allow("x") is True  # probe
        # Probe failure → re-opens with a fresh opened_at.
        cb.record_failure("x")
        assert cb._states["x"].state == CircuitState.OPEN
        # state_for() will not auto-roll back to HALF_OPEN until
        # reset_timeout elapses again.
        assert cb.state_for("x") == CircuitState.OPEN


class TestHookIntegration:
    async def test_on_tool_call_start_continue_when_closed(self) -> None:
        cb = CircuitBreaker(failure_threshold=2)
        action = await cb.on_tool_call_start(_span("x", phase=SpanPhase.START))
        assert action == HookAction.CONTINUE

    async def test_on_tool_call_start_skip_when_open(self) -> None:
        cb = CircuitBreaker(failure_threshold=1, reset_timeout_seconds=30.0)
        cb.record_failure("x")
        action = await cb.on_tool_call_start(_span("x", phase=SpanPhase.START))
        assert action == HookAction.SKIP

    async def test_on_tool_call_result_records_success(self) -> None:
        cb = CircuitBreaker(failure_threshold=2)
        cb.record_failure("x")
        await cb.on_tool_call_result(_span("x", phase=SpanPhase.COMPLETE, is_error=False))
        # Counter reset.
        assert cb._states["x"].consecutive_failures == 0

    async def test_on_tool_call_result_records_failure(self) -> None:
        cb = CircuitBreaker(failure_threshold=3)
        await cb.on_tool_call_result(_span("x", phase=SpanPhase.COMPLETE, is_error=True))
        await cb.on_tool_call_result(_span("x", phase=SpanPhase.COMPLETE, is_error=True))
        await cb.on_tool_call_result(_span("x", phase=SpanPhase.COMPLETE, is_error=True))
        assert cb.state_for("x") == CircuitState.OPEN
