"""CircuitBreaker state machine."""

from __future__ import annotations

import pytest

from kaos_agents.action.circuit import CircuitBreaker, CircuitState
from kaos_agents.events.spans import Span, SpanPhase, SpanSubject
from kaos_agents.hooks.base import HookAction


def _span(
    tool_name: str,
    *,
    phase: SpanPhase,
    is_error: bool = False,
    result_summary: str = "",
) -> Span:
    return Span(
        timestamp=0.0,
        sequence=0,
        session_id="s1",
        run_id="r1",
        subject=SpanSubject.TOOL_CALL,
        phase=phase,
        span_id="span-1",
        name=f"tool.{tool_name}",
        attributes={
            "tool_name": tool_name,
            "call_id": "c1",
            "is_error": is_error,
            "result_summary": result_summary,
        },
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
        # An informative non-error result must reset the counter. Empty
        # result_summary would now be caught by is_uninformative_result
        # (#506) and counted as failure, so pass real content.
        await cb.on_tool_call_result(
            _span(
                "x",
                phase=SpanPhase.COMPLETE,
                is_error=False,
                result_summary='{"data": "ok"}',
            )
        )
        # Counter reset.
        assert cb._states["x"].consecutive_failures == 0

    async def test_on_tool_call_result_records_failure(self) -> None:
        cb = CircuitBreaker(failure_threshold=3)
        await cb.on_tool_call_result(_span("x", phase=SpanPhase.COMPLETE, is_error=True))
        await cb.on_tool_call_result(_span("x", phase=SpanPhase.COMPLETE, is_error=True))
        await cb.on_tool_call_result(_span("x", phase=SpanPhase.COMPLETE, is_error=True))
        assert cb.state_for("x") == CircuitState.OPEN


class TestUninformativeCountsAsFailure:
    """The #506 extension — N consecutive zero-result returns trip the breaker.

    Replays the empirical shape from session
    ``01KS2DEBYT341F1F16B3BRQRV0`` where each ``kaos-web-search`` call
    returned ``is_error=False`` with body
    ``"No results found for: <query>"`` and the loop ran 12 of them
    before exhausting iteration budget.
    """

    async def test_consecutive_no_results_trips_breaker(self) -> None:
        cb = CircuitBreaker(failure_threshold=3, reset_timeout_seconds=30.0)
        for _ in range(3):
            await cb.on_tool_call_result(
                _span(
                    "kaos-web-search",
                    phase=SpanPhase.COMPLETE,
                    is_error=False,
                    result_summary="No results found for: site:federalreserve.gov FOMC 2026",
                )
            )
        assert cb.state_for("kaos-web-search") == CircuitState.OPEN

    async def test_session_deb_replay_trips_before_budget_exhaustion(self) -> None:
        """12 zero-result web searches must trip the breaker well before
        the 12th call — verifying the loop would NOT have spun in session
        DEB had the breaker been wired."""
        cb = CircuitBreaker(failure_threshold=5, reset_timeout_seconds=30.0)
        zero_result_bodies = [
            "No results found for: Federal Reserve federal funds rate",
            "No results found for: site:federalreserve.gov FOMC 2026",
            "No results found for: federal funds rate target range",
            "No results found for: FOMC calendar 2026",
            "No results found for: Federal Reserve Board target range",
            "No results found for: federalreserve.gov monetary policy 2026",
        ]
        for i, body in enumerate(zero_result_bodies):
            await cb.on_tool_call_result(
                _span(
                    "kaos-web-search",
                    phase=SpanPhase.COMPLETE,
                    is_error=False,
                    result_summary=body,
                )
            )
            if i + 1 >= 5:
                assert cb.state_for("kaos-web-search") == CircuitState.OPEN, (
                    f"breaker should have tripped by call {i + 1}"
                )

    async def test_informative_result_resets_consecutive_count(self) -> None:
        """A single informative return must reset the counter."""
        cb = CircuitBreaker(failure_threshold=3, reset_timeout_seconds=30.0)
        await cb.on_tool_call_result(
            _span(
                "kaos-web-search",
                phase=SpanPhase.COMPLETE,
                result_summary="No results found for: q1",
            )
        )
        await cb.on_tool_call_result(
            _span(
                "kaos-web-search",
                phase=SpanPhase.COMPLETE,
                result_summary="No results found for: q2",
            )
        )
        # Informative — counter resets.
        await cb.on_tool_call_result(
            _span(
                "kaos-web-search",
                phase=SpanPhase.COMPLETE,
                result_summary='Found 18 matches for "FOMC 2026" on federalreserve.gov',
            )
        )
        # Two more empties — still under threshold.
        await cb.on_tool_call_result(
            _span(
                "kaos-web-search",
                phase=SpanPhase.COMPLETE,
                result_summary="No results found for: q4",
            )
        )
        await cb.on_tool_call_result(
            _span(
                "kaos-web-search",
                phase=SpanPhase.COMPLETE,
                result_summary="No results found for: q5",
            )
        )
        assert cb.state_for("kaos-web-search") == CircuitState.CLOSED

    async def test_opt_out_via_constructor_keeps_legacy_behavior(self) -> None:
        """With ``uninformative_counts_as_failure=False`` the breaker
        treats zero-result calls as successes (the pre-#506 behavior)."""
        cb = CircuitBreaker(
            failure_threshold=3,
            reset_timeout_seconds=30.0,
            uninformative_counts_as_failure=False,
        )
        for _ in range(10):
            await cb.on_tool_call_result(
                _span(
                    "kaos-web-search",
                    phase=SpanPhase.COMPLETE,
                    result_summary="No results found for: anything",
                )
            )
        assert cb.state_for("kaos-web-search") == CircuitState.CLOSED

    async def test_uninformative_and_error_share_threshold(self) -> None:
        """Errors and uninformative returns both increment the same
        per-tool ``consecutive_failures`` counter — mixing them must
        trip at the threshold."""
        cb = CircuitBreaker(failure_threshold=3, reset_timeout_seconds=30.0)
        await cb.on_tool_call_result(_span("x", phase=SpanPhase.COMPLETE, is_error=True))
        await cb.on_tool_call_result(
            _span(
                "x",
                phase=SpanPhase.COMPLETE,
                is_error=False,
                result_summary="No results found",
            )
        )
        await cb.on_tool_call_result(
            _span(
                "x",
                phase=SpanPhase.COMPLETE,
                is_error=False,
                result_summary='{"total_matches": 0, "results": []}',
            )
        )
        assert cb.state_for("x") == CircuitState.OPEN

    async def test_extra_pattern_extends_predicate(self) -> None:
        """A caller-supplied pattern fires alongside the defaults."""
        import re

        cb = CircuitBreaker(
            failure_threshold=2,
            reset_timeout_seconds=30.0,
            extra_uninformative_patterns=(re.compile(r"\bquota\s+exhausted\b", re.IGNORECASE),),
        )
        # First call: default pattern fires (No results).
        await cb.on_tool_call_result(
            _span("x", phase=SpanPhase.COMPLETE, result_summary="No results found.")
        )
        # Second call: caller's pattern fires.
        await cb.on_tool_call_result(
            _span(
                "x",
                phase=SpanPhase.COMPLETE,
                result_summary="Service degraded — quota exhausted for tenant Y.",
            )
        )
        assert cb.state_for("x") == CircuitState.OPEN
