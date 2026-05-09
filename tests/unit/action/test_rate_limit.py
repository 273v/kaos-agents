"""RateLimiter token-bucket math + glob matching + hook contract."""

from __future__ import annotations

from kaos_agents.action.rate_limit import RateLimiter, _TokenBucket
from kaos_agents.events.spans import Span, SpanPhase, SpanSubject
from kaos_agents.hooks.base import HookAction


def _tool_call_start(tool_name: str) -> Span:
    return Span(
        timestamp=0.0,
        sequence=0,
        session_id="s1",
        run_id="r1",
        subject=SpanSubject.TOOL_CALL,
        phase=SpanPhase.START,
        span_id="span-1",
        name=f"tool.{tool_name}",
        attributes={"tool_name": tool_name, "call_id": "c1"},
    )


class TestTokenBucket:
    def test_capacity_exhaustion(self) -> None:
        bucket = _TokenBucket(capacity=2, refill_per_sec=1.0)
        assert bucket.consume() is True
        assert bucket.consume() is True
        assert bucket.consume() is False

    def test_refill_after_wait(self) -> None:
        bucket = _TokenBucket(capacity=2, refill_per_sec=1.0)
        bucket.consume()
        bucket.consume()
        assert bucket.consume() is False
        # Force-age the bucket without sleeping in tests.
        bucket.last_refill -= 1.1
        assert bucket.consume() is True

    def test_refill_clamped_to_capacity(self) -> None:
        bucket = _TokenBucket(capacity=2, refill_per_sec=10.0)
        bucket.consume()
        bucket.last_refill -= 100.0
        # Even after a long wait, only `capacity` tokens are available.
        assert bucket.consume() is True
        assert bucket.consume() is True
        assert bucket.consume() is False


class TestRateLimiterAllow:
    def test_unlimited_when_unconfigured(self) -> None:
        limiter = RateLimiter()
        for _ in range(50):
            assert limiter.allow("any-tool") is True

    def test_default_rate_capacity_then_refused(self) -> None:
        limiter = RateLimiter(default_rate=(2, 1.0))
        assert limiter.allow("kaos-source-pacer-fetch") is True
        assert limiter.allow("kaos-source-pacer-fetch") is True
        assert limiter.allow("kaos-source-pacer-fetch") is False

    def test_glob_match_overrides_default(self) -> None:
        limiter = RateLimiter(
            rates={"kaos-source-*": (1, 1.0)},
            default_rate=(10, 10.0),
        )
        assert limiter.allow("kaos-source-pacer-fetch") is True
        # Same bucket — same pattern.
        assert limiter.allow("kaos-source-edgar-fetch") is False
        # Non-matching tool falls through to default.
        assert limiter.allow("kaos-pdf-render") is True

    def test_first_glob_wins(self) -> None:
        limiter = RateLimiter(
            rates={
                "kaos-pdf-*": (1, 1.0),
                "kaos-pdf-render": (10, 10.0),
            }
        )
        # Insertion order means "kaos-pdf-*" matches first; the bucket
        # capacity is 1 not 10.
        assert limiter.allow("kaos-pdf-render") is True
        assert limiter.allow("kaos-pdf-render") is False


class TestRateLimiterRefill:
    def test_refill_after_wait_via_internal_clock(self) -> None:
        limiter = RateLimiter(default_rate=(2, 1.0))
        assert limiter.allow("x") is True
        assert limiter.allow("x") is True
        assert limiter.allow("x") is False
        # Drive the bucket's last_refill back by 1.1s without sleeping.
        bucket = limiter._buckets["__default__"]
        bucket.last_refill -= 1.1
        assert limiter.allow("x") is True


class TestRateLimiterHook:
    async def test_hook_continue_when_allowed(self) -> None:
        limiter = RateLimiter(default_rate=(1, 1.0))
        action = await limiter.on_tool_call_start(_tool_call_start("kaos-pdf-render"))
        assert action == HookAction.CONTINUE

    async def test_hook_skip_when_exhausted(self) -> None:
        limiter = RateLimiter(default_rate=(1, 1.0))
        await limiter.on_tool_call_start(_tool_call_start("kaos-pdf-render"))
        action = await limiter.on_tool_call_start(_tool_call_start("kaos-pdf-render"))
        assert action == HookAction.SKIP

    async def test_hook_unknown_tool_name_continue(self) -> None:
        # Span without tool_name attribute → CONTINUE (we can't gate
        # what we can't identify).
        span = Span(
            timestamp=0.0,
            sequence=0,
            session_id="s1",
            run_id="r1",
            subject=SpanSubject.TOOL_CALL,
            phase=SpanPhase.START,
            span_id="span-1",
            attributes={},
        )
        limiter = RateLimiter(default_rate=(0, 0.0))
        action = await limiter.on_tool_call_start(span)
        assert action == HookAction.CONTINUE
