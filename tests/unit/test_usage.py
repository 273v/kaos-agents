"""Tests for ``InvocationUsage`` + ``UsageObserved`` event.

Phase 5.0 primitives. These pin down:
- The value type's algebra (identity, associative addition).
- The llm-core duck-typed bridge (works without importing kaos_llm_core).
- Event serialization round-trip through the wire registry.
"""

from __future__ import annotations

from types import SimpleNamespace

from kaos_agents.events import ALL_EVENT_TYPES, UsageObserved
from kaos_agents.usage import ZERO_USAGE, InvocationUsage


class TestInvocationUsageAlgebra:
    def test_zero_is_identity(self) -> None:
        u = InvocationUsage(input_tokens=7, output_tokens=11, total_tokens=18, cost_usd=0.01)
        assert (u + ZERO_USAGE) == u
        assert (ZERO_USAGE + u) == u

    def test_addition_is_pointwise(self) -> None:
        a = InvocationUsage(input_tokens=1, output_tokens=2, total_tokens=3, cost_usd=0.10)
        b = InvocationUsage(input_tokens=5, output_tokens=7, total_tokens=12, cost_usd=0.25)
        c = a + b
        assert c == InvocationUsage(input_tokens=6, output_tokens=9, total_tokens=15, cost_usd=0.35)

    def test_sum_accumulates_many(self) -> None:
        items = [
            InvocationUsage(input_tokens=i, output_tokens=2 * i, total_tokens=3 * i, cost_usd=0.0)
            for i in range(1, 6)
        ]
        total = sum(items, ZERO_USAGE)
        assert total.input_tokens == 1 + 2 + 3 + 4 + 5
        assert total.output_tokens == 2 * (1 + 2 + 3 + 4 + 5)
        assert total.total_tokens == 3 * (1 + 2 + 3 + 4 + 5)


class TestFromLLMUsage:
    """The duck-typed factory accepts a SimpleNamespace shaped like
    ``kaos_llm_core.TokenUsage``."""

    def test_from_simple_namespace(self) -> None:
        raw = SimpleNamespace(input_tokens=100, output_tokens=50, total_tokens=150, cost_usd=0.042)
        u = InvocationUsage.from_llm_usage(raw)
        assert u == InvocationUsage(
            input_tokens=100, output_tokens=50, total_tokens=150, cost_usd=0.042
        )

    def test_from_invocation_reads_usage_attr(self) -> None:
        raw = SimpleNamespace(
            usage=SimpleNamespace(input_tokens=3, output_tokens=4, total_tokens=7, cost_usd=0.001)
        )
        u = InvocationUsage.from_invocation(raw)
        assert u.total_tokens == 7

    def test_partial_fields_default_to_zero(self) -> None:
        """Providers that don't report cost still produce a valid usage."""
        raw = SimpleNamespace(input_tokens=5, output_tokens=2, total_tokens=7)
        u = InvocationUsage.from_llm_usage(raw)
        assert u.cost_usd == 0.0

    def test_none_fields_coerce_to_zero(self) -> None:
        """A provider reporting ``input_tokens=None`` (happens on retries)
        doesn't crash the factory."""
        raw = SimpleNamespace(input_tokens=None, output_tokens=10, total_tokens=10, cost_usd=None)
        u = InvocationUsage.from_llm_usage(raw)
        assert u.input_tokens == 0
        assert u.output_tokens == 10
        assert u.cost_usd == 0.0


class TestUsageObservedEvent:
    def test_event_registered(self) -> None:
        """UsageObserved must appear in ALL_EVENT_TYPES so wire
        serializers handle it automatically."""
        assert UsageObserved in ALL_EVENT_TYPES

    def test_event_defaults(self) -> None:
        ev = UsageObserved(timestamp=1.0, sequence=0, session_id="s", run_id="r")
        assert ev.total_tokens == 0
        assert ev.cost_usd == 0.0
        assert ev.source == ""

    def test_event_carries_full_usage(self) -> None:
        ev = UsageObserved(
            timestamp=1.0,
            sequence=0,
            session_id="s",
            run_id="r",
            input_tokens=100,
            output_tokens=50,
            total_tokens=150,
            cost_usd=0.042,
            source="react",
        )
        assert ev.input_tokens + ev.output_tokens == ev.total_tokens
        # Round-trip through our value type.
        usage = InvocationUsage.from_llm_usage(ev)
        assert usage.total_tokens == 150
