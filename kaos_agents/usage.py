"""Per-turn token/cost usage aggregation.

The agent's ``TurnComplete`` event used to ship ``tokens_used=0`` and
``cost_usd=0.0`` as hard-coded placeholders (see the pre-Phase-5.0
``# TODO: aggregate from sub-events`` in ``agent.py``). Meanwhile every
``Call``/``Program`` in kaos-llm-core already returned an ``Invocation``
with real provider-reported usage on ``invocation.usage``. This module
is the one-layer-up plumbing that carries those numbers from the LLM
layer to the agent's turn boundary — no new accounting logic, just a
tight value type consumers can accumulate.

``InvocationUsage`` mirrors ``kaos_llm_core.programs._invocation.TokenUsage``
field-for-field. We duplicate the type rather than importing it so
kaos-agents keeps its clean-separation invariant: the LLM layer is
still optional (``[llm]`` extra) and the turn loop can aggregate usage
without forcing ``kaos_llm_core`` into the import graph for agents
that never make an LLM call.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class InvocationUsage:
    """Per-invocation token + cost summary.

    Frozen so callers can safely cache or pass across awaits. Addition
    is pointwise so accumulating across multiple LLM calls in a single
    turn is one expression: ``total = sum(usages, InvocationUsage.ZERO)``
    or ``a + b``.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0

    def __add__(self, other: InvocationUsage) -> InvocationUsage:
        return InvocationUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
            cost_usd=self.cost_usd + other.cost_usd,
        )

    @classmethod
    def from_llm_usage(cls, usage: Any) -> InvocationUsage:
        """Build from a ``kaos_llm_core.programs._invocation.TokenUsage``.

        Accepts any duck-typed object exposing ``input_tokens`` /
        ``output_tokens`` / ``total_tokens`` / ``cost_usd`` attributes,
        so tests can pass a simple ``SimpleNamespace`` without pulling
        in the llm-core dependency.
        """
        return cls(
            input_tokens=int(getattr(usage, "input_tokens", 0) or 0),
            output_tokens=int(getattr(usage, "output_tokens", 0) or 0),
            total_tokens=int(getattr(usage, "total_tokens", 0) or 0),
            cost_usd=float(getattr(usage, "cost_usd", 0.0) or 0.0),
        )

    @classmethod
    def from_invocation(cls, invocation: Any) -> InvocationUsage:
        """Build from a kaos-llm-core ``Invocation`` (reads ``.usage``)."""
        return cls.from_llm_usage(getattr(invocation, "usage", None) or ZERO_USAGE)


# Identity for addition; ``sum(usages, ZERO_USAGE)`` is the canonical
# accumulation idiom (single allocation, no None checks).
ZERO_USAGE = InvocationUsage()
