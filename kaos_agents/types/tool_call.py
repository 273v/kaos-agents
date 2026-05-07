"""Per-turn tool execution record.

:class:`ToolExecution` is the canonical structured record for one
tool invocation during a turn — name, arguments, result, timing,
token + cost usage, and plan/step correlation. It supersedes the
former ``ToolCallRecord`` (shallow record without timing/usage)
and acts as the data-rich source from which the wire-side
:class:`ToolCallSummary` (in :mod:`kaos_agents.events.tools`) is
projected.

The split keeps two consumer paths clean:

- **Internal** (memory.ACTIONS items, audit, graph triples) reads
  :class:`ToolExecution` directly. Has full arguments + result +
  timing.
- **Wire** (``TurnSummary.tool_calls`` event field, SSE/JSONL) ships
  :class:`ToolCallSummary`. Compact (no full arguments / result
  string) for stream consumers that just want telemetry.

Mirrors kelvin-agent's ``ActionExecution`` shape (timing, exception,
duration, plan_id) plus the WS-Phase-5 token + cost fields kaos
introduced via :class:`InvocationUsage`.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from kaos_agents.events.tools import ToolCallSummary


@dataclass(frozen=True, slots=True)
class ToolExecution:
    """Canonical structured record of one tool invocation.

    ``plan_id`` / ``step_id`` correlate the call to a plan-execute
    step. Both default to ``None`` for chat / research / direct-
    respond tool calls. ``started_at`` / ``ended_at`` are
    ``time.monotonic()`` floats — same clock used by the event stream
    for replay-safe ordering. Wall-clock timestamps live on the events
    themselves.
    """

    tool_name: str
    call_id: str = ""
    arguments: tuple[tuple[str, Any], ...] = ()  # Immutable key-value pairs
    result_summary: str = ""
    is_error: bool = False

    # Timing — set by the agent loop when the tool returns.
    started_at: float | None = None  # time.monotonic() at start
    ended_at: float | None = None  # time.monotonic() at end
    duration_ms: float = 0.0  # ended_at - started_at, in milliseconds

    # Per-call LLM usage — non-zero only when the tool itself drove
    # an LLM (e.g. RAG verifier inside rag-query, or a delegated
    # sub-agent). Plain tools stay at zero.
    cost_usd: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0

    # Plan / step correlation — kelvin's ActionExecution.plan_id
    # pattern. None for tool calls outside a plan.
    plan_id: str | None = None
    step_id: str | None = None

    # Free-form metadata — useful for tool-author extensions that
    # don't deserve their own typed field. Default empty.
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict_args(
        cls,
        tool_name: str,
        arguments: dict[str, Any],
        result_summary: str = "",
        *,
        call_id: str = "",
        is_error: bool = False,
        started_at: float | None = None,
        ended_at: float | None = None,
        duration_ms: float = 0.0,
        cost_usd: float = 0.0,
        input_tokens: int = 0,
        output_tokens: int = 0,
        plan_id: str | None = None,
        step_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ToolExecution:
        """Create from a mutable dict (convenience for callers).

        Sorts ``arguments`` for deterministic equality + hashing.
        """
        return cls(
            tool_name=tool_name,
            call_id=call_id,
            arguments=tuple(sorted(arguments.items())),
            result_summary=result_summary,
            is_error=is_error,
            started_at=started_at,
            ended_at=ended_at,
            duration_ms=duration_ms,
            cost_usd=cost_usd,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            plan_id=plan_id,
            step_id=step_id,
            metadata=metadata or {},
        )

    def to_dict(self) -> dict[str, Any]:
        """Plain-Python projection. Used by Memory.ACTIONS metadata
        side-channel so structured tool data is queryable from the
        memory store without re-parsing the rendered string content.
        """
        return {
            "tool_name": self.tool_name,
            "call_id": self.call_id,
            "arguments": [list(pair) for pair in self.arguments],
            "result_summary": self.result_summary,
            "is_error": self.is_error,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "duration_ms": self.duration_ms,
            "cost_usd": self.cost_usd,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "plan_id": self.plan_id,
            "step_id": self.step_id,
            "metadata": dict(self.metadata),
        }

    def to_summary(self) -> ToolCallSummary:
        """Wire-side projection — :class:`ToolCallSummary` for
        :class:`TurnSummary.tool_calls`. Drops arguments + result
        text + started_at/ended_at; keeps the per-call telemetry
        downstream consumers actually use."""
        # Local import to dodge tool_call <-> events.tools cycle on
        # module load. Both modules can be imported lazily.
        from kaos_agents.events.tools import ToolCallSummary

        return ToolCallSummary(
            tool_name=self.tool_name,
            call_id=self.call_id,
            is_error=self.is_error,
            duration_ms=self.duration_ms,
            cost_usd=self.cost_usd,
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
            plan_id=self.plan_id,
            step_id=self.step_id,
        )

    @classmethod
    def begin(
        cls,
        tool_name: str,
        *,
        call_id: str = "",
        arguments: dict[str, Any] | None = None,
        plan_id: str | None = None,
        step_id: str | None = None,
    ) -> ToolExecution:
        """Construct a "started" execution record with ``started_at``
        set to ``time.monotonic()``. Caller fills in the rest at end."""
        return cls.from_dict_args(
            tool_name=tool_name,
            call_id=call_id,
            arguments=arguments or {},
            started_at=time.monotonic(),
            plan_id=plan_id,
            step_id=step_id,
        )

    def with_completion(
        self,
        result_summary: str,
        *,
        is_error: bool = False,
        cost_usd: float = 0.0,
        input_tokens: int = 0,
        output_tokens: int = 0,
        ended_at: float | None = None,
    ) -> ToolExecution:
        """Return a new execution with completion fields filled in.

        ``ended_at`` defaults to ``time.monotonic()``; ``duration_ms``
        is computed from ``started_at`` if both are available.
        """
        end = ended_at if ended_at is not None else time.monotonic()
        if self.started_at is not None:
            duration_ms = (end - self.started_at) * 1000.0
        else:
            duration_ms = self.duration_ms

        return ToolExecution(
            tool_name=self.tool_name,
            call_id=self.call_id,
            arguments=self.arguments,
            result_summary=result_summary,
            is_error=is_error,
            started_at=self.started_at,
            ended_at=end,
            duration_ms=duration_ms,
            cost_usd=cost_usd,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            plan_id=self.plan_id,
            step_id=self.step_id,
            metadata=dict(self.metadata),
        )


# Back-compat alias — chunk A2 deletes ToolCallRecord but keeps the
# old name importable for one transition release. Tests + downstream
# code should migrate to ``ToolExecution`` directly. The alias is
# removed in Track 3 chunk C (closeout sweep).
ToolCallRecord = ToolExecution


__all__ = ["ToolCallRecord", "ToolExecution"]
