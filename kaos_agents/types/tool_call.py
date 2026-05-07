"""Per-turn tool call record."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class ToolCallRecord:
    """Record of a tool call made during a turn.

    ``plan_id`` and ``step_id`` correlate the call to a plan-execute
    step when the call fired inside a multi-step plan. Both default to
    ``None`` for chat / research / direct-respond tool calls. This is
    the kelvin-agent ``ActionExecution.plan_id`` pattern — explicit
    causality rather than parsing event timestamps.
    """

    tool_name: str
    arguments: tuple[tuple[str, Any], ...] = ()  # Immutable key-value pairs
    result_summary: str = ""
    is_error: bool = False
    plan_id: str | None = None
    step_id: str | None = None

    @classmethod
    def from_dict_args(
        cls,
        tool_name: str,
        arguments: dict[str, Any],
        result_summary: str = "",
        *,
        is_error: bool = False,
        plan_id: str | None = None,
        step_id: str | None = None,
    ) -> ToolCallRecord:
        """Create from a mutable dict (convenience for callers)."""
        return cls(
            tool_name=tool_name,
            arguments=tuple(sorted(arguments.items())),
            result_summary=result_summary,
            is_error=is_error,
            plan_id=plan_id,
            step_id=step_id,
        )
