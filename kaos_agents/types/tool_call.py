"""Per-turn tool call record."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class ToolCallRecord:
    """Record of a tool call made during a turn."""

    tool_name: str
    arguments: tuple[tuple[str, Any], ...] = ()  # Immutable key-value pairs
    result_summary: str = ""
    is_error: bool = False

    @classmethod
    def from_dict_args(
        cls,
        tool_name: str,
        arguments: dict[str, Any],
        result_summary: str = "",
        *,
        is_error: bool = False,
    ) -> ToolCallRecord:
        """Create from a mutable dict (convenience for callers)."""
        return cls(
            tool_name=tool_name,
            arguments=tuple(sorted(arguments.items())),
            result_summary=result_summary,
            is_error=is_error,
        )
