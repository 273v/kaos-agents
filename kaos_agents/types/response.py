"""The single-turn agent response."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from kaos_agents.types.intents import IntentResult
from kaos_agents.types.tool_call import ToolCallRecord


@dataclass(frozen=True, slots=True)
class AgentResponse:
    """The result of a single agent turn.

    Returned by ``BaseAgent.turn()`` — contains the response text,
    any artifacts produced, and metadata about what happened.
    """

    text: str
    intent: IntentResult
    tool_calls: tuple[ToolCallRecord, ...] = ()
    artifacts: tuple[str, ...] = ()  # Artifact URIs produced
    turn_number: int = 0
    tokens_used: int = 0
    # metadata is a tuple of key-value pairs for true immutability
    metadata: tuple[tuple[str, Any], ...] = ()

    @classmethod
    def create(
        cls,
        text: str,
        intent: IntentResult,
        *,
        tool_calls: tuple[ToolCallRecord, ...] = (),
        artifacts: tuple[str, ...] = (),
        turn_number: int = 0,
        tokens_used: int = 0,
        metadata: dict[str, Any] | None = None,
    ) -> AgentResponse:
        """Create with dict metadata (convenience for callers)."""
        meta = tuple(sorted(metadata.items())) if metadata else ()
        return cls(
            text=text,
            intent=intent,
            tool_calls=tool_calls,
            artifacts=artifacts,
            turn_number=turn_number,
            tokens_used=tokens_used,
            metadata=meta,
        )
