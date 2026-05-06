"""Agent response and intent models — pure data types."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum, unique
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from kaos_agents.usage import InvocationUsage


@unique
class IntentType(StrEnum):
    """Classification of user intent for dispatch routing."""

    RESPOND = "respond"  # Simple conversational response (no tools)
    TOOL_USE = "tool_use"  # Needs tool calling via ReAct
    RESEARCH = "research"  # Document Q&A via RAG pipeline
    PLAN = "plan"  # Multi-step plan needed
    CLARIFY = "clarify"  # Need more information from user


@dataclass(frozen=True, slots=True)
class IntentResult:
    """Result of intent classification.

    Carries the intent enum, confidence, reasoning text, and (when the
    classifier was an LLM) the typed ``InvocationUsage`` so the caller
    can plumb it to ``TurnComplete.usage`` / the session cost ceiling.
    ``usage`` is ``None`` for the heuristic-fallback path that never
    invoked an LLM.
    """

    intent: IntentType
    confidence: float  # 0.0 to 1.0
    reasoning: str  # Why this intent was chosen
    usage: InvocationUsage | None = None  # Per-classification token + cost

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            # Use object.__setattr__ because frozen
            object.__setattr__(self, "confidence", max(0.0, min(1.0, self.confidence)))


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


@dataclass(frozen=True, slots=True)
class AgentResponse:
    """The result of a single agent turn.

    Returned by BaseAgent.turn() — contains the response text,
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
