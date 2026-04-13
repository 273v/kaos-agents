"""Agent response and intent models — pure data types."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum, unique
from typing import Any


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
    """Result of intent classification."""

    intent: IntentType
    confidence: float  # 0.0 - 1.0
    reasoning: str  # Why this intent was chosen


@dataclass(frozen=True, slots=True)
class ToolCallRecord:
    """Record of a tool call made during a turn."""

    tool_name: str
    arguments: dict[str, Any]
    result_summary: str
    is_error: bool = False


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
    metadata: dict[str, Any] = field(default_factory=dict)
