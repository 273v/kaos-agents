"""Intent classification types — what the user wants the agent to do."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum, unique
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from kaos_agents.types.usage import InvocationUsage


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
