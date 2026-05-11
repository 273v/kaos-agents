"""The single-turn agent response."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from kaos_agents.types.intents import IntentResult
from kaos_agents.types.tool_call import ToolExecution


@dataclass(frozen=True, slots=True)
class AgentResponse:
    """The result of a single agent turn.

    Returned by ``BaseAgent.turn()`` — contains the response text,
    any artifacts produced, and metadata about what happened.

    Sprint-3 #10 (transparency lens): ``cost_usd`` and ``total_tokens``
    are first-class attributes so consumers can read them with one
    attribute access — no event-stream draining required. They mirror
    the values aggregated from ``TurnSummary.cost_usd`` /
    ``TurnSummary.tokens_used`` (which themselves aggregate every
    ``UsageObserved`` event for the turn). For backward compatibility,
    the same numbers are ALSO written into ``metadata`` under the
    ``cost_usd`` / ``total_tokens`` keys, so any caller already walking
    the metadata tuple keeps working unchanged. The headline attribute
    is the source of truth; the metadata mirror is a convenience.
    """

    text: str
    intent: IntentResult
    tool_calls: tuple[ToolExecution, ...] = ()
    artifacts: tuple[str, ...] = ()  # Artifact URIs produced
    turn_number: int = 0
    tokens_used: int = 0
    # metadata is a tuple of key-value pairs for true immutability
    metadata: tuple[tuple[str, Any], ...] = ()
    # Sprint-3 #10 — transparency lens. First-class headline fields
    # so ``response.cost_usd`` / ``response.total_tokens`` work without
    # draining the event stream.
    cost_usd: float = 0.0
    total_tokens: int = 0

    @classmethod
    def create(
        cls,
        text: str,
        intent: IntentResult,
        *,
        tool_calls: tuple[ToolExecution, ...] = (),
        artifacts: tuple[str, ...] = (),
        turn_number: int = 0,
        tokens_used: int = 0,
        metadata: dict[str, Any] | None = None,
        cost_usd: float = 0.0,
        total_tokens: int | None = None,
    ) -> AgentResponse:
        """Create with dict metadata (convenience for callers).

        Sprint-3 #10: ``cost_usd`` + ``total_tokens`` are mirrored into
        ``metadata`` so legacy consumers (those that walk
        ``dict(response.metadata).get("cost_usd")``) keep working. When
        ``total_tokens`` is omitted, it defaults to ``tokens_used`` —
        the two are semantically the same number (the turn's total
        token spend) and we don't want callers to have to set both.
        """
        meta_dict = dict(metadata) if metadata else {}
        # Mirror the headline fields into metadata for backward compat.
        # If the caller already supplied them in metadata, the explicit
        # kwargs win (they're the source of truth).
        effective_total_tokens = total_tokens if total_tokens is not None else tokens_used
        meta_dict.setdefault("cost_usd", cost_usd)
        meta_dict.setdefault("total_tokens", effective_total_tokens)
        # Overwrite if the kwargs were passed explicitly non-default.
        if cost_usd:
            meta_dict["cost_usd"] = cost_usd
        if total_tokens is not None:
            meta_dict["total_tokens"] = effective_total_tokens
        meta = tuple(sorted(meta_dict.items()))
        return cls(
            text=text,
            intent=intent,
            tool_calls=tool_calls,
            artifacts=artifacts,
            turn_number=turn_number,
            tokens_used=tokens_used,
            metadata=meta,
            cost_usd=cost_usd,
            total_tokens=effective_total_tokens,
        )
