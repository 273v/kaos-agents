"""Convert a :class:`KaosEvent` stream to an :class:`AgentResponse`.

Used by the default :meth:`KaosAgent.turn` implementation. Scans the
collected events for the canonical end-of-turn markers
(:class:`TurnSummary` value event + :class:`Span(TURN, COMPLETE)`
boundary + :class:`IntentClassified` decision) and assembles the
typed response.

Lives in :mod:`kaos_agents.runtime` (not :mod:`kaos_agents.base`)
because it depends on the concrete event subtypes — ``base/`` should
only depend on :class:`KaosEvent` directly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from kaos_agents.events import (
    IntentClassified,
    Span,
    SpanPhase,
    SpanSubject,
    TextDelta,
    TurnSummary,
)
from kaos_agents.types.intents import IntentResult, IntentType
from kaos_agents.types.response import AgentResponse
from kaos_agents.types.tool_call import ToolExecution

if TYPE_CHECKING:
    from kaos_agents.base.event import KaosEvent


def events_to_response(events: list[KaosEvent], session_id: str) -> AgentResponse:
    """Assemble an :class:`AgentResponse` from a collected event stream.

    Scans for the canonical end-of-turn markers and falls back
    gracefully when the stream is incomplete.

    Args:
        events: All events emitted during the turn, in order.
        session_id: Session this turn belonged to (for the response's
            ``metadata`` field).

    Returns:
        :class:`AgentResponse` populated from the stream's terminal
        events. Never raises — partial / failed turns produce a
        best-effort response with whatever fields were observed.
    """
    turn_summary: TurnSummary | None = None
    intent_event: IntentClassified | None = None
    turn_number = 0
    tool_call_records: list[ToolExecution] = []

    for event in events:
        if isinstance(event, TurnSummary):
            turn_summary = event
        elif isinstance(event, IntentClassified):
            intent_event = event
        elif isinstance(event, Span):
            if event.subject == SpanSubject.TURN and event.phase == SpanPhase.START:
                # Turn number lives on the START span's attributes.
                turn_number = int(event.attributes.get("turn_number", 0) or 0)
            elif event.subject == SpanSubject.TOOL_CALL and event.phase == SpanPhase.COMPLETE:
                attrs = event.attributes
                tool_call_records.append(
                    ToolExecution.from_dict_args(
                        tool_name=str(attrs.get("tool_name", "")),
                        arguments={},  # Args are on the START span; we don't replay them here
                        result_summary=str(attrs.get("result_summary", "")),
                        is_error=bool(attrs.get("is_error", False)),
                    )
                )

    # IntentResult: from the classification event if seen, else default.
    intent = IntentResult(
        intent=IntentType(intent_event.intent) if intent_event else IntentType.RESPOND,
        confidence=intent_event.confidence if intent_event else 0.0,
        reasoning=intent_event.reasoning if intent_event else "",
    )

    # Response text + token totals from TurnSummary (the canonical
    # end-of-turn aggregate). Fall back to concatenated TextDelta
    # content when the turn errored before TurnSummary fired.
    #
    # Sprint-3 #10 (transparency lens): also pull cost_usd off the
    # TurnSummary so the AgentResponse carries it as a first-class
    # attribute. tokens_used and total_tokens are the same number at
    # the turn level (TurnSummary.tokens_used is the aggregate across
    # every UsageObserved event for the turn); we surface it under
    # both names for ergonomic API consistency.
    if turn_summary is not None:
        text = turn_summary.text
        tokens_used = turn_summary.tokens_used
        cost_usd = float(turn_summary.cost_usd or 0.0)
    else:
        text = "".join(event.content for event in events if isinstance(event, TextDelta))
        tokens_used = 0
        cost_usd = 0.0

    return AgentResponse.create(
        text=text,
        intent=intent,
        tool_calls=tuple(tool_call_records),
        turn_number=turn_number,
        tokens_used=tokens_used,
        cost_usd=cost_usd,
        total_tokens=tokens_used,
        metadata={"session_id": session_id},
    )


__all__ = ["events_to_response"]
