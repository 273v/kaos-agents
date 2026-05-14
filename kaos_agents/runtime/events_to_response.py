"""Convert a :class:`KaosEvent` stream to an :class:`AgentResponse`.

The single canonical drain — used by both :meth:`KaosAgent.turn`
(the ABC default) and :meth:`Runner.turn` (the public Runner
streaming wrapper). Scans the collected events for the end-of-turn
markers (:class:`TurnSummary` value event + :class:`Span(TURN, COMPLETE)`
boundary + :class:`IntentClassified` decision + any
:class:`RunError`) and assembles the typed response.

PA14 (issue #166): consolidated from two parallel implementations
(``Runner.turn`` inline drain + this helper) into one. The
``Runner.turn`` body now collects events into a list and delegates
to :func:`events_to_response`, so the two paths cannot drift.

Resolved D1/D2/D3 divergences:

- D1: :class:`RunError` events now surface as ``metadata.error_type``
  and ``metadata.error_message`` keys (the canonical
  ``Runner.turn`` behavior wins).
- D2: when no :class:`IntentClassified` event fires, the default
  ``IntentResult.reasoning`` is the descriptive string
  ``"no IntentClassified event (run aborted early or errored)"``
  (canonical ``Runner.turn`` behavior).
- D3: when a :class:`TurnSummary` event fires with empty text, the
  drain falls back to concatenated :class:`TextDelta` content so
  partial / errored turns surface whatever streamed text was
  produced (canonical ``Runner.turn`` behavior).

Lives in :mod:`kaos_agents.runtime` (not :mod:`kaos_agents.base`)
because it depends on the concrete event subtypes — ``base/`` should
only depend on :class:`KaosEvent` directly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from kaos_agents.events import (
    IntentClassified,
    RunError,
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


# Default reasoning string when no IntentClassified event was emitted.
# Exposed as a module-level constant so callers (and tests) can refer
# to the canonical value without copy-pasting the literal.
_DEFAULT_NO_INTENT_REASONING = "no IntentClassified event (run aborted early or errored)"


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
    run_error: RunError | None = None
    turn_number = 0
    tool_call_records: list[ToolExecution] = []
    text_parts: list[str] = []

    for event in events:
        if isinstance(event, TurnSummary):
            turn_summary = event
        elif isinstance(event, IntentClassified):
            intent_event = event
        elif isinstance(event, RunError):
            # D1: keep the last RunError. ``metadata.error_*`` below
            # surfaces it so callers can distinguish abnormal turns
            # from clean ones without walking the event stream.
            run_error = event
        elif isinstance(event, TextDelta):
            # D3: collect deltas as we go so the empty-TurnSummary.text
            # fallback can use them without re-walking the list.
            text_parts.append(event.content)
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
    # D2: when no IntentClassified fires, use the descriptive default
    # reasoning so callers can tell "errored early" apart from
    # "classifier produced an empty reasoning string".
    intent = IntentResult(
        intent=IntentType(intent_event.intent) if intent_event else IntentType.RESPOND,
        confidence=intent_event.confidence if intent_event else 0.0,
        reasoning=intent_event.reasoning if intent_event else _DEFAULT_NO_INTENT_REASONING,
    )

    # Response text + token totals from TurnSummary (the canonical
    # end-of-turn aggregate). Fall back to concatenated TextDelta
    # content when (a) the turn errored before TurnSummary fired, or
    # (b) D3: the TurnSummary fired but its ``text`` is empty (partial
    # / errored turns can emit TextDeltas before an early TurnSummary).
    #
    # Sprint-3 #10 (transparency lens): also pull cost_usd off the
    # TurnSummary so the AgentResponse carries it as a first-class
    # attribute. tokens_used and total_tokens are the same number at
    # the turn level (TurnSummary.tokens_used is the aggregate across
    # every UsageObserved event for the turn); we surface it under
    # both names for ergonomic API consistency.
    if turn_summary is not None:
        text = turn_summary.text or "".join(text_parts)
        tokens_used = turn_summary.tokens_used
        cost_usd = float(turn_summary.cost_usd or 0.0)
    else:
        text = "".join(text_parts)
        tokens_used = 0
        cost_usd = 0.0

    metadata: dict[str, object] = {"session_id": session_id}
    if run_error is not None:
        # D1: surface RunError fields so callers can branch on them
        # without inspecting the event stream. Mirrors the canonical
        # ``Runner.turn`` body before consolidation.
        metadata["error_type"] = run_error.error_type
        metadata["error_message"] = run_error.message

    return AgentResponse.create(
        text=text,
        intent=intent,
        tool_calls=tuple(tool_call_records),
        turn_number=turn_number,
        tokens_used=tokens_used,
        cost_usd=cost_usd,
        total_tokens=tokens_used,
        metadata=metadata,
    )


__all__ = ["events_to_response"]
