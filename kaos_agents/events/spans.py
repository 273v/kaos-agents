"""Span — the universal phase-boundary event.

A span represents the lifecycle of any unit of agent work — a run, a
turn, a plan step, a tool call, a sub-agent invocation, a handoff, an
LLM call. One :class:`Span` event fires per phase transition
(START / PROGRESS / COMPLETE / ERROR / CANCELLED) per unit of work.

Aligns with OpenTelemetry semantics:
- ``span_id`` / ``parent_span_id`` build a span tree
- ``name`` is the operation identifier (``"tool.kaos-web-fetch"``,
  ``"step.s001"``, ``"turn.1"``)
- ``duration_ms`` is filled on COMPLETE / ERROR / CANCELLED
- ``error_type`` / ``error_message`` are filled on ERROR
- ``attributes`` is the OTel-style free-form attribute bag for
  subject-specific payload (tool args, step description, hand-off
  reason, etc.)

Why a single ``Span`` rather than ``TurnStart`` / ``StepStart`` /
``ToolCallStart`` / ... pairs:

- The previous taxonomy had asymmetries (``HandoffStart`` with no
  ``HandoffComplete``, no error/cancelled events for any subject) and
  duplicated the start/complete dance for every subject.
- A uniform Span makes hooks trivial: filter by subject, phase, or
  span tree without learning the specific class.
- Adding a new subject (e.g. RAG_QUERY) is an enum value plus
  emission code — no new class to register, document, test, or hold
  in the registry.

Concrete value events that carry *facts* beyond a phase boundary
(:class:`IntentClassified`, :class:`PlanProposed`, :class:`CitationFound`,
:class:`UsageObserved`, :class:`TurnSummary`) keep typed classes — see
the corresponding modules.
"""

from __future__ import annotations

from enum import StrEnum, unique
from typing import Any

from pydantic import Field

from kaos_agents.events._intermediates import LifecycleEvent


@unique
class SpanSubject(StrEnum):
    """What kind of work this span represents."""

    RUN = "run"  # The whole run (one or more turns)
    TURN = "turn"  # One agent turn
    STEP = "step"  # One plan step
    TOOL_CALL = "tool_call"  # One tool invocation
    SUBAGENT = "subagent"  # A delegated sub-agent
    HANDOFF = "handoff"  # Lateral hand-off to another agent
    LLM_CALL = "llm_call"  # One LLM invocation (transport-level)
    JUDGE = "judge"  # One Evaluate.evaluate_semantic LLM judgment call
    ROUTE = "route"  # One Route primitive decision (control-flow boundary)


@unique
class SpanPhase(StrEnum):
    """Where in its lifecycle the span is."""

    START = "start"
    PROGRESS = "progress"  # Long-running spans emit PROGRESS in between
    COMPLETE = "complete"
    ERROR = "error"
    CANCELLED = "cancelled"


class Span(LifecycleEvent):
    """A phase boundary in the lifecycle of any unit of agent work.

    A typical run produces a tree of spans:

        Span(RUN, START)
          Span(TURN, START, parent=run)
            Span(STEP, START, parent=turn)
              Span(TOOL_CALL, START, parent=step)
              Span(TOOL_CALL, COMPLETE, parent=step, duration_ms=...)
            Span(STEP, COMPLETE, parent=turn, duration_ms=...)
          Span(TURN, COMPLETE, parent=run, duration_ms=...)
        Span(RUN, COMPLETE, duration_ms=...)

    Plus one :class:`TurnSummary` value event alongside each
    ``Span(TURN, COMPLETE)`` carrying the typed turn-end aggregates.

    Consumers route by ``(subject, phase)`` rather than ``isinstance``::

        if isinstance(event, Span) and event.subject == SpanSubject.TOOL_CALL:
            if event.phase == SpanPhase.START:
                ui.show_tool_call(event.name, event.attributes)
            elif event.phase == SpanPhase.COMPLETE:
                ui.complete_tool_call(event.span_id, event.duration_ms)
    """

    subject: SpanSubject
    phase: SpanPhase
    span_id: str
    parent_span_id: str | None = None

    # Operation name — namespaced like "tool.kaos-web-fetch", "step.s001",
    # "turn.1". Optional but strongly encouraged for span-tree readability.
    name: str = ""

    # Filled on COMPLETE / ERROR / CANCELLED. None during START / PROGRESS.
    duration_ms: float | None = None

    # Filled on ERROR. None for all other phases.
    error_type: str | None = None
    error_message: str | None = None

    # Subject-specific payload. Keys are convention-driven per subject:
    #   TURN.START          -> {"turn_number": int}
    #   STEP.START           -> {"step_id": str, "description": str}
    #   STEP.COMPLETE        -> {"step_id": str, "result_summary": str, "is_error": bool}
    #   TOOL_CALL.START      -> {"tool_name": str, "call_id": str,
    #                            "arguments": tuple[tuple[str, str], ...]}
    #   TOOL_CALL.COMPLETE   -> {"tool_name": str, "call_id": str,
    #                            "result_summary": str, "is_error": bool}
    #   SUBAGENT.START       -> {"subagent_name": str, "task": str}
    #   SUBAGENT.COMPLETE    -> {"subagent_name": str, "result_summary": str,
    #                            "tokens_used": int}
    #   HANDOFF.START        -> {"from_agent": str, "to_agent": str, "reason": str}
    #   JUDGE.START          -> {"step_id": str, "expected": str (<=200 chars),
    #                            "result_preview": str (<=200 chars)}
    #   JUDGE.COMPLETE       -> {"step_id": str, "matched": bool,
    #                            "confidence": float, "reasoning": str (<=200 chars),
    #                            "mode": "structural" | "semantic"}
    #   ROUTE.COMPLETE       -> {"step_id": str, "decision": str (Decision.value),
    #                            "reason": str (<=200 chars),
    #                            "judgment_matched": bool,
    #                            "judgment_confidence": float,
    #                            "replan_count": int}
    attributes: dict[str, Any] = Field(default_factory=dict)


__all__ = ["Span", "SpanPhase", "SpanSubject"]
