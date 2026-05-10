"""KaosEvent stream — the universal event types for streaming agent execution.

The :class:`KaosEvent` ABC lives in :mod:`kaos_agents.base.event`. This
package owns the concrete event subtypes (15 today) split by domain and
the serde helpers that ship them across SSE / JSONL / WebSocket / MCP
wires.

The taxonomy (chunk 4):

- **Stream deltas** (3) — :class:`TextDelta`, :class:`ThinkingDelta`,
  :class:`ToolCallArgsDelta`. Token-by-token rendering events.
- **Spans** (1 class, 7 subjects, 5 phases) — :class:`Span` covers
  every phase boundary uniformly: ``Span(subject, phase, span_id, ...)``.
  Replaces the old ``TurnStart``/``TurnComplete`` /
  ``StepStart``/``StepComplete`` / ``ToolCallStart``/``ToolCallResult`` /
  ``HandoffStart`` / ``SubagentStart``/``SubagentComplete`` zoo.
- **Value events** (6) — :class:`IntentClassified`, :class:`PlanProposed`,
  :class:`CitationFound`, :class:`UsageObserved`,
  :class:`EvidenceInsufficient`, :class:`GroundingRefusalTriggered`.
  Carry typed *facts* beyond a phase boundary.
- **Aggregates** (1) — :class:`TurnSummary`. Fires alongside
  ``Span(TURN, COMPLETE)`` with the typed turn-end aggregate
  (text / intent / tool_calls / token + cost totals).
- **Memory** (1) — :class:`MemoryEvent` (with :class:`MemoryEventKind`
  enum). Replaces the old overloaded ``MemoryUpdated``.
- **Errors** (2) — :class:`RunError`, :class:`BudgetExceeded`. Distinct
  typed subtypes; control-flow consumers branch on them specifically.
- **Control flow** (1) — :class:`ToolCallApprovalRequired`. Pause/resume
  signal with persistence semantics; not a phase boundary.

15 concrete event classes total. Down from 19 in the pre-Span taxonomy
and far fewer asymmetries — every subject has uniform start / progress /
complete / error / cancelled phases, and adding a new subject is one
:class:`SpanSubject` enum value plus emission code.

Public surface: this package re-exports every event class, plus
:class:`EventEmitter`, ``emit_usage_observed``, ``emit_span_*``
helpers, and the four serde helpers — so callers continue to use
``from kaos_agents.events import X``.
"""

from __future__ import annotations

from kaos_agents.base.event import KaosEvent
from kaos_agents.events._intermediates import LifecycleEvent
from kaos_agents.events.budget import BudgetExceeded
from kaos_agents.events.collector import (
    EventCollector,
    active_collector,
    collect_events,
    push_event,
)
from kaos_agents.events.emitter import (
    EventEmitter,
    emit_thinking_from_invocation,
    emit_usage_observed,
)
from kaos_agents.events.escalation import EscalationRequired
from kaos_agents.events.lifecycle import (
    IntentClassified,
    RunError,
    TurnSummary,
    UsageObserved,
)
from kaos_agents.events.memory import MemoryEvent, MemoryEventKind
from kaos_agents.events.plan import (
    PlanProposed,
    PlanStepSummary,
)
from kaos_agents.events.research import (
    CitationFound,
    EvidenceInsufficient,
    GroundingRefusalTriggered,
)
from kaos_agents.events.serde import (
    deserialize_event,
    deserialize_event_json,
    event_type_name,
    serialize_event,
    serialize_event_json,
)
from kaos_agents.events.spans import (
    Span,
    SpanPhase,
    SpanSubject,
)
from kaos_agents.events.stream import (
    StreamDelta,
    TextDelta,
    ThinkingDelta,
    ToolCallArgsDelta,
)
from kaos_agents.events.tools import (
    ToolCallApprovalRequired,
    ToolCallSummary,
)

# All concrete event types, in stable order (used by tests + listings).
ALL_EVENT_TYPES: tuple[type[KaosEvent], ...] = (
    # Stream deltas
    TextDelta,
    ThinkingDelta,
    ToolCallArgsDelta,
    # Spans (one class, many subjects/phases)
    Span,
    # Value events
    IntentClassified,
    PlanProposed,
    CitationFound,
    UsageObserved,
    EvidenceInsufficient,
    GroundingRefusalTriggered,
    # Aggregates
    TurnSummary,
    # Memory
    MemoryEvent,
    # Errors
    RunError,
    BudgetExceeded,
    # Control flow
    ToolCallApprovalRequired,
)


__all__ = [
    "ALL_EVENT_TYPES",
    "BudgetExceeded",
    "CitationFound",
    "EscalationRequired",
    "EventCollector",
    "EventEmitter",
    "EvidenceInsufficient",
    "GroundingRefusalTriggered",
    "IntentClassified",
    "KaosEvent",
    "LifecycleEvent",
    "MemoryEvent",
    "MemoryEventKind",
    "PlanProposed",
    "PlanStepSummary",
    "RunError",
    "Span",
    "SpanPhase",
    "SpanSubject",
    "StreamDelta",
    "TextDelta",
    "ThinkingDelta",
    "ToolCallApprovalRequired",
    "ToolCallArgsDelta",
    "ToolCallSummary",
    "TurnSummary",
    "UsageObserved",
    "active_collector",
    "collect_events",
    "deserialize_event",
    "deserialize_event_json",
    "emit_thinking_from_invocation",
    "emit_usage_observed",
    "event_type_name",
    "push_event",
    "serialize_event",
    "serialize_event_json",
]
