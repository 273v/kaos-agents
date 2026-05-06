"""KaosEvent subclasses — the universal event types for streaming agent execution.

The :class:`KaosEvent` base ABC lives in :mod:`kaos_agents.base.event`. This
module owns the concrete event subtypes (19 today) and the serde helpers
that ship them across SSE / JSONL / WebSocket / MCP wires.

Every consumer surface (Python SDK, SSE, JSONL, MCP, WebSocket) ships and
receives the same KaosEvent stream. Events fall into two conceptual
categories distinguished by ``isinstance``:

- **Stream deltas** (TextDelta, ThinkingDelta, ToolCallArgsDelta) —
  low-level, real-time, token-by-token events.
- **Lifecycle events** (TurnStart, ToolCallStart, StepComplete, etc.) —
  high-level, semantic events for UI state machines, hooks, audit.

Consumers pick their granularity:
- A streaming text UI handles TextDelta.
- A progress dashboard handles ToolCallStart/StepComplete.
- An audit logger handles everything.

Design grounded in competitive research across 6 frameworks:
- Discriminated union of typed events (OpenAI, Pydantic-AI, CrewAI pattern)
- Two-level granularity (OpenAI, Pydantic-AI, LangGraph pattern)
- Tool calls as separate events (4 of 6 frameworks)
- Sequence counter for replay ordering (CrewAI pattern)
- agent_id for multi-agent attribution (LangGraph, CrewAI, ADK pattern)
- Error as event + exception (ADK, CrewAI pattern)

See docs/design/kaos-agents-streaming-research.md for the full analysis.
"""

from __future__ import annotations

import json
import time
from typing import TYPE_CHECKING, Any

from kaos_core.types.content import KaosModel
from pydantic import ConfigDict, ValidationError

from kaos_agents.base.event import KaosEvent
from kaos_agents.errors import EventDeserializationError, EventSerializationError
from kaos_agents.registry.event_registry import default_event_registry as _event_registry

if TYPE_CHECKING:
    from kaos_agents.types.usage import InvocationUsage


# ---------------------------------------------------------------------------
# Supporting types — nested inside event payloads (not events themselves)
# ---------------------------------------------------------------------------


class ToolCallSummary(KaosModel):
    """Compact summary of a tool call for TurnComplete.

    ``cost_usd`` aggregates the LLM cost attributable to this tool
    invocation when the tool itself drove an LLM call (e.g. RAG's
    rag-query verifier, or a delegated sub-agent). Plain tools that
    don't call an LLM stay at ``0.0``. Threaded through Phase 5.x for
    per-tool-call cost telemetry (P8 / N2).
    """

    model_config = ConfigDict(frozen=True, extra="forbid", populate_by_name=True)

    tool_name: str
    call_id: str
    is_error: bool = False
    duration_ms: float = 0.0
    cost_usd: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0


class PlanStepSummary(KaosModel):
    """Compact summary of a plan step for PlanProposed."""

    model_config = ConfigDict(frozen=True, extra="forbid", populate_by_name=True)

    step_id: str
    description: str
    tool_name: str | None = None


# ---------------------------------------------------------------------------
# Intermediate categories — enable isinstance(e, StreamDelta) dispatch
# ---------------------------------------------------------------------------


class StreamDelta(KaosEvent):
    """Base class for low-level, real-time streaming events.

    Consumers that care about token-by-token display handle these
    (TextDelta, ThinkingDelta, ToolCallArgsDelta).

    ``isinstance(event, StreamDelta)`` is the canonical way to route
    real-time display events.
    """

    @classmethod
    def event_category(cls) -> str:
        return "stream"


class LifecycleEvent(KaosEvent):
    """Base class for high-level, semantic lifecycle events.

    Consumers that drive UI state machines, logging, hooks, audit
    trails — anything that needs "what just happened" rather than
    "what tokens just arrived" — handle these.

    ``isinstance(event, LifecycleEvent)`` is the canonical way to
    route lifecycle events.
    """


# ---------------------------------------------------------------------------
# Stream deltas — low-level, real-time
# ---------------------------------------------------------------------------


class TextDelta(StreamDelta):
    """Streaming text token(s) from the LLM.

    Emitted as the model generates text. Consumers concatenate
    sequential TextDelta.content to build the full response.
    """

    content: str = ""


class ThinkingDelta(StreamDelta):
    """Streaming thinking/reasoning tokens (extended thinking).

    Separate from TextDelta so UIs can render thinking in a
    collapsible section or different style.
    """

    content: str = ""


class ToolCallArgsDelta(StreamDelta):
    """Streaming tool call argument tokens.

    Emitted as the model generates tool call arguments.
    Allows UIs to show partial argument JSON as it builds.
    """

    call_id: str = ""
    content: str = ""


# ---------------------------------------------------------------------------
# Lifecycle events — high-level, semantic
# ---------------------------------------------------------------------------

# --- Turn lifecycle ---


class TurnStart(LifecycleEvent):
    """Emitted at the beginning of a turn, after memory hydration."""

    turn_number: int = 0


class TurnComplete(LifecycleEvent):
    """Emitted at the end of a turn, after memory persistence.

    Contains the final response and aggregate metrics. ``tokens_used``
    is the turn's total (== ``input_tokens + output_tokens``); the
    split fields were added in Phase 5.0 when real provider-reported
    usage started flowing through the turn loop. Pre-5.0 both fields
    were hard-coded to 0.
    """

    text: str = ""
    intent: str = ""
    tool_calls: tuple[ToolCallSummary, ...] = ()
    tokens_used: int = 0
    cost_usd: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0


class UsageObserved(LifecycleEvent):
    """Emitted each time an LLM invocation completes during a turn.

    Streaming handlers yield one ``UsageObserved`` per completed
    ``Call`` / ``Program`` with its ``Invocation.usage`` surfaced verbatim.
    The turn loop accumulates these into the aggregate fields on the
    final ``TurnComplete``. Downstream hooks (OTel, cost-aware routing,
    budget enforcement) can consume individual events for fine-grained
    attribution — which program fired, at which step, for how much.

    ``source`` is a short string the emitter chose to distinguish
    concurrent sub-invocations (e.g. ``"react"``, ``"respond"``,
    ``"classify"``, or the name of a delegated subagent). It's
    informational, not a routing key.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0
    source: str = ""


class RunError(LifecycleEvent):
    """Emitted when an error occurs during execution.

    Also raised as an exception after yielding, so in-process
    consumers can catch it either way.

    Fields follow the KAOS agent-friendly error pattern:
    (1) what went wrong, (2) how to fix it, (3) alternative approach.
    """

    error_type: str = ""
    message: str = ""
    recovery_hint: str = ""


# --- Intent ---


class IntentClassified(LifecycleEvent):
    """Emitted after intent classification completes."""

    intent: str = ""  # IntentType value
    confidence: float = 0.0
    reasoning: str = ""


# --- Tool execution ---


class ToolCallStart(LifecycleEvent):
    """Emitted when a tool call begins execution."""

    call_id: str = ""
    tool_name: str = ""
    arguments: tuple[tuple[str, str], ...] = ()


class ToolCallResult(LifecycleEvent):
    """Emitted when a tool call completes."""

    call_id: str = ""
    tool_name: str = ""
    result_summary: str = ""
    is_error: bool = False
    duration_ms: float = 0.0


class ToolCallApprovalRequired(LifecycleEvent):
    """Emitted when a tool needs human approval before execution.

    The run pauses after yielding this event. Resume via Runner.resume()
    with the run_state_ref. See Phase 6 of the streaming roadmap.
    """

    call_id: str = ""
    tool_name: str = ""
    arguments: tuple[tuple[str, str], ...] = ()
    reason: str = ""
    run_state_ref: str = ""  # VFS path or ID for serialized RunState


# --- Planning ---


class PlanProposed(LifecycleEvent):
    """Emitted when the planner produces a plan before execution."""

    steps: tuple[PlanStepSummary, ...] = ()
    strategy: str = ""  # "direct", "decompose", "rolling", "adaptive"


class StepStart(LifecycleEvent):
    """Emitted when a plan step begins execution."""

    step_id: str = ""
    description: str = ""


class StepComplete(LifecycleEvent):
    """Emitted when a plan step finishes execution."""

    step_id: str = ""
    result_summary: str = ""
    is_error: bool = False
    duration_ms: float = 0.0


# --- Research ---


class CitationFound(LifecycleEvent):
    """Emitted when RAG verifies a claim against source documents."""

    claim: str = ""
    source_uri: str = ""
    confidence: float = 0.0
    verified: bool = False


class EvidenceInsufficient(LifecycleEvent):
    """Emitted when RAG cannot find sufficient evidence to answer.

    This is the "refuses when uncertain" pattern (Everlaw Deep Dive).
    """

    reason: str = ""
    what_would_resolve: str = ""


class GroundingRefusalTriggered(LifecycleEvent):
    """Emitted when the Agent's refusal_policy collapses an Answer.

    Fires when a ``GroundedAnswer`` comes back as ``Answer[T]`` but the
    answer's confidence is below ``refusal_policy.min_confidence``, or
    when span verification fails with ``require_verification=True``.
    The Answer is replaced with ``InsufficientEvidence`` downstream.

    Listeners (logging hooks, plan-execute replan logic) use this to
    surface the refusal and decide whether to retry, replan, or stop.
    """

    original_confidence: float = 0.0
    min_confidence: float = 0.0
    reason: str = ""


# --- Memory ---


class MemoryUpdated(LifecycleEvent):
    """Emitted when a memory section is modified."""

    section: str = ""  # MemoryType value
    action: str = ""  # "add", "evict", "summarize"
    item_count: int = 0


# --- Delegation (Phase 7 — included in type system now) ---


class HandoffStart(LifecycleEvent):
    """Emitted when control transfers to another agent."""

    from_agent: str = ""
    to_agent: str = ""
    reason: str = ""


class SubagentStart(LifecycleEvent):
    """Emitted when a sub-agent is spawned for an isolated task."""

    subagent_name: str = ""
    task: str = ""


class SubagentComplete(LifecycleEvent):
    """Emitted when a sub-agent finishes and returns results."""

    subagent_name: str = ""
    result_summary: str = ""
    tokens_used: int = 0


# ---------------------------------------------------------------------------
# Type registry + serialization
# ---------------------------------------------------------------------------

# All concrete event types (order doesn't matter for the registry).
ALL_EVENT_TYPES: tuple[type[KaosEvent], ...] = (
    TextDelta,
    ThinkingDelta,
    ToolCallArgsDelta,
    TurnStart,
    TurnComplete,
    RunError,
    IntentClassified,
    ToolCallStart,
    ToolCallResult,
    ToolCallApprovalRequired,
    PlanProposed,
    StepStart,
    StepComplete,
    CitationFound,
    EvidenceInsufficient,
    GroundingRefusalTriggered,
    MemoryUpdated,
    HandoffStart,
    SubagentStart,
    SubagentComplete,
    UsageObserved,
)

# Wire-format dispatch is backed by
# :data:`kaos_agents.registry.event_registry.default_event_registry`
# (imported at module top). Every :class:`KaosEvent` subclass
# auto-registers via ``__init_subclass__`` at class-creation time, so
# importing this module populates the registry as a side effect.


def event_type_name(event: KaosEvent) -> str:
    """Get the wire-format type string for an event instance."""
    return type(event).event_type()


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


def serialize_event(event: KaosEvent) -> dict[str, Any]:
    """Serialize a KaosEvent to a JSON-safe dict with a ``type`` discriminator.

    The ``type`` field is the snake_case version of the class name.
    Nested ``KaosModel`` payloads (ToolCallSummary, PlanStepSummary)
    are serialized as dicts. Tuples become lists.

    Example::

        >>> e = TurnStart(timestamp=1.0, sequence=0, session_id="s", run_id="r", turn_number=1)
        >>> serialize_event(e)
        {'type': 'turn_start', 'timestamp': 1.0, 'sequence': 0, ...}
    """
    try:
        data = event.model_dump(mode="json")
    except ValidationError as exc:
        raise EventSerializationError(
            f"Failed to serialize {type(event).__name__}: {exc}. "
            "Ensure all event fields contain JSON-serializable values."
        ) from exc
    data["type"] = event_type_name(event)
    return data


def deserialize_event(data: dict[str, Any]) -> KaosEvent:
    """Reconstruct a KaosEvent from a serialized dict.

    Raises:
        EventDeserializationError: If the ``type`` field is missing or unknown,
            or if required fields are missing or have incorrect types.

    Example::

        >>> d = {'type': 'turn_start', 'timestamp': 1.0, 'sequence': 0,
        ...      'session_id': 's', 'run_id': 'r', 'turn_number': 1}
        >>> deserialize_event(d)
        TurnStart(timestamp=1.0, sequence=0, session_id='s', run_id='r',
                  agent_id=None, turn_number=1)
    """
    type_name = data.get("type")
    if not type_name:
        raise EventDeserializationError(
            "Event dict missing 'type' field. "
            "Every serialized event must include a 'type' field with the snake_case event name. "
            "Use serialize_event() to produce correctly formatted dicts."
        )

    cls = _event_registry.get(type_name)
    if cls is None:
        valid = ", ".join(_event_registry.list_types())
        raise EventDeserializationError(
            f"Unknown event type '{type_name}'. "
            f"Valid types: {valid}. "
            "Check that the event type name is snake_case (e.g., 'turn_start', not 'TurnStart')."
        )

    payload = {k: v for k, v in data.items() if k != "type"}
    try:
        return cls.model_validate(payload)
    except ValidationError as exc:
        required = ", ".join(name for name, info in cls.model_fields.items() if info.is_required())
        raise EventDeserializationError(
            f"Failed to construct {cls.__name__}: {exc}. "
            f"Required fields: {required or '(none)'}. "
            f"Provided fields: {', '.join(payload) or '(none)'}."
        ) from exc


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------


def serialize_event_json(event: KaosEvent) -> str:
    """Serialize a KaosEvent to a compact JSON string.

    Raises:
        EventSerializationError: If the event contains values that
            cannot be serialized to JSON (e.g., custom objects).
    """
    try:
        return json.dumps(serialize_event(event), separators=(",", ":"))
    except (TypeError, ValueError) as exc:
        raise EventSerializationError(
            f"Failed to serialize {type(event).__name__} to JSON: {exc}. "
            "Ensure all event fields contain JSON-serializable values "
            "(str, int, float, bool, None, tuple, dict)."
        ) from exc


def deserialize_event_json(data: str) -> KaosEvent:
    """Deserialize a KaosEvent from a JSON string.

    Raises:
        EventDeserializationError: If the JSON is invalid or the event
            cannot be reconstructed.
    """
    try:
        parsed = json.loads(data)
    except json.JSONDecodeError as exc:
        raise EventDeserializationError(
            f"Invalid JSON at position {exc.pos}: {exc.msg}. "
            "Ensure the input is valid JSON. "
            "Use serialize_event_json() to produce correctly formatted strings."
        ) from exc
    if not isinstance(parsed, dict):
        raise EventDeserializationError(
            f"Expected a JSON object, got {type(parsed).__name__}. "
            "Each event must be a JSON object with a 'type' field. "
            "Use serialize_event_json() to produce correctly formatted strings."
        )
    return deserialize_event(parsed)


# ---------------------------------------------------------------------------
# Event factory helper
# ---------------------------------------------------------------------------


class EventEmitter:
    """Helper that auto-fills timestamp, sequence, session_id, run_id.

    Used by the agent loop to emit events without repeating boilerplate.

    Usage::

        emitter = EventEmitter(session_id="abc", run_id="run_01")
        yield emitter.emit(TurnStart, turn_number=1)
        yield emitter.emit(IntentClassified, intent="tool_use", confidence=0.9)
    """

    __slots__ = ("_agent_id", "_run_id", "_sequence", "_session_id")

    def __init__(
        self,
        session_id: str,
        run_id: str,
        agent_id: str | None = None,
    ) -> None:
        self._session_id = session_id
        self._run_id = run_id
        self._agent_id = agent_id
        self._sequence = 0

    def emit(self, cls: type[KaosEvent], **kwargs: Any) -> KaosEvent:
        """Create an event instance with auto-filled base fields.

        Args:
            cls: The event class to instantiate.
            **kwargs: Event-specific fields.

        Returns:
            A fully-initialized event instance.
        """
        seq = self._sequence
        self._sequence += 1
        return cls(
            timestamp=time.monotonic(),
            sequence=seq,
            session_id=self._session_id,
            run_id=self._run_id,
            agent_id=kwargs.pop("agent_id", self._agent_id),
            **kwargs,
        )

    @property
    def sequence(self) -> int:
        """Current sequence counter (next event will have this value)."""
        return self._sequence


def emit_usage_observed(
    emitter: EventEmitter, usage: InvocationUsage, *, source: str = ""
) -> KaosEvent:
    """Build a ``UsageObserved`` event from an ``InvocationUsage``.

    Returns ``KaosEvent`` (not ``UsageObserved``) because ``emitter.emit``
    is declared to return the common base type. Downstream consumers
    narrow with ``isinstance(event, UsageObserved)`` as usual.

    Centralized so pattern-level dispatchers (chat, plan, research)
    don't each re-learn the field mapping.
    """
    return emitter.emit(
        UsageObserved,
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        total_tokens=usage.total_tokens,
        cost_usd=usage.cost_usd,
        source=source,
    )
