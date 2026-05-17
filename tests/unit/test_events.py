"""Tests for the KaosEvent model — the new Span-based taxonomy.

Covers:
- Round-trip serialization (serialize -> deserialize) for every event type
- Type discrimination via isinstance (StreamDelta vs LifecycleEvent)
- Sequence counter monotonicity via EventEmitter
- JSON string serialization/deserialization
- snake_case type field mapping
- Error handling for invalid inputs
- Frozen immutability
- Forward compatibility (unknown fields ignored on deserialize)
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from kaos_agents.errors import EventDeserializationError
from kaos_agents.events import (
    ALL_EVENT_TYPES,
    BudgetExceeded,
    CapabilityRequested,
    CitationFound,
    EventEmitter,
    EvidenceInsufficient,
    GoalChecked,
    GroundingRefusalTriggered,
    IntentClassified,
    KaosEvent,
    LifecycleEvent,
    LoopTerminated,
    MemoryEvent,
    MemoryEventKind,
    PatternMismatch,
    PlanProposed,
    PlanStepSummary,
    RunError,
    Span,
    SpanPhase,
    SpanSubject,
    StreamDelta,
    TextDelta,
    ThinkingDelta,
    ToolCallApprovalRequired,
    ToolCallArgsDelta,
    ToolCallSummary,
    ToolPolicyElevated,
    TurnSummary,
    UsageObserved,
    deserialize_event,
    deserialize_event_json,
    event_type_name,
    serialize_event,
    serialize_event_json,
)

# ---------------------------------------------------------------------------
# Fixtures: one instance of every event type with realistic data
# ---------------------------------------------------------------------------

# Common base args — typed explicitly so ty can narrow **kwargs.
_TS = 1.0
_SID = "sess-1"
_RID = "run-1"


def _make_all_events() -> list[KaosEvent]:
    """Create one instance of every concrete event type."""
    return [
        TextDelta(timestamp=_TS, sequence=0, session_id=_SID, run_id=_RID, content="Hello"),
        ThinkingDelta(
            timestamp=_TS, sequence=1, session_id=_SID, run_id=_RID, content="Let me think..."
        ),
        ToolCallArgsDelta(
            timestamp=_TS,
            sequence=2,
            session_id=_SID,
            run_id=_RID,
            call_id="tc_01",
            content='{"query":',
        ),
        # Span — one instance covers the universal phase-boundary surface.
        # Subject/phase combinations are exercised under TestSpan below.
        Span(
            timestamp=_TS,
            sequence=3,
            session_id=_SID,
            run_id=_RID,
            subject=SpanSubject.TURN,
            phase=SpanPhase.START,
            span_id="span0001",
            name="turn.1",
            attributes={"turn_number": 1},
        ),
        TurnSummary(
            timestamp=_TS,
            sequence=4,
            session_id=_SID,
            run_id=_RID,
            text="Done.",
            intent="respond",
            tool_calls=(
                ToolCallSummary(
                    tool_name="kaos-source-fr-search", call_id="tc_01", duration_ms=500
                ),
            ),
            tokens_used=100,
            cost_usd=0.001,
        ),
        RunError(
            timestamp=_TS,
            sequence=5,
            session_id=_SID,
            run_id=_RID,
            error_type="timeout",
            message="Timed out",
            recovery_hint="Retry",
        ),
        PatternMismatch(
            timestamp=_TS,
            sequence=52,
            session_id=_SID,
            run_id=_RID,
            classified_intent="plan",
            agent_pattern="chat",
            recommended_pattern="plan",
            fallback_handler="_handle_tool_use",
            rationale="intent=plan but agent pattern=chat — redirecting to ReAct",
        ),
        BudgetExceeded(
            timestamp=_TS,
            sequence=51,
            session_id=_SID,
            run_id=_RID,
            kind="cost",
            limit=1.0,
            actual=1.5,
            reason="Plan exceeded cost cap",
        ),
        IntentClassified(
            timestamp=_TS,
            sequence=6,
            session_id=_SID,
            run_id=_RID,
            intent="tool_use",
            confidence=0.92,
            reasoning="Has tools",
        ),
        ToolCallApprovalRequired(
            timestamp=_TS,
            sequence=9,
            session_id=_SID,
            run_id=_RID,
            call_id="tc_02",
            tool_name="kaos-web-delete",
            arguments=(("url", "https://example.com"),),
            reason="Destructive tool",
            run_state_ref="vfs://runs/run-1/state.json",
        ),
        PlanProposed(
            timestamp=_TS,
            sequence=10,
            session_id=_SID,
            run_id=_RID,
            steps=(
                PlanStepSummary(
                    step_id="s1", description="Search FR", tool_name="kaos-source-fr-search"
                ),
                PlanStepSummary(step_id="s2", description="Extract text"),
            ),
            strategy="adaptive",
        ),
        CitationFound(
            timestamp=_TS,
            sequence=13,
            session_id=_SID,
            run_id=_RID,
            claim="EPA issued 3 enforcement actions",
            source_uri="doc:fr-2026-001",
            confidence=0.95,
            verified=True,
        ),
        EvidenceInsufficient(
            timestamp=_TS,
            sequence=14,
            session_id=_SID,
            run_id=_RID,
            reason="No documents address this topic",
            what_would_resolve="Load EPA enforcement action documents",
        ),
        GroundingRefusalTriggered(
            timestamp=_TS,
            sequence=141,
            session_id=_SID,
            run_id=_RID,
            original_confidence=0.4,
            min_confidence=0.7,
            reason="Answer confidence 0.40 below policy threshold 0.70",
        ),
        MemoryEvent(
            timestamp=_TS,
            sequence=15,
            session_id=_SID,
            run_id=_RID,
            kind=MemoryEventKind.ADDED,
            section="messages",
            item_count=5,
        ),
        UsageObserved(
            timestamp=_TS,
            sequence=19,
            session_id=_SID,
            run_id=_RID,
            input_tokens=120,
            output_tokens=80,
            total_tokens=200,
            cost_usd=0.004,
            source="react",
        ),
        ToolPolicyElevated(
            timestamp=_TS,
            sequence=20,
            session_id=_SID,
            run_id=_RID,
            elevated_groups=["web"],
            kept_groups=["documents", "web"],
            previous_allowed=["documents"],
            rationale="user asked to search live",
            iteration=1,
        ),
        CapabilityRequested(
            timestamp=_TS,
            sequence=21,
            session_id=_SID,
            run_id=_RID,
            requested_groups=["browser"],
            justification="page is JS-rendered, need Chromium",
            iteration=1,
            previous_allowed=["documents"],
        ),
        GoalChecked(
            timestamp=_TS,
            sequence=22,
            session_id=_SID,
            run_id=_RID,
            kind="satisfied",
            rationale="grounded in tool calls + citations",
            next_action="",
            missing="",
            confidence=0.92,
            iteration=1,
            cost_usd=0.0001,
            latency_ms=50.0,
        ),
        LoopTerminated(
            timestamp=_TS,
            sequence=23,
            session_id=_SID,
            run_id=_RID,
            reason="satisfied",
            iterations_used=1,
            elevations_used=1,
            cost_usd=0.01,
            wall_clock_ms=500.0,
        ),
    ]


ALL_EVENTS = _make_all_events()


# ---------------------------------------------------------------------------
# Test: every concrete type is represented in ALL_EVENT_TYPES registry
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestEventRegistry:
    def test_all_event_types_registered(self) -> None:
        """Every event class in ALL_EVENT_TYPES is a subclass of KaosEvent."""
        for cls in ALL_EVENT_TYPES:
            assert issubclass(cls, KaosEvent), f"{cls.__name__} is not an KaosEvent subclass"

    def test_all_concrete_types_have_fixtures(self) -> None:
        """Every registered event type has a fixture in ALL_EVENTS."""
        fixture_types = {type(e) for e in ALL_EVENTS}
        for cls in ALL_EVENT_TYPES:
            assert cls in fixture_types, f"No fixture for {cls.__name__}"

    def test_no_duplicate_type_names(self) -> None:
        """Every event type maps to a unique snake_case name."""
        names = [event_type_name(e) for e in ALL_EVENTS]
        assert len(names) == len(set(names)), f"Duplicate type names: {names}"


# ---------------------------------------------------------------------------
# Test: snake_case type names
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestTypeNames:
    @pytest.mark.parametrize(
        ("cls", "expected"),
        [
            (TextDelta, "text_delta"),
            (ThinkingDelta, "thinking_delta"),
            (ToolCallArgsDelta, "tool_call_args_delta"),
            (TurnSummary, "turn_summary"),
            (RunError, "run_error"),
            (PatternMismatch, "pattern_mismatch"),
            (BudgetExceeded, "budget_exceeded"),
            (IntentClassified, "intent_classified"),
            (ToolCallApprovalRequired, "tool_call_approval_required"),
            (PlanProposed, "plan_proposed"),
            (CitationFound, "citation_found"),
            (EvidenceInsufficient, "evidence_insufficient"),
            (GroundingRefusalTriggered, "grounding_refusal_triggered"),
            (MemoryEvent, "memory_event"),
        ],
    )
    def test_snake_case_name(self, cls: type[KaosEvent], expected: str) -> None:
        # MemoryEvent requires the kind discriminator at construction time.
        if cls is MemoryEvent:
            event = cls(
                timestamp=1.0,
                sequence=0,
                session_id="s",
                run_id="r",
                kind=MemoryEventKind.ADDED,
            )
        else:
            event = cls(timestamp=1.0, sequence=0, session_id="s", run_id="r")
        assert event_type_name(event) == expected

    def test_span_snake_case_name(self) -> None:
        e = Span(
            timestamp=1.0,
            sequence=0,
            session_id="s",
            run_id="r",
            subject=SpanSubject.TURN,
            phase=SpanPhase.START,
            span_id="abc",
        )
        assert event_type_name(e) == "span"


# ---------------------------------------------------------------------------
# Test: round-trip serialization for every event type
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSerialization:
    @pytest.mark.parametrize("event", ALL_EVENTS, ids=lambda e: type(e).__name__)
    def test_round_trip(self, event: KaosEvent) -> None:
        """serialize -> deserialize produces an equal event."""
        data = serialize_event(event)
        restored = deserialize_event(data)
        assert restored == event

    @pytest.mark.parametrize("event", ALL_EVENTS, ids=lambda e: type(e).__name__)
    def test_json_round_trip(self, event: KaosEvent) -> None:
        """JSON string serialize -> deserialize produces an equal event."""
        json_str = serialize_event_json(event)
        restored = deserialize_event_json(json_str)
        assert restored == event

    @pytest.mark.parametrize("event", ALL_EVENTS, ids=lambda e: type(e).__name__)
    def test_type_field_present(self, event: KaosEvent) -> None:
        """Serialized dict has a 'type' field."""
        data = serialize_event(event)
        assert "type" in data
        assert data["type"] == event_type_name(event)

    @pytest.mark.parametrize("event", ALL_EVENTS, ids=lambda e: type(e).__name__)
    def test_json_valid(self, event: KaosEvent) -> None:
        """Serialized JSON is valid JSON."""
        json_str = serialize_event_json(event)
        parsed = json.loads(json_str)
        assert isinstance(parsed, dict)
        assert "type" in parsed

    @pytest.mark.parametrize("event", ALL_EVENTS, ids=lambda e: type(e).__name__)
    def test_base_fields_present(self, event: KaosEvent) -> None:
        """All base fields are present in serialized output."""
        data = serialize_event(event)
        assert "timestamp" in data
        assert "sequence" in data
        assert "session_id" in data
        assert "run_id" in data
        assert "agent_id" in data


# ---------------------------------------------------------------------------
# Test: type discrimination via isinstance
# ---------------------------------------------------------------------------

# Stream deltas
_STREAM_DELTA_TYPES = (TextDelta, ThinkingDelta, ToolCallArgsDelta)
# Lifecycle events (everything else)
_LIFECYCLE_TYPES = tuple(cls for cls in ALL_EVENT_TYPES if cls not in _STREAM_DELTA_TYPES)


@pytest.mark.unit
class TestTypeDiscrimination:
    def test_stream_deltas_are_agent_events(self) -> None:
        for cls in _STREAM_DELTA_TYPES:
            e = cls(timestamp=1.0, sequence=0, session_id="s", run_id="r")
            assert isinstance(e, KaosEvent)

    def test_lifecycle_events_are_agent_events(self) -> None:
        for cls in _LIFECYCLE_TYPES:
            if cls is Span:
                e = cls(
                    timestamp=1.0,
                    sequence=0,
                    session_id="s",
                    run_id="r",
                    subject=SpanSubject.TURN,
                    phase=SpanPhase.START,
                    span_id="x",
                )
            elif cls is MemoryEvent:
                e = cls(
                    timestamp=1.0,
                    sequence=0,
                    session_id="s",
                    run_id="r",
                    kind=MemoryEventKind.ADDED,
                )
            else:
                e = cls(timestamp=1.0, sequence=0, session_id="s", run_id="r")
            assert isinstance(e, KaosEvent)

    def test_stream_deltas_are_stream_delta_subclass(self) -> None:
        """Every concrete stream delta IS a StreamDelta via isinstance."""
        for cls in _STREAM_DELTA_TYPES:
            e = cls(timestamp=1.0, sequence=0, session_id="s", run_id="r")
            assert isinstance(e, StreamDelta)
            assert not isinstance(e, LifecycleEvent)

    def test_lifecycle_events_are_lifecycle_subclass(self) -> None:
        """Every concrete lifecycle event IS a LifecycleEvent via isinstance."""
        for cls in _LIFECYCLE_TYPES:
            if cls is Span:
                e = cls(
                    timestamp=1.0,
                    sequence=0,
                    session_id="s",
                    run_id="r",
                    subject=SpanSubject.TURN,
                    phase=SpanPhase.START,
                    span_id="x",
                )
            elif cls is MemoryEvent:
                e = cls(
                    timestamp=1.0,
                    sequence=0,
                    session_id="s",
                    run_id="r",
                    kind=MemoryEventKind.ADDED,
                )
            else:
                e = cls(timestamp=1.0, sequence=0, session_id="s", run_id="r")
            assert isinstance(e, LifecycleEvent)
            assert not isinstance(e, StreamDelta)

    def test_stream_delta_not_lifecycle(self) -> None:
        """Stream deltas and lifecycle events are distinguishable via isinstance."""
        stream_events = [e for e in ALL_EVENTS if isinstance(e, StreamDelta)]
        lifecycle_events = [e for e in ALL_EVENTS if isinstance(e, LifecycleEvent)]
        assert len(stream_events) == 3
        assert len(lifecycle_events) == len(ALL_EVENTS) - 3
        # Mutually exclusive
        for e in stream_events:
            assert e not in lifecycle_events


# ---------------------------------------------------------------------------
# Test: Span — the universal phase-boundary event
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSpan:
    def test_round_trip_for_each_subject(self) -> None:
        """Every SpanSubject + SpanPhase combination round-trips cleanly."""
        for subject in SpanSubject:
            for phase in SpanPhase:
                e = Span(
                    timestamp=1.0,
                    sequence=0,
                    session_id="s",
                    run_id="r",
                    subject=subject,
                    phase=phase,
                    span_id=f"id-{subject.value}-{phase.value}",
                    name=f"{subject.value}.{phase.value}",
                    attributes={"k": "v"},
                )
                restored = deserialize_event_json(serialize_event_json(e))
                assert isinstance(restored, Span)
                assert restored.subject == subject
                assert restored.phase == phase
                assert restored.span_id == e.span_id
                assert restored.attributes == {"k": "v"}

    def test_complete_carries_duration(self) -> None:
        e = Span(
            timestamp=1.0,
            sequence=0,
            session_id="s",
            run_id="r",
            subject=SpanSubject.TOOL_CALL,
            phase=SpanPhase.COMPLETE,
            span_id="x",
            duration_ms=250.5,
        )
        restored = deserialize_event_json(serialize_event_json(e))
        assert isinstance(restored, Span)
        assert restored.duration_ms == 250.5


# ---------------------------------------------------------------------------
# Test: EventEmitter
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestEventEmitter:
    def test_sequence_monotonic(self) -> None:
        """EventEmitter produces monotonically increasing sequence numbers."""
        emitter = EventEmitter(session_id="s", run_id="r")
        events = [
            emitter.span_start(SpanSubject.TURN, name="turn.1", attributes={"turn_number": 1}),
            emitter.emit(IntentClassified, intent="respond", confidence=0.9, reasoning="simple"),
            emitter.emit(TextDelta, content="Hello"),
            emitter.emit(TurnSummary, text="Hello", intent="respond"),
        ]
        sequences = [e.sequence for e in events]
        assert sequences == [0, 1, 2, 3]

    def test_base_fields_filled(self) -> None:
        emitter = EventEmitter(session_id="test-sess", run_id="test-run")
        event = emitter.emit(TextDelta, content="hi")
        assert event.session_id == "test-sess"
        assert event.run_id == "test-run"
        assert event.agent_id is None
        assert event.timestamp > 0

    def test_agent_id_propagated(self) -> None:
        emitter = EventEmitter(session_id="s", run_id="r", agent_id="sub-agent-1")
        event = emitter.emit(TextDelta, content="hi")
        assert event.agent_id == "sub-agent-1"

    def test_agent_id_override(self) -> None:
        """Per-event agent_id overrides the emitter default."""
        emitter = EventEmitter(session_id="s", run_id="r", agent_id="default")
        event = emitter.span_start(
            SpanSubject.SUBAGENT,
            attributes={"subagent_name": "writer", "task": "draft"},
        )
        # agent_id forwarding for span_start uses the emitter default.
        assert event.agent_id == "default"

    def test_sequence_property(self) -> None:
        emitter = EventEmitter(session_id="s", run_id="r")
        assert emitter.sequence == 0
        emitter.emit(TextDelta, content="a")
        assert emitter.sequence == 1
        emitter.emit(TextDelta, content="b")
        assert emitter.sequence == 2

    def test_span_complete_matches_start(self) -> None:
        """span_complete with span_id from span_start preserves identity."""
        emitter = EventEmitter(session_id="s", run_id="r")
        start = emitter.span_start(
            SpanSubject.TOOL_CALL,
            name="tool.x",
            attributes={"tool_name": "x", "call_id": "c1"},
        )
        complete = emitter.span_complete(
            SpanSubject.TOOL_CALL,
            span_id=start.span_id,
            name="tool.x",
            duration_ms=10.0,
            attributes={"tool_name": "x", "call_id": "c1", "is_error": False},
        )
        assert complete.span_id == start.span_id
        assert complete.subject == SpanSubject.TOOL_CALL
        assert complete.phase == SpanPhase.COMPLETE


# ---------------------------------------------------------------------------
# Test: error handling
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestErrorHandling:
    def test_missing_type_field(self) -> None:
        """Missing 'type' field raises EventDeserializationError with guidance."""
        with pytest.raises(EventDeserializationError, match="missing 'type' field"):
            deserialize_event({"timestamp": 1.0, "sequence": 0})

    def test_unknown_type(self) -> None:
        """Unknown event type raises EventDeserializationError with valid types."""
        with pytest.raises(EventDeserializationError, match="Unknown event type"):
            deserialize_event({"type": "nonexistent_event"})

    def test_unknown_type_lists_valid(self) -> None:
        """Error message for unknown type includes the list of valid types."""
        with pytest.raises(EventDeserializationError, match="span"):
            deserialize_event({"type": "nonexistent_event"})

    def test_missing_required_fields(self) -> None:
        """Missing required fields raises EventDeserializationError with field names."""
        with pytest.raises(EventDeserializationError, match="Required fields"):
            deserialize_event({"type": "span"})

    def test_forward_compat_ignores_unknown_fields(self) -> None:
        """Unknown fields in the dict are silently ignored."""
        data = {
            "type": "text_delta",
            "timestamp": 1.0,
            "sequence": 0,
            "session_id": "s",
            "run_id": "r",
            "content": "hi",
            "future_field": "should be ignored",
        }
        event = deserialize_event(data)
        assert isinstance(event, TextDelta)
        assert event.content == "hi"

    def test_invalid_json_string(self) -> None:
        """Invalid JSON raises EventDeserializationError with position."""
        with pytest.raises(EventDeserializationError, match="Invalid JSON"):
            deserialize_event_json("{not valid json}")

    def test_json_non_object(self) -> None:
        """JSON array (not object) raises EventDeserializationError."""
        with pytest.raises(EventDeserializationError, match="Expected a JSON object"):
            deserialize_event_json("[1, 2, 3]")

    def test_empty_type_field(self) -> None:
        """Empty string type field raises EventDeserializationError."""
        with pytest.raises(EventDeserializationError, match="missing 'type' field"):
            deserialize_event({"type": "", "timestamp": 1.0})

    def test_error_is_kaos_agent_error(self) -> None:
        """EventDeserializationError inherits from KaosAgentError."""
        from kaos_agents.errors import KaosAgentError

        with pytest.raises(KaosAgentError):
            deserialize_event({"type": "nonexistent"})


# ---------------------------------------------------------------------------
# Test: frozen immutability
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestImmutability:
    @pytest.mark.parametrize("event", ALL_EVENTS, ids=lambda e: type(e).__name__)
    def test_frozen(self, event: KaosEvent) -> None:
        """Events are frozen — attributes cannot be reassigned.

        Pydantic v2 ``model_config = ConfigDict(frozen=True)`` raises
        :class:`pydantic.ValidationError` on ``setattr``; the legacy
        dataclass path raised :class:`AttributeError`. Accept either so
        the test survives the chunk-2 dataclass→pydantic migration.
        """
        from pydantic import ValidationError

        with pytest.raises((AttributeError, ValidationError)):
            setattr(event, "sequence", 999)  # noqa: B010


# ---------------------------------------------------------------------------
# Test: nested dataclass serialization
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestNestedSerialization:
    def test_tool_call_summary_round_trip(self) -> None:
        event = TurnSummary(
            timestamp=1.0,
            sequence=0,
            session_id="s",
            run_id="r",
            text="Done",
            intent="tool_use",
            tool_calls=(
                ToolCallSummary(tool_name="tool-a", call_id="c1", is_error=False, duration_ms=100),
                ToolCallSummary(tool_name="tool-b", call_id="c2", is_error=True, duration_ms=50),
            ),
        )
        data = serialize_event(event)
        restored = deserialize_event(data)
        assert restored == event
        assert isinstance(restored, TurnSummary)
        assert len(restored.tool_calls) == 2
        assert restored.tool_calls[0].tool_name == "tool-a"
        assert restored.tool_calls[1].is_error is True

    def test_plan_step_summary_round_trip(self) -> None:
        event = PlanProposed(
            timestamp=1.0,
            sequence=0,
            session_id="s",
            run_id="r",
            steps=(
                PlanStepSummary(step_id="s1", description="Step 1", tool_name="tool-x"),
                PlanStepSummary(step_id="s2", description="Step 2"),
            ),
            strategy="decompose",
        )
        data = serialize_event(event)
        restored = deserialize_event(data)
        assert restored == event
        assert isinstance(restored, PlanProposed)
        assert len(restored.steps) == 2
        assert restored.steps[0].tool_name == "tool-x"
        assert restored.steps[1].tool_name is None

    def test_span_attributes_round_trip(self) -> None:
        """Span.attributes survives JSON serialization (dict of arbitrary values)."""
        event = Span(
            timestamp=1.0,
            sequence=0,
            session_id="s",
            run_id="r",
            subject=SpanSubject.TOOL_CALL,
            phase=SpanPhase.START,
            span_id="abc",
            name="tool.kaos-source-fr-search",
            attributes={
                "tool_name": "kaos-source-fr-search",
                "call_id": "tc_01",
                "arguments": [["query", "EPA"], ["limit", "10"]],
            },
        )
        json_str = serialize_event_json(event)
        restored = deserialize_event_json(json_str)
        assert restored == event
        assert isinstance(restored, Span)
        assert restored.attributes["tool_name"] == "kaos-source-fr-search"


# ---------------------------------------------------------------------------
# Benchmark: serialization throughput
# ---------------------------------------------------------------------------


@pytest.mark.benchmark(group="events")
class TestSerializationBenchmark:
    """Benchmark serialize/deserialize to ensure events are fast enough for streaming.

    Target: serialize + deserialize a typical event in < 50us (20k events/sec).
    This matters because a streaming agent may emit hundreds of events per turn.
    """

    def test_serialize_simple_event(self, benchmark: Any) -> None:
        """Benchmark serializing a TextDelta (simplest event)."""
        event = TextDelta(
            timestamp=1.0, sequence=42, session_id="bench", run_id="run-1", content="Hello world"
        )
        benchmark(serialize_event, event)

    def test_serialize_complex_event(self, benchmark: Any) -> None:
        """Benchmark serializing a TurnSummary (most complex event)."""
        event = TurnSummary(
            timestamp=1.0,
            sequence=42,
            session_id="bench",
            run_id="run-1",
            text="Full response text with some content.",
            intent="tool_use",
            tool_calls=(
                ToolCallSummary(
                    tool_name="kaos-source-fr-search", call_id="tc_01", duration_ms=500
                ),
                ToolCallSummary(
                    tool_name="kaos-web-get-markdown", call_id="tc_02", duration_ms=300
                ),
            ),
            tokens_used=1500,
            cost_usd=0.003,
        )
        benchmark(serialize_event, event)

    def test_json_round_trip(self, benchmark: Any) -> None:
        """Benchmark full JSON serialize + deserialize cycle for a Span."""
        event = Span(
            timestamp=1.0,
            sequence=7,
            session_id="bench",
            run_id="run-1",
            subject=SpanSubject.TOOL_CALL,
            phase=SpanPhase.COMPLETE,
            span_id="abc123",
            name="tool.kaos-source-fr-search",
            duration_ms=1200.5,
            attributes={
                "tool_name": "kaos-source-fr-search",
                "call_id": "tc_01",
                "result_summary": "Found 3 results",
                "is_error": False,
            },
        )

        def round_trip() -> None:
            json_str = serialize_event_json(event)
            deserialize_event_json(json_str)

        benchmark(round_trip)
