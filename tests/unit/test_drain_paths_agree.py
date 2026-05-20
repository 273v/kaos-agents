"""Assert the event → :class:`AgentResponse` drain paths agree.

KC17-P1-4 / PA14 (issue #166). The codebase historically carried
two parallel drain implementations, with an unenforced "must agree"
invariant called out in :mod:`kaos_agents.runtime.runner`:

1. ``Runner.turn`` — :mod:`kaos_agents.runtime.runner` (inline drain
   plus reconstruction of :class:`AgentResponse`).
2. ``events_to_response`` — :mod:`kaos_agents.runtime.events_to_response`
   (the standalone helper consumed by ``KaosAgent.turn``).

PA14 consolidates them: ``Runner.turn`` now collects events into a
list and delegates to :func:`events_to_response`. There is one
canonical drain. This test still exercises all three entry points
(``Runner.turn``, ``KaosAgent.turn``, and the helper directly) so a
future change that breaks the delegation fails loudly.

# Resolved divergences (D1 / D2 / D3)

Pre-PA14 the two paths disagreed on three default-shaped axes. The
PA14 consolidation aligned all three on the canonical
``Runner.turn`` behavior, and the corresponding tests below are now
asserted to PASS rather than xfail:

- **D1 ``RunError`` metadata**: when a :class:`RunError` event
  fires, ``metadata.error_type`` and ``metadata.error_message`` are
  populated. (Helper now mirrors Runner.turn.)
- **D2 default ``IntentResult.reasoning``**: when no
  :class:`IntentClassified` fires, ``intent.reasoning`` is the
  descriptive sentinel
  ``"no IntentClassified event (run aborted early or errored)"``.
- **D3 empty ``TurnSummary.text`` fallback**: when a
  :class:`TurnSummary` event fires with ``text=""``, the drain
  falls back to concatenated :class:`TextDelta` content. Matters
  for partial / errored turns that emit TextDeltas before an
  early TurnSummary.

The convergent ``EQUIVALENT_STREAMS`` corpus covers the happy paths
where all three drains agree byte-for-byte; the per-divergence
tests below pin the canonical behavior so any future regression
fails loudly.
"""

from __future__ import annotations

from typing import Any

import pytest

from kaos_agents.base.agent import KaosAgent
from kaos_agents.base.event import KaosEvent
from kaos_agents.config import Agent
from kaos_agents.events import (
    EventEmitter,
    IntentClassified,
    RunError,
    SpanSubject,
    TextDelta,
    TurnSummary,
)
from kaos_agents.runtime.events_to_response import events_to_response
from kaos_agents.runtime.runner import Runner
from kaos_agents.types import AgentResponse
from kaos_agents.types.metadata import AgentMetadata

# --------------------------------------------------------------------------
# Test doubles for the three drain entry points
# --------------------------------------------------------------------------


class _ScriptedAgent(KaosAgent):
    """Concrete :class:`KaosAgent` whose ``run`` yields a fixed event list.

    Used to drive path 2 (the ABC default ``turn``). The constructor
    captures the scripted event list; ``run`` simply yields each event.
    """

    def __init__(self, events: list[KaosEvent]) -> None:
        self._events = events

    @classmethod
    def metadata(cls) -> AgentMetadata:
        return AgentMetadata(name="scripted-agent", description="test double")

    async def run(self, message: str, session_id: str, **_kwargs):  # type: ignore[override]
        for ev in self._events:
            yield ev


def _build_scripted_runner(scripted: list[KaosEvent]) -> Runner:
    """Return a :class:`Runner` whose ``run`` yields ``scripted``.

    The base ``Runner.turn`` body delegates to ``self.run(...)``. By
    constructing a fresh subclass that overrides ``run`` to yield our
    scripted events we exercise the real ``Runner.turn`` drain logic
    without spinning up an internal agent, runtime, VFS, or hook
    dispatcher. A fresh subclass per call lets the override close
    over ``scripted`` instead of needing to mutate ``Runner.__slots__``.
    """

    class _Inline(Runner):
        async def run(self, message: str, session_id: str, **_kwargs):  # type: ignore[override]
            for ev in scripted:
                yield ev

    return _Inline(Agent())


# --------------------------------------------------------------------------
# Drain invokers — one per path
# --------------------------------------------------------------------------


async def _drain_runner_turn(events: list[KaosEvent], session_id: str) -> AgentResponse:
    """Path A: invoke ``Runner.turn`` with a scripted ``run`` stream."""
    runner = _build_scripted_runner(events)
    return await runner.turn("test-message", session_id)


async def _drain_agent_turn(events: list[KaosEvent], session_id: str) -> AgentResponse:
    """Path B: invoke ``KaosAgent.turn`` (the ABC default)."""
    agent = _ScriptedAgent(events)
    return await agent.turn("test-message", session_id)


def _drain_helper(events: list[KaosEvent], session_id: str) -> AgentResponse:
    """Path C: invoke :func:`events_to_response` directly."""
    return events_to_response(events, session_id)


# --------------------------------------------------------------------------
# Normalization for cross-path comparison
# --------------------------------------------------------------------------


def _normalize(response: AgentResponse) -> dict[str, Any]:
    """Project :class:`AgentResponse` to a dict that excludes
    nondeterministic and divergence-irrelevant fields.

    Fields kept (the contract we care about):
        - text
        - intent.intent (the enum value)
        - intent.confidence
        - tool_calls: tuple of (tool_name, is_error, result_summary)
          — `arguments` are dropped at the wire boundary by all
          three paths (only the START span carries them) so they
          shouldn't drift between drains
        - turn_number
        - tokens_used
        - total_tokens
        - cost_usd
        - metadata.session_id (the one universal metadata key)

    Fields explicitly dropped:
        - intent.reasoning — D2 divergence, asserted separately
        - intent.usage — never populated by any drain
        - artifacts — set to ``()`` by all three drains, no info
        - metadata.error_type / metadata.error_message — D1
          divergence, asserted separately
        - tool_calls' nondeterministic timing/usage fields
          (started_at, ended_at, duration_ms, cost_usd,
          input_tokens, output_tokens) — these aren't on the
          COMPLETE span's attributes and the three drains all
          default them to zero / None, so they don't drift either,
          but we project away to keep this future-proof
    """
    metadata = dict(response.metadata)
    return {
        "text": response.text,
        "intent_type": response.intent.intent.value,
        "intent_confidence": response.intent.confidence,
        "tool_calls": tuple(
            (tc.tool_name, tc.is_error, tc.result_summary) for tc in response.tool_calls
        ),
        "turn_number": response.turn_number,
        "tokens_used": response.tokens_used,
        "total_tokens": response.total_tokens,
        "cost_usd": response.cost_usd,
        "session_id": metadata.get("session_id"),
    }


# --------------------------------------------------------------------------
# Synthetic event-stream corpus
# --------------------------------------------------------------------------


SESSION_ID = "drain-paths-test"


def _emitter() -> EventEmitter:
    return EventEmitter(session_id=SESSION_ID, run_id="run-1")


def _stream_minimal_turn_summary() -> list[KaosEvent]:
    """A TurnSummary with text + tokens + cost. No intent classification.

    Forces the "default IntentResult" path (no IntentClassified seen).
    Asserts D2 (default reasoning string) under xfail.
    """
    em = _emitter()
    turn_start = em.span_start(SpanSubject.TURN, name="turn.1", attributes={"turn_number": 1})
    turn_complete = em.span_complete(
        SpanSubject.TURN, span_id=turn_start.span_id, name="turn.1", duration_ms=100.0
    )
    summary = em.emit(
        TurnSummary,
        text="hello world",
        intent="respond",
        tool_calls=(),
        tokens_used=42,
        cost_usd=0.0123,
    )
    return [turn_start, turn_complete, summary]


def _stream_text_deltas_only() -> list[KaosEvent]:
    """TextDeltas with no TurnSummary. Tests the fallback path.

    Both paths should concatenate the deltas. No TurnSummary means
    tokens_used=0 and cost_usd=0.0 on both.
    """
    em = _emitter()
    return [
        em.emit(TextDelta, content="part 1 "),
        em.emit(TextDelta, content="part 2 "),
        em.emit(TextDelta, content="part 3"),
    ]


def _stream_intent_then_summary() -> list[KaosEvent]:
    """IntentClassified followed by TurnSummary — happy path."""
    em = _emitter()
    turn_start = em.span_start(SpanSubject.TURN, name="turn.2", attributes={"turn_number": 2})
    intent = em.emit(
        IntentClassified,
        intent="tool_use",
        confidence=0.87,
        reasoning="user asked to fetch a URL",
    )
    turn_complete = em.span_complete(
        SpanSubject.TURN, span_id=turn_start.span_id, name="turn.2", duration_ms=250.0
    )
    summary = em.emit(
        TurnSummary,
        text="Fetched the URL.",
        intent="tool_use",
        tool_calls=(),
        tokens_used=120,
        cost_usd=0.005,
    )
    return [turn_start, intent, turn_complete, summary]


def _stream_with_tool_call() -> list[KaosEvent]:
    """Turn with one tool-call complete span. tool_calls list non-empty."""
    em = _emitter()
    turn_start = em.span_start(SpanSubject.TURN, name="turn.3", attributes={"turn_number": 3})
    intent = em.emit(
        IntentClassified,
        intent="tool_use",
        confidence=0.95,
        reasoning="needs tool",
    )
    tc_start = em.span_start(
        SpanSubject.TOOL_CALL,
        name="tool.kaos-web-fetch",
        attributes={
            "tool_name": "kaos-web-fetch",
            "call_id": "tc_1",
            "arguments": (("url", "https://example.com"),),
        },
    )
    tc_complete = em.span_complete(
        SpanSubject.TOOL_CALL,
        span_id=tc_start.span_id,
        name="tool.kaos-web-fetch",
        duration_ms=50.0,
        attributes={
            "tool_name": "kaos-web-fetch",
            "call_id": "tc_1",
            "result_summary": "HTTP 200 (1024 bytes)",
            "is_error": False,
        },
    )
    turn_complete = em.span_complete(
        SpanSubject.TURN, span_id=turn_start.span_id, name="turn.3", duration_ms=500.0
    )
    summary = em.emit(
        TurnSummary,
        text="Fetched example.com (1024 bytes).",
        intent="tool_use",
        tool_calls=(),
        tokens_used=200,
        cost_usd=0.01,
    )
    return [turn_start, intent, tc_start, tc_complete, turn_complete, summary]


def _stream_multi_tool_calls() -> list[KaosEvent]:
    """Two tool calls, one success one error."""
    em = _emitter()
    turn_start = em.span_start(SpanSubject.TURN, name="turn.4", attributes={"turn_number": 4})
    intent = em.emit(
        IntentClassified,
        intent="tool_use",
        confidence=0.9,
        reasoning="multi-tool",
    )

    tc1_start = em.span_start(
        SpanSubject.TOOL_CALL,
        name="tool.kaos-source-fetch",
        attributes={"tool_name": "kaos-source-fetch", "call_id": "tc_a"},
    )
    tc1_complete = em.span_complete(
        SpanSubject.TOOL_CALL,
        span_id=tc1_start.span_id,
        attributes={
            "tool_name": "kaos-source-fetch",
            "call_id": "tc_a",
            "result_summary": "OK",
            "is_error": False,
        },
    )

    tc2_start = em.span_start(
        SpanSubject.TOOL_CALL,
        name="tool.kaos-web-fetch",
        attributes={"tool_name": "kaos-web-fetch", "call_id": "tc_b"},
    )
    tc2_complete = em.span_complete(
        SpanSubject.TOOL_CALL,
        span_id=tc2_start.span_id,
        attributes={
            "tool_name": "kaos-web-fetch",
            "call_id": "tc_b",
            "result_summary": "timeout after 30s",
            "is_error": True,
        },
    )

    turn_complete = em.span_complete(
        SpanSubject.TURN, span_id=turn_start.span_id, duration_ms=900.0
    )
    summary = em.emit(
        TurnSummary,
        text="Mixed results.",
        intent="tool_use",
        tool_calls=(),
        tokens_used=300,
        cost_usd=0.02,
    )
    return [
        turn_start,
        intent,
        tc1_start,
        tc1_complete,
        tc2_start,
        tc2_complete,
        turn_complete,
        summary,
    ]


def _stream_text_delta_then_turn_summary() -> list[KaosEvent]:
    """TextDeltas during the turn + final TurnSummary.

    All three drains should prefer the TurnSummary text over the
    concatenated deltas (the standard happy-path streaming case).
    """
    em = _emitter()
    turn_start = em.span_start(SpanSubject.TURN, name="turn.5", attributes={"turn_number": 5})
    intent = em.emit(IntentClassified, intent="respond", confidence=0.99, reasoning="chat")
    delta_a = em.emit(TextDelta, content="Hel")
    delta_b = em.emit(TextDelta, content="lo")
    turn_complete = em.span_complete(SpanSubject.TURN, span_id=turn_start.span_id)
    # TurnSummary.text wins over the streamed deltas.
    summary = em.emit(
        TurnSummary,
        text="Hello, world!",
        intent="respond",
        tokens_used=10,
        cost_usd=0.0001,
    )
    return [turn_start, intent, delta_a, delta_b, turn_complete, summary]


def _stream_empty() -> list[KaosEvent]:
    """Empty stream — every drain falls back to all defaults."""
    return []


def _stream_only_intent_no_summary() -> list[KaosEvent]:
    """IntentClassified fires but no TurnSummary (e.g. early pause)."""
    em = _emitter()
    intent = em.emit(IntentClassified, intent="plan", confidence=0.7, reasoning="needs planning")
    return [intent]


def _stream_turn_number_only() -> list[KaosEvent]:
    """Just a Span(TURN, START) — confirms turn_number is read from it."""
    em = _emitter()
    return [
        em.span_start(SpanSubject.TURN, name="turn.7", attributes={"turn_number": 7}),
    ]


EQUIVALENT_STREAMS: list[tuple[str, list[KaosEvent]]] = [
    ("text_deltas_only", _stream_text_deltas_only()),
    ("intent_then_summary", _stream_intent_then_summary()),
    ("with_tool_call", _stream_with_tool_call()),
    ("multi_tool_calls", _stream_multi_tool_calls()),
    ("text_delta_then_turn_summary", _stream_text_delta_then_turn_summary()),
    ("empty", _stream_empty()),
    ("only_intent_no_summary", _stream_only_intent_no_summary()),
    ("turn_number_only", _stream_turn_number_only()),
]


# --------------------------------------------------------------------------
# Convergent contract: the three drains agree on the projected dict
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "label,events",
    EQUIVALENT_STREAMS,
    ids=[label for label, _ in EQUIVALENT_STREAMS],
)
async def test_three_drains_agree_on_normalized_response(
    label: str, events: list[KaosEvent]
) -> None:
    """All three drain paths produce the same normalized
    :class:`AgentResponse` for the convergent corpus.

    Normalized = projected through :func:`_normalize`, which drops
    nondeterministic fields and the three known divergence axes
    documented in the module docstring (those have their own xfail
    tests below).
    """
    response_a = await _drain_runner_turn(events, SESSION_ID)
    response_b = await _drain_agent_turn(events, SESSION_ID)
    response_c = _drain_helper(events, SESSION_ID)

    norm_a = _normalize(response_a)
    norm_b = _normalize(response_b)
    norm_c = _normalize(response_c)

    assert norm_a == norm_b, (
        f"Runner.turn vs KaosAgent.turn disagreed on '{label}':\n"
        f"  Runner.turn   : {norm_a}\n"
        f"  KaosAgent.turn: {norm_b}"
    )
    assert norm_b == norm_c, (
        f"KaosAgent.turn vs events_to_response disagreed on '{label}':\n"
        f"  KaosAgent.turn      : {norm_b}\n"
        f"  events_to_response  : {norm_c}"
    )


# --------------------------------------------------------------------------
# Canonical-behavior assertions for the consolidated drain. Each was
# an xfail divergence pre-PA14; now they assert the agreed canonical
# behavior across all three drain entry points.
# --------------------------------------------------------------------------


async def test_d1_run_error_surfaces_in_metadata() -> None:
    """PA14 canonical: a :class:`RunError` populates ``metadata.error_*``.

    All three drain entry points surface ``error_type`` and
    ``error_message`` so callers can distinguish abnormal turns
    from clean ones without walking the event stream.
    """
    em = _emitter()
    err = em.emit(
        RunError,
        error_type="ToolTimeout",
        message="tool fetch timed out",
        recovery_hint="retry with a longer timeout",
    )
    events = [err]

    response_a = await _drain_runner_turn(events, SESSION_ID)
    response_b = await _drain_agent_turn(events, SESSION_ID)
    response_c = _drain_helper(events, SESSION_ID)

    for label, resp in (
        ("Runner.turn", response_a),
        ("KaosAgent.turn", response_b),
        ("events_to_response", response_c),
    ):
        meta = dict(resp.metadata)
        assert meta.get("error_type") == "ToolTimeout", (
            f"{label}: expected error_type=ToolTimeout, got {meta.get('error_type')!r}"
        )
        assert meta.get("error_message") == "tool fetch timed out", (
            f"{label}: expected error_message='tool fetch timed out', "
            f"got {meta.get('error_message')!r}"
        )


async def test_d2_default_intent_reasoning_is_canonical() -> None:
    """PA14 canonical: when no :class:`IntentClassified` fires, the
    drain populates ``intent.reasoning`` with the descriptive sentinel
    so partial / errored turns are distinguishable from "classifier
    returned an empty reasoning string"."""
    events = _stream_minimal_turn_summary()
    expected = "no IntentClassified event (run aborted early or errored)"
    response_a = await _drain_runner_turn(events, SESSION_ID)
    response_b = await _drain_agent_turn(events, SESSION_ID)
    response_c = _drain_helper(events, SESSION_ID)
    for label, resp in (
        ("Runner.turn", response_a),
        ("KaosAgent.turn", response_b),
        ("events_to_response", response_c),
    ):
        assert resp.intent.reasoning == expected, (
            f"{label}: expected intent.reasoning={expected!r}, got {resp.intent.reasoning!r}"
        )


async def test_d3_empty_turn_summary_text_falls_back_to_deltas() -> None:
    """PA14 canonical: when a :class:`TurnSummary` fires with empty
    text, the drain falls back to the concatenated :class:`TextDelta`
    content (preserves partial / errored output).
    """
    em = _emitter()
    turn_start = em.span_start(SpanSubject.TURN, name="turn.x", attributes={"turn_number": 1})
    delta_a = em.emit(TextDelta, content="streamed ")
    delta_b = em.emit(TextDelta, content="content")
    turn_complete = em.span_complete(SpanSubject.TURN, span_id=turn_start.span_id)
    # Note: text="" — the fallback trigger.
    summary = em.emit(
        TurnSummary,
        text="",
        intent="respond",
        tokens_used=5,
        cost_usd=0.0,
    )
    events = [turn_start, delta_a, delta_b, turn_complete, summary]

    response_a = await _drain_runner_turn(events, SESSION_ID)
    response_b = await _drain_agent_turn(events, SESSION_ID)
    response_c = _drain_helper(events, SESSION_ID)
    for label, resp in (
        ("Runner.turn", response_a),
        ("KaosAgent.turn", response_b),
        ("events_to_response", response_c),
    ):
        assert resp.text == "streamed content", (
            f"{label}: expected text='streamed content' (delta fallback), got {resp.text!r}"
        )
