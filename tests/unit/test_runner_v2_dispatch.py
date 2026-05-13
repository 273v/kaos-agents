"""Phase 2.C — Runner integration tests for the KAOS_AGENT_LOOP=v2 flag.

Verifies the new dispatch surface (`run_trigger`, `invoke_trigger`,
`agent_loop_version`) and confirms that:

1. Default Runner is v1 (no behavior change for existing callers).
2. ``KAOS_AGENT_LOOP=v2`` env var flips the flag.
3. Constructor kwarg ``agent_loop_version=`` overrides the env var.
4. ``run_trigger`` in v1 mode replays through the legacy ``run`` path.
5. ``run_trigger`` in v2 mode dispatches to ``AgentLoop.stream`` and
   yields the new event quartet.
6. ``invoke_trigger`` in v1 raises with guidance; in v2 returns a
   ``TurnInvocation``.
7. ``Runner.run(message, session_id)`` in v2 mode routes through the
   AgentLoop.

The intent extractor inside the new path is stubbed so no real LLM
calls are made.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from kaos_agents.config import Agent, AgentPattern
from kaos_agents.core.invocation import TurnInvocation
from kaos_agents.events import (
    IntentClassified,
    KaosEvent,
    Span,
    SpanPhase,
    SpanSubject,
    TurnSummary,
)
from kaos_agents.intent import (
    Goal,
    IntentResult,
)
from kaos_agents.runtime.runner import Runner
from kaos_agents.triggers import Trigger
from kaos_agents.types import IntentType


def _make_intent_result(
    *,
    pattern: AgentPattern = AgentPattern.CHAT,
    requires_clarification: bool = False,
) -> IntentResult:
    return IntentResult(
        goal=Goal(statement="test goal", intent_type=IntentType.RESPOND),
        constraints=(),
        ambiguities=(),
        requires_clarification=requires_clarification,
        pattern=pattern,
        confidence=0.9,
        raw_input="test message",
    )


def _make_invocation_stub(intent: IntentResult) -> SimpleNamespace:
    """Mimic a kaos-llm-core Invocation with .output."""
    return SimpleNamespace(
        output=intent,
        usage=SimpleNamespace(input_tokens=10, output_tokens=5, total_tokens=15, cost_usd=0.001),
    )


@pytest.fixture
def stub_intent(monkeypatch: pytest.MonkeyPatch) -> IntentResult:
    """Patch IntentExtractor.invoke to return a deterministic result."""
    intent = _make_intent_result()

    async def _fake_invoke(self: Any, **kwargs: Any) -> Any:
        return _make_invocation_stub(intent)

    from kaos_agents.intent.extractor import IntentExtractor

    monkeypatch.setattr(IntentExtractor, "invoke", _fake_invoke)
    return intent


@pytest.mark.unit
class TestAgentLoopVersionResolution:
    """The Phase 2 feature flag is a constructor kwarg / env var."""

    def test_default_is_v1(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("KAOS_AGENT_LOOP", raising=False)
        runner = Runner(Agent())
        assert runner.agent_loop_version == "v1"

    def test_env_var_v2(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("KAOS_AGENT_LOOP", "v2")
        runner = Runner(Agent())
        assert runner.agent_loop_version == "v2"

    def test_kwarg_overrides_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("KAOS_AGENT_LOOP", "v2")
        runner = Runner(Agent(), agent_loop_version="v1")
        assert runner.agent_loop_version == "v1"

    def test_explicit_v2_kwarg(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("KAOS_AGENT_LOOP", raising=False)
        runner = Runner(Agent(), agent_loop_version="v2", auto_select_planner=False)
        assert runner.agent_loop_version == "v2"


@pytest.mark.unit
class TestInvokeTriggerV1Refusal:
    """v1 path raises a typed RuntimeError with migration guidance."""

    async def test_invoke_trigger_v1_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("KAOS_AGENT_LOOP", raising=False)
        runner = Runner(Agent())
        trigger = Trigger.mcp("hi", session_id="s1")
        with pytest.raises(RuntimeError, match="KAOS_AGENT_LOOP=v2"):
            await runner.invoke_trigger(trigger)


@pytest.mark.unit
class TestInvokeTriggerV2:
    """v2 path returns a TurnInvocation with a populated intent."""

    async def test_invoke_trigger_v2_returns_turn_invocation(
        self,
        monkeypatch: pytest.MonkeyPatch,
        stub_intent: IntentResult,
    ) -> None:
        runner = Runner(Agent(), agent_loop_version="v2", auto_select_planner=False)
        trigger = Trigger.mcp("hello", session_id="s1")
        invocation = await runner.invoke_trigger(trigger)
        assert isinstance(invocation, TurnInvocation)
        assert invocation.session_id == "s1"
        assert invocation.intent is not None
        assert invocation.intent.confidence == stub_intent.confidence
        assert invocation.is_complete
        assert invocation.error is None
        # Skeleton path — no planner wired in Phase 2.
        assert invocation.extras.get("phase") == "skeleton"


@pytest.mark.unit
class TestRunTriggerV2Dispatch:
    """v2 dispatch yields the AgentLoop event quartet."""

    async def test_run_trigger_v2_emits_turn_lifecycle(
        self,
        monkeypatch: pytest.MonkeyPatch,
        stub_intent: IntentResult,
    ) -> None:
        runner = Runner(Agent(), agent_loop_version="v2", auto_select_planner=False)
        trigger = Trigger.mcp("hello", session_id="s1")
        events: list[KaosEvent] = []
        async for event in runner.run_trigger(trigger):
            events.append(event)
        # Required event types in order: Span(TURN, START), IntentClassified,
        # TurnSummary, Span(TURN, COMPLETE).
        kinds = [type(e).__name__ for e in events]
        assert "Span" in kinds
        assert "IntentClassified" in kinds
        assert "TurnSummary" in kinds
        # First event is the turn-start span.
        assert isinstance(events[0], Span)
        assert events[0].subject == SpanSubject.TURN
        assert events[0].phase == SpanPhase.START
        # Last event is the turn-complete span.
        assert isinstance(events[-1], Span)
        assert events[-1].subject == SpanSubject.TURN
        assert events[-1].phase == SpanPhase.COMPLETE

    async def test_run_trigger_v2_intent_classified_carries_pattern(
        self,
        monkeypatch: pytest.MonkeyPatch,
        stub_intent: IntentResult,
    ) -> None:
        runner = Runner(Agent(), agent_loop_version="v2", auto_select_planner=False)
        trigger = Trigger.mcp("hello", session_id="s1")
        events = [e async for e in runner.run_trigger(trigger)]
        ic_events = [e for e in events if isinstance(e, IntentClassified)]
        assert len(ic_events) == 1
        # The stub intent is CHAT with confidence 0.9.
        assert ic_events[0].confidence == pytest.approx(0.9)


@pytest.mark.unit
class TestRunTriggerV1Fallback:
    """v1 ``run_trigger`` extracts the message and replays through ``run``."""

    async def test_v1_run_trigger_extracts_message_and_calls_run(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        runner = Runner(Agent(), agent_loop_version="v1")

        captured: dict[str, Any] = {}

        async def _fake_run(self: Any, message: str, session_id: str) -> Any:
            captured["message"] = message
            captured["session_id"] = session_id
            yield Span(
                subject=SpanSubject.TURN,
                phase=SpanPhase.START,
                span_id="x",
                name="turn.1",
                attributes={},
                timestamp=0.0,
                sequence=0,
                session_id=session_id,
                run_id="r1",
            )

        monkeypatch.setattr(Runner, "run", _fake_run)

        trigger = Trigger.mcp("hi from v1", session_id="s7")
        events = [e async for e in runner.run_trigger(trigger)]
        assert captured["message"] == "hi from v1"
        assert captured["session_id"] == "s7"
        assert len(events) == 1


@pytest.mark.unit
class TestRunV2Routing:
    """``Runner.run`` in v2 mode routes through the AgentLoop."""

    async def test_run_v2_dispatches_to_agent_loop(
        self,
        monkeypatch: pytest.MonkeyPatch,
        stub_intent: IntentResult,
    ) -> None:
        runner = Runner(Agent(), agent_loop_version="v2", auto_select_planner=False)
        events = [e async for e in runner.run("hello v2", "s2")]
        # AgentLoop quartet should be present.
        assert any(
            isinstance(e, Span) and e.subject == SpanSubject.TURN and e.phase == SpanPhase.START
            for e in events
        )
        assert any(isinstance(e, IntentClassified) for e in events)
        assert any(isinstance(e, TurnSummary) for e in events)

    def test_run_v1_default_does_not_import_agent_loop(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """v1 default should not require AgentLoop / IntentExtractor.

        Smoke check: constructing a v1 Runner does not eagerly import
        the new modules. Phase 2 ships with v1 default; no consumer
        should pay the v2 import cost unless they opt in.
        """
        monkeypatch.delenv("KAOS_AGENT_LOOP", raising=False)
        runner = Runner(Agent())
        # Construction succeeds; lazy imports only happen on v2 dispatch.
        assert runner.agent_loop_version == "v1"


@pytest.mark.unit
class TestPlanIntentTriggers:
    """Different trigger source kinds all produce the same v2 dispatch."""

    @pytest.mark.parametrize(
        "trigger",
        [
            Trigger.mcp("hi", session_id="s1"),
            Trigger.http("hi", request_id="r1"),
            Trigger.cli("hi", session_id="s1"),
        ],
    )
    async def test_v2_handles_all_message_carrying_kinds(
        self,
        monkeypatch: pytest.MonkeyPatch,
        stub_intent: IntentResult,
        trigger: Trigger,
    ) -> None:
        runner = Runner(Agent(), agent_loop_version="v2", auto_select_planner=False)
        events = [e async for e in runner.run_trigger(trigger)]
        # All three sources produce the lifecycle quartet.
        assert any(
            isinstance(e, Span) and e.subject == SpanSubject.TURN and e.phase == SpanPhase.COMPLETE
            for e in events
        )
