"""Tests for Runner execution engine (Phase 2).

Covers:
- Runner.run() yields events (delegates to internal agent)
- Runner.turn() returns AgentResponse (backward compat)
- Runner constructs correct internal agent for each pattern
- Runner.agent property returns the config
- Multiple Runners can share one Agent config
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from kaos_agents.config import Agent, AgentPattern
from kaos_agents.events import (
    IntentClassified,
    KaosEvent,
    Span,
    SpanPhase,
    SpanSubject,
    TurnSummary,
)
from kaos_agents.runtime.runner import Runner
from kaos_agents.types import AgentResponse, IntentResult, IntentType


def _make_runner(pattern: AgentPattern = AgentPattern.CHAT) -> Runner:
    """Create a Runner with in-memory VFS (no runtime)."""
    agent = Agent(pattern=pattern)
    return Runner(agent)


@pytest.mark.unit
class TestRunnerProperties:
    def test_agent_property(self) -> None:
        agent = Agent(name="test-agent", model="anthropic:claude-sonnet-4-6")
        runner = Runner(agent)
        assert runner.agent is agent
        assert runner.agent.name == "test-agent"

    def test_shared_agent_config(self) -> None:
        """Multiple Runners can share the same Agent config."""
        agent = Agent(name="shared")
        runner_a = Runner(agent)
        runner_b = Runner(agent)
        assert runner_a.agent is runner_b.agent


@pytest.mark.unit
class TestRunnerRun:
    @pytest.mark.asyncio
    async def test_run_yields_events(self) -> None:
        """Runner.run() yields KaosEvent objects."""
        runner = _make_runner()
        mock_intent = IntentResult(intent=IntentType.RESPOND, confidence=1.0, reasoning="test")

        # Patch the internal agent's classify and dispatch
        with (
            patch(
                "kaos_agents.runtime.agent.BaseAgent._classify",
                new_callable=AsyncMock,
                return_value=mock_intent,
            ),
            patch(
                "kaos_agents.runtime.agent.BaseAgent._dispatch",
                new_callable=AsyncMock,
                return_value=("Hello!", []),
            ),
        ):
            events: list[KaosEvent] = []
            async for event in runner.run("Hi", "test-session"):
                events.append(event)

        assert len(events) >= 3
        first = events[0]
        last = events[-1]
        assert isinstance(first, Span)
        assert first.subject == SpanSubject.TURN
        assert first.phase == SpanPhase.START
        # Last event is the TurnSummary (which fires after Span(TURN, COMPLETE))
        assert isinstance(last, TurnSummary)

    @pytest.mark.asyncio
    async def test_run_chat_pattern(self) -> None:
        """Runner with CHAT pattern constructs ChatAgent internally."""
        runner = _make_runner(AgentPattern.CHAT)
        mock_intent = IntentResult(intent=IntentType.RESPOND, confidence=1.0, reasoning="test")

        with (
            patch(
                "kaos_agents.runtime.agent.BaseAgent._classify",
                new_callable=AsyncMock,
                return_value=mock_intent,
            ),
            patch(
                "kaos_agents.runtime.agent.BaseAgent._dispatch",
                new_callable=AsyncMock,
                return_value=("Response", []),
            ),
        ):
            events: list[KaosEvent] = []
            async for event in runner.run("Hello", "s1"):
                events.append(event)

        assert any(isinstance(e, IntentClassified) for e in events)


@pytest.mark.unit
class TestRunnerTurn:
    @pytest.mark.asyncio
    async def test_turn_returns_agent_response(self) -> None:
        """Runner.turn() collects events and returns AgentResponse."""
        runner = _make_runner()
        mock_intent = IntentResult(intent=IntentType.RESPOND, confidence=1.0, reasoning="test")

        with (
            patch(
                "kaos_agents.runtime.agent.BaseAgent._classify",
                new_callable=AsyncMock,
                return_value=mock_intent,
            ),
            patch(
                "kaos_agents.runtime.agent.BaseAgent._dispatch",
                new_callable=AsyncMock,
                return_value=("Hello back!", []),
            ),
        ):
            response = await runner.turn("Hello", "test-session")

        assert isinstance(response, AgentResponse)
        assert response.text == "Hello back!"

    @pytest.mark.asyncio
    async def test_turn_plan_pattern(self) -> None:
        """Runner with PLAN pattern works via turn()."""
        agent = Agent(pattern=AgentPattern.PLAN)
        runner = Runner(agent)
        mock_intent = IntentResult(intent=IntentType.RESPOND, confidence=1.0, reasoning="test")

        with (
            patch(
                "kaos_agents.runtime.agent.BaseAgent._classify",
                new_callable=AsyncMock,
                return_value=mock_intent,
            ),
            patch(
                "kaos_agents.runtime.agent.BaseAgent._dispatch",
                new_callable=AsyncMock,
                return_value=("Plan result", []),
            ),
        ):
            response = await runner.turn("Do something complex", "s1")

        assert isinstance(response, AgentResponse)
        assert response.text == "Plan result"

    @pytest.mark.asyncio
    async def test_turn_research_pattern(self) -> None:
        """Runner with RESEARCH pattern works via turn()."""
        agent = Agent(pattern=AgentPattern.RESEARCH)
        runner = Runner(agent)
        mock_intent = IntentResult(intent=IntentType.RESPOND, confidence=1.0, reasoning="test")

        with (
            patch(
                "kaos_agents.runtime.agent.BaseAgent._classify",
                new_callable=AsyncMock,
                return_value=mock_intent,
            ),
            patch(
                "kaos_agents.runtime.agent.BaseAgent._dispatch",
                new_callable=AsyncMock,
                return_value=("Research result", []),
            ),
        ):
            response = await runner.turn("What does the contract say?", "s1")

        assert isinstance(response, AgentResponse)
        assert response.text == "Research result"


# ---------------------------------------------------------------------------
# Gap 9: Runner passes ToolAnnotations from runtime to PermissionPolicy
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRunnerAnnotationLookup:
    def test_lookup_annotations_from_runtime(self) -> None:
        """Runner._lookup_annotations fetches annotations via runtime.tools."""
        from kaos_core import KaosRuntime
        from kaos_core.base.tool import KaosTool
        from kaos_core.types.metadata import (
            ToolAnnotations,
            ToolCapability,
            ToolCategory,
            ToolMetadata,
        )

        class _ReadOnlyTool(KaosTool):
            @property
            def metadata(self) -> ToolMetadata:
                return ToolMetadata(
                    name="kaos-test-read",
                    display_name="Test",
                    description="test",
                    category=ToolCategory.TEXT,
                    capability=ToolCapability.QUERY,
                    module_name="test",
                    version="0.1.0",
                    annotations=ToolAnnotations(readOnlyHint=True),
                    input_schema=[],
                )

            async def execute(self, inputs, context=None):  # type: ignore[no-untyped-def, override]
                from kaos_core.types.results import ToolResult

                return ToolResult.create_text("ok")

        runtime = KaosRuntime()
        runtime.tools.register_tool(_ReadOnlyTool())

        agent = Agent()
        runner = Runner(agent, runtime=runtime)
        annotations = runner._lookup_annotations("kaos-test-read")
        assert annotations is not None
        assert annotations.readOnlyHint is True

    def test_lookup_annotations_missing_returns_none(self) -> None:
        from kaos_core import KaosRuntime

        runtime = KaosRuntime()
        agent = Agent()
        runner = Runner(agent, runtime=runtime)
        assert runner._lookup_annotations("nonexistent") is None

    def test_lookup_annotations_no_runtime(self) -> None:
        agent = Agent()
        runner = Runner(agent)
        assert runner._lookup_annotations("anything") is None

    @pytest.mark.asyncio
    async def test_permission_policy_receives_annotations_via_runner(self) -> None:
        """End-to-end: destructiveHint triggers ASK via annotations lookup."""
        from kaos_core import KaosRuntime
        from kaos_core.base.tool import KaosTool
        from kaos_core.types.metadata import (
            ToolAnnotations,
            ToolCapability,
            ToolCategory,
            ToolMetadata,
        )

        from kaos_agents.events import ToolCallApprovalRequired
        from kaos_agents.hooks import KaosHook
        from kaos_agents.runtime.permissions import PermissionPolicy

        class _DestructiveTool(KaosTool):
            @property
            def metadata(self) -> ToolMetadata:
                return ToolMetadata(
                    name="kaos-test-delete",
                    display_name="Delete",
                    description="destructive",
                    category=ToolCategory.TEXT,
                    capability=ToolCapability.TRANSFORM,
                    module_name="test",
                    version="0.1.0",
                    annotations=ToolAnnotations(destructiveHint=True),
                    input_schema=[],
                )

            async def execute(self, inputs, context=None):  # type: ignore[no-untyped-def, override]
                from kaos_core.types.results import ToolResult

                return ToolResult.create_text("deleted")

        runtime = KaosRuntime()
        runtime.tools.register_tool(_DestructiveTool())

        # Simulate the agent emitting a ToolCallStart for the destructive tool
        class _InjectHook(KaosHook):
            async def on_turn_start(self, event):  # type: ignore[no-untyped-def, override]
                pass

        agent = Agent()
        policy = PermissionPolicy()  # empty rules — rely on annotations
        runner = Runner(agent, runtime=runtime, permission_policy=policy)

        # Stub the internal agent's run() to emit a tool-call START span directly
        async def _fake_run(self_, message, session_id):  # type: ignore[no-untyped-def]
            from kaos_agents.events import EventEmitter

            emitter = EventEmitter(session_id=session_id, run_id="r")
            yield emitter.span_start(
                SpanSubject.TURN,
                name="turn.1",
                attributes={"turn_number": 1},
            )
            yield emitter.span_start(
                SpanSubject.TOOL_CALL,
                name="tool.kaos-test-delete",
                attributes={
                    "tool_name": "kaos-test-delete",
                    "call_id": "tc_1",
                    "arguments": (),
                },
            )

        with patch("kaos_agents.runtime.agent.BaseAgent.run", _fake_run):
            events = []
            async for event in runner.run("delete please", "s-annot"):
                events.append(event)

        # Runner should have paused with ToolCallApprovalRequired
        approvals = [e for e in events if isinstance(e, ToolCallApprovalRequired)]
        assert len(approvals) == 1
        assert approvals[0].tool_name == "kaos-test-delete"
        # The raw tool-call START span should NOT appear (Runner returned
        # before yielding it).
        tool_starts = [
            e
            for e in events
            if isinstance(e, Span)
            and e.subject == SpanSubject.TOOL_CALL
            and e.phase == SpanPhase.START
        ]
        assert len(tool_starts) == 0
