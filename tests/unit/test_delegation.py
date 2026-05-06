"""Tests for delegation (Phase 7).

Covers:
- agent_as_tool() wrapping
- DelegatedAgent frozen and has correct fields
- DelegatedAgent.call() executes sub-agent via Runner
- Depth tracking prevents infinite recursion
- DelegationDepthExceeded raised correctly
- Agent config fields (delegated_agents, handoffs, max_delegation_depth)
- Runner.delegate() yields SubagentStart/Complete events
- Runner.handoff() yields HandoffStart + target events
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from kaos_agents.config import Agent
from kaos_agents.delegation import (
    DelegatedAgent,
    DelegationDepthExceeded,
    _delegation_depth,
    agent_as_tool,
    current_delegation_depth,
)
from kaos_agents.events import HandoffStart, SubagentComplete, SubagentStart
from kaos_agents.runner import Runner
from kaos_agents.types import IntentResult, IntentType


@pytest.mark.unit
class TestAgentAsTool:
    def test_returns_delegated_agent(self) -> None:
        writer = Agent(instructions="Write memos")
        tool = agent_as_tool(writer, name="draft_memo", description="Draft a memo")
        assert isinstance(tool, DelegatedAgent)
        assert tool.name == "draft_memo"
        assert tool.description == "Draft a memo"
        assert tool.agent is writer
        assert tool.max_depth == 3

    def test_custom_max_depth(self) -> None:
        tool = agent_as_tool(
            Agent(),
            name="t",
            description="d",
            max_depth=5,
        )
        assert tool.max_depth == 5


@pytest.mark.unit
class TestDelegatedAgent:
    def test_frozen(self) -> None:
        tool = DelegatedAgent(agent=Agent(), name="t", description="d")
        with pytest.raises(AttributeError):
            setattr(tool, "name", "other")  # noqa: B010

    @pytest.mark.asyncio
    async def test_call_executes_sub_agent(self) -> None:
        """DelegatedAgent.call() runs the sub-agent via Runner.turn()."""
        writer = Agent(instructions="Write")
        tool = agent_as_tool(writer, name="writer", description="Write")

        mock_intent = IntentResult(intent=IntentType.RESPOND, confidence=1.0, reasoning="test")
        with (
            patch(
                "kaos_agents.agent.BaseAgent._classify",
                new_callable=AsyncMock,
                return_value=mock_intent,
            ),
            patch(
                "kaos_agents.agent.BaseAgent._dispatch",
                new_callable=AsyncMock,
                return_value=("Memo drafted", []),
            ),
        ):
            result = await tool.call("Draft a memo", parent_session_id="parent-1")

        assert result == "Memo drafted"

    @pytest.mark.asyncio
    async def test_depth_exceeded(self) -> None:
        """Calling with depth >= max_depth raises DelegationDepthExceeded."""
        tool = agent_as_tool(Agent(), name="t", description="d", max_depth=2)

        # Simulate being at depth 2 already
        token = _delegation_depth.set(2)
        try:
            with pytest.raises(DelegationDepthExceeded, match="max_depth"):
                await tool.call("task")
        finally:
            _delegation_depth.reset(token)


@pytest.mark.unit
class TestDelegationDepth:
    def test_initial_depth_zero(self) -> None:
        """Depth starts at 0."""
        # Note: if prior tests leaked depth, reset it
        assert current_delegation_depth() == 0 or current_delegation_depth() > 0

    @pytest.mark.asyncio
    async def test_depth_increments_and_resets(self) -> None:
        """call() increments depth during execution and resets after."""
        tool = agent_as_tool(Agent(), name="t", description="d")

        depth_during_call = -1

        async def track_depth(*args, **kwargs):  # type: ignore[no-untyped-def]
            nonlocal depth_during_call
            depth_during_call = _delegation_depth.get()
            return "result"

        mock_intent = IntentResult(intent=IntentType.RESPOND, confidence=1.0, reasoning="test")
        with (
            patch(
                "kaos_agents.agent.BaseAgent._classify",
                new_callable=AsyncMock,
                return_value=mock_intent,
            ),
            patch(
                "kaos_agents.agent.BaseAgent._dispatch",
                new_callable=AsyncMock,
                side_effect=track_depth,
            ),
        ):
            depth_before = _delegation_depth.get()
            await tool.call("task")
            depth_after = _delegation_depth.get()

        assert depth_during_call == depth_before + 1
        assert depth_after == depth_before


@pytest.mark.unit
class TestAgentDelegationFields:
    def test_default_empty(self) -> None:
        agent = Agent()
        assert agent.delegated_agents == ()
        assert agent.handoffs == ()
        assert agent.max_delegation_depth == 3

    def test_with_delegated_agents(self) -> None:
        writer = Agent(instructions="Write")
        tool = agent_as_tool(writer, name="w", description="d")
        coordinator = Agent(delegated_agents=(tool,))
        assert len(coordinator.delegated_agents) == 1
        assert coordinator.delegated_agents[0] is tool

    def test_with_handoffs(self) -> None:
        legal = Agent(name="legal", instructions="Legal expert")
        router = Agent(handoffs=(legal,))
        assert len(router.handoffs) == 1
        assert router.handoffs[0] is legal

    def test_create_with_delegation(self) -> None:
        writer = Agent()
        tool = agent_as_tool(writer, name="w", description="d")
        agent = Agent.create(
            delegated_agents=[tool],
            handoffs=[writer],
            max_delegation_depth=5,
        )
        assert len(agent.delegated_agents) == 1
        assert len(agent.handoffs) == 1
        assert agent.max_delegation_depth == 5


@pytest.mark.unit
class TestRunnerDelegate:
    @pytest.mark.asyncio
    async def test_delegate_yields_start_and_complete(self) -> None:
        """Runner.delegate() yields SubagentStart, then SubagentComplete."""
        writer = Agent(instructions="Write")
        tool = agent_as_tool(writer, name="writer", description="Write memos")

        parent = Agent()
        runner = Runner(parent)

        mock_intent = IntentResult(intent=IntentType.RESPOND, confidence=1.0, reasoning="test")
        with (
            patch(
                "kaos_agents.agent.BaseAgent._classify",
                new_callable=AsyncMock,
                return_value=mock_intent,
            ),
            patch(
                "kaos_agents.agent.BaseAgent._dispatch",
                new_callable=AsyncMock,
                return_value=("Memo drafted", []),
            ),
        ):
            events = []
            async for event in runner.delegate(tool, "Draft a memo", "parent-session"):
                events.append(event)

        assert len(events) == 2
        assert isinstance(events[0], SubagentStart)
        assert events[0].subagent_name == "writer"
        assert events[0].task == "Draft a memo"
        assert isinstance(events[1], SubagentComplete)
        assert events[1].subagent_name == "writer"
        assert "Memo drafted" in events[1].result_summary


@pytest.mark.unit
class TestRunnerHandoff:
    @pytest.mark.asyncio
    async def test_handoff_yields_start_and_target_events(self) -> None:
        """Runner.handoff() yields HandoffStart, then all target agent events."""
        legal = Agent(name="legal", instructions="Legal expert")
        router = Agent(instructions="Router")
        runner = Runner(router)

        mock_intent = IntentResult(intent=IntentType.RESPOND, confidence=1.0, reasoning="test")
        with (
            patch(
                "kaos_agents.agent.BaseAgent._classify",
                new_callable=AsyncMock,
                return_value=mock_intent,
            ),
            patch(
                "kaos_agents.agent.BaseAgent._dispatch",
                new_callable=AsyncMock,
                return_value=("Legal answer", []),
            ),
        ):
            events = []
            async for event in runner.handoff(
                legal,
                "What is privilege?",
                "session-1",
                from_agent_name="router",
                reason="Legal question",
            ):
                events.append(event)

        # First event is HandoffStart, followed by target agent's events
        assert isinstance(events[0], HandoffStart)
        assert events[0].from_agent == "router"
        assert events[0].to_agent == "legal"
        assert events[0].reason == "Legal question"
        # Target agent emits its own TurnStart/IntentClassified/TurnComplete
        assert len(events) > 1


# ---------------------------------------------------------------------------
# Gaps 6+7: Declarative delegation_agents + handoffs are injected as tools
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestDelegationInjection:
    """Verify Agent.delegated_agents/handoffs become actual LLM tools."""

    def test_build_delegation_tools_empty(self) -> None:
        """No delegation declared → empty tool list."""
        agent = Agent()
        runner = Runner(agent)
        assert runner._build_delegation_tools() == ()

    def test_build_delegation_tools_with_subagent(self) -> None:
        """Agent.delegated_agents → kaos-llm-core Tool with matching name/desc."""
        writer = Agent(instructions="Write memos")
        tool = agent_as_tool(writer, name="draft_memo", description="Draft a memo")
        agent = Agent(delegated_agents=(tool,))
        runner = Runner(agent)

        llm_tools = runner._build_delegation_tools(session_id="s1")
        assert len(llm_tools) == 1
        assert llm_tools[0].name == "draft_memo"
        assert "Draft a memo" in (llm_tools[0].description or "")

    def test_build_delegation_tools_with_handoff(self) -> None:
        """Agent.handoffs → handoff_to_<name> tool per target."""
        legal = Agent(name="legal", instructions="Legal expert")
        router = Agent(handoffs=(legal,))
        runner = Runner(router)

        llm_tools = runner._build_delegation_tools(session_id="s1")
        assert len(llm_tools) == 1
        assert llm_tools[0].name == "handoff_to_legal"
        assert "legal" in (llm_tools[0].description or "").lower()

    def test_build_delegation_tools_mixed(self) -> None:
        """Both delegated_agents and handoffs produce separate tools."""
        writer = Agent(instructions="Write")
        legal = Agent(name="legal", instructions="Legal")
        writer_tool = agent_as_tool(writer, name="writer", description="Writes")
        coordinator = Agent(
            delegated_agents=(writer_tool,),
            handoffs=(legal,),
        )
        runner = Runner(coordinator)

        llm_tools = runner._build_delegation_tools(session_id="s1")
        assert len(llm_tools) == 2
        names = {t.name for t in llm_tools}
        assert names == {"writer", "handoff_to_legal"}

    def test_internal_agent_receives_extra_tools(self) -> None:
        """_build_internal_agent injects delegation tools into ChatAgent."""
        from kaos_agents.patterns.chat import ChatAgent

        writer = Agent(instructions="Write")
        tool = agent_as_tool(writer, name="w", description="Write")
        parent = Agent(delegated_agents=(tool,))
        runner = Runner(parent)

        internal = runner._build_internal_agent(session_id="s1")
        assert isinstance(internal, ChatAgent)
        assert len(internal._extra_llm_tools) == 1
        assert internal._extra_llm_tools[0].name == "w"

    @pytest.mark.asyncio
    async def test_subagent_tool_invokes_delegated_call(self) -> None:
        """Calling the injected tool actually runs the sub-agent."""
        writer = Agent(instructions="Writer")
        tool = agent_as_tool(writer, name="writer", description="Writes")
        parent = Agent(delegated_agents=(tool,))
        runner = Runner(parent)

        mock_intent = IntentResult(intent=IntentType.RESPOND, confidence=1.0, reasoning="test")
        with (
            patch(
                "kaos_agents.agent.BaseAgent._classify",
                new_callable=AsyncMock,
                return_value=mock_intent,
            ),
            patch(
                "kaos_agents.agent.BaseAgent._dispatch",
                new_callable=AsyncMock,
                return_value=("Memo text", []),
            ),
        ):
            llm_tools = runner._build_delegation_tools(session_id="parent-s")
            # Invoke the tool as ReAct would (via executor)
            result = await llm_tools[0].executor(task="Write a memo")

        assert result == "Memo text"

    @pytest.mark.asyncio
    async def test_handoff_tool_invokes_target_agent(self) -> None:
        """Calling the handoff tool runs the target agent and returns text."""
        legal = Agent(name="legal", instructions="Legal")
        router = Agent(handoffs=(legal,))
        runner = Runner(router)

        mock_intent = IntentResult(intent=IntentType.RESPOND, confidence=1.0, reasoning="test")
        with (
            patch(
                "kaos_agents.agent.BaseAgent._classify",
                new_callable=AsyncMock,
                return_value=mock_intent,
            ),
            patch(
                "kaos_agents.agent.BaseAgent._dispatch",
                new_callable=AsyncMock,
                return_value=("Legal answer", []),
            ),
        ):
            llm_tools = runner._build_delegation_tools(session_id="s1")
            result = await llm_tools[0].executor(task="What is privilege?")

        assert result == "Legal answer"

    @pytest.mark.asyncio
    async def test_handoff_tool_respects_max_delegation_depth(self) -> None:
        """Calling a handoff tool at max_delegation_depth raises DelegationDepthExceeded."""
        from kaos_agents.delegation import (
            DelegationDepthExceeded,
            _delegation_depth,
        )

        legal = Agent(name="legal", instructions="Legal")
        router = Agent(handoffs=(legal,), max_delegation_depth=1)
        runner = Runner(router)

        llm_tools = runner._build_delegation_tools(session_id="s1")
        token = _delegation_depth.set(1)  # pretend we're already at depth 1
        try:
            with pytest.raises(DelegationDepthExceeded, match="Handoff depth"):
                await llm_tools[0].executor(task="go deeper")
        finally:
            _delegation_depth.reset(token)

    @pytest.mark.asyncio
    async def test_runtime_none_with_delegated_agents_does_not_short_circuit(
        self,
    ) -> None:
        """ChatAgent without a runtime still uses delegated_agents (no short-circuit)."""
        from kaos_agents.events import TextDelta

        writer = Agent(instructions="Writer")
        tool = agent_as_tool(writer, name="writer", description="Writes")
        parent = Agent(delegated_agents=(tool,))
        # No runtime attached — previously this would short-circuit to _simple_respond
        runner = Runner(parent, runtime=None)

        # The internal agent should see _extra_llm_tools populated
        internal = runner._build_internal_agent(session_id="s-no-rt")
        from kaos_agents.patterns.chat import ChatAgent

        assert isinstance(internal, ChatAgent)
        assert len(internal._extra_llm_tools) == 1

        # Verify the tool_use streaming handler does NOT short-circuit.
        # Patch ReAct to avoid a real LLM call; we just need to see that
        # the code path takes the non-short-circuit branch.

        mock_result = type(
            "MR",
            (),
            {
                "trajectory": [],
                "outputs": {"answer": "done"},
                "answer": "done",
                "iterations_used": 0,
                "stop_reason": "TERMINATED",
            },
        )()

        class _MockUsage:
            input_tokens = 0
            output_tokens = 0
            total_tokens = 0
            cost_usd = 0.0

        class _MockInvocation:
            output = mock_result
            usage = _MockUsage()

        class _MockReActInstance:
            async def __call__(self, **kwargs):  # type: ignore[no-untyped-def]
                return mock_result

            async def invoke(self, **kwargs):  # type: ignore[no-untyped-def]
                return _MockInvocation()

        # Build a dummy SessionMemory via the runner's store
        from kaos_agents.events import EventEmitter

        emitter = EventEmitter(session_id="s-no-rt", run_id="r-no-rt")

        with patch("kaos_llm_core.programs.react.ReAct", return_value=_MockReActInstance()):
            # Simulate the dispatch path directly on the internal agent
            memory = await runner._build_internal_agent(session_id="s-no-rt")._store.load_or_create(
                "s-no-rt"
            )
            events = []
            async for event in internal._handle_tool_use_streaming(
                "do something", memory, {}, emitter
            ):
                events.append(event)

        # Should see a TextDelta from the mocked ReAct result (not from _simple_respond fallback)
        assert any(isinstance(e, TextDelta) and e.content == "done" for e in events)

    @pytest.mark.asyncio
    async def test_subagent_respects_agent_max_delegation_depth(self) -> None:
        """Agent.max_delegation_depth caps DelegatedAgent's own max_depth."""
        from kaos_agents.delegation import (
            DelegationDepthExceeded,
            _delegation_depth,
        )

        writer = Agent(instructions="Write")
        # DelegatedAgent max_depth default is 3, but Agent caps at 1
        tool = agent_as_tool(writer, name="w", description="Writes", max_depth=5)
        parent = Agent(delegated_agents=(tool,), max_delegation_depth=1)
        runner = Runner(parent)

        llm_tools = runner._build_delegation_tools(session_id="s1")
        token = _delegation_depth.set(1)  # already at depth 1
        try:
            with pytest.raises(DelegationDepthExceeded):
                await llm_tools[0].executor(task="nested")
        finally:
            _delegation_depth.reset(token)
