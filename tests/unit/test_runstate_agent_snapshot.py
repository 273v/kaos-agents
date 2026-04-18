"""Regression tests for WS-0.3 — resume preserves agent config.

Pre-fix bug: ``api.py:/v1/runs/{id}/approve`` reconstructed a default
``Agent()`` on every resume, silently dropping the original pattern /
model / tools / instructions. A paused plan-pattern run resumed as a
default chat-pattern run.

Post-fix: ``RunState`` carries an :class:`AgentSnapshot` captured at
pause time; the approve endpoint calls ``state.agent_config.to_agent()``
to rebuild the original Agent.

Coverage:

1. ``AgentSnapshot.from_agent(agent)`` captures every serializable field.
2. Snapshot round-trips through ``to_dict`` / ``from_dict``.
3. ``RunState.to_json`` / ``from_json`` round-trip the snapshot.
4. ``Runner._pause_for_approval`` populates ``RunState.agent_config``.
5. ``AgentSnapshot.to_agent()`` rebuilds the original Agent shape.
"""

from __future__ import annotations

import pytest

from kaos_agents.config import Agent, AgentPattern
from kaos_agents.interrupts import AgentSnapshot, PendingToolCall, RunState


@pytest.mark.unit
class TestAgentSnapshotRoundTrip:
    def test_from_agent_captures_all_fields(self) -> None:
        agent = Agent.create(
            pattern=AgentPattern.PLAN,
            instructions="Regulatory monitor agent.",
            model="anthropic:claude-haiku-4-5",
            tools=["kaos-source-*", "kaos-web-fetch"],
            max_tools=20,
            max_react_iterations=5,
            max_plan_steps=15,
            rag_top_k=8,
            rag_max_retries=1,
            max_delegation_depth=4,
            name="regulatory-agent",
        )
        snapshot = AgentSnapshot.from_agent(agent)

        assert snapshot.pattern == "plan"
        assert snapshot.instructions == "Regulatory monitor agent."
        assert snapshot.model == "anthropic:claude-haiku-4-5"
        assert snapshot.tools == ("kaos-source-*", "kaos-web-fetch")
        assert snapshot.max_tools == 20
        assert snapshot.max_react_iterations == 5
        assert snapshot.max_plan_steps == 15
        assert snapshot.rag_top_k == 8
        assert snapshot.rag_max_retries == 1
        assert snapshot.max_delegation_depth == 4
        assert snapshot.name == "regulatory-agent"

    def test_dict_round_trip(self) -> None:
        original = AgentSnapshot(
            pattern="research",
            instructions="Legal research.",
            model="openai:gpt-5.4-nano",
            tools=("kaos-source-*",),
            max_tools=30,
            max_react_iterations=None,
            max_plan_steps=None,
            rag_top_k=10,
            rag_max_retries=2,
            max_delegation_depth=3,
            name="legal",
        )
        data = original.to_dict()
        reconstructed = AgentSnapshot.from_dict(data)
        assert reconstructed == original

    def test_to_agent_rebuilds_original(self) -> None:
        """A paused plan-pattern Agent must NOT come back as default chat."""
        original = Agent.create(
            pattern=AgentPattern.PLAN,
            instructions="Monitor EDGAR filings.",
            model="anthropic:claude-haiku-4-5",
            tools=["kaos-source-edgar-*"],
            max_plan_steps=20,
        )
        snapshot = AgentSnapshot.from_agent(original)
        rebuilt = snapshot.to_agent()

        assert rebuilt.pattern == AgentPattern.PLAN, (
            f"resumed pattern was {rebuilt.pattern!r}, expected PLAN — "
            "WS-0.3 regression: pattern not preserved through snapshot"
        )
        assert rebuilt.instructions == "Monitor EDGAR filings."
        assert rebuilt.model == "anthropic:claude-haiku-4-5"
        assert rebuilt.tools == ("kaos-source-edgar-*",)
        assert rebuilt.max_plan_steps == 20


@pytest.mark.unit
class TestRunStateAgentConfigRoundTrip:
    def test_to_json_from_json_preserves_agent_config(self) -> None:
        snapshot = AgentSnapshot(
            pattern="plan",
            instructions="Deep research.",
            model="anthropic:claude-haiku-4-5",
            tools=("kaos-web-*",),
            max_plan_steps=12,
        )
        state = RunState(
            run_id="r_abc123",
            session_id="s_xyz",
            pending_tool_call=PendingToolCall(call_id="tc1", tool_name="x"),
            original_message="find filings",
            agent_config=snapshot,
        )
        serialized = state.to_json()
        deserialized = RunState.from_json(serialized)

        assert deserialized.agent_config is not None, (
            "RunState.from_json dropped the agent_config field"
        )
        assert deserialized.agent_config == snapshot

    def test_from_json_without_agent_config_returns_none(self) -> None:
        """Backward compat — older serialized RunStates have no
        agent_config field; load must succeed with None."""
        state = RunState(
            run_id="r_old",
            session_id="s_old",
            original_message="legacy message",
        )
        serialized = state.to_json()
        # The field is omitted from the JSON because we only write it
        # when not None — confirm load round-trips None.
        deserialized = RunState.from_json(serialized)
        assert deserialized.agent_config is None


@pytest.mark.unit
class TestRunnerPauseCapturesSnapshot:
    """End-to-end: when the Runner pauses, the persisted RunState must
    carry an AgentSnapshot reflecting the Agent the Runner was
    constructed with."""

    @pytest.mark.asyncio
    async def test_pause_persists_agent_snapshot(self) -> None:
        from kaos_core.types.enums import IsolationMode, StorageBackend
        from kaos_core.vfs.core import VirtualFileSystem
        from kaos_core.vfs.models import VFSConfig

        from kaos_agents.events import EventEmitter, ToolCallStart
        from kaos_agents.runner import Runner

        vfs = VirtualFileSystem(
            config=VFSConfig(
                default_backend=StorageBackend.MEMORY,
                isolation_mode=IsolationMode.GLOBAL,
            )
        )
        agent = Agent.create(
            pattern=AgentPattern.PLAN,
            instructions="Audit EDGAR filings.",
            model="anthropic:claude-haiku-4-5",
            tools=["kaos-source-*"],
            max_plan_steps=10,
        )
        runner = Runner(agent, vfs=vfs)

        emitter = EventEmitter(session_id="s_pause", run_id="r_pause_test")
        tool_event = emitter.emit(
            ToolCallStart,
            call_id="tc1",
            tool_name="kaos-source-edgar-lookup",
            arguments=(("cik", "0000320193"),),
        )

        approval = await runner._pause_for_approval(
            tool_event,
            session_id="s_pause",
            message="audit AAPL filings",
            event_count=0,
            emitted=[],
            reason="destructive tool — needs human approval",
        )

        # Load the persisted RunState back from VFS and assert the
        # snapshot round-tripped with the original Agent config.
        from kaos_agents.interrupts import load_run_state

        state = await load_run_state("r_pause_test", vfs)
        assert state.agent_config is not None, (
            "Runner._pause_for_approval did not persist agent_config — WS-0.3 regression."
        )
        assert state.agent_config.pattern == "plan"
        assert state.agent_config.instructions == "Audit EDGAR filings."
        assert state.agent_config.model == "anthropic:claude-haiku-4-5"
        assert state.agent_config.tools == ("kaos-source-*",)
        assert state.agent_config.max_plan_steps == 10

        # The approval event itself carries the state reference.
        assert approval.run_id == "r_pause_test"
