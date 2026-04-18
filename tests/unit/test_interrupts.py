"""Tests for run interruption and resumption (Phase 6).

Covers:
- RunState JSON round-trip serialization
- PendingToolCall data integrity
- RunState.from_json error handling (invalid JSON, missing fields)
- RunState with and without pending_tool_call
- RunState with metadata
"""

from __future__ import annotations

import pytest

from kaos_agents.errors import EventDeserializationError
from kaos_agents.interrupts import PendingToolCall, RunState


@pytest.mark.unit
class TestPendingToolCall:
    def test_frozen(self) -> None:
        ptc = PendingToolCall(call_id="tc_01", tool_name="test-tool")
        with pytest.raises(AttributeError):
            setattr(ptc, "call_id", "other")  # noqa: B010

    def test_defaults(self) -> None:
        ptc = PendingToolCall(call_id="tc_01", tool_name="test-tool")
        assert ptc.arguments == ()
        assert ptc.reason == ""


@pytest.mark.unit
class TestRunState:
    def test_round_trip(self) -> None:
        """to_json → from_json produces equal state."""
        state = RunState(
            run_id="run_abc",
            session_id="session_1",
            pending_tool_call=PendingToolCall(
                call_id="tc_01",
                tool_name="kaos-web-delete",
                arguments=(("url", "https://example.com"),),
                reason="Destructive tool",
            ),
            event_count=7,
        )
        json_str = state.to_json()
        restored = RunState.from_json(json_str)
        assert restored.run_id == state.run_id
        assert restored.session_id == state.session_id
        assert restored.event_count == state.event_count
        assert restored.pending_tool_call is not None
        assert restored.pending_tool_call.call_id == "tc_01"
        assert restored.pending_tool_call.tool_name == "kaos-web-delete"
        assert restored.pending_tool_call.arguments == (("url", "https://example.com"),)
        assert restored.pending_tool_call.reason == "Destructive tool"

    def test_round_trip_no_pending(self) -> None:
        """Round-trip without a pending tool call."""
        state = RunState(run_id="run_1", session_id="sess_1", event_count=3)
        json_str = state.to_json()
        restored = RunState.from_json(json_str)
        assert restored.run_id == "run_1"
        assert restored.pending_tool_call is None

    def test_round_trip_with_metadata(self) -> None:
        """Round-trip with metadata."""
        state = RunState(
            run_id="run_1",
            session_id="sess_1",
            metadata=(("key", "value"), ("num", 42)),
        )
        json_str = state.to_json()
        restored = RunState.from_json(json_str)
        assert len(restored.metadata) == 2

    def test_frozen(self) -> None:
        state = RunState(run_id="r", session_id="s")
        with pytest.raises(AttributeError):
            setattr(state, "run_id", "other")  # noqa: B010

    def test_from_json_invalid_json(self) -> None:
        with pytest.raises(EventDeserializationError, match="Invalid RunState JSON"):
            RunState.from_json("{not valid}")

    def test_from_json_not_object(self) -> None:
        with pytest.raises(EventDeserializationError, match="Expected a JSON object"):
            RunState.from_json("[1, 2]")

    def test_from_json_missing_run_id(self) -> None:
        with pytest.raises(EventDeserializationError, match="missing required field"):
            RunState.from_json('{"session_id": "s"}')

    def test_from_json_missing_session_id(self) -> None:
        with pytest.raises(EventDeserializationError, match="missing required field"):
            RunState.from_json('{"run_id": "r"}')

    def test_round_trip_with_vfs_refs(self) -> None:
        """memory_snapshot_ref, event_log_ref, and original_message round-trip."""
        state = RunState(
            run_id="r1",
            session_id="s1",
            memory_snapshot_ref="kaos-agents/runs/r1/memory.json",
            event_log_ref="kaos-agents/runs/r1/events.jsonl",
            original_message="please do the thing",
        )
        restored = RunState.from_json(state.to_json())
        assert restored.memory_snapshot_ref == "kaos-agents/runs/r1/memory.json"
        assert restored.event_log_ref == "kaos-agents/runs/r1/events.jsonl"
        assert restored.original_message == "please do the thing"


@pytest.mark.unit
class TestRunStateVFSPersistence:
    """Cross-process resume via VFS-persisted RunState."""

    @pytest.mark.asyncio
    async def test_save_and_load_run_state(self) -> None:
        from kaos_core.types.enums import StorageBackend
        from kaos_core.vfs.core import VirtualFileSystem
        from kaos_core.vfs.models import VFSConfig

        from kaos_agents.interrupts import load_run_state, save_run_state

        vfs = VirtualFileSystem(config=VFSConfig(default_backend=StorageBackend.MEMORY))
        state = RunState(
            run_id="run_xyz",
            session_id="sess_1",
            pending_tool_call=PendingToolCall(
                call_id="tc_1",
                tool_name="dangerous-tool",
            ),
            event_count=5,
            original_message="please continue",
        )
        path = await save_run_state(state, vfs)
        assert "run_xyz" in path

        loaded = await load_run_state("run_xyz", vfs)
        assert loaded.run_id == "run_xyz"
        assert loaded.session_id == "sess_1"
        assert loaded.event_count == 5
        assert loaded.original_message == "please continue"
        assert loaded.pending_tool_call is not None
        assert loaded.pending_tool_call.tool_name == "dangerous-tool"

    @pytest.mark.asyncio
    async def test_load_run_state_missing_raises(self) -> None:
        from kaos_core.types.enums import StorageBackend
        from kaos_core.vfs.core import VirtualFileSystem
        from kaos_core.vfs.models import VFSConfig

        from kaos_agents.interrupts import load_run_state

        vfs = VirtualFileSystem(config=VFSConfig(default_backend=StorageBackend.MEMORY))
        with pytest.raises(EventDeserializationError, match="No persisted RunState"):
            await load_run_state("nonexistent", vfs)

    @pytest.mark.asyncio
    async def test_save_and_load_event_log(self) -> None:
        from kaos_core.types.enums import StorageBackend
        from kaos_core.vfs.core import VirtualFileSystem
        from kaos_core.vfs.models import VFSConfig

        from kaos_agents.events import IntentClassified, TurnStart
        from kaos_agents.interrupts import load_event_log, save_event_log

        vfs = VirtualFileSystem(config=VFSConfig(default_backend=StorageBackend.MEMORY))
        events = [
            TurnStart(timestamp=1.0, sequence=0, session_id="s", run_id="r1", turn_number=1),
            IntentClassified(
                timestamp=1.1,
                sequence=1,
                session_id="s",
                run_id="r1",
                intent="tool_use",
                confidence=0.9,
                reasoning="needs tools",
            ),
        ]
        await save_event_log(events, "r1", vfs)

        loaded = await load_event_log("r1", vfs)
        assert len(loaded) == 2
        assert isinstance(loaded[0], TurnStart)
        assert isinstance(loaded[1], IntentClassified)
        assert loaded[1].confidence == 0.9


@pytest.mark.unit
class TestRunnerResume:
    """End-to-end pause + resume via Runner.run() + Runner.resume()."""

    @pytest.mark.asyncio
    async def test_resume_denied_yields_run_error(self) -> None:
        """resume(approved=False) yields a RunError event explaining the denial."""
        from kaos_core.types.enums import StorageBackend
        from kaos_core.vfs.core import VirtualFileSystem
        from kaos_core.vfs.models import VFSConfig

        from kaos_agents.config import Agent
        from kaos_agents.events import RunError
        from kaos_agents.runner import Runner

        vfs = VirtualFileSystem(config=VFSConfig(default_backend=StorageBackend.MEMORY))
        runner = Runner(Agent(), vfs=vfs)

        state = RunState(
            run_id="r_deny",
            session_id="s_deny",
            pending_tool_call=PendingToolCall(call_id="tc", tool_name="dangerous"),
        )

        events = []
        async for event in runner.resume(state, approved=False):
            events.append(event)

        assert len(events) == 1
        assert isinstance(events[0], RunError)
        assert "denied" in events[0].message.lower()
        assert events[0].error_type == "approval_denied"

    @pytest.mark.asyncio
    async def test_pause_persists_run_state_to_vfs(self) -> None:
        """When permission policy returns ASK, RunState is written to VFS."""
        from unittest.mock import patch

        from kaos_core.types.enums import StorageBackend
        from kaos_core.vfs.core import VirtualFileSystem
        from kaos_core.vfs.models import VFSConfig

        from kaos_agents.config import Agent
        from kaos_agents.events import (
            EventEmitter,
            ToolCallApprovalRequired,
            ToolCallStart,
            TurnStart,
        )
        from kaos_agents.interrupts import (
            event_log_iter_path,
            load_run_state,
            run_state_path,
        )
        from kaos_agents.permissions import (
            PermissionDecision,
            PermissionPolicy,
            PermissionRule,
        )
        from kaos_agents.runner import Runner

        vfs = VirtualFileSystem(config=VFSConfig(default_backend=StorageBackend.MEMORY))
        policy = PermissionPolicy(
            rules=(PermissionRule(pattern="dangerous-*", action=PermissionDecision.ASK),)
        )
        runner = Runner(Agent(), vfs=vfs, permission_policy=policy)

        async def _fake_run(self_, message, session_id):  # type: ignore[no-untyped-def]
            emitter = EventEmitter(session_id=session_id, run_id="run_pause_test")
            yield emitter.emit(TurnStart, turn_number=1)
            yield emitter.emit(
                ToolCallStart,
                call_id="tc_1",
                tool_name="dangerous-delete",
                arguments=(),
            )

        with patch("kaos_agents.agent.BaseAgent.run", _fake_run):
            events = []
            async for event in runner.run("kill it", "sess_pause"):
                events.append(event)

        # Final event should be ToolCallApprovalRequired
        approval = [e for e in events if isinstance(e, ToolCallApprovalRequired)]
        assert len(approval) == 1

        # RunState should be persisted
        loaded = await load_run_state("run_pause_test", vfs)
        assert loaded.session_id == "sess_pause"
        assert loaded.original_message == "kill it"
        assert loaded.pending_tool_call is not None
        assert loaded.pending_tool_call.tool_name == "dangerous-delete"

        # The approval event's run_state_ref should point to the persisted path
        assert approval[0].run_state_ref == run_state_path("run_pause_test")

        # The event log file should exist with the pre-pause TurnStart
        assert await vfs.exists(event_log_iter_path("run_pause_test"))
