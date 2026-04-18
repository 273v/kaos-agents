"""Run interruption and resumption — durable pause/resume for approval.

When a tool requires human approval, the Runner:
1. Serializes the in-memory ``SessionMemory`` to a VFS path
2. Writes the events emitted so far as JSONL to a VFS path
3. Constructs a ``RunState`` referencing those VFS paths
4. Persists the ``RunState`` itself to VFS keyed by ``run_id``
5. Yields a ``ToolCallApprovalRequired`` event whose ``run_state_ref``
   is the VFS path of the persisted RunState
6. Stops yielding (the run is paused)

The caller can later resume by:
- ``Runner.resume(run_state, approved=True)`` — continues the run
- ``Runner.resume(run_state, approved=False)`` — yields a RunError

The state survives process restarts because everything (memory, event
log, RunState) lives in the VFS, not in process memory.

This follows the OpenAI Agents SDK pattern where interrupted runs
produce a serializable state object.
"""

from __future__ import annotations

import json
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from kaos_core.logging import get_logger

from kaos_agents.errors import EventDeserializationError

if TYPE_CHECKING:
    from kaos_core.vfs.core import VirtualFileSystem

    from kaos_agents.events import AgentEvent

logger = get_logger(__name__)

# VFS path layout for run state persistence
_RUN_STATE_PREFIX = "kaos-agents/runs"


def run_state_path(run_id: str) -> str:
    """VFS path for a persisted RunState by run_id."""
    return f"{_RUN_STATE_PREFIX}/{run_id}/state.json"


def memory_snapshot_path(run_id: str) -> str:
    """VFS path for the memory snapshot at pause time."""
    return f"{_RUN_STATE_PREFIX}/{run_id}/memory.json"


def event_log_path(run_id: str) -> str:
    """VFS path for the JSONL event log emitted before the pause."""
    return f"{_RUN_STATE_PREFIX}/{run_id}/events.jsonl"


@dataclass(frozen=True, slots=True)
class PendingToolCall:
    """A tool call that is awaiting approval."""

    call_id: str
    tool_name: str
    arguments: tuple[tuple[str, str], ...] = ()
    reason: str = ""


@dataclass(frozen=True, slots=True)
class AgentSnapshot:
    """Serializable subset of :class:`kaos_agents.config.Agent`.

    Captured at pause time so ``Runner.resume()`` (and the cross-process
    approval endpoint) can reconstruct the original Agent rather than
    falling back to defaults. Fixes WS-0.3: a paused plan-pattern run
    would previously resume as a default chat-pattern run, silently
    dropping model / tool_filter / instructions.

    Fields mirror :class:`kaos_agents.config.Agent` minus:

    - ``provider`` (``ProviderConfig`` instances are not trivially
      JSON-serializable; deferred to a follow-on).
    - ``settings`` (``KaosAgentSettings`` is resolved at Runner
      construction; the resume endpoint re-resolves from env).
    - ``delegated_agents`` / ``handoffs`` (nested Agent trees; deferred).

    ``pattern`` is stored as the ``AgentPattern`` string value for
    trivial JSON serialization.
    """

    pattern: str = "chat"
    instructions: str = "You are a helpful assistant."
    model: str = ""
    tools: tuple[str, ...] = ()
    max_tools: int | None = None
    max_react_iterations: int | None = None
    max_plan_steps: int | None = None
    rag_top_k: int | None = None
    rag_max_retries: int | None = None
    max_delegation_depth: int = 3
    name: str | None = None

    @classmethod
    def from_agent(cls, agent: Any) -> AgentSnapshot:
        """Capture a snapshot from a live :class:`Agent` instance."""
        return cls(
            pattern=getattr(agent.pattern, "value", str(agent.pattern)),
            instructions=getattr(agent, "instructions", "You are a helpful assistant."),
            model=getattr(agent, "model", ""),
            tools=tuple(getattr(agent, "tools", ()) or ()),
            max_tools=getattr(agent, "max_tools", None),
            max_react_iterations=getattr(agent, "max_react_iterations", None),
            max_plan_steps=getattr(agent, "max_plan_steps", None),
            rag_top_k=getattr(agent, "rag_top_k", None),
            rag_max_retries=getattr(agent, "rag_max_retries", None),
            max_delegation_depth=getattr(agent, "max_delegation_depth", 3),
            name=getattr(agent, "name", None),
        )

    def to_agent(self) -> Any:
        """Rebuild an :class:`Agent` from this snapshot.

        ``provider`` / ``settings`` / ``delegated_agents`` / ``handoffs``
        are lost — they are not in the snapshot today. Providers can be
        re-attached externally; settings re-resolve from env.
        """
        # Local import avoids a circular import at module load time.
        from kaos_agents.config import Agent, AgentPattern

        kwargs: dict[str, Any] = {
            "pattern": AgentPattern(self.pattern),
            "instructions": self.instructions,
            "tools": self.tools,
            "max_tools": self.max_tools,
            "max_react_iterations": self.max_react_iterations,
            "max_plan_steps": self.max_plan_steps,
            "rag_top_k": self.rag_top_k,
            "rag_max_retries": self.rag_max_retries,
            "max_delegation_depth": self.max_delegation_depth,
            "name": self.name,
        }
        if self.model:
            kwargs["model"] = self.model
        return Agent.create(**kwargs)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        return {
            "pattern": self.pattern,
            "instructions": self.instructions,
            "model": self.model,
            "tools": list(self.tools),
            "max_tools": self.max_tools,
            "max_react_iterations": self.max_react_iterations,
            "max_plan_steps": self.max_plan_steps,
            "rag_top_k": self.rag_top_k,
            "rag_max_retries": self.rag_max_retries,
            "max_delegation_depth": self.max_delegation_depth,
            "name": self.name,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AgentSnapshot:
        """Deserialize from a dict produced by :meth:`to_dict`."""
        tools_raw = data.get("tools", [])
        return cls(
            pattern=data.get("pattern", "chat"),
            instructions=data.get("instructions", "You are a helpful assistant."),
            model=data.get("model", ""),
            tools=tuple(tools_raw) if tools_raw else (),
            max_tools=data.get("max_tools"),
            max_react_iterations=data.get("max_react_iterations"),
            max_plan_steps=data.get("max_plan_steps"),
            rag_top_k=data.get("rag_top_k"),
            rag_max_retries=data.get("rag_max_retries"),
            max_delegation_depth=data.get("max_delegation_depth", 3),
            name=data.get("name"),
        )


@dataclass(frozen=True, slots=True)
class RunState:
    """Serializable snapshot of an interrupted run.

    Created when the Runner pauses for approval. Contains everything
    needed to resume execution after the human responds.

    Usage::

        # Serialize to persist
        state_json = run_state.to_json()

        # Deserialize to resume
        state = RunState.from_json(state_json)
    """

    run_id: str
    session_id: str
    pending_tool_call: PendingToolCall | None = None
    event_count: int = 0  # Number of events emitted before pause
    # VFS path to the serialized memory snapshot at pause time. Empty
    # string means no memory snapshot was persisted (e.g., for unit tests
    # or when VFS is unavailable).
    memory_snapshot_ref: str = ""
    # VFS path to the serialized event log (JSONL) of events emitted
    # before the pause. Empty string means no event log was persisted.
    event_log_ref: str = ""
    # The original user message — needed to resume the run.
    original_message: str = ""
    created_at: float = field(default_factory=time.time)
    metadata: tuple[tuple[str, Any], ...] = ()
    # WS-0.3: snapshot of the original Agent config at pause time so
    # resume can rebuild the Runner with the correct pattern / model /
    # tools / instructions. None means "no snapshot captured" (older
    # RunStates or unit tests) — the caller falls back to default Agent.
    agent_config: AgentSnapshot | None = None

    def to_json(self) -> str:
        """Serialize to a JSON string for VFS persistence."""
        data: dict[str, Any] = {
            "run_id": self.run_id,
            "session_id": self.session_id,
            "event_count": self.event_count,
            "memory_snapshot_ref": self.memory_snapshot_ref,
            "event_log_ref": self.event_log_ref,
            "original_message": self.original_message,
            "created_at": self.created_at,
        }
        if self.pending_tool_call is not None:
            data["pending_tool_call"] = {
                "call_id": self.pending_tool_call.call_id,
                "tool_name": self.pending_tool_call.tool_name,
                "arguments": [list(pair) for pair in self.pending_tool_call.arguments],
                "reason": self.pending_tool_call.reason,
            }
        else:
            data["pending_tool_call"] = None
        if self.metadata:
            data["metadata"] = [list(pair) for pair in self.metadata]
        if self.agent_config is not None:
            data["agent_config"] = self.agent_config.to_dict()
        return json.dumps(data, separators=(",", ":"))

    @classmethod
    def from_json(cls, data: str) -> RunState:
        """Deserialize from a JSON string.

        Raises:
            EventDeserializationError: If the JSON is invalid or
                required fields are missing.
        """
        try:
            parsed = json.loads(data)
        except json.JSONDecodeError as exc:
            raise EventDeserializationError(
                f"Invalid RunState JSON at position {exc.pos}: {exc.msg}. "
                "Ensure the input is valid JSON produced by RunState.to_json(). "
                "Check that the VFS path is correct."
            ) from exc

        if not isinstance(parsed, dict):
            raise EventDeserializationError(
                f"Expected a JSON object for RunState, got {type(parsed).__name__}. "
                "RunState must be a JSON object with run_id and session_id fields."
            )

        pending = None
        ptc_data = parsed.get("pending_tool_call")
        if ptc_data is not None and isinstance(ptc_data, dict):
            pending = PendingToolCall(
                call_id=ptc_data.get("call_id", ""),
                tool_name=ptc_data.get("tool_name", ""),
                arguments=tuple(tuple(p) for p in ptc_data.get("arguments", [])),
                reason=ptc_data.get("reason", ""),
            )

        meta_raw = parsed.get("metadata", [])
        metadata = tuple(tuple(p) for p in meta_raw) if meta_raw else ()

        agent_config_raw = parsed.get("agent_config")
        agent_config = (
            AgentSnapshot.from_dict(agent_config_raw)
            if isinstance(agent_config_raw, dict)
            else None
        )

        try:
            return cls(
                run_id=parsed["run_id"],
                session_id=parsed["session_id"],
                pending_tool_call=pending,
                event_count=parsed.get("event_count", 0),
                memory_snapshot_ref=parsed.get("memory_snapshot_ref", ""),
                event_log_ref=parsed.get("event_log_ref", ""),
                original_message=parsed.get("original_message", ""),
                created_at=parsed.get("created_at", 0.0),
                metadata=metadata,
                agent_config=agent_config,
            )
        except KeyError as exc:
            raise EventDeserializationError(
                f"RunState missing required field: {exc}. "
                "Required fields: run_id, session_id. "
                "Use RunState.to_json() to produce correctly formatted state."
            ) from exc


# ---------------------------------------------------------------------------
# VFS persistence helpers (stand-alone for cross-process resume)
# ---------------------------------------------------------------------------


async def save_run_state(state: RunState, vfs: VirtualFileSystem) -> str:
    """Persist a RunState JSON to VFS at the standard run_id path.

    Returns the VFS path so callers can include it in event payloads.
    """
    path = run_state_path(state.run_id)
    await vfs.write(path, state.to_json().encode())
    logger.debug("interrupts.save_run_state: run_id=%s path=%s", state.run_id, path)
    return path


async def load_run_state(run_id: str, vfs: VirtualFileSystem) -> RunState:
    """Load a previously persisted RunState by run_id.

    Raises:
        EventDeserializationError: If the RunState file doesn't exist or
            is malformed.
    """
    path = run_state_path(run_id)
    if not await vfs.exists(path):
        raise EventDeserializationError(
            f"No persisted RunState found for run_id={run_id!r} at path {path}. "
            "Check that the run was actually paused (a RunState is only written "
            "when the Runner pauses for approval). "
            "Verify the VFS configuration matches the one used at pause time."
        )
    raw = await vfs.read(path)
    return RunState.from_json(raw.decode())


async def save_event_log(events: Sequence[AgentEvent], run_id: str, vfs: VirtualFileSystem) -> str:
    """Persist an event log as JSONL to VFS.

    Used at pause time to capture all events emitted before the pause.
    On resume, the events are replayed before the run continues.
    """
    from kaos_agents.events import serialize_event_json

    path = event_log_path(run_id)
    payload = "\n".join(serialize_event_json(e) for e in events) + ("\n" if events else "")
    await vfs.write(path, payload.encode())
    logger.debug(
        "interrupts.save_event_log: run_id=%s events=%d path=%s",
        run_id,
        len(events),
        path,
    )
    return path


async def load_event_log(run_id: str, vfs: VirtualFileSystem) -> list[AgentEvent]:
    """Load a previously persisted event log as JSONL.

    Returns an empty list if the log doesn't exist (resume without
    replay).
    """
    from kaos_agents.events import deserialize_event_json

    path = event_log_path(run_id)
    if not await vfs.exists(path):
        return []
    raw = (await vfs.read(path)).decode()
    events: list[AgentEvent] = []
    for line in raw.splitlines():
        if not line.strip():
            continue
        events.append(deserialize_event_json(line))
    return events


def event_log_iter_path(run_id: str) -> str:
    """Public accessor for the event log path (avoids import of private helper)."""
    return event_log_path(run_id)
