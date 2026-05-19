"""Runner — the execution engine for Agent configurations.

The Runner takes an ``Agent`` (frozen config) and runtime dependencies,
then drives the agent loop. It is the single entry point for both
streaming and blocking execution.

Separation of concerns:
- ``Agent``: what the agent is (instructions, model, tools, pattern)
- ``Runner``: how the agent runs (runtime, context, hooks, state)

Usage::

    agent = Agent(
        instructions="You are a research assistant.",
        model="anthropic:claude-sonnet-4-6",
        tools=("kaos-source-*", "kaos-web-*"),
        pattern=AgentPattern.PLAN,
    )
    runner = Runner(agent, runtime=runtime)

    # Streaming
    async for event in runner.run("Find recent policy changes", "session-1"):
        ...

    # Blocking
    response = await runner.turn("Find EPA enforcement actions", "session-1")
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any

from kaos_core.logging import get_logger

from kaos_agents.config import Agent, AgentPattern
from kaos_agents.events import (
    EventEmitter,
    KaosEvent,
    RunError,
    Span,
    SpanPhase,
    SpanSubject,
    ToolCallApprovalRequired,
)
from kaos_agents.hooks import HookAction, KaosHook, dispatch_hook
from kaos_agents.memory.store import SessionStore
from kaos_agents.runtime.delegation import DelegatedAgent
from kaos_agents.runtime.interrupts import (
    PendingToolCall,
    RunState,
    memory_snapshot_path,
    save_event_log,
    save_run_state,
)
from kaos_agents.runtime.permissions import PermissionPolicy
from kaos_agents.types import AgentResponse, PermissionDecision

if TYPE_CHECKING:
    from kaos_core.base.context import KaosContext
    from kaos_core.registry.container import KaosRuntime
    from kaos_core.vfs.core import VirtualFileSystem

logger = get_logger(__name__)


class Runner:
    """Executes an Agent's turn loop.

    The Runner owns the execution dependencies (runtime, context, VFS)
    and constructs the appropriate internal agent pattern based on
    ``Agent.pattern``. Multiple Runners can share one Agent config.

    Args:
        agent: Frozen agent configuration.
        runtime: KaosRuntime for tool execution. None for tool-free agents.
        context: KaosContext for request-level metadata. None for standalone use.
        vfs: VirtualFileSystem for memory persistence. Defaults to runtime.vfs
            or an in-memory VFS if neither is available.
    """

    __slots__ = (
        "_agent",
        "_agent_loop_version",
        "_auto_select_planner",
        "_context",
        "_corpus",
        "_hooks",
        "_permission_policy",
        "_runtime",
        "_settings",
        "_unsafe_bypass",
        "_vfs",
    )

    def __init__(
        self,
        agent: Agent,
        *,
        runtime: KaosRuntime | None = None,
        context: KaosContext | None = None,
        vfs: VirtualFileSystem | None = None,
        hooks: tuple[KaosHook, ...] = (),
        permission_policy: PermissionPolicy | None = None,
        corpus: Any | None = None,
        agent_loop_version: str | None = None,
        auto_select_planner: bool = True,
        unsafe_bypass: bool = False,
    ) -> None:
        self._agent = agent
        self._runtime = runtime
        self._settings = agent.resolve_settings()
        self._vfs = vfs or _resolve_vfs(runtime)
        self._hooks = hooks
        # KC17-P0-2: when no policy is supplied, install the default-safe
        # policy (which still escalates destructiveHint / humanConfirmation
        # to ASK). The pre-KC17 behaviour of "no policy → skip all checks"
        # let API/MCP callers fire destructive tools without approval. The
        # ``unsafe_bypass=True`` escape hatch is for tests + internal
        # benchmarks ONLY — MUST NOT be used in production.
        self._unsafe_bypass = unsafe_bypass
        if unsafe_bypass:
            if permission_policy is not None:
                logger.warning(
                    "Runner: unsafe_bypass=True ignores the provided "
                    "permission_policy. This MUST NOT be used in production."
                )
            self._permission_policy = None
        elif permission_policy is None:
            self._permission_policy = PermissionPolicy.default_safe()
        else:
            self._permission_policy = permission_policy
        self._corpus = corpus
        # Phase 2 feature flag: KAOS_AGENT_LOOP=v2 routes Runner.run /
        # Runner.run_trigger / Runner.invoke_trigger through the new
        # AgentLoop. Default v1 keeps the existing BaseAgent path so
        # zero existing tests change behavior.
        self._agent_loop_version = (
            agent_loop_version
            if agent_loop_version is not None
            else os.environ.get("KAOS_AGENT_LOOP", "v1")
        )
        # Phase 3.D: when v2 is active, auto_select_planner controls
        # whether the AgentLoop picks a Planner from intent.pattern.
        # Default True (Resolved Decision #3 enables classifier-driven
        # selection). Tests that want the skeleton path pass False.
        self._auto_select_planner = auto_select_planner

        # Ensure a context exists when corpus is provided (tools need it)
        if context is None and corpus is not None:
            from kaos_core.base.context import KaosContext

            context = KaosContext.create()
        if context is not None and corpus is not None:
            context._config["_corpus"] = corpus
        self._context = context

    @property
    def agent(self) -> Agent:
        """The agent configuration this Runner executes."""
        return self._agent

    @property
    def agent_loop_version(self) -> str:
        """The active loop dispatch version (``"v1"`` or ``"v2"``).

        Resolved at construction time from the ``agent_loop_version``
        kwarg or ``KAOS_AGENT_LOOP`` env var. Phase 2 default is
        ``"v1"`` (legacy ``BaseAgent`` path); ``"v2"`` opts into the
        new :class:`AgentLoop` (with the Phase-2 caveat that planners
        / termination / escalation / governance subsystems are not yet
        wired, so v2 produces a minimal skeleton output).
        """
        return self._agent_loop_version

    async def run_trigger(self, trigger: Any) -> AsyncIterator[KaosEvent]:
        """Trigger-source-agnostic streaming entry point (Phase 2+).

        Dispatches to either the legacy v1 BaseAgent path or the new
        v2 :class:`AgentLoop` based on :attr:`agent_loop_version`.

        v1 fallback: extracts the message from ``trigger.payload`` and
        replays through :meth:`run`.

        v2: constructs an :class:`AgentLoop` and yields events from
        its ``stream(trigger)`` method.

        ``trigger`` is typed ``Any`` to avoid importing
        :class:`kaos_agents.triggers.Trigger` at runtime in the v1 path
        (keeps the legacy hot path light).
        """
        if self._agent_loop_version == "v2":
            async for event in self._run_via_agent_loop(trigger):
                yield event
            return
        # v1 fallback — shape-tolerant message extraction.
        message = ""
        try:
            payload = trigger.payload or {}
            for key in ("message", "goal", "reason"):
                value = payload.get(key)
                if value:
                    message = str(value)
                    break
        except AttributeError:
            pass
        session_id = getattr(trigger, "source_id", None) or _mint_session_id()
        async for event in self.run(message, session_id):
            yield event

    async def invoke_trigger(self, trigger: Any) -> Any:
        """Blocking trigger entry returning a :class:`TurnInvocation`.

        v2-only — v1 has no canonical TurnInvocation surface, so the
        v1 path raises :class:`RuntimeError` with guidance to either
        opt into v2 or use :meth:`turn`.
        """
        if self._agent_loop_version != "v2":
            raise RuntimeError(
                "Runner.invoke_trigger requires KAOS_AGENT_LOOP=v2 "
                "(or agent_loop_version='v2' on Runner.__init__). The "
                "v1 BaseAgent path has no TurnInvocation surface — use "
                "Runner.turn(message, session_id) instead."
            )
        loop = self._build_agent_loop()
        return await loop.invoke(trigger=trigger)

    async def _run_via_agent_loop(self, trigger: Any) -> AsyncIterator[KaosEvent]:
        """Phase-2 v2 dispatch path: build an AgentLoop and stream it.

        Phase 2 ships the wiring with no planners / termination /
        escalation / governance configured. The loop produces a
        well-formed ``Span(TURN, START)`` / ``IntentClassified`` /
        ``TurnSummary`` / ``Span(TURN, COMPLETE)`` quartet and an
        empty-output :class:`TurnInvocation` (extras["phase"]="skeleton").
        Phases 3-5 wire the missing subsystems.
        """
        loop = self._build_agent_loop()
        async for event in loop.stream(trigger):
            yield event

    def _build_agent_loop(self) -> Any:
        """Construct an :class:`AgentLoop` from this Runner's config.

        Phase 2: only IntentExtractor + hooks + permission_policy +
        envelope_hash. Phases 3-5 add planners, memory tier, judges,
        escalation policy, governance.

        Lazy import of :mod:`kaos_agents.loop` and
        :mod:`kaos_agents.intent` to keep the v1 import graph light.
        """
        import contextlib

        from kaos_agents.intent import IntentExtractor
        from kaos_agents.loop import AgentLoop

        model = getattr(self._agent, "model", None) or "anthropic:claude-haiku-4-5"
        envelope_hash = ""
        with contextlib.suppress(Exception):  # Phase 4 hardens this best-effort path
            envelope_hash = self._agent.to_envelope().agent_hash()

        # Bridge runtime tools into kaos-llm-core Tool instances so the
        # auto-selected ReActPlanner / HierarchicalPlanner sub-agents
        # can actually call them. Without this, v2 selects a planner
        # with empty tools (kaos-llm-core warns "ReAct(tools=[]) is a
        # degenerate construction") and produces empty answers — bug
        # #3 in the v2 audit + Phase 6 cutover Gate A blocker.
        bridged_tools: tuple[Any, ...] = ()
        if self._runtime is not None:
            from kaos_agents.actions.tool_bridge import bridge_runtime_tools

            filter_names = list(self._agent.tools) if self._agent.tools else None
            bridged_tools = tuple(
                bridge_runtime_tools(
                    self._runtime,
                    self._context,
                    filter_names=filter_names,
                    max_tools=self._agent.max_tools or self._settings.max_tools,
                    permission_policy=self._permission_policy,
                )
            )

        return AgentLoop(
            intent_extractor=IntentExtractor(model=model),
            hooks=self._hooks,
            permission_policy=self._permission_policy,
            agent_envelope_hash=envelope_hash,
            auto_select_planner=self._auto_select_planner,
            tools=bridged_tools,
            # 0.1.0a17: hand the runtime to AgentLoop so prepare_turn
            # can render the tool-group catalog into IntentSignature's
            # ``available_tool_groups`` input. Without this, the
            # classifier's catalog-aware bullets all reduce to no-ops.
            runtime=self._runtime,
        )

    async def run(self, message: str, session_id: str) -> AsyncIterator[KaosEvent]:
        """Execute a turn, yielding events progressively.

        This is the primary streaming entry point. Constructs the
        appropriate internal agent based on ``Agent.pattern`` and
        delegates to its ``run()`` method.

        Hooks are dispatched for each event before yielding. If a hook
        returns ``HookAction.SKIP`` for a tool call event, the event
        is suppressed. See ``kaos_agents.hooks`` for details.

        Args:
            message: The user's message.
            session_id: Session identifier for memory persistence.

        Yields:
            KaosEvent subclass instances in execution order.
        """
        # Phase 2 feature flag: when KAOS_AGENT_LOOP=v2, route through
        # the new AgentLoop instead of the legacy BaseAgent path.
        # Constructs an MCPToolTrigger and delegates to run_trigger so
        # all entry points fan in to the same AgentLoop dispatch.
        if self._agent_loop_version == "v2":
            from kaos_agents.triggers import Trigger

            trigger = Trigger.mcp(message, session_id=session_id)
            async for event in self._run_via_agent_loop(trigger):
                yield event
            return

        internal = self._build_internal_agent(session_id=session_id)
        event_count = 0
        emitted: list[KaosEvent] = []  # tracked for pause persistence
        async for event in internal.run(message, session_id):
            # Detect tool-call START spans for hook/permission gating.
            is_tool_start = (
                isinstance(event, Span)
                and event.subject == SpanSubject.TOOL_CALL
                and event.phase == SpanPhase.START
            )

            # Run hooks
            if self._hooks:
                action = await dispatch_hook(self._hooks, event)
                if action == HookAction.SKIP:
                    continue
                if action == HookAction.REQUIRE_APPROVAL and is_tool_start:
                    assert isinstance(event, Span)  # type-narrow for ty
                    approval = await self._pause_for_approval(
                        event,
                        session_id=session_id,
                        message=message,
                        event_count=event_count,
                        emitted=emitted,
                        reason="Hook requested approval",
                    )
                    yield approval
                    return  # Pause — caller must resume

            # Check permission policy on tool call events.
            # Look up ToolAnnotations from the runtime so readOnlyHint,
            # destructiveHint, humanConfirmationRequired flow through to
            # the policy evaluation.
            if self._permission_policy and is_tool_start:
                assert isinstance(event, Span)  # type-narrow for ty
                tool_name = str(event.attributes.get("tool_name", ""))
                annotations = self._lookup_annotations(tool_name)
                decision = self._permission_policy.evaluate(tool_name, annotations)
                if decision == PermissionDecision.DENY:
                    continue  # Suppress denied tool call
                if decision == PermissionDecision.ASK:
                    reason = f"Permission policy requires approval for {tool_name}"
                    approval = await self._pause_for_approval(
                        event,
                        session_id=session_id,
                        message=message,
                        event_count=event_count,
                        emitted=emitted,
                        reason=reason,
                    )
                    yield approval
                    return  # Pause

            event_count += 1
            emitted.append(event)
            yield event

    async def _pause_for_approval(
        self,
        tool_event: Span,
        *,
        session_id: str,
        message: str,
        event_count: int,
        emitted: list[KaosEvent],
        reason: str,
    ) -> ToolCallApprovalRequired:
        """Persist run state to VFS and build the ToolCallApprovalRequired event.

        Writes:
        - Memory snapshot to ``kaos-agents/runs/{run_id}/memory.json``
        - Event log (JSONL) to ``kaos-agents/runs/{run_id}/events.jsonl``
        - RunState to ``kaos-agents/runs/{run_id}/state.json``

        The returned event's ``run_state_ref`` is the VFS path of the
        persisted RunState, allowing cross-process resume.
        """
        run_id = tool_event.run_id
        attrs = tool_event.attributes
        tool_name = str(attrs.get("tool_name", ""))
        call_id = str(attrs.get("call_id", ""))
        raw_args = attrs.get("arguments", ()) or ()
        # Normalize arguments into the wire-shape tuple-of-tuples used by
        # PendingToolCall and ToolCallApprovalRequired.
        if isinstance(raw_args, dict):
            args_tuple: tuple[tuple[str, str], ...] = tuple(
                (str(k), str(v)) for k, v in raw_args.items()
            )
        else:
            args_tuple = tuple((str(k), str(v)) for k, v in raw_args)

        # Persist memory snapshot to VFS at the run-state path
        store = SessionStore(self._vfs)
        try:
            memory = await store.load_or_create(session_id)
            mem_path = memory_snapshot_path(run_id)
            import json as _json

            mem_payload = _json.dumps(memory.to_dict(), separators=(",", ":"), default=str).encode()
            await self._vfs.write(mem_path, mem_payload)
        except Exception as exc:
            logger.warning(
                "Runner._pause_for_approval: memory snapshot failed (continuing): %s", exc
            )
            mem_path = ""

        # Persist event log
        try:
            log_path = await save_event_log(emitted, run_id, self._vfs)
        except Exception as exc:
            logger.warning("Runner._pause_for_approval: event log save failed: %s", exc)
            log_path = ""

        # Build and persist the RunState
        # WS-0.3: capture an AgentSnapshot so cross-process resume
        # (e.g. POST /v1/runs/{id}/approve) rebuilds the Runner with the
        # original pattern/model/tools/instructions rather than falling
        # back to a default chat Agent.
        from kaos_agents.runtime.interrupts import AgentSnapshot

        agent_config = AgentSnapshot.from_agent(self._agent)

        state = RunState(
            run_id=run_id,
            session_id=session_id,
            pending_tool_call=PendingToolCall(
                call_id=call_id,
                tool_name=tool_name,
                arguments=args_tuple,
                reason=reason,
            ),
            event_count=event_count,
            memory_snapshot_ref=mem_path,
            event_log_ref=log_path,
            original_message=message,
            agent_config=agent_config,
        )
        try:
            state_path = await save_run_state(state, self._vfs)
        except Exception as exc:
            logger.warning("Runner._pause_for_approval: run state save failed: %s", exc)
            state_path = state.to_json()  # fallback: embed inline

        return ToolCallApprovalRequired(
            timestamp=tool_event.timestamp,
            sequence=tool_event.sequence,
            session_id=session_id,
            run_id=run_id,
            call_id=call_id,
            tool_name=tool_name,
            arguments=args_tuple,
            reason=reason,
            run_state_ref=state_path,
        )

    async def resume(
        self,
        run_state: RunState,
        *,
        approved: bool,
    ) -> AsyncIterator[KaosEvent]:
        """Resume an interrupted run after human approval.

        - ``approved=True``: Re-runs the original message on the same
          session. The internal agent's memory is restored from the
          snapshot so context is preserved. Tools that were previously
          subject to ASK rules will execute normally on this resume —
          the caller is expected to have policy that doesn't re-pause
          the same tool, or to remove/modify the rule before resuming.
        - ``approved=False``: Yields a single RunError event explaining
          that the user denied the approval, then returns.

        Args:
            run_state: The RunState produced when the run paused.
            approved: True to continue, False to abort with RunError.

        Yields:
            KaosEvent stream of the continuation (or a RunError on denial).
        """
        emitter = EventEmitter(session_id=run_state.session_id, run_id=run_state.run_id)

        if not approved:
            tool_name = (
                run_state.pending_tool_call.tool_name if run_state.pending_tool_call else "unknown"
            )
            yield emitter.emit(
                RunError,
                error_type="approval_denied",
                message=(f"Human approval denied for tool call '{tool_name}'."),
                recovery_hint=(
                    "If this denial was a mistake, call resume() again with approved=True. "
                    "Otherwise, restart the run with a different message or different tools."
                ),
            )
            return

        # Replay pre-pause events first so the consumer sees the full timeline
        from kaos_agents.runtime.interrupts import load_event_log

        try:
            replay = await load_event_log(run_state.run_id, self._vfs)
            for event in replay:
                yield event
        except Exception as exc:
            logger.warning("Runner.resume: event log replay failed (continuing): %s", exc)

        # Restore memory snapshot if available, then re-run the original message.
        # The run continues from where it left off conceptually; in practice
        # we re-execute the turn with the same message and the restored memory.
        if run_state.memory_snapshot_ref:
            try:
                raw = await self._vfs.read(run_state.memory_snapshot_ref)
                import json as _json

                snapshot_data = _json.loads(raw.decode())
                # Re-save to the standard session path so the internal
                # SessionStore.load_or_create picks it up
                from kaos_agents.memory.store import _session_path

                await self._vfs.write(
                    _session_path(run_state.session_id),
                    _json.dumps(snapshot_data, separators=(",", ":"), default=str).encode(),
                )
            except Exception as exc:
                logger.warning(
                    "Runner.resume: memory snapshot restore failed (continuing fresh): %s",
                    exc,
                )

        # Continue with the original message
        async for event in self.run(run_state.original_message, run_state.session_id):
            yield event

    async def turn(self, message: str, session_id: str) -> AgentResponse:
        """Execute a turn, returning the final :class:`AgentResponse`.

        WS-0.2: this path was previously a bypass that called
        ``internal.turn(...)`` directly, skipping the Runner's hook
        dispatch + permission policy evaluation. The MCP tool surface
        and the JSON API both use ``turn()``, so the bypass silently
        disabled safety policy for every non-streaming caller.

        After WS-0.2: ``turn()`` drains ``run()`` and reconstructs the
        :class:`AgentResponse` from the emitted events. Hooks fire,
        permission policy evaluates, and SSE + JSON + MCP clients get
        identical safety semantics.

        PA14 (issue #166): the inline drain that used to live in this
        body has been consolidated into the canonical
        :func:`events_to_response` helper. ``Runner.turn`` now collects
        events into a list and delegates, so the two paths cannot
        drift on cost/tokens/error metadata. The 3 known divergences
        (D1 RunError metadata, D2 default intent reasoning, D3 empty
        TurnSummary fallback) have been resolved in the helper.

        Args:
            message: The user's message.
            session_id: Session identifier for memory persistence.

        Returns:
            AgentResponse with the agent's reply and metadata aggregated
            from the streamed event sequence.
        """
        # Local import avoids a circular-import at module load time —
        # kaos_agents.events imports from the same package.
        from kaos_agents.runtime.events_to_response import events_to_response

        events: list[KaosEvent] = []
        async for event in self.run(message, session_id):
            events.append(event)
        return events_to_response(events, session_id)

    async def delegate(
        self,
        delegated: DelegatedAgent,
        task: str,
        session_id: str,
    ) -> AsyncIterator[KaosEvent]:
        """Execute a delegated sub-agent, yielding start/complete events.

        Wraps ``DelegatedAgent.call()`` with event emission so the
        sub-agent's execution is visible in the parent's event stream.

        Yields:
            SubagentStart event, then SubagentComplete with the result.

        Raises:
            DelegationDepthExceeded: If nested delegation exceeds max_depth.
        """
        from kaos_agents.runtime.agent import _generate_run_id

        run_id = _generate_run_id()
        emitter = EventEmitter(session_id=session_id, run_id=run_id)

        sub_span = emitter.span_start(
            SpanSubject.SUBAGENT,
            attributes={"subagent_name": delegated.name, "task": task},
        )
        yield sub_span

        try:
            result_text = await delegated.call(
                task,
                parent_session_id=session_id,
                runtime=self._runtime,
                context=self._context,
                vfs=self._vfs,
            )
        except Exception as exc:
            logger.warning("Runner.delegate: sub-agent '%s' failed: %s", delegated.name, exc)
            raise

        yield emitter.span_complete(
            SpanSubject.SUBAGENT,
            span_id=sub_span.span_id,
            attributes={
                "subagent_name": delegated.name,
                "result_summary": result_text[: self._settings.result_summary_max_chars],
                "tokens_used": 0,
            },
        )

    async def handoff(
        self,
        target: Agent,
        message: str,
        session_id: str,
        *,
        from_agent_name: str = "",
        reason: str = "",
    ) -> AsyncIterator[KaosEvent]:
        """Transfer control to another agent, sharing the session.

        Unlike ``delegate()`` which creates an isolated sub-session,
        ``handoff()`` uses the same session_id — the target agent sees
        the parent's memory.

        Yields:
            HandoffStart event, then all events from the target agent's run.
        """
        from kaos_agents.runtime.agent import _generate_run_id

        run_id = _generate_run_id()
        emitter = EventEmitter(session_id=session_id, run_id=run_id)

        target_name = target.name or "unnamed"
        yield emitter.span_start(
            SpanSubject.HANDOFF,
            attributes={
                "from_agent": from_agent_name,
                "to_agent": target_name,
                "reason": reason,
            },
        )

        target_runner = Runner(
            target,
            runtime=self._runtime,
            context=self._context,
            vfs=self._vfs,
            hooks=self._hooks,
            permission_policy=self._permission_policy,
        )
        async for event in target_runner.run(message, session_id):
            yield event

    def _lookup_annotations(self, tool_name: str) -> Any:
        """Fetch ToolAnnotations for a tool from the runtime's registry.

        Returns None if no runtime is configured or the tool is not registered.
        Returned annotations let PermissionPolicy apply readOnlyHint/
        destructiveHint/humanConfirmationRequired rules correctly.
        """
        if self._runtime is None:
            return None
        try:
            tool = self._runtime.tools.get_tool(tool_name)
        except Exception as exc:
            logger.debug(
                "Runner._lookup_annotations: tool registry lookup failed for %s: %s",
                tool_name,
                exc,
            )
            return None
        if tool is None:
            return None
        try:
            return tool.metadata.annotations
        except AttributeError:
            return None

    def _build_delegation_tools(self, session_id: str | None = None) -> tuple[Any, ...]:
        """Convert Agent.delegated_agents and Agent.handoffs to kaos-llm-core Tools.

        Called before dispatching to the internal agent so the parent's
        ReAct loop can invoke sub-agents and handoff targets via normal
        tool-calling. Each tool is a small async closure that forwards
        the call to the appropriate Runner helper.

        Args:
            session_id: Parent session id for sub-session derivation.
                Unused for handoffs (they share the parent session).

        Returns:
            Tuple of kaos-llm-core Tool instances, empty if no delegation
            is declared on the Agent.
        """
        delegated = self._agent.delegated_agents
        handoffs = self._agent.handoffs
        if not delegated and not handoffs:
            return ()

        # Lazy import — kaos-llm-core is an optional dep
        try:
            from kaos_llm_core import Tool
        except ImportError:
            logger.warning(
                "Runner._build_delegation_tools: kaos-llm-core not installed; "
                "delegated_agents and handoffs are ignored. "
                "Install with: pip install 'kaos-agents[llm]'"
            )
            return ()

        parent_session = session_id or ""
        runtime = self._runtime
        context = self._context
        vfs = self._vfs
        tools: list[Any] = []

        # Agent-level max_delegation_depth acts as the floor for any
        # DelegatedAgent that left its own max_depth at the default (3).
        # This lets the Agent config globally cap recursion across all
        # delegated_agents without editing each one.
        agent_max_depth = self._agent.max_delegation_depth

        # Sub-agents (agent-as-tool pattern).
        # Use a factory closure to capture each DelegatedAgent; keep
        # _subagent_invoke signature to `task: str` only so
        # Tool.from_callable doesn't reject on unsupported hint types.
        def _make_subagent_tool(da: DelegatedAgent) -> Any:
            # Apply agent-level cap: the effective max_depth is the
            # minimum of the DelegatedAgent's own limit and the Agent's.
            effective_da = da
            if da.max_depth != agent_max_depth and agent_max_depth < da.max_depth:
                from dataclasses import replace as _replace

                effective_da = _replace(da, max_depth=agent_max_depth)

            async def _subagent_invoke(task: str) -> str:
                """Invoke a sub-agent with a task; returns its response text."""
                return await effective_da.call(
                    task,
                    parent_session_id=parent_session,
                    runtime=runtime,
                    context=context,
                    vfs=vfs,
                )

            safe_name = da.name.replace("-", "_").replace(" ", "_")
            return Tool.from_callable(
                _subagent_invoke,
                name=safe_name,
                description=da.description,
            )

        for da in delegated:
            tools.append(_make_subagent_tool(da))

        # Handoffs (flat routing via a handoff tool).
        # Each handoff target becomes a handoff_to_<name> tool.
        # Handoffs participate in the same delegation-depth ContextVar so
        # handoff loops are bounded by max_delegation_depth.
        parent_hooks = self._hooks

        def _make_handoff_tool(target: Agent) -> Any:
            target_name = target.name or "handoff_target"
            target_safe_name = target_name.replace("-", "_").replace(" ", "_")

            async def _handoff_invoke(task: str) -> str:
                """Hand off to another agent with the given task; returns its response."""
                from kaos_agents.runtime.delegation import (
                    DelegationDepthExceeded,
                    _delegation_depth,
                )

                current = _delegation_depth.get()
                if current >= agent_max_depth:
                    raise DelegationDepthExceeded(
                        f"Handoff depth {current} reached max_delegation_depth "
                        f"{agent_max_depth} while routing to '{target_name}'. "
                        f"Increase Agent.max_delegation_depth, or flatten the "
                        f"handoff graph to avoid recursion. "
                        f"Alternative: use delegated_agents (sub-agents) if you "
                        f"need deeper hierarchies with per-agent depth limits."
                    )

                sub_runner = Runner(
                    target,
                    runtime=runtime,
                    context=context,
                    vfs=vfs,
                    hooks=parent_hooks,
                )
                token = _delegation_depth.set(current + 1)
                try:
                    response = await sub_runner.turn(task, parent_session)
                finally:
                    _delegation_depth.reset(token)
                return response.text

            return Tool.from_callable(
                _handoff_invoke,
                name=f"handoff_to_{target_safe_name}",
                description=(
                    f"Transfer control to '{target_name}' agent. Provide the task to delegate."
                ),
            )

        for target in handoffs:
            tools.append(_make_handoff_tool(target))

        return tuple(tools)

    def _build_single_delegation_tool(
        self, da: DelegatedAgent, session_id: str | None = None
    ) -> Any | None:
        """Build a single kaos-llm-core Tool from a DelegatedAgent.

        Used by ``_build_internal_agent`` to inject auto-created delegation
        tools (e.g., the RetrievalAgent) alongside user-declared ones.
        Returns None if kaos-llm-core is not available.
        """
        try:
            from kaos_llm_core import Tool
        except ImportError:
            return None

        parent_session = session_id or ""
        runtime = self._runtime
        context = self._context
        vfs = self._vfs

        async def _subagent_invoke(task: str) -> str:
            return await da.call(
                task,
                parent_session_id=parent_session,
                runtime=runtime,
                context=context,
                vfs=vfs,
            )

        safe_name = da.name.replace("-", "_").replace(" ", "_")
        return Tool.from_callable(
            _subagent_invoke,
            name=safe_name,
            description=da.description,
        )

    def _build_internal_agent(self, session_id: str | None = None) -> _InternalAgent:
        """Construct the appropriate internal agent based on pattern.

        Args:
            session_id: Session id — used to derive sub-session ids for
                delegated agents. None when constructing the internal
                agent without a specific session context.
        """
        pattern = self._agent.pattern
        model = self._agent.effective_model()
        tool_filter = self._agent.tool_filter()
        extra_tools = self._build_delegation_tools(session_id=session_id)

        # WS-0.1: thread the Runner's permission policy into the internal
        # agent so it applies at the tool-executor level — before any
        # side effect is committed. The runner's post-hoc check on
        # ``ToolCallStart`` events is retained as defense in depth.
        permission_policy = self._permission_policy

        if pattern == AgentPattern.RESEARCH:
            from kaos_agents.patterns.research import ResearchAgent

            # Auto-inject RetrievalAgent as a delegation tool so the
            # ResearchAgent can dynamically search documents during ReAct
            # turns (e.g., after RAG returns InsufficientEvidence).
            research_extra_tools = extra_tools
            try:
                from kaos_agents.retrieval_agent import create_retrieval_agent

                retrieval_tool = create_retrieval_agent(self._runtime, model=model)
                # Build the delegation wrapper for the retrieval agent
                if retrieval_tool is not None:
                    from kaos_agents.runtime.delegation import DelegatedAgent

                    if isinstance(retrieval_tool, DelegatedAgent):
                        da_tool = self._build_single_delegation_tool(retrieval_tool, session_id)
                        if da_tool is not None:
                            research_extra_tools = (*extra_tools, da_tool)
            except Exception as exc:
                logger.debug("runner: retrieval agent not available: %s", exc)

            return ResearchAgent(
                self._vfs,
                runtime=self._runtime,
                context=self._context,
                model=model,
                tool_filter=tool_filter,
                max_tools=self._agent.max_tools,
                max_react_iterations=self._agent.max_react_iterations,
                rag_top_k=self._agent.rag_top_k or self._settings.rag_top_k,
                rag_max_retries=self._agent.rag_max_retries or self._settings.rag_max_retries,
                settings=self._settings,
                provider=self._agent.provider,
                extra_llm_tools=research_extra_tools,
                permission_policy=permission_policy,
                instructions=self._agent.instructions,
                corpus=self._corpus,
            )

        if pattern == AgentPattern.PLAN:
            from kaos_agents.patterns.plan_execute import PlanExecuteAgent

            return PlanExecuteAgent(
                self._vfs,
                runtime=self._runtime,
                context=self._context,
                model=model,
                tool_filter=tool_filter,
                max_tools=self._agent.max_tools,
                max_react_iterations=self._agent.max_react_iterations,
                max_plan_steps=self._agent.max_plan_steps,
                settings=self._settings,
                provider=self._agent.provider,
                extra_llm_tools=extra_tools,
                permission_policy=permission_policy,
                instructions=self._agent.instructions,
            )

        # Default: CHAT
        from kaos_agents.patterns.chat import ChatAgent

        return ChatAgent(
            self._vfs,
            runtime=self._runtime,
            context=self._context,
            model=model,
            tool_filter=tool_filter,
            max_tools=self._agent.max_tools,
            max_react_iterations=self._agent.max_react_iterations,
            settings=self._settings,
            provider=self._agent.provider,
            extra_llm_tools=extra_tools,
            permission_policy=permission_policy,
            instructions=self._agent.instructions,
        )


# Type alias for the internal agent (any BaseAgent subclass).
# Used only for _build_internal_agent return type annotation.
from kaos_agents.runtime.agent import BaseAgent as _InternalAgent  # noqa: E402


def _mint_session_id() -> str:
    """Mint a fresh session id for trigger dispatches that omit one.

    Used by :meth:`Runner.run_trigger` v1 fallback when the trigger
    has no ``source_id``. Phase 4+ may centralise session-id minting
    in the SessionStore.
    """
    from uuid import uuid4

    return f"session_{uuid4().hex[:12]}"


def _resolve_vfs(runtime: KaosRuntime | None) -> VirtualFileSystem:
    """Get VFS from runtime, falling back to the kaos-core disk-backed default.

    The disk-backed default matches ``kaos_core.KaosRuntime`` and
    ``kaos_core.vfs.core.VirtualFileSystem`` — both default to
    ``StorageBackend.DISK`` rooted at ``.kaos-vfs/``. Callers that need an
    in-memory VFS (most commonly tests) should construct one explicitly
    and pass it via ``KaosRuntime(vfs=...)`` or ``Runner(vfs=...)``.
    """
    if runtime is not None and hasattr(runtime, "vfs") and runtime.vfs is not None:
        return runtime.vfs

    from kaos_core.vfs.core import VirtualFileSystem

    return VirtualFileSystem()
