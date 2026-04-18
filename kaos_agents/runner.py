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

from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any

from kaos_core.logging import get_logger

from kaos_agents._constants import RESULT_SUMMARY_TRUNCATE
from kaos_agents.config import Agent, AgentPattern
from kaos_agents.delegation import DelegatedAgent
from kaos_agents.events import (
    AgentEvent,
    EventEmitter,
    HandoffStart,
    RunError,
    SubagentComplete,
    SubagentStart,
    ToolCallApprovalRequired,
    ToolCallStart,
)
from kaos_agents.hooks import BaseHook, HookAction, dispatch_hook
from kaos_agents.interrupts import (
    PendingToolCall,
    RunState,
    memory_snapshot_path,
    save_event_log,
    save_run_state,
)
from kaos_agents.memory.store import SessionStore
from kaos_agents.models import AgentResponse
from kaos_agents.permissions import PermissionDecision, PermissionPolicy

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
        "_context",
        "_hooks",
        "_permission_policy",
        "_runtime",
        "_settings",
        "_vfs",
    )

    def __init__(
        self,
        agent: Agent,
        *,
        runtime: KaosRuntime | None = None,
        context: KaosContext | None = None,
        vfs: VirtualFileSystem | None = None,
        hooks: tuple[BaseHook, ...] = (),
        permission_policy: PermissionPolicy | None = None,
        corpus: Any | None = None,
    ) -> None:
        self._agent = agent
        self._runtime = runtime
        self._context = context
        self._settings = agent.resolve_settings()
        self._vfs = vfs or _resolve_vfs(runtime)
        self._hooks = hooks
        self._permission_policy = permission_policy
        self._corpus = corpus

    @property
    def agent(self) -> Agent:
        """The agent configuration this Runner executes."""
        return self._agent

    async def run(self, message: str, session_id: str) -> AsyncIterator[AgentEvent]:
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
            AgentEvent subclass instances in execution order.
        """
        internal = self._build_internal_agent(session_id=session_id)
        event_count = 0
        emitted: list[AgentEvent] = []  # tracked for pause persistence
        async for event in internal.run(message, session_id):
            # Run hooks
            if self._hooks:
                action = await dispatch_hook(self._hooks, event)
                if action == HookAction.SKIP:
                    continue
                if action == HookAction.REQUIRE_APPROVAL and isinstance(event, ToolCallStart):
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
            if self._permission_policy and isinstance(event, ToolCallStart):
                annotations = self._lookup_annotations(event.tool_name)
                decision = self._permission_policy.evaluate(event.tool_name, annotations)
                if decision == PermissionDecision.DENY:
                    continue  # Suppress denied tool call
                if decision == PermissionDecision.ASK:
                    reason = f"Permission policy requires approval for {event.tool_name}"
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
        tool_event: ToolCallStart,
        *,
        session_id: str,
        message: str,
        event_count: int,
        emitted: list[AgentEvent],
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
        from kaos_agents.interrupts import AgentSnapshot

        agent_config = AgentSnapshot.from_agent(self._agent)

        state = RunState(
            run_id=run_id,
            session_id=session_id,
            pending_tool_call=PendingToolCall(
                call_id=tool_event.call_id,
                tool_name=tool_event.tool_name,
                arguments=tool_event.arguments,
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
            call_id=tool_event.call_id,
            tool_name=tool_event.tool_name,
            arguments=tool_event.arguments,
            reason=reason,
            run_state_ref=state_path,
        )

    async def resume(
        self,
        run_state: RunState,
        *,
        approved: bool,
    ) -> AsyncIterator[AgentEvent]:
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
            AgentEvent stream of the continuation (or a RunError on denial).
        """
        emitter = EventEmitter(session_id=run_state.session_id, run_id=run_state.run_id)

        if not approved:
            yield emitter.emit(
                RunError,
                error_type="approval_denied",
                message=(
                    f"Human approval denied for tool call "
                    f"'{run_state.pending_tool_call.tool_name if run_state.pending_tool_call else 'unknown'}'."
                ),
                recovery_hint=(
                    "If this denial was a mistake, call resume() again with approved=True. "
                    "Otherwise, restart the run with a different message or different tools."
                ),
            )
            return

        # Replay pre-pause events first so the consumer sees the full timeline
        from kaos_agents.interrupts import load_event_log

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

        Args:
            message: The user's message.
            session_id: Session identifier for memory persistence.

        Returns:
            AgentResponse with the agent's reply and metadata aggregated
            from the streamed event sequence.
        """
        # Local import avoids a circular-import at module load time —
        # kaos_agents.events imports from the same package.
        from kaos_agents.events import (
            IntentClassified,
            RunError,
            TextDelta,
            ToolCallResult,
            TurnComplete,
            TurnStart,
        )
        from kaos_agents.models import IntentResult, IntentType, ToolCallRecord

        text_parts: list[str] = []
        tool_calls: list[ToolCallRecord] = []
        intent_result: IntentResult | None = None
        turn_complete: TurnComplete | None = None
        turn_start_number: int = 0
        run_error: RunError | None = None

        async for event in self.run(message, session_id):
            if isinstance(event, TextDelta):
                text_parts.append(event.content)
            elif isinstance(event, ToolCallResult):
                tool_calls.append(
                    ToolCallRecord.from_dict_args(
                        tool_name=event.tool_name,
                        arguments={},
                        result_summary=event.result_summary,
                        is_error=event.is_error,
                    )
                )
            elif isinstance(event, IntentClassified):
                intent_result = IntentResult(
                    intent=IntentType(event.intent),
                    confidence=event.confidence,
                    reasoning=event.reasoning,
                )
            elif isinstance(event, TurnStart):
                turn_start_number = event.turn_number
            elif isinstance(event, TurnComplete):
                turn_complete = event
            elif isinstance(event, RunError):
                run_error = event

        # Prefer TurnComplete's text (which already aggregates TextDelta
        # content and includes pattern-specific post-processing); fall
        # back to concatenated deltas when TurnComplete was not emitted
        # (e.g. early pause, run_error).
        if turn_complete is not None and turn_complete.text:
            text = turn_complete.text
        else:
            text = "".join(text_parts)

        if intent_result is None:
            # No classification event — synthesize a default so AgentResponse
            # stays well-typed. Happens on error paths before intent fires.
            intent_result = IntentResult(
                intent=IntentType.RESPOND,
                confidence=0.0,
                reasoning="no IntentClassified event (run aborted early or errored)",
            )

        metadata: dict[str, Any] = {"session_id": session_id}
        if run_error is not None:
            metadata["error_type"] = run_error.error_type
            metadata["error_message"] = run_error.message

        tokens_used = turn_complete.tokens_used if turn_complete is not None else 0
        return AgentResponse.create(
            text=text,
            intent=intent_result,
            tool_calls=tuple(tool_calls),
            turn_number=turn_start_number,
            tokens_used=tokens_used,
            metadata=metadata,
        )

    async def delegate(
        self,
        delegated: DelegatedAgent,
        task: str,
        session_id: str,
    ) -> AsyncIterator[AgentEvent]:
        """Execute a delegated sub-agent, yielding start/complete events.

        Wraps ``DelegatedAgent.call()`` with event emission so the
        sub-agent's execution is visible in the parent's event stream.

        Yields:
            SubagentStart event, then SubagentComplete with the result.

        Raises:
            DelegationDepthExceeded: If nested delegation exceeds max_depth.
        """
        from kaos_agents.agent import _generate_run_id

        run_id = _generate_run_id()
        emitter = EventEmitter(session_id=session_id, run_id=run_id)

        yield emitter.emit(
            SubagentStart,
            subagent_name=delegated.name,
            task=task,
        )

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

        yield emitter.emit(
            SubagentComplete,
            subagent_name=delegated.name,
            result_summary=result_text[:RESULT_SUMMARY_TRUNCATE],
            tokens_used=0,
        )

    async def handoff(
        self,
        target: Agent,
        message: str,
        session_id: str,
        *,
        from_agent_name: str = "",
        reason: str = "",
    ) -> AsyncIterator[AgentEvent]:
        """Transfer control to another agent, sharing the session.

        Unlike ``delegate()`` which creates an isolated sub-session,
        ``handoff()`` uses the same session_id — the target agent sees
        the parent's memory.

        Yields:
            HandoffStart event, then all events from the target agent's run.
        """
        from kaos_agents.agent import _generate_run_id

        run_id = _generate_run_id()
        emitter = EventEmitter(session_id=session_id, run_id=run_id)

        target_name = target.name or "unnamed"
        yield emitter.emit(
            HandoffStart,
            from_agent=from_agent_name,
            to_agent=target_name,
            reason=reason,
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
                from kaos_agents.delegation import (
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
                    from kaos_agents.delegation import DelegatedAgent

                    if isinstance(retrieval_tool, DelegatedAgent):
                        da_tool = self._build_single_delegation_tool(
                            retrieval_tool, session_id
                        )
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
from kaos_agents.agent import BaseAgent as _InternalAgent  # noqa: E402


def _resolve_vfs(runtime: KaosRuntime | None) -> VirtualFileSystem:
    """Get VFS from runtime, falling back to in-memory VFS."""
    if runtime is not None and hasattr(runtime, "vfs") and runtime.vfs is not None:
        return runtime.vfs

    from kaos_core.types.enums import StorageBackend
    from kaos_core.vfs.core import VirtualFileSystem
    from kaos_core.vfs.models import VFSConfig

    config = VFSConfig(default_backend=StorageBackend.MEMORY)
    return VirtualFileSystem(config=config)
