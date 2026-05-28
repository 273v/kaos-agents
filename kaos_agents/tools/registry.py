"""MCP tools for kaos-agents.

Exposes agent capabilities through the standard KaosTool → MCP adapter:
- kaos-agent-chat: Single-turn conversational agent with optional tool calling
- kaos-agent-plan: Multi-step plan-execute agent for complex goals
- kaos-agent-memory-query: Read session memory contents
- kaos-agent-memory-clear: Clear a session's memory
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from kaos_core.base.tool import KaosTool
from kaos_core.logging import get_logger
from kaos_core.types.enums import StorageBackend
from kaos_core.types.metadata import ToolAnnotations, ToolCapability, ToolCategory, ToolMetadata
from kaos_core.types.parameters import ParameterSchema
from kaos_core.types.results import ToolResult
from kaos_core.vfs.core import VirtualFileSystem
from kaos_core.vfs.models import VFSConfig

if TYPE_CHECKING:
    from kaos_core.base.context import KaosContext
    from kaos_core.registry.container import KaosRuntime

logger = get_logger(__name__)

_MODULE = "kaos-agents"
_VERSION = "0.1.0"

_AGENT_READ_ANNOTATIONS = ToolAnnotations(
    readOnlyHint=True,
    destructiveHint=False,
    idempotentHint=True,
    openWorldHint=True,
)

_AGENT_WRITE_ANNOTATIONS = ToolAnnotations(
    readOnlyHint=False,
    destructiveHint=False,
    idempotentHint=False,
    openWorldHint=True,
)

_AGENT_DESTRUCTIVE_ANNOTATIONS = ToolAnnotations(
    readOnlyHint=False,
    destructiveHint=True,
    idempotentHint=True,
    openWorldHint=False,
)


def _get_vfs(runtime: KaosRuntime | None) -> VirtualFileSystem:
    """Get VFS from runtime, falling back to in-memory VFS."""
    if runtime is not None and hasattr(runtime, "vfs") and runtime.vfs is not None:
        return runtime.vfs
    config = VFSConfig(default_backend=StorageBackend.MEMORY)
    return VirtualFileSystem(config=config)


def _resolve_caller_session_id(
    inputs: dict[str, Any],
    context: KaosContext | None,
    *,
    field: str = "session_id",
) -> tuple[str, ToolResult | None]:
    """Resolve the session id a memory tool should operate on.

    Memory tools (`kaos-agent-memory-query`, `kaos-agent-memory-clear`,
    `kaos-agent-memory-search`) historically accepted
    ``inputs["session_id"]`` and ignored ``context.session_id``. A
    caller scoped to session A could therefore read or destroy session
    B's memory by passing B's id as a string parameter — a horizontal
    privilege break that flatly contradicts kaos-core's session-scoped
    VFS / artifact / tool model (see ``kaos_core.tools`` VFS tools,
    which derive the scope from ``context.session_id``).

    Resolution rule:

    - When the calling :class:`KaosContext` has a non-empty
      ``session_id``, that is the authoritative scope. ``inputs[field]``,
      if present, MUST equal it or the call is refused. A confused or
      hostile caller passing a sibling id gets an actionable refusal
      naming the mismatch, never a silent retarget onto the requested
      session.
    - When the calling context is ``None`` (CLI / standalone tests /
      stub callers), ``inputs[field]`` is the only signal — refuse if
      empty.

    Returns ``(session_id, None)`` on success or
    ``("", ToolResult.create_error(...))`` on failure. Callers MUST
    return the error ToolResult unchanged and short-circuit before any
    SessionStore / VFS access.
    """
    requested = (inputs.get(field) or "").strip()
    ctx_sid = ""
    if context is not None:
        sid_attr = getattr(context, "session_id", None)
        if sid_attr:
            ctx_sid = str(sid_attr).strip()

    if ctx_sid:
        if requested and requested != ctx_sid:
            return "", ToolResult.create_error(
                f"Refused: '{field}={requested!r}' does not match the "
                f"calling session ({ctx_sid!r}). Memory tools always "
                f"operate on the caller's own session — pass no "
                f"'{field}' argument to use the calling session, or "
                f"pass the same id."
            )
        return ctx_sid, None

    if not requested:
        return "", ToolResult.create_error(
            f"Missing '{field}' parameter and no calling-session "
            f"context. Provide a session id (the caller must scope to "
            f"a session)."
        )
    return requested, None


def _resolve_invocation_session_id(
    inputs: dict[str, Any],
    context: KaosContext | None,
    *,
    field: str = "session_id",
) -> tuple[str, ToolResult | None]:
    """Resolve session id for top-level agent-invocation tools.

    ``AgentChatTool`` / ``AgentPlanTool`` are externally-driven endpoints:
    an MCP client, HTTP API caller, or CLI invokes them with the
    session_id they want the agent to run in. Unlike memory tools (which
    read/destroy existing session memory and need the strict B0.1
    same-session guard from :func:`_resolve_caller_session_id`), these
    tools CREATE / RESUME a session on behalf of the external caller.

    Resolution rule:

    - ``inputs[field]`` is authoritative when present — the external
      caller picked the session it wants the agent to run in.
    - When ``inputs[field]`` is absent, fall back to ``context.session_id``
      (in-process sub-agent invocation, e.g. ``agent_as_tool``).
    - When both are absent, refuse with the same actionable message as
      the memory-tool resolver.

    Returns ``(session_id, None)`` on success or
    ``("", ToolResult.create_error(...))`` on failure.
    """
    requested = (inputs.get(field) or "").strip()
    if requested:
        return requested, None
    if context is not None:
        sid_attr = getattr(context, "session_id", None)
        if sid_attr:
            return str(sid_attr).strip(), None
    return "", ToolResult.create_error(
        f"Missing '{field}' parameter and no calling-session "
        f"context. Provide a session id (the caller must scope to "
        f"a session)."
    )


def _build_invocation_context(
    context: KaosContext | None,
    *,
    runtime: KaosRuntime | None,
    session_id: str,
) -> KaosContext | None:
    """Return a :class:`KaosContext` whose ``session_id`` is ``session_id``.

    Closes the B0.1-class split-brain that
    :func:`_resolve_invocation_session_id` opened for the agent-invocation
    tools (``AgentChatTool`` / ``AgentPlanTool``). The resolver lets
    ``inputs["session_id"]`` win over ``context.session_id`` — necessary
    for MCP ergonomics (the MCP context's ``session_id`` is the caller's
    client identity, not the agent chat session). But the rest of the
    pipeline (hydration's ``caller_session_id``, memory persistence,
    bridged tools' ``context.session_id``, VFS namespace, artifact-guard
    scope) MUST see the same session id the resolver picked — otherwise
    a caller scoped to ``"attacker"`` could push memory writes and tool
    invocations into ``"victim"`` while bridged runtime tools still ran
    under attacker's scope.

    Resolution rule:

    - When ``context.session_id`` already matches ``session_id``, return
      the original context unchanged (cheap, preserves caller-supplied
      ``default_vfs_namespace``, ``trace_id``, etc.).
    - When ``context`` exists but its ``session_id`` differs, derive a
      child context via :meth:`KaosContext.create_child_context` with
      the resolved ``session_id`` so every downstream boundary resolves
      uniformly against the same scope.
    - When ``context`` is ``None`` (CLI / standalone callers), build a
      fresh ``KaosContext`` from the runtime so bridged tools and
      hydration still see a valid scoped context.

    The original calling context's identity (e.g. the MCP ``client_id``)
    is preserved on the child via ``metadata["caller_session_id"]`` so
    audit hooks that need the caller-vs-target distinction can still
    inspect it without affecting boundary resolution.
    """
    from kaos_core.base.context import KaosContext as _KaosContext

    if context is not None:
        ctx_sid = (getattr(context, "session_id", "") or "").strip()
        if ctx_sid == session_id:
            return context
        return context.create_child_context(
            session_id=session_id,
            metadata={"caller_session_id": ctx_sid} if ctx_sid else {},
        )
    return _KaosContext.create(session_id=session_id, runtime=runtime)


def _surfacing_error_from_status(status: dict[str, Any]) -> str | None:
    """Return a ToolResult.create_error message when the run hit an
    actionable infrastructure failure, else ``None``.

    Implements the contract from probe 4b in
    ``docs/design/skeptic-prod-ops-findings.md``: auth-failure,
    rate-limit, service-unavailable, transport, and context-too-large
    errors MUST surface as ``isError=True`` rather than the historic
    silent ``text=""`` empty success that broke SOC2 alerting.

    The message follows the agent-friendly error contract (see
    ``CLAUDE.md``): (1) what went wrong, (2) how to fix it, (3)
    alternative tool. The runtime emits ``error_type`` on the
    :class:`RunError` event as the structured kind (e.g.
    ``"auth_failure"``); we cross-check against the surfacing set so
    a non-surfacing RunError (e.g. one we couldn't classify and
    emitted as ``error_type=type(exc).__name__``) still falls through
    to the normal success envelope.
    """
    from kaos_agents.errors import SURFACING_FAILURE_KINDS

    event = status.get("run_error_event")
    if event is None:
        return None
    error_type = getattr(event, "error_type", "") or ""
    if error_type not in SURFACING_FAILURE_KINDS:
        return None
    message = getattr(event, "message", "") or ""
    recovery_hint = getattr(event, "recovery_hint", "") or ""
    summary = f"Agent run failed: {error_type}."
    if message:
        summary = f"{summary} {message}"
    if recovery_hint:
        summary = f"{summary} {recovery_hint}"
    return summary


async def _run_turn_with_status(
    runner: Any,
    message: str,
    session_id: str,
    *,
    is_internal_iteration: bool = False,
) -> tuple[Any, dict]:
    """Drain ``runner.run()`` once, building both an AgentResponse and a
    status dict that surfaces budget / pause events at the tool boundary.

    Runner.turn() folds events into AgentResponse but does NOT expose
    BudgetExceeded or ToolCallApprovalRequired in its return value, so
    the MCP tool surface previously had no way to tell callers "the run
    paused" or "the run blew its budget." We do the same fold here plus
    a side-channel status dict so the tool result can report it.

    ``is_internal_iteration`` is forwarded to ``runner.run()`` for the
    AgenticLoop iteration-leak fix
    (docs/plans/2026-05-19-agentic-loop-honesty.md §3.1.a).
    """
    from kaos_agents.events import (
        BudgetExceeded,
        IntentClassified,
        RunError,
        Span,
        SpanPhase,
        SpanSubject,
        TextDelta,
        ToolCallApprovalRequired,
        TurnSummary,
    )
    from kaos_agents.types import (
        AgentResponse,
        IntentResult,
        IntentType,
        ToolExecution,
    )

    text_parts: list[str] = []
    tool_calls: list[ToolExecution] = []
    intent_result: IntentResult | None = None
    turn_summary: TurnSummary | None = None
    turn_start_number = 0
    run_error: RunError | None = None
    status: dict[str, Any] = {
        "budget_exceeded": False,
        "budget_kind": None,
        "paused_for_approval": False,
        "pending_tool_name": None,
        "run_state_ref": None,
        # Probe 4b: a RunError observed during the run that the tool
        # wrapper must surface as ToolResult.isError=True. Stored as the
        # raw event so the surfacing helper can read error_type +
        # message + recovery_hint without re-walking the stream.
        "run_error_event": None,
        # Sprint-3 #9 — the actual cost spent on this turn. Read off
        # the TurnSummary event's cost_usd field. The tool wrapper
        # uses this to compare against the configured cap and emit
        # BudgetExceeded post-call when the cap was breached. None
        # until the turn produces a TurnSummary (or 0.0 if the run
        # never made an LLM call).
        "cost_usd": 0.0,
    }

    async for event in runner.run(message, session_id, is_internal_iteration=is_internal_iteration):
        if isinstance(event, TextDelta):
            text_parts.append(event.content)
        elif (
            isinstance(event, Span)
            and event.subject == SpanSubject.TOOL_CALL
            and event.phase == SpanPhase.COMPLETE
        ):
            attrs = event.attributes
            tool_calls.append(
                ToolExecution.from_dict_args(
                    tool_name=str(attrs.get("tool_name", "")),
                    arguments={},
                    result_summary=str(attrs.get("result_summary", "")),
                    is_error=bool(attrs.get("is_error", False)),
                )
            )
        elif isinstance(event, IntentClassified):
            intent_result = IntentResult(
                intent=IntentType(event.intent),
                confidence=event.confidence,
                reasoning=event.reasoning,
            )
        elif (
            isinstance(event, Span)
            and event.subject == SpanSubject.TURN
            and event.phase == SpanPhase.START
        ):
            turn_start_number = int(event.attributes.get("turn_number", 0) or 0)
        elif isinstance(event, TurnSummary):
            turn_summary = event
        elif isinstance(event, RunError):
            run_error = event
            status["run_error_event"] = event
        elif isinstance(event, BudgetExceeded):
            status["budget_exceeded"] = True
            status["budget_kind"] = event.kind
        elif isinstance(event, ToolCallApprovalRequired):
            status["paused_for_approval"] = True
            status["pending_tool_name"] = event.tool_name
            status["run_state_ref"] = event.run_state_ref

    if turn_summary is not None and turn_summary.text:
        text = turn_summary.text
    else:
        text = "".join(text_parts)
    if turn_summary is not None:
        status["cost_usd"] = float(turn_summary.cost_usd or 0.0)
    if intent_result is None:
        intent_result = IntentResult(
            intent=IntentType.RESPOND,
            confidence=0.0,
            reasoning="no IntentClassified event (run paused, aborted, or errored)",
        )
    metadata: dict = {"session_id": session_id}
    if run_error is not None:
        metadata["error_type"] = run_error.error_type
        metadata["error_message"] = run_error.message
    tokens_used = turn_summary.tokens_used if turn_summary is not None else 0
    cost_usd = float(turn_summary.cost_usd or 0.0) if turn_summary is not None else 0.0
    # Sprint-3 #10 — also push total_tokens into the side-channel
    # status dict so the tool surface can include it in
    # structuredContent without re-reading the events stream.
    status["total_tokens"] = tokens_used
    response = AgentResponse.create(
        text=text,
        intent=intent_result,
        tool_calls=tuple(tool_calls),
        turn_number=turn_start_number,
        tokens_used=tokens_used,
        cost_usd=cost_usd,
        total_tokens=tokens_used,
        metadata=metadata,
    )
    return response, status


class AgentChatTool(KaosTool):
    """Run a single conversational turn with optional tool calling.

    The ChatAgent dispatches to ReAct when tools are available and the
    user's message requires tool use. Otherwise it responds conversationally.
    Session memory persists across turns via VFS.
    """

    @property
    def metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name="kaos-agent-chat",
            display_name="Agent Chat",
            description=(
                "Run a single conversational turn with a KAOS agent. "
                "Supports tool calling via ReAct when tools are registered on the runtime. "
                "Session memory persists across turns — use the same session_id for multi-turn. "
                "For multi-step planning tasks, use kaos-agent-plan instead. "
                "TRANSPARENCY (Sprint-3 #10): structuredContent carries "
                "``cost_usd: float`` (USD spent on this turn) and "
                "``total_tokens: int`` (aggregate input+output tokens) "
                "as top-level fields. Both numbers come from the same "
                "TurnSummary aggregate that feeds AgentResponse.cost_usd "
                "and AgentResponse.total_tokens — the MCP surface and "
                "the in-process surface return identical numbers."
            ),
            category=ToolCategory.TEXT,
            capability=ToolCapability.TRANSFORM,
            module_name=_MODULE,
            version=_VERSION,
            annotations=_AGENT_WRITE_ANNOTATIONS,
            input_schema=[
                ParameterSchema(
                    name="message",
                    type="string",
                    description="The user's message to the agent.",
                ),
                ParameterSchema(
                    name="session_id",
                    type="string",
                    description=(
                        "Session identifier for memory persistence. "
                        "Use the same ID across turns for conversation continuity."
                    ),
                ),
                ParameterSchema(
                    name="model",
                    type="string",
                    description=(
                        "LLM model for the agent. Default: anthropic:claude-haiku-4-5. "
                        "Use a stronger model (anthropic:claude-sonnet-4-6) for complex tasks."
                    ),
                    required=False,
                ),
                ParameterSchema(
                    name="tool_filter",
                    type="string",
                    description=(
                        "Comma-separated list of tool names to make available. "
                        "Default: all registered tools. "
                        "Example: kaos-source-fr-search,kaos-web-get-markdown"
                    ),
                    required=False,
                ),
                ParameterSchema(
                    name="max_cost_usd",
                    type="number",
                    description=(
                        "Soft cost ceiling for this turn (USD). "
                        "Checked AFTER the turn's LLM call completes; "
                        "when the actual spend exceeds the cap, the "
                        "structuredContent payload carries "
                        "``budget_exceeded=true`` and a "
                        "``BudgetExceeded(kind='cost')`` event is "
                        "emitted. The ChatAgent's underlying ReAct "
                        "loop is one logical LLM invocation — the "
                        "agent cannot abort it mid-flight, so a "
                        "single turn's spend may overshoot by one "
                        "call's worth. Worst-case overshoot is "
                        "bounded by the model's per-call cost; "
                        "typical overshoot is within 5-25% of the "
                        "cap. For STRICT enforcement across many "
                        "smaller calls, use kaos-agent-findings "
                        "(chunk-level cap, <5% overshoot). For "
                        "multi-step plan execution, use "
                        "kaos-agent-plan (per-step cap). Set null "
                        "or omit for no cap."
                    ),
                    required=False,
                ),
                ParameterSchema(
                    name="require_approval_for_tools",
                    type="string",
                    description=(
                        "Comma-separated glob patterns of tool names that "
                        "require explicit approval (ASK rule). When the agent "
                        "tries to call a matching tool, the Runner pauses "
                        "and the response text will be empty (no streaming "
                        "approve/resume from this tool surface). Example: "
                        "kaos-source-fr-*,kaos-web-*"
                    ),
                    required=False,
                ),
                ParameterSchema(
                    name="instructions",
                    type="string",
                    description=(
                        "System-prompt override for the agent (persona, "
                        "rubric, refusal policy). Mirrors the CLI's "
                        "--instructions flag. Use for legal-review, "
                        "senior-counsel, or domain-specific personas. "
                        "Example: 'You are senior counsel; flag every "
                        "deviation by exact filename and section.'"
                    ),
                    required=False,
                ),
            ],
        )

    async def execute(
        self, inputs: dict[str, Any], context: KaosContext | None = None
    ) -> ToolResult:
        message = inputs.get("message", "")
        session_id, err = _resolve_invocation_session_id(inputs, context)
        if err is not None:
            return err
        model = inputs.get("model")
        tool_filter_str = inputs.get("tool_filter")
        max_cost_usd_raw = inputs.get("max_cost_usd")
        max_cost_usd: float | None = None if max_cost_usd_raw is None else float(max_cost_usd_raw)
        if max_cost_usd is not None and max_cost_usd <= 0.0:
            return ToolResult.create_error(
                f"max_cost_usd must be > 0 when set, got {max_cost_usd}. "
                "Pass null (or omit) to disable the cap."
            )
        approval_patterns_str = inputs.get("require_approval_for_tools")
        instructions = inputs.get("instructions") or None

        if not message:
            return ToolResult.create_error(
                "Missing 'message' parameter. "
                "Provide the user's message text. "
                'Example: {"message": "What is the Federal Register?"}'
            )
        if not session_id:
            return ToolResult.create_error(
                "Missing 'session_id' parameter. "
                "Provide a unique session identifier for memory persistence. "
                'Example: {"session_id": "research-epa-rules"}'
            )

        try:
            from kaos_agents.config import Agent, AgentPattern
            from kaos_agents.runtime.runner import Runner
            from kaos_agents.settings import DEFAULT_MODEL, KaosAgentSettings

            runtime = context.runtime if context else None

            # B0.1 — derive a per-invocation context whose ``session_id``
            # matches the resolved chat session, so memory hydration, the
            # bridged tools' ``context.session_id``, VFS namespace lookups,
            # and the artifact-guard scope all land on the SAME session.
            # Without this the resolver above can pick ``inputs.session_id``
            # while the bridged runtime tools still see ``context.session_id``
            # — the split-brain reopened the B0.1 horizontal-isolation gap
            # that ``_resolve_caller_session_id`` closed for memory tools.
            invocation_context = _build_invocation_context(
                context, runtime=runtime, session_id=session_id
            )

            tool_filter = (
                tuple(t.strip() for t in tool_filter_str.split(",") if t.strip())
                if tool_filter_str
                else ()
            )

            agent_settings = None
            if max_cost_usd is not None:
                # Sprint-3 #9 — also tighten max_react_iterations as
                # a function of the cap. The chat-pattern cannot
                # abort ReAct mid-flight, so the only way to bound
                # the worst-case overshoot inside a single turn is
                # to cap ReAct iterations proportional to how many
                # LLM calls fit in the budget. A typical Haiku call
                # on a short prompt is ~$0.001-0.005; round to
                # $0.005 per call as a conservative ceiling. The
                # result keeps overshoot within ±5% of the cap on
                # the small-cap test path (probe 2 reproduction).
                # When the cap is generous (>= $0.05) we leave the
                # default iterations alone — the cap won't bind.
                react_iter_budget: int | None = None
                if max_cost_usd < 0.05:
                    # Force-stop after one ReAct turn — sufficient
                    # for the small-cap honesty test, and ReAct's
                    # max_iterations >= 1 guard accepts this.
                    react_iter_budget = max(1, int(max_cost_usd / 0.005))
                kwargs_for_settings: dict[str, Any] = {
                    "plan_max_cost_usd": max_cost_usd,
                }
                if react_iter_budget is not None:
                    kwargs_for_settings["max_react_iterations"] = react_iter_budget
                agent_settings = KaosAgentSettings(**kwargs_for_settings)

            permission_policy = None
            if approval_patterns_str:
                from kaos_agents.runtime.permissions import PermissionPolicy
                from kaos_agents.types.permissions import PermissionDecision, PermissionRule

                patterns = tuple(p.strip() for p in approval_patterns_str.split(",") if p.strip())
                permission_policy = PermissionPolicy(
                    rules=tuple(
                        PermissionRule(
                            pattern=pat,
                            action=PermissionDecision.ASK,
                            reason="per-request require_approval_for_tools",
                        )
                        for pat in patterns
                    )
                )

            agent_kwargs: dict[str, Any] = {
                "pattern": AgentPattern.CHAT,
                "model": model or DEFAULT_MODEL,
                "tools": tool_filter,
                "settings": agent_settings,
            }
            if instructions:
                agent_kwargs["instructions"] = instructions
            agent_config = Agent(**agent_kwargs)
            runner = Runner(
                agent_config,
                runtime=runtime,
                context=invocation_context,
                permission_policy=permission_policy,
            )

            # PA5: auto-hydrate any VFS artifact references found in the
            # user message into SessionMemory.DOCUMENTS BEFORE the runner
            # loads the session. The runner's internal agent calls
            # ``SessionStore.load_or_create(session_id)`` on every turn,
            # so we have to write through the store so the runner sees
            # the hydrated content on the same turn.
            hydrated_artifacts: tuple = ()
            try:
                from kaos_agents.memory.store import SessionStore
                from kaos_agents.runtime.artifact_hydration import (
                    hydrate_artifacts_from_message,
                )

                vfs = _get_vfs(runtime)
                store = SessionStore(vfs)
                memory_for_hydration = await store.load_or_create(session_id)
                hydrated_artifacts = await hydrate_artifacts_from_message(
                    message,
                    memory=memory_for_hydration,
                    runtime=runtime,
                    caller_session_id=session_id,
                )
                if hydrated_artifacts:
                    await store.save(memory_for_hydration)
                    logger.info(
                        "kaos-agent-chat.auto_hydrate: session=%s injected=%d artifact(s)",
                        session_id,
                        len(hydrated_artifacts),
                    )
            except Exception as exc:
                # Auto-hydration is best-effort. A failure here MUST NOT
                # break the chat turn — if the user's intent doesn't
                # actually depend on the artifact, the turn still works.
                # We log a WARNING so the failure is visible without
                # blocking the response.
                logger.warning("kaos-agent-chat.auto_hydrate: failed: %s", exc)

            response, status = await _run_turn_with_status(runner, message, session_id)

            # Surface an actionable RunError as ToolResult.isError=True.
            # See skeptic probe 4b: an auth/rate-limit/transport failure
            # silently producing ``text=""`` + ``isError=False`` breaks
            # SOC2 alerting. Each surfacing kind names the credential or
            # provider and the recovery path.
            surfacing = _surfacing_error_from_status(status)
            if surfacing is not None:
                return ToolResult.create_error(surfacing)

            # Sprint-3 #9 (transparency — the tool keeps its own
            # contract). The chat-pattern's underlying ReAct call
            # is one logical LLM invocation; we cannot abort it
            # mid-flight from this tool surface, but we MUST honor
            # the post-call contract: when the actual spend exceeds
            # the cap, surface budget_exceeded=True and emit a
            # BudgetExceeded(kind='cost') event so observers (SOC2
            # audit, MCP wire stream, OTel) see the breach instead
            # of a silent overshoot. The probe-2 +21% overshoot
            # bug was the tool silently dropping this check; this
            # is the fix.
            actual_cost = float(status.get("cost_usd") or 0.0)
            if max_cost_usd is not None and actual_cost > max_cost_usd:
                status["budget_exceeded"] = True
                if not status.get("budget_kind"):
                    status["budget_kind"] = "cost"
                logger.info(
                    "kaos-agent-chat.budget_exceeded: actual=$%.4f > "
                    "cap=$%.4f (chat pattern is one ReAct call; "
                    "overshoot bounded by per-call cost)",
                    actual_cost,
                    max_cost_usd,
                )

            result_data: dict[str, Any] = {
                "text": response.text,
                "turn_number": response.turn_number,
                "intent": response.intent.intent.value if response.intent else "unknown",
                "tool_calls": [
                    {
                        "tool_name": tc.tool_name,
                        "result_summary": tc.result_summary,
                        "is_error": tc.is_error,
                    }
                    for tc in response.tool_calls
                ],
                "budget_exceeded": status["budget_exceeded"],
                "paused_for_approval": status["paused_for_approval"],
                # Sprint-3 #9 — surface the actual measured cost so
                # consumers can verify the contract. This is the
                # source of truth; ``budget_exceeded`` is derived
                # from this vs. the configured cap.
                "cost_usd": actual_cost,
                # Sprint-3 #10 (transparency lens) — total tokens for
                # this turn, aggregated from every UsageObserved event
                # via TurnSummary.tokens_used. Matches the value on
                # AgentResponse.total_tokens for in-process callers.
                "total_tokens": int(status.get("total_tokens") or 0),
            }
            if max_cost_usd is not None:
                result_data["max_cost_usd"] = max_cost_usd
            if status["paused_for_approval"]:
                result_data["pending_tool_name"] = status["pending_tool_name"]
                result_data["run_state_ref"] = status["run_state_ref"]
            if status["budget_exceeded"]:
                result_data["budget_kind"] = status["budget_kind"]
            # PA5: report what was auto-hydrated this turn so consumers
            # (tests, the audit pipeline, downstream observers) can see
            # which artifact references were picked up without re-reading
            # session memory.
            if hydrated_artifacts:
                result_data["hydrated_artifacts"] = [
                    {
                        "artifact_id": h.artifact_id,
                        "uri": h.uri,
                        "name": h.name,
                        "size": h.size,
                        "mime_type": h.mime_type,
                        "tier": h.tier,
                        "bytes_loaded": h.bytes_loaded,
                    }
                    for h in hydrated_artifacts
                ]

            summary = response.text if response.text else "(empty response)"
            if status["budget_exceeded"]:
                summary = f"[BudgetExceeded: {status['budget_kind']}] {summary}"
            elif status["paused_for_approval"]:
                summary = f"[ApprovalRequired: {status['pending_tool_name']}] {summary}"
            elif response.tool_calls:
                tools_used = ", ".join(tc.tool_name for tc in response.tool_calls)
                summary = f"[Tools: {tools_used}] {summary}"

            return ToolResult.create_success(output=result_data, summary=summary)

        except Exception as exc:
            logger.warning("kaos-agent-chat: failed: %s", exc)
            return ToolResult.create_error(
                f"Agent chat failed: {exc}. "
                "Check that the session_id is valid and the LLM API key is configured. "
                "For simpler queries, try kaos-agent-chat with no tool_filter."
            )


class AgentPlanTool(KaosTool):
    """Run a multi-step plan-execute agent for complex goals.

    Uses the adaptive planning strategy: simple goals execute directly,
    complex goals are decomposed into steps and executed with tools.
    """

    @property
    def metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name="kaos-agent-plan",
            display_name="Agent Plan-Execute",
            description=(
                "Run a plan-execute agent for complex multi-step goals. "
                "Decomposes the goal into steps, executes them with "
                "available tools, "
                "and synthesizes results. For simple conversational queries, "
                "use kaos-agent-chat. "
                "Session memory persists — prior plan failures inform "
                "future attempts via Reflexion. "
                "TRANSPARENCY (Sprint-3 #10): structuredContent carries "
                "``cost_usd: float`` (USD spent across all plan steps) "
                "and ``total_tokens: int`` (aggregate input+output "
                "tokens). Numbers match AgentResponse.cost_usd / "
                "AgentResponse.total_tokens — the in-process and MCP "
                "surfaces are aligned."
            ),
            category=ToolCategory.TEXT,
            capability=ToolCapability.TRANSFORM,
            module_name=_MODULE,
            version=_VERSION,
            annotations=_AGENT_WRITE_ANNOTATIONS,
            input_schema=[
                ParameterSchema(
                    name="message",
                    type="string",
                    description="The user's goal or multi-step request.",
                ),
                ParameterSchema(
                    name="session_id",
                    type="string",
                    description=(
                        "Session identifier for memory persistence. "
                        "Use the same ID across turns for context continuity."
                    ),
                ),
                ParameterSchema(
                    name="model",
                    type="string",
                    description=(
                        "LLM model for planning and execution. "
                        "Default: anthropic:claude-haiku-4-5. "
                        "Use anthropic:claude-sonnet-4-6 for complex multi-step research."
                    ),
                    required=False,
                ),
                ParameterSchema(
                    name="tool_filter",
                    type="string",
                    description=(
                        "Comma-separated list of tool names to make available for planning. "
                        "Default: all registered tools."
                    ),
                    required=False,
                ),
                ParameterSchema(
                    name="max_steps",
                    type="integer",
                    description="Maximum plan steps. Default: 20.",
                    required=False,
                    constraints={"min": 1, "max": 50},
                ),
                ParameterSchema(
                    name="max_cost_usd",
                    type="number",
                    description=(
                        "Strict per-plan cost ceiling in USD. Threads "
                        "to KaosAgentSettings.plan_max_cost_usd. The "
                        "plan-execute path checks the cap AFTER each "
                        "step completes and aborts BEFORE dispatching "
                        "the next step when the cap is breached. The "
                        "Runner emits BudgetExceeded(kind='cost') and "
                        "stops; the response carries "
                        "``budget_exceeded=true``. Worst-case "
                        "overshoot is one step's worth of cost (step "
                        "granularity, not call granularity). Set "
                        "null to disable. Use 0.001 to force a budget "
                        "exceedance in tests."
                    ),
                    required=False,
                ),
                ParameterSchema(
                    name="require_approval_for_tools",
                    type="string",
                    description=(
                        "Comma-separated glob patterns for tools that "
                        "require approval (ASK rule). When matched, the "
                        "Runner pauses; response text will be empty."
                    ),
                    required=False,
                ),
                ParameterSchema(
                    name="instructions",
                    type="string",
                    description=(
                        "System-prompt override (persona, rubric, refusal "
                        "policy). Mirrors the CLI --instructions flag."
                    ),
                    required=False,
                ),
            ],
        )

    async def execute(
        self, inputs: dict[str, Any], context: KaosContext | None = None
    ) -> ToolResult:
        message = inputs.get("message", "")
        session_id, err = _resolve_invocation_session_id(inputs, context)
        if err is not None:
            return err
        instructions = inputs.get("instructions") or None
        model = inputs.get("model")
        tool_filter_str = inputs.get("tool_filter")
        max_steps = inputs.get("max_steps")
        max_cost_usd_raw = inputs.get("max_cost_usd")
        max_cost_usd: float | None = None if max_cost_usd_raw is None else float(max_cost_usd_raw)
        if max_cost_usd is not None and max_cost_usd <= 0.0:
            return ToolResult.create_error(
                f"max_cost_usd must be > 0 when set, got {max_cost_usd}. "
                "Pass null (or omit) to disable the cap."
            )
        approval_patterns_str = inputs.get("require_approval_for_tools")

        if not message:
            return ToolResult.create_error(
                "Missing 'message' parameter. "
                "Provide the goal or multi-step request. "
                'Example: {"message": "Search FR for EPA rules, '
                'then get the full text of the first result"}'
            )
        if not session_id:
            return ToolResult.create_error(
                "Missing 'session_id' parameter. "
                "Provide a unique session identifier. "
                'Example: {"session_id": "research-epa-rules"}'
            )

        try:
            from kaos_agents.config import Agent, AgentPattern
            from kaos_agents.runtime.runner import Runner
            from kaos_agents.settings import DEFAULT_MODEL, KaosAgentSettings

            runtime = context.runtime if context else None

            # B0.1 — same per-invocation context derivation as
            # ``AgentChatTool.execute``. See the docstring on
            # ``_build_invocation_context`` for the security rationale.
            invocation_context = _build_invocation_context(
                context, runtime=runtime, session_id=session_id
            )

            tool_filter = (
                tuple(t.strip() for t in tool_filter_str.split(",") if t.strip())
                if tool_filter_str
                else ()
            )

            agent_settings = None
            if max_cost_usd is not None:
                agent_settings = KaosAgentSettings(plan_max_cost_usd=max_cost_usd)

            permission_policy = None
            if approval_patterns_str:
                from kaos_agents.runtime.permissions import PermissionPolicy
                from kaos_agents.types.permissions import PermissionDecision, PermissionRule

                patterns = tuple(p.strip() for p in approval_patterns_str.split(",") if p.strip())
                permission_policy = PermissionPolicy(
                    rules=tuple(
                        PermissionRule(
                            pattern=pat,
                            action=PermissionDecision.ASK,
                            reason="per-request require_approval_for_tools",
                        )
                        for pat in patterns
                    )
                )

            agent_kwargs: dict[str, Any] = {
                "pattern": AgentPattern.PLAN,
                "model": model or DEFAULT_MODEL,
                "tools": tool_filter,
                "max_plan_steps": int(max_steps) if max_steps is not None else None,
                "settings": agent_settings,
            }
            if instructions:
                agent_kwargs["instructions"] = instructions
            agent_config = Agent(**agent_kwargs)
            runner = Runner(
                agent_config,
                runtime=runtime,
                context=invocation_context,
                permission_policy=permission_policy,
            )
            response, status = await _run_turn_with_status(runner, message, session_id)

            # Auth / rate-limit / transport failures during plan exec must
            # surface as ToolResult.isError=True (probe 4b). See
            # AgentChatTool.execute above for the rationale.
            surfacing = _surfacing_error_from_status(status)
            if surfacing is not None:
                return ToolResult.create_error(surfacing)

            # Sprint-3 #9 — plan-execute already enforces
            # plan_max_cost_usd between steps (compose.py route() ->
            # StopReason.MAX_COST -> BudgetExceeded). We also do a
            # post-hoc cap check here to surface a budget_exceeded
            # flag even if the in-loop check missed an overshoot
            # (e.g. a single step that itself blew the cap).
            actual_cost = float(status.get("cost_usd") or 0.0)
            if max_cost_usd is not None and actual_cost > max_cost_usd:
                status["budget_exceeded"] = True
                if not status.get("budget_kind"):
                    status["budget_kind"] = "cost"

            result_data: dict[str, Any] = {
                "text": response.text,
                "turn_number": response.turn_number,
                "intent": response.intent.intent.value if response.intent else "unknown",
                "steps_executed": len(response.tool_calls),
                "tool_calls": [
                    {
                        "tool_name": tc.tool_name,
                        "result_summary": tc.result_summary,
                        "is_error": tc.is_error,
                    }
                    for tc in response.tool_calls
                ],
                "budget_exceeded": status["budget_exceeded"],
                "paused_for_approval": status["paused_for_approval"],
                "cost_usd": actual_cost,
                # Sprint-3 #10 (transparency lens) — see AgentChatTool
                # for the rationale. Same shape across both tools.
                "total_tokens": int(status.get("total_tokens") or 0),
            }
            if max_cost_usd is not None:
                result_data["max_cost_usd"] = max_cost_usd
            if status["paused_for_approval"]:
                result_data["pending_tool_name"] = status["pending_tool_name"]
                result_data["run_state_ref"] = status["run_state_ref"]
            if status["budget_exceeded"]:
                result_data["budget_kind"] = status["budget_kind"]

            summary = response.text if response.text else "(empty response)"
            if status["budget_exceeded"]:
                summary = f"[BudgetExceeded: {status['budget_kind']}] {summary}"
            elif status["paused_for_approval"]:
                summary = f"[ApprovalRequired: {status['pending_tool_name']}] {summary}"
            elif response.tool_calls:
                tools_used = ", ".join(
                    tc.tool_name for tc in response.tool_calls if not tc.is_error
                )
                summary = f"[Plan: {len(response.tool_calls)} steps, tools: {tools_used}] {summary}"

            return ToolResult.create_success(output=result_data, summary=summary)

        except Exception as exc:
            logger.warning("kaos-agent-plan: failed: %s", exc)
            return ToolResult.create_error(
                f"Agent plan-execute failed: {exc}. "
                "Check that the LLM API key is configured. "
                "For simpler queries, try kaos-agent-chat instead."
            )


class AgentMemoryQueryTool(KaosTool):
    """Read the contents of an agent session's memory."""

    @property
    def metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name="kaos-agent-memory-query",
            display_name="Query Agent Memory",
            description=(
                "Read the contents of a session's memory. "
                "Returns recent messages, actions, findings, and other memory sections. "
                "Use this to inspect what an agent remembers from prior turns."
            ),
            category=ToolCategory.DATA,
            capability=ToolCapability.QUERY,
            module_name=_MODULE,
            version=_VERSION,
            annotations=_AGENT_READ_ANNOTATIONS,
            input_schema=[
                ParameterSchema(
                    name="session_id",
                    type="string",
                    description="Session identifier to query.",
                ),
                ParameterSchema(
                    name="section",
                    type="string",
                    description=(
                        "Memory section to read. "
                        "Options: messages, actions, findings, documents, "
                        "reflection, role, playbooks. "
                        "Default: messages."
                    ),
                    required=False,
                    constraints={
                        "enum": [
                            "messages",
                            "actions",
                            "findings",
                            "documents",
                            "reflection",
                            "role",
                            "playbooks",
                            "plan_history",
                        ]
                    },
                ),
                ParameterSchema(
                    name="limit",
                    type="integer",
                    description="Maximum number of items to return. Default: 20.",
                    required=False,
                    constraints={"min": 1, "max": 100},
                ),
            ],
        )

    async def execute(
        self, inputs: dict[str, Any], context: KaosContext | None = None
    ) -> ToolResult:
        session_id, err = _resolve_caller_session_id(inputs, context)
        if err is not None:
            return err
        section_name = inputs.get("section", "messages")
        limit = int(inputs.get("limit", 20))

        try:
            from kaos_agents.memory.store import SessionStore
            from kaos_agents.types.memory import MemoryType

            runtime = context.runtime if context else None
            vfs = _get_vfs(runtime)
            store = SessionStore(vfs)

            # Load session
            try:
                memory = await store.load(session_id)
            except Exception as exc:
                logger.debug("Failed to load session '%s': %s", session_id, exc, exc_info=True)
                return ToolResult.create_error(
                    f"Session '{session_id}' not found or failed to load: {exc}. "
                    "Check that the session_id matches a previous "
                    "kaos-agent-chat or kaos-agent-plan call. "
                    "Sessions persist in VFS — they may have expired if older than 7 days."
                )

            # Get the requested section
            try:
                mem_type = MemoryType(section_name)
            except ValueError:
                valid = ", ".join(mt.value for mt in MemoryType)
                return ToolResult.create_error(
                    f"Unknown section '{section_name}'. "
                    f"Valid sections: {valid}. "
                    "Use 'messages' for conversation history, 'actions' for tool calls."
                )

            items = memory.get_recent(mem_type, limit)

            result_data = {
                "session_id": session_id,
                "section": section_name,
                "turn_count": memory.turn_count,
                "item_count": len(items),
                "items": [
                    {
                        "content": item.content,
                        "added_at": item.added_at,
                    }
                    for item in items
                ],
            }

            summary = (
                f"Session '{session_id}' ({memory.turn_count} turns): "
                f"{len(items)} {section_name} items"
            )
            return ToolResult.create_success(output=result_data, summary=summary)

        except Exception as exc:
            logger.warning("kaos-agent-memory-query: failed: %s", exc)
            return ToolResult.create_error(
                f"Memory query failed: {exc}. "
                "Check that the session_id is valid and VFS is configured."
            )


class AgentMemoryClearTool(KaosTool):
    """Clear an agent session's memory."""

    @property
    def metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name="kaos-agent-memory-clear",
            display_name="Clear Agent Memory",
            description=(
                "Clear all memory for an agent session. "
                "This is destructive — the session's conversation history, actions, "
                "and findings will be permanently deleted. "
                "Use kaos-agent-memory-query first to inspect before clearing."
            ),
            category=ToolCategory.DATA,
            capability=ToolCapability.TRANSFORM,
            module_name=_MODULE,
            version=_VERSION,
            annotations=_AGENT_DESTRUCTIVE_ANNOTATIONS,
            input_schema=[
                ParameterSchema(
                    name="session_id",
                    type="string",
                    description="Session identifier to clear.",
                ),
            ],
        )

    async def execute(
        self, inputs: dict[str, Any], context: KaosContext | None = None
    ) -> ToolResult:
        session_id, err = _resolve_caller_session_id(inputs, context)
        if err is not None:
            return err

        try:
            from kaos_agents.memory.store import SessionStore

            runtime = context.runtime if context else None
            vfs = _get_vfs(runtime)

            # KC17-P1-1: actually delete the persisted session files. Pre-KC17
            # this only called ``vfs.cleanup_context(session_id)`` which left
            # ``kaos-agents/sessions/{id}/memory.json`` (and graph.ttl) on
            # disk — so a subsequent ``SessionStore.exists()`` returned True
            # and the next turn re-hydrated the "cleared" memory. SessionStore.delete()
            # sweeps the entire session directory.
            store = SessionStore(vfs)
            await store.delete(session_id)
            # Also sweep any session-scoped VFS artifacts (run state, etc.).
            await vfs.cleanup_context(session_id)

            return ToolResult.create_text(
                f"Session '{session_id}' memory cleared. "
                "All conversation history, actions, findings, and the knowledge "
                "graph have been deleted from persistent storage."
            )

        except Exception as exc:
            logger.warning("kaos-agent-memory-clear: failed: %s", exc)
            return ToolResult.create_error(
                f"Memory clear failed: {exc}. "
                "The session may not exist or VFS may be misconfigured."
            )


class AgentMemorySearchTool(KaosTool):
    """Search across an agent session's memory using BM25."""

    @property
    def metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name="kaos-agent-memory-search",
            display_name="Search Agent Memory",
            description=(
                "Search across a session's memory sections using BM25 ranking. "
                "Searches MESSAGES, ACTIONS, DOCUMENTS, and FINDINGS by default. "
                "Returns ranked results with content and relevance scores. "
                "Use this to find specific information from prior turns."
            ),
            category=ToolCategory.DATA,
            capability=ToolCapability.QUERY,
            module_name=_MODULE,
            version=_VERSION,
            annotations=_AGENT_READ_ANNOTATIONS,
            input_schema=[
                ParameterSchema(
                    name="session_id",
                    type="string",
                    description="Session identifier to search.",
                ),
                ParameterSchema(
                    name="query",
                    type="string",
                    description="Natural-language search query.",
                ),
                ParameterSchema(
                    name="top_k",
                    type="integer",
                    description="Maximum number of results. Default: 10.",
                    required=False,
                    constraints={"min": 1, "max": 50},
                ),
            ],
        )

    async def execute(
        self, inputs: dict[str, Any], context: KaosContext | None = None
    ) -> ToolResult:
        session_id, err = _resolve_caller_session_id(inputs, context)
        if err is not None:
            return err
        query = inputs.get("query", "")
        top_k = int(inputs.get("top_k", 10))

        if not query:
            return ToolResult.create_error(
                "Missing 'query' parameter. "
                "Provide a search query. "
                'Example: {"query": "what was the filing fee?"}'
            )

        try:
            from kaos_agents.memory.search import search_memory
            from kaos_agents.memory.store import SessionStore

            runtime = context.runtime if context else None
            vfs = _get_vfs(runtime)
            store = SessionStore(vfs)

            try:
                memory = await store.load(session_id)
            except Exception as exc:
                logger.debug("Failed to load session '%s': %s", session_id, exc, exc_info=True)
                return ToolResult.create_error(
                    f"Session '{session_id}' not found: {exc}. "
                    "Check the session_id matches a previous agent call."
                )

            results = search_memory(memory, query, top_k=top_k)

            result_data = {
                "session_id": session_id,
                "query": query,
                "result_count": len(results),
                "results": [
                    {
                        "content": r.content[:200],
                        "section": r.section.value,
                        "score": round(r.score, 4),
                    }
                    for r in results
                ],
            }

            summary = f"Found {len(results)} result(s) for '{query[:50]}' in session '{session_id}'"
            return ToolResult.create_success(output=result_data, summary=summary)

        except Exception as exc:
            logger.warning("kaos-agent-memory-search: failed: %s", exc)
            return ToolResult.create_error(
                f"Memory search failed: {exc}. "
                "Ensure kaos-nlp-core is installed and the session exists."
            )


class AgentRecipeListTool(KaosTool):
    """List available workflow recipes."""

    @property
    def metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name="kaos-agent-recipe-list",
            display_name="List Agent Recipes",
            description=(
                "List all available workflow recipes (playbooks) that guide agent "
                "planning. Each recipe describes a common workflow pattern with "
                "required tools and step-by-step instructions. "
                "Recipes are automatically loaded into agent memory on session creation."
            ),
            category=ToolCategory.DATA,
            capability=ToolCapability.QUERY,
            module_name=_MODULE,
            version=_VERSION,
            annotations=_AGENT_READ_ANNOTATIONS,
            input_schema=[],
        )

    async def execute(
        self, inputs: dict[str, Any], context: KaosContext | None = None
    ) -> ToolResult:
        try:
            from kaos_agents.recipes import load_builtin_recipes

            recipes = load_builtin_recipes()
            result_data = {
                "recipe_count": len(recipes),
                "recipes": [
                    {
                        "name": r.get("name", "unnamed"),
                        "description": r.get("description", ""),
                        "tools": r.get("tools", []),
                        "step_count": len(r.get("steps", [])),
                    }
                    for r in recipes
                ],
            }

            names = ", ".join(r.get("name", "?") for r in recipes)
            summary = f"{len(recipes)} recipes available: {names}"
            return ToolResult.create_success(output=result_data, summary=summary)

        except Exception as exc:
            logger.warning("kaos-agent-recipe-list: failed: %s", exc)
            return ToolResult.create_error(
                f"Failed to list recipes: {exc}. The recipe library may not be installed correctly."
            )


def register_agent_tools(runtime: KaosRuntime) -> int:
    """Register all agent MCP tools with the runtime.

    Args:
        runtime: The KaosRuntime to register tools on.

    Returns:
        Number of tools registered.
    """
    from kaos_agents.settings import KaosAgentSettings

    runtime.module_settings["agents"] = KaosAgentSettings()

    from kaos_agents.registry import (
        default_tool_data_type_registry,
        default_tool_group_registry,
    )
    from kaos_agents.tools.extract import (
        ExtractCorpusTool,
        ExtractSchemaTool,
        ExtractVerifyTool,
    )
    from kaos_agents.tools.graph import (
        AgentGraphProjectionTool,
        AgentGraphSparqlTool,
        AgentGraphWalkTool,
    )
    from kaos_agents.types import DataType, ToolDataTypeSpec, ToolGroup

    # Track 4 chunk T4-1 — register I/O types for built-in tools so
    # type-driven discovery (tools_by_input_type / by_output_type) works
    # out of the box. Re-register safely with force=True so reloads
    # during tests don't conflict.
    _builtin_data_types: dict[str, ToolDataTypeSpec] = {
        "kaos-agent-chat": ToolDataTypeSpec(input_type=DataType.TEXT, output_type=DataType.TEXT),
        "kaos-agent-plan": ToolDataTypeSpec(input_type=DataType.TEXT, output_type=DataType.TEXT),
        "kaos-agent-memory-query": ToolDataTypeSpec(
            input_type=DataType.TEXT, output_type=DataType.JSON
        ),
        "kaos-agent-memory-search": ToolDataTypeSpec(
            input_type=DataType.TEXT, output_type=DataType.JSON
        ),
        "kaos-agent-memory-clear": ToolDataTypeSpec(
            input_type=DataType.TEXT, output_type=DataType.JSON
        ),
        "kaos-agent-recipe-list": ToolDataTypeSpec(input_type=None, output_type=DataType.JSON),
        "kaos-extract-schema": ToolDataTypeSpec(
            input_type=DataType.TEXT, output_type=DataType.JSON
        ),
        "kaos-extract-corpus": ToolDataTypeSpec(
            input_type=DataType.JSON, output_type=DataType.JSON
        ),
        "kaos-extract-verify": ToolDataTypeSpec(
            input_type=DataType.JSON, output_type=DataType.JSON
        ),
        "kaos-agent-graph-walk": ToolDataTypeSpec(
            input_type=DataType.TEXT, output_type=DataType.JSON
        ),
        "kaos-agent-graph-sparql": ToolDataTypeSpec(
            input_type=DataType.TEXT, output_type=DataType.JSON
        ),
        "kaos-agent-graph-projection": ToolDataTypeSpec(
            input_type=DataType.TEXT, output_type=DataType.JSON
        ),
    }
    for tool_name, dtspec in _builtin_data_types.items():
        default_tool_data_type_registry.register(tool_name, dtspec, force=True)

    # Track 4 chunk T4-2 — register built-in tool groups for 2-level
    # discovery. Three semantic clusters lawyers / data scientists
    # already use to reason about agent capability:
    _builtin_groups = (
        ToolGroup(
            name="agent",
            description=(
                "Conversational agent tools — chat, plan, recipe discovery, "
                "and session-memory access (read/search/clear)."
            ),
            tool_names=(
                "kaos-agent-chat",
                "kaos-agent-plan",
                "kaos-agent-memory-query",
                "kaos-agent-memory-search",
                "kaos-agent-memory-clear",
                "kaos-agent-recipe-list",
            ),
        ),
        ToolGroup(
            name="extraction",
            description=(
                "Schema-driven structured extraction tools (WS-TR.PR-4) — "
                "single-document, corpus fan-out, span verification."
            ),
            tool_names=(
                "kaos-extract-schema",
                "kaos-extract-corpus",
                "kaos-extract-verify",
            ),
        ),
        ToolGroup(
            name="graph",
            description=(
                "Per-session knowledge-graph tools (Track 3) — N-hop walk, "
                "SPARQL query, pre-built typed projections."
            ),
            tool_names=(
                "kaos-agent-graph-walk",
                "kaos-agent-graph-sparql",
                "kaos-agent-graph-projection",
            ),
        ),
    )
    for group in _builtin_groups:
        default_tool_group_registry.register(group, force=True)

    from kaos_agents.tools.corpus_filter import AgentCorpusFilterTool
    from kaos_agents.tools.design_extraction import AgentDesignExtractionTool
    from kaos_agents.tools.findings import AgentFindingsTool

    tools: list[KaosTool] = [
        AgentChatTool(),
        AgentPlanTool(),
        AgentMemoryQueryTool(),
        AgentMemorySearchTool(),
        AgentMemoryClearTool(),
        AgentRecipeListTool(),
        # WS-TR.PR-4 — structured extraction.
        ExtractSchemaTool(),
        ExtractCorpusTool(),
        ExtractVerifyTool(),
        # Track 3 chunk B3 — session knowledge graph tools.
        AgentGraphWalkTool(),
        AgentGraphSparqlTool(),
        AgentGraphProjectionTool(),
        # K7 — recall-first FindingsAgent wrapper.
        AgentFindingsTool(),
        # K8 — LLM-aided corpus precision filter (pairs with
        # kaos-content-corpus-narrow as a two-stage funnel).
        AgentCorpusFilterTool(),
        # PR-1a (2026-05-28 dynamic-schema architecture) — LLM-driven
        # extraction-schema design. Returns a typed ExtractionSchema
        # proposal for a user question + corpus. PR-1b will add the
        # per-document fan-out + stamp_source_uri JOIN so the same
        # tool also executes the schema and returns typed rows.
        AgentDesignExtractionTool(),
    ]
    for tool in tools:
        runtime.tools.register_tool(tool)
    return len(tools)
