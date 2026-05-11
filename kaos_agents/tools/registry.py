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


async def _run_turn_with_status(runner: Any, message: str, session_id: str) -> tuple[Any, dict]:
    """Drain ``runner.run()`` once, building both an AgentResponse and a
    status dict that surfaces budget / pause events at the tool boundary.

    Runner.turn() folds events into AgentResponse but does NOT expose
    BudgetExceeded or ToolCallApprovalRequired in its return value, so
    the MCP tool surface previously had no way to tell callers "the run
    paused" or "the run blew its budget." We do the same fold here plus
    a side-channel status dict so the tool result can report it.
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
    status = {
        "budget_exceeded": False,
        "budget_kind": None,
        "paused_for_approval": False,
        "pending_tool_name": None,
        "run_state_ref": None,
    }

    async for event in runner.run(message, session_id):
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
    response = AgentResponse.create(
        text=text,
        intent=intent_result,
        tool_calls=tuple(tool_calls),
        turn_number=turn_start_number,
        tokens_used=tokens_used,
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
                "For multi-step planning tasks, use kaos-agent-plan instead."
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
                        "Per-turn cost ceiling in USD. Threads to "
                        "KaosAgentSettings.plan_max_cost_usd. When the run "
                        "exceeds this cap, the Runner emits BudgetExceeded "
                        "and stops. Use values like 0.001 to force a budget "
                        "exceedance for testing."
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
        session_id = inputs.get("session_id", "")
        model = inputs.get("model")
        tool_filter_str = inputs.get("tool_filter")
        max_cost_usd = inputs.get("max_cost_usd")
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

            tool_filter = (
                tuple(t.strip() for t in tool_filter_str.split(",") if t.strip())
                if tool_filter_str
                else ()
            )

            agent_settings = None
            if max_cost_usd is not None:
                agent_settings = KaosAgentSettings(plan_max_cost_usd=float(max_cost_usd))

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
                context=context,
                permission_policy=permission_policy,
            )
            response, status = await _run_turn_with_status(runner, message, session_id)

            result_data = {
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
            }
            if status["paused_for_approval"]:
                result_data["pending_tool_name"] = status["pending_tool_name"]
                result_data["run_state_ref"] = status["run_state_ref"]
            if status["budget_exceeded"]:
                result_data["budget_kind"] = status["budget_kind"]

            summary = response.text[:500] if response.text else "(empty response)"
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
                "future attempts via Reflexion."
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
                        "Per-plan cost ceiling in USD. Threads to "
                        "KaosAgentSettings.plan_max_cost_usd. When the plan "
                        "exceeds this cap, the Runner emits BudgetExceeded "
                        "and stops. Use 0.001 to force a budget exceedance."
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
        session_id = inputs.get("session_id", "")
        instructions = inputs.get("instructions") or None
        model = inputs.get("model")
        tool_filter_str = inputs.get("tool_filter")
        max_steps = inputs.get("max_steps")
        max_cost_usd = inputs.get("max_cost_usd")
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

            tool_filter = (
                tuple(t.strip() for t in tool_filter_str.split(",") if t.strip())
                if tool_filter_str
                else ()
            )

            agent_settings = None
            if max_cost_usd is not None:
                agent_settings = KaosAgentSettings(plan_max_cost_usd=float(max_cost_usd))

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
                context=context,
                permission_policy=permission_policy,
            )
            response, status = await _run_turn_with_status(runner, message, session_id)

            result_data = {
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
            }
            if status["paused_for_approval"]:
                result_data["pending_tool_name"] = status["pending_tool_name"]
                result_data["run_state_ref"] = status["run_state_ref"]
            if status["budget_exceeded"]:
                result_data["budget_kind"] = status["budget_kind"]

            summary = response.text[:500] if response.text else "(empty response)"
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
        session_id = inputs.get("session_id", "")
        section_name = inputs.get("section", "messages")
        limit = int(inputs.get("limit", 20))

        if not session_id:
            return ToolResult.create_error(
                "Missing 'session_id' parameter. "
                "Provide the session ID to query. "
                'Example: {"session_id": "research-epa-rules"}'
            )

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
        session_id = inputs.get("session_id", "")

        if not session_id:
            return ToolResult.create_error(
                "Missing 'session_id' parameter. "
                "Provide the session ID to clear. "
                "Use kaos-agent-memory-query to inspect memory before clearing."
            )

        try:
            runtime = context.runtime if context else None
            vfs = _get_vfs(runtime)

            # Delete session files from VFS
            await vfs.cleanup_context(session_id)

            return ToolResult.create_text(
                f"Session '{session_id}' memory cleared. "
                "All conversation history, actions, and findings have been deleted."
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
        session_id = inputs.get("session_id", "")
        query = inputs.get("query", "")
        top_k = int(inputs.get("top_k", 10))

        if not session_id:
            return ToolResult.create_error(
                "Missing 'session_id' parameter. "
                "Provide the session ID to search. "
                'Example: {"session_id": "research-epa", "query": "filing fee"}'
            )
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
    ]
    for tool in tools:
        runtime.tools.register_tool(tool)
    return len(tools)
