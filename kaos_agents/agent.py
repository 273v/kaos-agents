"""BaseAgent — the core agent loop.

Stateless agent that orchestrates memory, tool calling, and LLM dispatch.
The agent is reconstructed per MCP call from session_id. All persistent
state lives in SessionMemory, which hydrates from VFS.

Two execution modes:
- ``run()`` — streaming: yields ``AgentEvent`` objects progressively
- ``turn()`` — blocking: collects all events, returns ``AgentResponse``

The 8-step turn (both modes share the same logic):
1. Hydrate memory from store (or create fresh)
2. Begin turn (clear ephemeral sections)
3. Add user message to MESSAGES section
4. Assemble context from memory
5. Classify intent
6. Dispatch to handler (respond, tool_use, research, plan, clarify)
7. Update memory (response, actions, findings)
8. End turn, persist, return response
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any

from kaos_core.logging import get_logger
from kaos_llm_core import InputField, OutputField, Signature

from kaos_agents._constants import FALLBACK_RECENT_MESSAGES
from kaos_agents.context.classify import classify_intent
from kaos_agents.events import (
    AgentEvent,
    EventEmitter,
    IntentClassified,
    MemoryUpdated,
    RunError,
    TextDelta,
    ToolCallResult,
    ToolCallStart,
    ToolCallSummary,
    TurnComplete,
    TurnStart,
    UsageObserved,
)
from kaos_agents.memory.session import SessionMemory
from kaos_agents.memory.store import SessionStore
from kaos_agents.memory.types import MemoryType
from kaos_agents.models import AgentResponse, IntentResult, IntentType, ToolCallRecord
from kaos_agents.settings import KaosAgentSettings
from kaos_agents.usage import ZERO_USAGE, InvocationUsage

# Default instruction for the respond handler. Module-level constant
# so it's auditable and overridable (subclasses can replace self._respond_instruction).
_DEFAULT_RESPOND_INSTRUCTION = "You are a helpful assistant."


class RespondSignature(Signature):
    """Generate a conversational response to the user's message.

    The agent's voice + tone is governed by the ``instructions=`` kwarg
    passed to the :class:`Call` (defaults to "You are a helpful
    assistant."). Subclasses or callers override the instructions to
    project a different persona without changing the I/O contract here.
    """

    message: str = InputField(description="The user's message.")
    conversation_history: str = InputField(description="Recent conversation history for context.")
    response: str = OutputField(description="Your response to the user.")


if TYPE_CHECKING:
    from kaos_core.vfs.core import VirtualFileSystem

    from kaos_agents.providers import ProviderConfig

logger = get_logger(__name__)


def _generate_run_id() -> str:
    """Generate a unique run ID for event correlation."""
    return f"run_{uuid.uuid4().hex[:12]}"


class BaseAgent:
    """Core agent with the 8-step turn loop.

    Subclasses (ChatAgent, PlanExecuteAgent) override dispatch handlers
    for each intent type. BaseAgent provides the loop scaffolding.

    Two execution modes:
    - ``run(message, session_id)`` yields ``AgentEvent`` progressively
    - ``turn(message, session_id)`` returns ``AgentResponse`` (backward compat)

    The agent is stateless — constructed per call, not per session.
    All state lives in SessionMemory.
    """

    def __init__(
        self,
        vfs: VirtualFileSystem,
        *,
        model: str | None = None,
        settings: KaosAgentSettings | None = None,
        provider: ProviderConfig | None = None,
        instructions: str | None = None,
    ) -> None:
        self._settings = KaosAgentSettings.resolve(settings)
        self._store = SessionStore(vfs)
        self._model = model or self._settings.default_llm_model
        self._provider = provider
        # WS-0.4: ``Agent.instructions`` is threaded from the top-level
        # ``Agent`` config through Runner → pattern classes → BaseAgent.
        # Prior to WS-0.4 this field was advertised as "core identity" on
        # the ``Agent`` public API (config.py:57) but never reached the
        # internal agents — every pattern used a hardcoded default. Now:
        #
        # - ``_simple_respond`` uses self._instructions when set.
        # - ``ChatAgent._handle_tool_use_streaming`` composes its ReAct
        #   instruction with the caller's instructions.
        # - Pattern-specific defaults (``_DEFAULT_RESPOND_INSTRUCTION``,
        #   ``_REACT_INSTRUCTION``) apply only when
        #   ``instructions is None``.
        self._instructions: str | None = instructions

    def _model_for_role(self, role: str) -> str:
        """Resolve the model to use for a specific role.

        If a ProviderConfig is attached, delegates to its role_models map.
        Otherwise returns the agent's default model.

        Special cases:
        - role='plan' falls back to settings.planning_llm_model when no
          provider is set (backward compat with the pre-provider path).
        """
        if self._provider is not None:
            from kaos_agents.providers import ModelRole

            try:
                return self._provider.model_for(ModelRole(role))
            except ValueError:
                return self._provider.default
        # Backward compat: planning has its own settings field
        if role == "plan":
            return self._settings.planning_llm_model
        return self._model

    async def run(self, message: str, session_id: str) -> AsyncIterator[AgentEvent]:
        """Execute a single agent turn, yielding events progressively.

        This is the primary streaming entry point. Yields ``AgentEvent``
        objects at each step of the 8-step loop. Consumers iterate:

            async for event in agent.run("Find EPA actions", "session-1"):
                match event:
                    case TurnStart(): ...
                    case ToolCallStart(): ...
                    case TurnComplete(): ...

        Args:
            message: The user's message.
            session_id: Session identifier for memory persistence.

        Yields:
            AgentEvent subclass instances in execution order.
        """
        run_id = _generate_run_id()
        emitter = EventEmitter(session_id=session_id, run_id=run_id)

        # Step 1: Hydrate memory
        memory = await self._store.load_or_create(session_id)
        logger.debug(
            "agent.step1_hydrate: session=%s total_tokens=%d turn_history=%d",
            session_id,
            memory.total_tokens,
            memory.turn_count,
        )

        # Step 2: Begin turn
        memory.begin_turn()
        turn_number = memory.turn_count + 1
        logger.debug("agent.step2_begin_turn: session=%s turn_number=%d", session_id, turn_number)

        yield emitter.emit(TurnStart, turn_number=turn_number)

        # Step 3: Add user message
        memory.add(MemoryType.MESSAGES, f"user: {message}")
        logger.debug("agent.step3_add_message: session=%s message_len=%d", session_id, len(message))

        # Step 4: Assemble context (query-aware when sections are large)
        from kaos_agents.context.assemble import assemble_context

        context_items = assemble_context(
            memory,
            message,
            sections=[
                MemoryType.MESSAGES,
                MemoryType.ACTIONS,
                MemoryType.FINDINGS,
                MemoryType.DOCUMENTS,
            ],
            total_budget_tokens=self._settings.default_context_budget_tokens,
            priority_order=[
                MemoryType.MESSAGES,
                MemoryType.FINDINGS,
                MemoryType.DOCUMENTS,
                MemoryType.ACTIONS,
            ],
            retrieval_threshold=self._settings.retrieval_threshold,
        )

        context_section_counts = {
            mt.value: len(items) for mt, items in context_items.items() if items
        }
        context_total_items = sum(context_section_counts.values())
        logger.debug(
            "agent.step4_assemble: session=%s sections=%s total_items=%d",
            session_id,
            context_section_counts,
            context_total_items,
        )

        # Step 5: Classify intent
        intent = await self._classify(message, memory, context_items)

        yield emitter.emit(
            IntentClassified,
            intent=intent.intent.value,
            confidence=intent.confidence,
            reasoning=intent.reasoning,
        )

        logger.debug(
            "agent.run: session=%s intent=%s confidence=%.2f",
            session_id,
            intent.intent.value,
            intent.confidence,
        )

        # Step 6: Dispatch to streaming handler — yields events from the handler
        response_text = ""
        tool_calls: list[ToolCallRecord] = []
        tool_call_summaries: list[ToolCallSummary] = []
        # Per-tool LLM usage attribution (P8 / N2). Keyed by tool_name —
        # ``UsageObserved.source`` is informational and the convention is
        # for tool implementations to set ``source`` to the tool name (or
        # a stable prefix matching the tool name) when the tool itself
        # drove an LLM call. Tools that don't call an LLM never emit a
        # UsageObserved and stay at zero attribution. Multiple calls to
        # the same tool sum into the same bucket.
        per_tool_usage: dict[str, InvocationUsage] = {}

        logger.debug(
            "agent.step6_dispatch: session=%s intent=%s pattern=%s",
            session_id,
            intent.intent.value,
            type(self).__name__,
        )

        turn_usage = ZERO_USAGE
        try:
            async for event in self._dispatch_streaming(
                intent, message, memory, context_items, emitter
            ):
                yield event
                # Collect response data from terminal events for memory update
                if isinstance(event, TextDelta):
                    response_text += event.content
                elif isinstance(event, ToolCallStart):
                    pass  # Tracked via ToolCallResult
                elif isinstance(event, ToolCallResult):
                    tool_calls.append(
                        ToolCallRecord.from_dict_args(
                            tool_name=event.tool_name,
                            arguments={},
                            result_summary=event.result_summary,
                            is_error=event.is_error,
                        )
                    )
                    tool_call_summaries.append(
                        ToolCallSummary(
                            tool_name=event.tool_name,
                            call_id=event.call_id,
                            is_error=event.is_error,
                            duration_ms=event.duration_ms,
                        )
                    )
                elif isinstance(event, UsageObserved):
                    turn_usage = turn_usage + InvocationUsage.from_llm_usage(event)
                    # Attribute to a tool when the source matches a tool
                    # name we've seen this turn (or starts with one — for
                    # sub-call sources like "rag-query.verifier"). Match
                    # is best-effort; unattributed usage stays in turn
                    # totals only.
                    src = (event.source or "").strip()
                    if src:
                        usage = InvocationUsage.from_llm_usage(event)
                        per_tool_usage[src] = per_tool_usage.get(src, ZERO_USAGE) + usage
        except Exception as exc:
            logger.warning("agent.run: dispatch failed: %s", exc)
            yield emitter.emit(
                RunError,
                error_type=type(exc).__name__,
                message=str(exc),
                recovery_hint="Check logs for details. Try a simpler query.",
            )

        # If no TextDelta events were yielded, get response from the non-streaming path.
        # This handles the case where _dispatch_streaming falls back to _dispatch.
        if not response_text:
            # The streaming handler may have set response_text via a different mechanism.
            # For BaseAgent's simple respond, we use the non-streaming path.
            pass

        logger.debug(
            "agent.step6_complete: session=%s response_len=%d tool_calls=%d",
            session_id,
            len(response_text),
            len(tool_calls),
        )

        # Step 7: Update memory
        if response_text:
            memory.add(MemoryType.MESSAGES, f"assistant: {response_text}")
        if tool_calls:
            for tc in tool_calls:
                summary = f"Tool: {tc.tool_name}({tc.arguments}) → {tc.result_summary}"
                memory.add(MemoryType.ACTIONS, summary)
            yield emitter.emit(
                MemoryUpdated,
                section=MemoryType.ACTIONS.value,
                action="add",
                item_count=len(tool_calls),
            )

        # Step 8: Summarize (if needed), end turn, persist
        try:
            n_summarized = await memory.summarize_turn(model=self._model)
            if n_summarized > 0:
                logger.debug("agent.run: summarized %d sections", n_summarized)
        except Exception as exc:
            logger.warning("agent.run: summarization failed (non-fatal): %s", exc)

        memory.end_turn()
        await self._store.save(memory)
        logger.debug(
            "agent.step8_persist: session=%s total_tokens=%d",
            session_id,
            memory.total_tokens,
        )

        # Backfill per-tool cost attribution into the summaries before
        # we emit TurnComplete. Builds a fresh tuple so the summaries
        # stay frozen-by-immutability — we just reconstruct each one
        # with the per-tool slice of usage on top of the base shape.
        attributed_summaries: list[ToolCallSummary] = []
        for s in tool_call_summaries:
            usage = per_tool_usage.get(s.tool_name, ZERO_USAGE)
            if usage is ZERO_USAGE:
                attributed_summaries.append(s)
                continue
            attributed_summaries.append(
                ToolCallSummary(
                    tool_name=s.tool_name,
                    call_id=s.call_id,
                    is_error=s.is_error,
                    duration_ms=s.duration_ms,
                    cost_usd=usage.cost_usd,
                    input_tokens=usage.input_tokens,
                    output_tokens=usage.output_tokens,
                )
            )

        yield emitter.emit(
            TurnComplete,
            text=response_text,
            intent=intent.intent.value,
            tool_calls=tuple(attributed_summaries),
            tokens_used=turn_usage.total_tokens,
            cost_usd=turn_usage.cost_usd,
            input_tokens=turn_usage.input_tokens,
            output_tokens=turn_usage.output_tokens,
        )

    async def turn(self, message: str, session_id: str) -> AgentResponse:
        """Execute a single agent turn (backward-compatible blocking mode).

        Collects all events from ``run()`` and returns the final response.
        Use ``run()`` for streaming.

        Args:
            message: The user's message.
            session_id: Session identifier for memory persistence.

        Returns:
            AgentResponse with the agent's reply and metadata.
        """
        events: list[AgentEvent] = []
        async for event in self.run(message, session_id):
            events.append(event)
        return _events_to_response(events, session_id)

    # -- Streaming dispatch (override in subclasses) ---------------------------

    async def _dispatch_streaming(
        self,
        intent: IntentResult,
        message: str,
        memory: SessionMemory,
        context_items: dict[MemoryType, list[Any]],
        emitter: EventEmitter,
    ) -> AsyncIterator[AgentEvent]:
        """Dispatch to the appropriate streaming handler based on intent.

        Yields AgentEvent instances. Subclasses override to add streaming
        for tool_use, research, plan handlers.

        Default implementation falls back to _dispatch() and yields
        the result as a single TextDelta.
        """
        # Default: use the non-streaming handlers and wrap the result.
        # Pre-Phase-5.0 callers (and test mocks) may return a 2-tuple
        # without usage — accept both shapes so we don't force every
        # downstream test fixture to be rewritten at once.
        dispatched = await self._dispatch(intent, message, memory, context_items)
        if len(dispatched) == 3:
            response_text, tool_calls, usage = dispatched
        else:
            response_text, tool_calls = dispatched  # type: ignore[misc]
            usage = ZERO_USAGE

        # Yield tool call events if any
        for tc in tool_calls:
            yield emitter.emit(
                ToolCallStart,
                call_id=tc.tool_name,  # Use tool_name as call_id for backward compat
                tool_name=tc.tool_name,
                arguments=tc.arguments,
            )
            yield emitter.emit(
                ToolCallResult,
                call_id=tc.tool_name,
                tool_name=tc.tool_name,
                result_summary=tc.result_summary,
                is_error=tc.is_error,
                duration_ms=0.0,
            )

        # Yield the response text
        if response_text:
            yield emitter.emit(TextDelta, content=response_text)

        # Surface real usage from the handler's LLM invocations so the
        # turn loop can roll it into TurnComplete. Zero usage (no LLM
        # call) still emits — downstream consumers distinguish "nothing
        # happened" from "missing data".
        yield emitter.emit(
            UsageObserved,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            total_tokens=usage.total_tokens,
            cost_usd=usage.cost_usd,
            source="dispatch",
        )

    # -- Overridable dispatch handlers ---------------------------------------

    async def _classify(
        self,
        message: str,
        memory: SessionMemory,
        context_items: dict[MemoryType, list[Any]] | None = None,
    ) -> IntentResult:
        """Classify user intent. Override for custom classification."""
        # Build context text from assembled items for the classifier
        context_text = ""
        if context_items:
            parts = []
            for _mt, items in context_items.items():
                if items:
                    parts.append("\n".join(item.content for item in items))
            context_text = "\n".join(parts)

        return await classify_intent(
            message,
            memory,
            model=self._model_for_role("classify"),
            context_text=context_text,
        )

    async def _dispatch(
        self,
        intent: IntentResult,
        message: str,
        memory: SessionMemory,
        context_items: dict[MemoryType, list[Any]],
    ) -> tuple[str, list[ToolCallRecord], InvocationUsage]:
        """Dispatch to the appropriate handler based on intent.

        Returns ``(response_text, tool_calls, usage)``. ``usage`` is the
        sum of token+cost spend across whatever sub-LLM-calls the handler
        made (:class:`InvocationUsage.ZERO_USAGE` when the handler
        bypassed the LLM entirely). Subclasses override specific
        handlers to add tool_use, research, plan.
        """
        if intent.intent == IntentType.RESPOND:
            return await self._handle_respond(message, memory, context_items)
        if intent.intent == IntentType.CLARIFY:
            return await self._handle_clarify(message, memory, context_items)
        if intent.intent == IntentType.TOOL_USE:
            return await self._handle_tool_use(message, memory, context_items)
        if intent.intent == IntentType.RESEARCH:
            return await self._handle_research(message, memory, context_items)
        if intent.intent == IntentType.PLAN:
            return await self._handle_plan(message, memory, context_items)

        # Fallback
        return await self._handle_respond(message, memory, context_items)

    async def _handle_respond(
        self,
        message: str,
        memory: SessionMemory,
        context_items: dict[MemoryType, list[Any]],
    ) -> tuple[str, list[ToolCallRecord], InvocationUsage]:
        """Handle simple conversational response. Uses a Call."""
        response, usage = await self._simple_respond(message, memory, context_items=context_items)
        return response, [], usage

    async def _handle_clarify(
        self,
        message: str,
        memory: SessionMemory,
        context_items: dict[MemoryType, list[Any]],
    ) -> tuple[str, list[ToolCallRecord], InvocationUsage]:
        """Handle clarification request."""
        response, usage = await self._simple_respond(
            message,
            memory,
            extra_instruction="The user's request is ambiguous. Ask a clarifying question.",
        )
        return response, [], usage

    async def _handle_tool_use(
        self,
        message: str,
        memory: SessionMemory,
        context_items: dict[MemoryType, list[Any]],
    ) -> tuple[str, list[ToolCallRecord], InvocationUsage]:
        """Handle tool-using request. Override in ChatAgent to use ReAct."""
        # BaseAgent falls back to simple response (no tools configured)
        return await self._handle_respond(message, memory, context_items)

    async def _handle_research(
        self,
        message: str,
        memory: SessionMemory,
        context_items: dict[MemoryType, list[Any]],
    ) -> tuple[str, list[ToolCallRecord], InvocationUsage]:
        """Handle research/document Q&A. Override to use RAG."""
        return await self._handle_respond(message, memory, context_items)

    async def _handle_plan(
        self,
        message: str,
        memory: SessionMemory,
        context_items: dict[MemoryType, list[Any]],
    ) -> tuple[str, list[ToolCallRecord], InvocationUsage]:
        """Handle multi-step plan. Override in PlanExecuteAgent."""
        return await self._handle_respond(message, memory, context_items)

    # -- Internal helpers ----------------------------------------------------

    async def _simple_respond(
        self,
        message: str,
        memory: SessionMemory,
        *,
        extra_instruction: str = "",
        context_items: dict[MemoryType, list[Any]] | None = None,
    ) -> tuple[str, InvocationUsage]:
        """Generate a simple text response via Call.

        Returns ``(response_text, usage)`` so callers can record token
        spend. Pre-Phase-5.0 this returned just ``str`` and the agent
        shipped ``tokens_used=0`` in every ``TurnComplete``; the usage
        is now provider-reported (pulled from ``Invocation.usage``).
        """
        from kaos_agents._llm_imports import require_llm_core

        require_llm_core()
        from kaos_llm_core import Call

        # Use pre-assembled context if available, otherwise build from memory
        if context_items:
            parts = []
            for mt, items in context_items.items():
                if items:
                    parts.append(
                        f"=== {mt.value.upper()} ===\n" + "\n".join(i.content for i in items)
                    )
            history = "\n\n".join(parts) if parts else "(new conversation)"
        else:
            recent = memory.get_recent(MemoryType.MESSAGES, FALLBACK_RECENT_MESSAGES)
            history = "\n".join(item.content for item in recent) if recent else "(new conversation)"

        instructions = self._instructions or _DEFAULT_RESPOND_INSTRUCTION
        if extra_instruction:
            instructions = f"{instructions} {extra_instruction}"

        call = Call(
            RespondSignature, model=self._model_for_role("respond"), instructions=instructions
        )
        # ``.invoke()`` returns the full Invocation so we can read
        # ``invocation.usage`` — the bare ``await call(...)`` path is
        # slightly cheaper but throws the usage record on the floor.
        invocation = await call.invoke(message=message, conversation_history=history)
        text = str(getattr(invocation.output, "response", "")) if invocation.output else ""
        return text, InvocationUsage.from_invocation(invocation)


# ---------------------------------------------------------------------------
# Event-to-response conversion (used by turn() backward compat)
# ---------------------------------------------------------------------------


def _events_to_response(events: list[AgentEvent], session_id: str) -> AgentResponse:
    """Convert a collected event stream to a single AgentResponse.

    Scans events for TurnComplete (final text + metrics) and IntentClassified
    (intent). Falls back gracefully if events are incomplete.
    """
    # Find the final TurnComplete and IntentClassified events
    turn_complete: TurnComplete | None = None
    intent_event: IntentClassified | None = None
    tool_call_records: list[ToolCallRecord] = []

    for event in events:
        if isinstance(event, TurnComplete):
            turn_complete = event
        elif isinstance(event, IntentClassified):
            intent_event = event
        elif isinstance(event, ToolCallResult):
            tool_call_records.append(
                ToolCallRecord.from_dict_args(
                    tool_name=event.tool_name,
                    arguments={},
                    result_summary=event.result_summary,
                    is_error=event.is_error,
                )
            )

    # Build IntentResult from the intent event
    intent = IntentResult(
        intent=IntentType(intent_event.intent) if intent_event else IntentType.RESPOND,
        confidence=intent_event.confidence if intent_event else 0.0,
        reasoning=intent_event.reasoning if intent_event else "",
    )

    # Build response from TurnComplete or fallback to concatenated TextDelta
    if turn_complete:
        text = turn_complete.text
        tokens_used = turn_complete.tokens_used
        turn_number = 0
        # Find turn number from TurnStart
        for event in events:
            if isinstance(event, TurnStart):
                turn_number = event.turn_number
                break
    else:
        # Fallback: concatenate all TextDelta content
        text = "".join(event.content for event in events if isinstance(event, TextDelta))
        tokens_used = 0
        turn_number = 0

    return AgentResponse.create(
        text=text,
        intent=intent,
        tool_calls=tuple(tool_call_records),
        turn_number=turn_number,
        tokens_used=tokens_used,
        metadata={"session_id": session_id},
    )
