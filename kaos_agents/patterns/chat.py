"""ChatAgent — conversation pattern with tool calling via ReAct.

Extends BaseAgent with:
- Tool-using responses via ReAct (inner loop)
- Runtime tool bridge (KaosTool → ReAct Tool)
- Tool call recording for memory
- Streaming tool call events (ToolCallStart/ToolCallResult) from ReAct trajectory
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any

from kaos_core.logging import get_logger
from kaos_llm_core import InputField, OutputField, Signature

from kaos_agents._constants import FALLBACK_RECENT_MESSAGES
from kaos_agents.actions.tool_bridge import bridge_runtime_tools
from kaos_agents.events import (
    EventEmitter,
    KaosEvent,
    Span,
    SpanPhase,
    SpanSubject,
    TextDelta,
    UsageObserved,
    emit_usage_observed,
)
from kaos_agents.patterns._tool_schema import (
    drop_tool_at_index,
    drop_tool_by_name,
    extract_invalid_tool_index,
    extract_invalid_tool_name,
    is_tool_schema_rejection,
)
from kaos_agents.runtime.agent import BaseAgent
from kaos_agents.types import (
    ZERO_USAGE,
    AgentMetadata,
    IntentResult,
    IntentType,
    InvocationUsage,
    ToolCallRecord,
    ToolExecution,
)
from kaos_agents.types.memory import MemoryType

_REACT_INSTRUCTION = (
    "Complete the user's request using the available tools. Be thorough "
    "and cite your sources.\n\n"
    "REFLECTION DISCIPLINE — at the start of every iteration, before "
    "you reach for another tool, answer this question to yourself: "
    "'Given what I have already retrieved or observed, can I answer "
    "the user's question NOW?' If yes, stop calling tools and produce "
    "the final answer. Do not fetch more data when the data you have "
    "is sufficient.\n\n"
    "AGGREGATION DISCIPLINE — if the user is asking a comparative or "
    "aggregate question (longest, shortest, biggest, most, fewest, "
    "average, count of X), and you have already retrieved the values "
    "you need to compare, COMPUTE the answer directly. Do not retrieve "
    "more passages hoping one of them contains the answer phrase — the "
    "answer is something you derive, not something to look up. Use "
    "kaos-retrieval-corpus-manifest for per-document stats and "
    "kaos-llm-core-compute (if available) for arithmetic over retrieved "
    "values.\n\n"
    "REFUSAL DISCIPLINE — only refuse with 'insufficient evidence' "
    "when the corpus genuinely lacks the underlying facts. Aggregation "
    "queries where the facts ARE present but no single passage IS the "
    "answer are not 'insufficient evidence' — they need aggregation, "
    "not more retrieval."
)

if TYPE_CHECKING:
    from kaos_core.base.context import KaosContext
    from kaos_core.registry.container import KaosRuntime
    from kaos_core.vfs.core import VirtualFileSystem

    from kaos_agents.memory.session import SessionMemory
    from kaos_agents.settings import KaosAgentSettings
    from kaos_agents.types.memory import MemoryItem
    from kaos_agents.types.providers import ProviderConfig

logger = get_logger(__name__)


def _extract_structured_content(result: Any) -> dict[str, Any] | None:
    """Recover the tool's ``structured_content`` dict from the
    tool-bridge's combined string return value.

    ``actions.tool_bridge`` returns ``f"{text}\n\n{json.dumps(structured)}"``
    so the ReAct LLM sees both the human-readable text AND the
    structured payload. The emitter needs the dict back so the wire
    can carry it (the SPA's ArtifactCard reads
    ``call.structured_content.artifact_id``). Returns ``None`` when
    the result doesn't carry a structured dict.
    """
    if result is None:
        return None
    if isinstance(result, dict):
        if result.get("error") is True:
            return None
        return result
    if not isinstance(result, str) or not result:
        return None
    import json as _json

    stripped = result.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        try:
            parsed = _json.loads(stripped)
            if isinstance(parsed, dict) and parsed.get("error") is not True:
                return parsed
        except _json.JSONDecodeError:
            pass
    sep = "\n\n"
    idx = result.rfind(sep + "{")
    if idx != -1:
        candidate = result[idx + len(sep) :].strip()
        if candidate.endswith("}"):
            try:
                parsed = _json.loads(candidate)
                if isinstance(parsed, dict) and parsed.get("error") is not True:
                    return parsed
            except _json.JSONDecodeError:
                pass
    return None


class ToolTaskSignature(Signature):
    """Complete the user's request using the available tools.

    The actual reasoning loop lives in :class:`ReAct` (which consumes
    this Signature). Treat this as the I/O contract: question + context
    in, final answer out. Tool selection is emergent from ReAct.
    """

    question: str = InputField(description="The user's request.")
    context: str = InputField(description="Conversation context.")
    answer: str = OutputField(description="Your final answer to the user.")


class ChatAgent(BaseAgent):
    """Conversational agent with tool calling.

    Extends BaseAgent to handle tool_use intent via ReAct. Bridges
    KaosRuntime tools for the ReAct inner loop.

    Usage:
        agent = ChatAgent(
            vfs=runtime.vfs,
            runtime=runtime,
            context=context,
            model="anthropic:claude-sonnet-4-6",
        )
        response = await agent.turn("Extract dates from report.pdf", session_id="abc")
    """

    @classmethod
    def metadata(cls) -> AgentMetadata:
        """Pattern identity — ``pattern="chat"``, supports respond /
        tool_use / clarify intents."""
        return AgentMetadata(
            name="chat-agent",
            description="Conversational agent with ReAct tool calling.",
            pattern="chat",
            tags=("chat", "tool-use", "react"),
        )

    def __init__(
        self,
        vfs: VirtualFileSystem,
        *,
        runtime: KaosRuntime | None = None,
        context: KaosContext | None = None,
        model: str | None = None,
        tool_filter: list[str] | None = None,
        max_tools: int | None = None,
        max_react_iterations: int | None = None,
        settings: KaosAgentSettings | None = None,
        provider: ProviderConfig | None = None,
        extra_llm_tools: tuple[Any, ...] = (),
        permission_policy: Any = None,
        instructions: str | None = None,
    ) -> None:
        super().__init__(
            vfs,
            model=model,
            settings=settings,
            provider=provider,
            instructions=instructions,
        )
        self._runtime = runtime
        self._context = context
        self._tool_filter = tool_filter
        self._max_tools = max_tools or self._settings.max_tools
        self._max_react_iterations = max_react_iterations or self._settings.max_react_iterations
        # Extra kaos-llm-core Tool instances to append to the bridged runtime tools.
        # Used by Runner to inject delegation (agent_as_tool) and handoff tools.
        self._extra_llm_tools: tuple[Any, ...] = extra_llm_tools
        # WS-0.1 pre-execution permission gate. Threaded into bridge_runtime_tools
        # so DENY / ASK decisions stop the underlying tool from running rather
        # than just annotating it after the fact.
        self._permission_policy = permission_policy

    async def _dispatch_streaming(
        self,
        intent: IntentResult,
        message: str,
        memory: SessionMemory,
        context_items: dict[MemoryType, list[Any]],
        emitter: EventEmitter,
    ) -> AsyncIterator[KaosEvent]:
        """Override dispatch to stream tool call events from ReAct.

        For TOOL_USE intent, yields ToolCallStart/ToolCallResult for each
        tool in the ReAct trajectory. For other intents, falls back to
        the BaseAgent default (which wraps _dispatch).
        """
        if intent.intent != IntentType.TOOL_USE:
            # Delegate non-tool intents to BaseAgent default
            async for event in super()._dispatch_streaming(
                intent, message, memory, context_items, emitter
            ):
                yield event
            return

        # TOOL_USE: stream tool call events from ReAct
        async for event in self._handle_tool_use_streaming(message, memory, context_items, emitter):
            yield event

    async def _handle_tool_use_streaming(
        self,
        message: str,
        memory: SessionMemory,
        context_items: dict[MemoryType, list[MemoryItem]],
        emitter: EventEmitter,
    ) -> AsyncIterator[KaosEvent]:
        """Handle tool-using request via ReAct, yielding events per tool call."""
        # If neither a runtime nor extra delegation tools are available,
        # fall back to a simple LLM response. Extra tools alone (delegation/
        # handoff) are enough to drive a tool-using ReAct loop even when
        # no KaosRuntime is attached.
        if self._runtime is None and not self._extra_llm_tools:
            logger.warning(
                "chat_agent: no runtime and no extra tools — falling back to simple response"
            )
            response, usage = await self._simple_respond(message, memory)
            if response:
                yield emitter.emit(TextDelta, content=response)
            yield emit_usage_observed(emitter, usage, source="respond-fallback")
            return

        from kaos_agents._llm_imports import require_llm_core

        require_llm_core()
        from kaos_llm_core.programs.react import ReAct

        tools: list[Any] = []
        if self._runtime is not None:
            tools.extend(
                bridge_runtime_tools(
                    self._runtime,
                    self._context,
                    filter_names=self._tool_filter,
                    max_tools=self._max_tools,
                    permission_policy=self._permission_policy,
                )
            )
        # Append extra tools (delegation / handoff) injected by the Runner
        tools.extend(self._extra_llm_tools)

        logger.debug(
            "chat_agent._handle_tool_use_streaming: tools_available=%d "
            "(runtime=%d, extra=%d), model=%s",
            len(tools),
            len(tools) - len(self._extra_llm_tools),
            len(self._extra_llm_tools),
            self._model_for_role("respond"),
        )

        if not tools:
            logger.warning("chat_agent: no tools available — falling back to simple response")
            response, usage = await self._simple_respond(message, memory)
            if response:
                yield emitter.emit(TextDelta, content=response)
            yield emit_usage_observed(emitter, usage, source="respond-fallback")
            return

        if context_items:
            parts = []
            for mt, items in context_items.items():
                if items:
                    parts.append(
                        f"=== {mt.value.upper()} ===\n" + "\n".join(i.content for i in items)
                    )
            context_text = "\n\n".join(parts) if parts else ""
        else:
            recent = memory.get_recent(MemoryType.MESSAGES, FALLBACK_RECENT_MESSAGES)
            context_text = "\n".join(item.content for item in recent) if recent else ""

        # WS-0.4: compose the caller's instructions (if any) with the
        # pattern-specific default so user-supplied identity survives
        # into the ReAct prompt. Caller instructions first — they are
        # the "core identity"; the default appends task-specific
        # guidance ("be thorough, cite sources") without clobbering.
        react_instructions = (
            f"{self._instructions}\n\n{_REACT_INSTRUCTION}"
            if self._instructions
            else _REACT_INSTRUCTION
        )
        # FIX-16: when ONE tool's JSON Schema is rejected at the
        # provider boundary (OpenAI's strict validator is the common
        # culprit — see FIX-14), drop the offending tool and retry
        # instead of failing the entire turn with no tools at all.
        # Cap the retry budget so a pathological catalog can't loop
        # forever; if every retry still hits a schema error or a
        # non-schema exception fires, fall through to the existing
        # react-fallback below.
        _MAX_SCHEMA_DROPS = 5
        current_tools = list(tools)
        already_dropped: set[str] = set()
        try:
            invocation = None
            result = None
            t_start = time.monotonic()
            for attempt in range(_MAX_SCHEMA_DROPS + 1):
                react = ReAct(
                    ToolTaskSignature,
                    tools=current_tools,
                    model=self._model_for_role("respond"),
                    max_iterations=self._max_react_iterations,
                    instructions=react_instructions,
                )
                try:
                    # ``.invoke()`` returns the full Invocation so we can surface
                    # ``invocation.usage`` for TurnComplete — ``.__call__()``
                    # returns the bare ReActResult and throws usage on the floor.
                    invocation = await react.invoke(question=message, context=context_text)
                    result = invocation.output
                    break
                except Exception as exc:
                    msg = str(exc)
                    if attempt >= _MAX_SCHEMA_DROPS or not is_tool_schema_rejection(msg):
                        raise
                    bad_name = extract_invalid_tool_name(msg)
                    if bad_name is None:
                        bad_index = extract_invalid_tool_index(msg)
                        if bad_index is None:
                            raise
                        current_tools, bad_name = drop_tool_at_index(current_tools, bad_index)
                    else:
                        current_tools = drop_tool_by_name(current_tools, bad_name)
                    if bad_name is None or bad_name in already_dropped or not current_tools:
                        # Re-raise → outer except below runs the
                        # react-fallback. Cases: index past EOL; the
                        # same tool failed twice (would loop); last
                        # tool standing got dropped.
                        raise
                    already_dropped.add(bad_name)
                    logger.warning(
                        "chat_agent: dropped tool %r due to schema rejection "
                        "(attempt %d/%d, %d tools remaining): %s",
                        bad_name,
                        attempt + 1,
                        _MAX_SCHEMA_DROPS,
                        len(current_tools),
                        exc,
                    )
            # Loop must exit via break or raise; defensive guard:
            if invocation is None or result is None:
                raise RuntimeError("ReAct retry loop exhausted without success or raise")
            t_total = (time.monotonic() - t_start) * 1000

            # Surface model reasoning blocks (Anthropic extended thinking,
            # OpenAI o1 reasoning, Google thinkingConfig) when the
            # underlying provider exposed them. Helper is a no-op when
            # `invocation.extras["native_thinking"]` is empty.
            from kaos_agents.events import emit_thinking_from_invocation

            thinking_event = emit_thinking_from_invocation(emitter, invocation)
            if thinking_event is not None:
                yield thinking_event

            # Emit tool call events from the trajectory
            total_tool_calls = 0
            for iteration in result.trajectory:
                for obs in iteration.tool_results:
                    total_tool_calls += 1
                    result_preview = str(obs.result)[:200] if obs.result else ""
                    call_id = obs.tool_call_id or obs.tool_name
                    args_tuple = tuple(sorted(obs.arguments.items())) if obs.arguments else ()
                    structured_content = _extract_structured_content(obs.result)
                    logger.debug(
                        "chat_agent.tool_call: tool=%s, is_error=%s, result_preview=%r",
                        obs.tool_name,
                        obs.is_error,
                        result_preview[:80],
                    )
                    tc_span = emitter.span_start(
                        SpanSubject.TOOL_CALL,
                        name=f"tool.{obs.tool_name}",
                        attributes={
                            "tool_name": obs.tool_name,
                            "call_id": call_id,
                            "arguments": args_tuple,
                        },
                    )
                    yield tc_span
                    complete_attrs: dict[str, Any] = {
                        "tool_name": obs.tool_name,
                        "call_id": call_id,
                        "result_summary": result_preview,
                        "is_error": obs.is_error,
                    }
                    if isinstance(structured_content, dict) and structured_content:
                        complete_attrs["structured_content"] = structured_content
                    yield emitter.span_complete(
                        SpanSubject.TOOL_CALL,
                        span_id=tc_span.span_id,
                        name=f"tool.{obs.tool_name}",
                        duration_ms=0.0,  # Per-tool timing not available from trajectory
                        attributes=complete_attrs,
                    )

            # Emit the final response
            response_text = str(result.answer) if result.outputs and result.answer else ""
            if not response_text and result.trajectory:
                response_text = result.trajectory[-1].text

            if response_text:
                yield emitter.emit(TextDelta, content=response_text)

            # ReAct rolls up usage across every sub-Call + tool-using
            # iteration into invocation.usage. Emit the rolled-up total
            # so the turn loop sees one consolidated UsageObserved.
            yield emit_usage_observed(
                emitter, InvocationUsage.from_invocation(invocation), source="react"
            )

            logger.debug(
                "chat_agent.react_complete: iterations=%d, tool_calls=%d, stop=%s, latency_ms=%.0f",
                result.iterations_used,
                total_tool_calls,
                result.stop_reason,
                t_total,
            )

        except Exception as exc:
            logger.warning("chat_agent: ReAct failed: %s", exc)
            response, usage = await self._simple_respond(
                message,
                memory,
                extra_instruction=f"A tool-calling attempt failed: {exc}. "
                "Respond helpfully without tools.",
            )
            if response:
                yield emitter.emit(TextDelta, content=response)
            yield emit_usage_observed(emitter, usage, source="react-fallback")

    async def _handle_tool_use(
        self,
        message: str,
        memory: SessionMemory,
        context_items: dict[MemoryType, list[MemoryItem]],
    ) -> tuple[str, list[ToolExecution], InvocationUsage]:
        """Handle tool-using request (non-streaming, backward compat).

        Delegates to _handle_tool_use_streaming and collects events,
        avoiding logic duplication. Aggregates UsageObserved events into
        the returned InvocationUsage so the outer turn loop sees real
        token/cost numbers on the fallback path too.
        """
        emitter = EventEmitter(session_id="internal", run_id="internal")

        response_text = ""
        tool_calls: list[ToolExecution] = []
        usage_total = ZERO_USAGE
        async for event in self._handle_tool_use_streaming(message, memory, context_items, emitter):
            if isinstance(event, TextDelta):
                response_text += event.content
            elif (
                isinstance(event, Span)
                and event.subject == SpanSubject.TOOL_CALL
                and event.phase == SpanPhase.COMPLETE
            ):
                attrs = event.attributes
                tool_calls.append(
                    ToolCallRecord.from_dict_args(
                        tool_name=str(attrs.get("tool_name", "")),
                        arguments={},
                        result_summary=str(attrs.get("result_summary", "")),
                        is_error=bool(attrs.get("is_error", False)),
                    )
                )
            elif isinstance(event, UsageObserved):
                usage_total = usage_total + InvocationUsage.from_llm_usage(event)

        return response_text, tool_calls, usage_total
