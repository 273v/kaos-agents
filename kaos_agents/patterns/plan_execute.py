"""PlanExecuteAgent — extends ChatAgent with planning strategies.

Adds PLAN intent handling via the adaptive strategy (ADaPT):
- Simple goals → direct execution
- Complex goals → hierarchical decomposition with parallel execution
- Failures → automatic fallback with Reflexion feedback
- Streaming: yields PlanProposed, StepStart, StepComplete events

This is the full-featured agent pattern that uses all 7 planning primitives.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any

from kaos_core.logging import get_logger

from kaos_agents._constants import (
    PLAN_PRIOR_REFLECTION_COUNT,
    RESULT_SUMMARY_TRUNCATE,
)
from kaos_agents.events import (
    EventEmitter,
    KaosEvent,
    PlanProposed,
    PlanStepSummary,
    Span,
    SpanPhase,
    SpanSubject,
    TextDelta,
    emit_usage_observed,
)
from kaos_agents.patterns.chat import ChatAgent
from kaos_agents.planning.recall import recall
from kaos_agents.types import ZERO_USAGE, IntentResult, IntentType, InvocationUsage, ToolCallRecord
from kaos_agents.types.memory import MemoryType
from kaos_agents.types.plan import StopReason

if TYPE_CHECKING:
    from kaos_core.base.context import KaosContext
    from kaos_core.registry.container import KaosRuntime
    from kaos_core.vfs.core import VirtualFileSystem

    from kaos_agents.memory.session import SessionMemory
    from kaos_agents.settings import KaosAgentSettings
    from kaos_agents.types.memory import MemoryItem
    from kaos_agents.types.providers import ProviderConfig

logger = get_logger(__name__)


class PlanExecuteAgent(ChatAgent):
    """Agent with full planning: adaptive strategy for PLAN intent.

    Extends ChatAgent (which handles RESPOND, CLARIFY, TOOL_USE) with
    plan-propose → execute → observe for complex multi-step goals.

    Usage:
        agent = PlanExecuteAgent(
            vfs=runtime.vfs,
            runtime=runtime,
            context=context,
            model="anthropic:claude-sonnet-4-6",
        )
        response = await agent.turn(
            "Research whether Tesla complies with EPA emission standards",
            session_id="abc",
        )
    """

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
        max_plan_steps: int | None = None,
        settings: KaosAgentSettings | None = None,
        provider: ProviderConfig | None = None,
        extra_llm_tools: tuple[Any, ...] = (),
        permission_policy: Any = None,
        instructions: str | None = None,
    ) -> None:
        super().__init__(
            vfs,
            runtime=runtime,
            context=context,
            model=model,
            tool_filter=tool_filter,
            max_tools=max_tools,
            max_react_iterations=max_react_iterations,
            settings=settings,
            provider=provider,
            extra_llm_tools=extra_llm_tools,
            permission_policy=permission_policy,
            instructions=instructions,
        )
        self._max_plan_steps = max_plan_steps or self._settings.plan_max_steps

    async def _dispatch_streaming(
        self,
        intent: IntentResult,
        message: str,
        memory: SessionMemory,
        context_items: dict[MemoryType, list[Any]],
        emitter: EventEmitter,
    ) -> AsyncIterator[KaosEvent]:
        """Override dispatch to stream plan execution events.

        For PLAN intent, yields PlanProposed, StepStart, StepComplete.
        For other intents, delegates to ChatAgent (which handles TOOL_USE)
        or BaseAgent (which handles RESPOND/CLARIFY).
        """
        if intent.intent != IntentType.PLAN:
            async for event in super()._dispatch_streaming(
                intent, message, memory, context_items, emitter
            ):
                yield event
            return

        async for event in self._handle_plan_streaming(message, memory, context_items, emitter):
            yield event

    async def _handle_plan_streaming(
        self,
        message: str,
        memory: SessionMemory,
        context_items: dict[MemoryType, list[MemoryItem]],
        emitter: EventEmitter,
    ) -> AsyncIterator[KaosEvent]:
        """Handle multi-step plan via adaptive strategy, yielding events."""
        from kaos_agents.actions.tool_bridge import bridge_runtime_tools
        from kaos_agents.planning.result_check import is_error_result
        from kaos_agents.planning.strategies.adaptive import execute_adaptive
        from kaos_agents.types.plan import PlanBudget

        tools_dict: dict[str, Any] = {}
        tool_descriptions: dict[str, str] = {}
        if self._runtime:
            llm_tools = bridge_runtime_tools(
                self._runtime,
                self._context,
                filter_names=self._tool_filter,
                max_tools=self._max_tools,
                permission_policy=self._permission_policy,
            )
            for t in llm_tools:
                tools_dict[t.name] = t
                tool_descriptions[t.name] = t.description or ""
        # Append extra tools (delegation / handoff) injected by the Runner
        for t in self._extra_llm_tools:
            tools_dict[t.name] = t
            tool_descriptions[t.name] = t.description or ""

        from kaos_agents.context.triage import triage_corpus

        ctx = recall(
            memory,
            [MemoryType.MESSAGES, MemoryType.FINDINGS, MemoryType.ACTIONS, MemoryType.REFLECTION],
            budget_tokens=self._settings.default_context_budget_tokens // 2,
            priority_order=[MemoryType.MESSAGES, MemoryType.FINDINGS],
        )

        triage = triage_corpus(
            memory,
            message,
            max_selected=20,
            threshold=self._settings.retrieval_threshold,
        )

        context_text = ctx.text
        if triage is not None:
            context_text = f"{triage.context_summary}\n\n{context_text}"
            n_documents = triage.selected_count
        elif memory.has_section(MemoryType.DOCUMENTS):
            n_documents = memory.section_item_count(MemoryType.DOCUMENTS)
        else:
            n_documents = 0

        prior_failures = ""
        reflections = memory.get_recent(MemoryType.REFLECTION, PLAN_PRIOR_REFLECTION_COUNT)
        if reflections:
            prior_failures = "\n".join(r.content for r in reflections)

        budget = PlanBudget(
            max_steps=self._settings.plan_max_steps,
            max_replans=self._settings.plan_max_replans,
            max_cost_usd=self._settings.plan_max_cost_usd,
            max_wall_clock_seconds=self._settings.plan_max_wall_clock_seconds,
        )

        result = await execute_adaptive(
            message,
            tools=tools_dict,
            tool_descriptions=tool_descriptions,
            context=context_text,
            prior_failures=prior_failures,
            model=self._model_for_role("plan"),
            budget=budget,
            complexity_threshold=self._settings.complexity_threshold,
            simple_word_threshold=self._settings.simple_goal_word_threshold,
            max_steps=self._max_plan_steps,
            confidence_threshold=self._settings.confidence_threshold,
            deepen_threshold=self._settings.deepen_threshold,
            tool_timeout_seconds=self._settings.tool_timeout_seconds,
            n_documents=n_documents,
        )

        # Planning's act.py already reads ``invocation.usage.total_tokens``
        # from each step and rolls them into ``budget.cost_usd`` /
        # ``budget.tokens_used``. ``ComposeResult`` surfaces those
        # aggregates as ``total_cost_usd`` / ``total_tokens``. Input/output
        # split is not tracked per-step today; emit the aggregate.
        yield emit_usage_observed(
            emitter,
            InvocationUsage(
                total_tokens=int(getattr(result, "total_tokens", 0) or 0),
                cost_usd=float(getattr(result, "total_cost_usd", 0.0) or 0.0),
            ),
            source="plan-execute",
        )

        # Recover step metadata from the plan graph
        step_tool_names: dict[str, str] = {}
        step_descriptions: dict[str, str] = {}
        if result.plan_json and result.plan_json != "{}":
            try:
                from kaos_agents.planning.graph import PlanGraph

                pg = PlanGraph.from_json(result.plan_json)
                for sid in pg.step_ids():
                    props = pg.get_step(sid)
                    if props:
                        if props.get("tool_name"):
                            step_tool_names[sid] = props["tool_name"]
                        if props.get("description"):
                            step_descriptions[sid] = props["description"]
            except Exception as exc:
                logger.warning("plan_execute: failed to parse plan graph: %s", exc)

        # Emit PlanProposed
        plan_steps = tuple(
            PlanStepSummary(
                step_id=sid,
                description=step_descriptions.get(sid, sid),
                tool_name=step_tool_names.get(sid),
            )
            for sid in result.step_results
        )
        if plan_steps:
            yield emitter.emit(PlanProposed, steps=plan_steps, strategy="adaptive")

        # Emit per-step events
        for step_id, step_result in result.step_results.items():
            desc = step_descriptions.get(step_id, step_id)
            step_is_error = is_error_result(str(step_result))

            step_span = emitter.span_start(
                SpanSubject.STEP,
                name=f"step.{step_id}",
                attributes={"step_id": step_id, "description": desc},
            )
            yield step_span
            yield emitter.span_complete(
                SpanSubject.STEP,
                span_id=step_span.span_id,
                name=f"step.{step_id}",
                duration_ms=0.0,
                attributes={
                    "step_id": step_id,
                    "result_summary": str(step_result)[:RESULT_SUMMARY_TRUNCATE],
                    "is_error": step_is_error,
                },
            )

        # Emit final response text
        if result.stop_reason == StopReason.SUCCESS and result.step_results:
            response = _synthesize_results(result.step_results)
        elif result.stop_reason == StopReason.SUCCESS:
            response = "Plan completed but no results were produced."
        else:
            response = (
                f"Plan execution stopped: {result.stop_reason.value}. "
                f"Completed {result.steps_executed} steps."
            )
            memory.add(
                MemoryType.REFLECTION,
                f"Plan failed ({result.stop_reason.value}) for goal: {message[:100]}. "
                f"Steps completed: {result.steps_executed}.",
            )

        if response:
            yield emitter.emit(TextDelta, content=response)

        logger.debug(
            "plan_execute: %d steps, stop=%s, cost=$%.4f",
            result.steps_executed,
            result.stop_reason.value,
            result.total_cost_usd,
        )

    async def _handle_plan(
        self,
        message: str,
        memory: SessionMemory,
        context_items: dict[MemoryType, list[MemoryItem]],
    ) -> tuple[str, list[ToolCallRecord], InvocationUsage]:
        """Handle multi-step plan (non-streaming, backward compat).

        Delegates to _handle_plan_streaming and collects events,
        avoiding logic duplication. Aggregates UsageObserved into the
        returned InvocationUsage for the outer turn loop.
        """
        from kaos_agents.events import EventEmitter, TextDelta, UsageObserved

        emitter = EventEmitter(session_id="internal", run_id="internal")

        response_text = ""
        tool_calls: list[ToolCallRecord] = []
        usage_total = ZERO_USAGE
        async for event in self._handle_plan_streaming(message, memory, context_items, emitter):
            if isinstance(event, TextDelta):
                response_text += event.content
            elif (
                isinstance(event, Span)
                and event.subject == SpanSubject.STEP
                and event.phase == SpanPhase.COMPLETE
            ):
                attrs = event.attributes
                step_id = str(attrs.get("step_id", ""))
                # Each completed plan step ships as a "tool call" record
                # in the legacy non-streaming response shape. Tag with
                # the plan_id (run-level) and step_id so downstream
                # consumers (UI, audit, OTel) can correlate to the
                # originating plan/step without parsing event timestamps.
                # ``Span.span_id`` of the matching plan-execute run is
                # not threaded here yet — Track 6 polishes this when
                # ContextVar-based span correlation lands.
                tool_calls.append(
                    ToolCallRecord.from_dict_args(
                        tool_name=step_id,
                        arguments={"step_id": step_id},
                        result_summary=str(attrs.get("result_summary", "")),
                        is_error=bool(attrs.get("is_error", False)),
                        step_id=step_id,
                    )
                )
            elif isinstance(event, UsageObserved):
                usage_total = usage_total + InvocationUsage.from_llm_usage(event)

        return response_text, tool_calls, usage_total


def _synthesize_results(results: dict[str, Any]) -> str:
    """Combine step results into a response summary."""
    from kaos_agents.planning.result_check import is_error_result

    parts = []
    for step_id, result in results.items():
        result_str = str(result)
        if result_str and not is_error_result(result_str):
            # Truncate long results
            if len(result_str) > 300:
                result_str = result_str[:300] + "..."
            parts.append(f"**{step_id}**: {result_str}")

    if parts:
        return "Plan completed with the following results:\n\n" + "\n\n".join(parts)
    return "Plan completed but results were empty or all errored."
