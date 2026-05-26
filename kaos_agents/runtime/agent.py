"""BaseAgent — the core agent loop.

Stateless agent that orchestrates memory, tool calling, and LLM dispatch.
The agent is reconstructed per MCP call from session_id. All persistent
state lives in SessionMemory, which hydrates from VFS.

Two execution modes:
- ``run()`` — streaming: yields ``KaosEvent`` objects progressively
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

import contextlib
import contextvars
import re
import time
import uuid
from collections.abc import AsyncIterator, Iterator
from typing import TYPE_CHECKING, Any

from kaos_core.logging import get_logger
from kaos_llm_core import InputField, OutputField, Signature

from kaos_agents._constants import FALLBACK_RECENT_MESSAGES
from kaos_agents.base.agent import KaosAgent
from kaos_agents.context.classify import classify_intent
from kaos_agents.events import (
    CitationFound,
    EventEmitter,
    IntentClassified,
    KaosEvent,
    MemoryEvent,
    MemoryEventKind,
    PatternMismatch,
    RunError,
    Span,
    SpanPhase,
    SpanSubject,
    TextDelta,
    TurnSummary,
    UsageObserved,
    collect_events,
    emit_memory_added,
    emit_usage_observed,
    use_emitter,
)
from kaos_agents.memory.session import SessionMemory
from kaos_agents.memory.store import SessionStore
from kaos_agents.settings import KaosAgentSettings
from kaos_agents.types import (
    ZERO_USAGE,
    IntentResult,
    IntentType,
    InvocationUsage,
    ToolExecution,
)
from kaos_agents.types.memory import MemoryType

# Default instruction for the respond handler. Module-level constant
# so it's auditable and overridable (subclasses can replace self._respond_instruction).
_DEFAULT_RESPOND_INSTRUCTION = "You are a helpful assistant."

# Defense-in-depth: scratchpad-tag closers that instruction-tuned
# models (Claude 4.x family, GPT-5.x) hallucinate when given an
# opener-only field marker. ChatCodec / XMLCodec are no longer used
# in the respond path, but a closer that bleeds into a JSON string
# value (or arrives via a future non-JSON codec) is still stripped
# here so the SSE wire stays clean. The patterns are conservative:
# only whole bracketed `[/name]` / `</name>` lines that look like
# field-name slugs (no internal whitespace).
_STRIP_SCRATCHPAD_RE = re.compile(
    r"^[ \t]*(?:\[/\w+\]|</\w+>)[ \t]*\n?",
    re.MULTILINE,
)


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

    from kaos_agents.types.providers import ProviderConfig

logger = get_logger(__name__)


def _generate_run_id() -> str:
    """Generate a unique run ID for event correlation."""
    return f"run_{uuid.uuid4().hex[:12]}"


class BaseAgent(KaosAgent):
    """Core agent with the 8-step turn loop.

    Canonical concrete implementation of :class:`KaosAgent`. Subclasses
    (ChatAgent, PlanExecuteAgent, ResearchAgent) override dispatch
    handlers for each intent type; BaseAgent provides the loop
    scaffolding shared by every pattern.

    Two execution modes:
    - ``run(message, session_id)`` yields ``KaosEvent`` progressively
    - ``turn(message, session_id)`` returns ``AgentResponse`` (default
      from :class:`KaosAgent` — collects events from ``run()`` and
      converts via :func:`events_to_response`)

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

    # ContextVar for per-call instruction overrides — replaces the legacy
    # ``self._instructions = augmented; try: ...; finally: self._instructions = saved``
    # mutation pattern. ContextVar propagates correctly across awaits and
    # is task-local, so concurrent invocations of the same agent instance
    # never see each other's overrides. Read via :attr:`instructions`.
    _INSTRUCTIONS_OVERRIDE: contextvars.ContextVar[str | None] = contextvars.ContextVar(
        "kaos_agents._instructions_override", default=None
    )

    @property
    def instructions(self) -> str | None:
        """Resolved instructions: per-call override (if set) or the agent default.

        Subclass dispatch paths (e.g. ResearchAgent's ReAct escalation) push a
        temporary augmented prompt via :meth:`override_instructions` instead of
        mutating ``self._instructions`` directly.
        """
        return self._INSTRUCTIONS_OVERRIDE.get() or self._instructions

    @classmethod
    @contextlib.contextmanager
    def override_instructions(cls, augmented: str | None) -> Iterator[None]:
        """Push a per-call instructions override for the lifetime of the block.

        ContextVar-backed so the override is task-local and propagates across
        awaits — safe for concurrent invocations of the same agent instance.
        """
        token = cls._INSTRUCTIONS_OVERRIDE.set(augmented)
        try:
            yield
        finally:
            cls._INSTRUCTIONS_OVERRIDE.reset(token)

    def _model_for_role(self, role: str) -> str:
        """Resolve the model to use for a specific role.

        If a ProviderConfig is attached, delegates to its role_models map.
        Otherwise returns the agent's default model.

        Special cases:
        - role='plan' falls back to settings.planning_llm_model when no
          provider is set (backward compat with the pre-provider path).
        """
        if self._provider is not None:
            from kaos_agents.types.providers import ModelRole

            try:
                return self._provider.model_for(ModelRole(role))
            except ValueError:
                return self._provider.default
        # Backward compat: planning has its own settings field
        if role == "plan":
            return self._settings.planning_llm_model
        # Research pattern gets a sonnet-grade default — retrieval-query
        # quality scales with model strength (observed BM25 top_score
        # 6.4 → 16.9 between haiku-4-5 and sonnet-4-6 / gpt-5.4-mini on
        # identical legal-corpus queries) and aggregate synthesis is
        # materially more reliable.
        if role == "research":
            return self._settings.research_llm_model
        return self._model

    async def run(
        self,
        message: str,
        session_id: str,
        *,
        is_internal_iteration: bool = False,
    ) -> AsyncIterator[KaosEvent]:
        """Execute a single agent turn, yielding events progressively.

        This is the primary streaming entry point. Yields ``KaosEvent``
        objects at each step of the 8-step loop. Consumers iterate:

            async for event in agent.run("Find EPA actions", "session-1"):
                match event:
                    case Span(subject=SpanSubject.TURN, phase=SpanPhase.START): ...
                    case Span(subject=SpanSubject.TOOL_CALL, phase=SpanPhase.START): ...
                    case TurnSummary(): ...

        Args:
            message: The user's message.
            session_id: Session identifier for memory persistence.
            is_internal_iteration: When True, the caller is replaying the
                SAME user message under an outer multi-iteration loop
                (e.g. kaos-agents' AgenticLoop critic-driven replan). The
                agent will NOT persist the user message to
                ``SessionMemory.MESSAGES`` (it's already there from
                iteration 1) and will NOT persist the intermediate
                assistant response. The caller is expected to call
                ``POST /v1/sessions/{id}/memory/messages/turn`` once after
                the loop terminates so memory has exactly one user entry
                plus one assistant entry per turn. Default False keeps
                single-turn callers (CLI, MCP tool, single-shot API
                clients) unchanged. See the iteration-leak fix in
                ``docs/plans/2026-05-19-agentic-loop-honesty.md`` §3.1.a.

        Yields:
            KaosEvent subclass instances in execution order.
        """
        run_id = _generate_run_id()
        emitter = EventEmitter(session_id=session_id, run_id=run_id)

        # Open an event collector so span auto-stitching works:
        # - Each span_start pushes its span_id onto the collector's stack.
        # - The next span_start (e.g. TOOL_CALL inside a TURN) reads the
        #   stack top to synthesize parent_span_id automatically.
        # - span_complete pops.
        # Without this, every span_start lands as a root (parent_span_id
        # null) and OTel tracing flattens the turn → step → tool_call
        # tree. The collector's events list is not consumed — events
        # still flow out via `yield`. This block must wrap every span
        # emission, including those inside dispatched patterns
        # (chat / plan_execute / research) which receive the same
        # emitter and run inside this generator's frame.
        # Theme A (2026-05-25): also publish ``emitter`` to the
        # ``_active_emitter_var`` ContextVar via ``use_emitter`` so deep
        # helpers (capabilities.retrieve._invoke_tool, actions.tool_bridge
        # asyncio.timeout handler, planning.act._act_tool asyncio.timeout
        # handler) can call ``active_emitter().emit(RunError, ...)`` /
        # ``span_error(SpanSubject.TOOL_CALL, ...)`` without the emitter
        # being threaded through their signatures. The collector scope
        # is what captures the events; the emitter scope is what lets
        # those helpers produce events with full base-field metadata.
        with collect_events(), use_emitter(emitter):
            async for event in self._run_inner(
                message,
                session_id,
                emitter,
                run_id,
                is_internal_iteration=is_internal_iteration,
            ):
                yield event

    async def _run_inner(
        self,
        message: str,
        session_id: str,
        emitter: EventEmitter,
        run_id: str,
        *,
        is_internal_iteration: bool = False,
    ) -> AsyncIterator[KaosEvent]:
        """Body of run() — extracted so the outer generator can wrap it
        in a `with collect_events()` block for span-tree stitching."""
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

        turn_name = f"turn.{turn_number}"
        turn_t0 = time.monotonic()
        turn_span = emitter.span_start(
            SpanSubject.TURN,
            name=turn_name,
            attributes={"turn_number": turn_number},
        )
        turn_span_id = turn_span.span_id
        yield turn_span

        # Step 3: Add user message.
        # Skipped when this run is an internal critic-driven replay of
        # the same user turn (AgenticLoop iteration 2+). The canonical
        # write is performed once at loop exit via POST
        # /v1/sessions/{id}/memory/messages/turn. See the iteration-leak
        # fix in docs/plans/2026-05-19-agentic-loop-honesty.md §3.1.a.
        if not is_internal_iteration:
            memory.add(MemoryType.MESSAGES, f"user: {message}")
            emit_memory_added(MemoryType.MESSAGES.value, item_count=1)
        logger.debug(
            "agent.step3_add_message: session=%s message_len=%d internal=%s",
            session_id,
            len(message),
            is_internal_iteration,
        )

        # WU-G.2 / #352 — set the sticky corpus flag whenever the
        # DOCUMENTS section is non-empty entering this turn. The flag
        # persists with the SessionMemory snapshot so a follow-up turn
        # whose total-budget trim drops every DOCUMENTS body still
        # sees a "corpus reachable via search_memory" handle. Calling
        # this here (BEFORE assemble_context) is what lets
        # ``assemble_context``'s default ``pin_corpus_handles=None``
        # see the flag.
        if (
            memory.has_section(MemoryType.DOCUMENTS)
            and memory.section_item_count(MemoryType.DOCUMENTS) > 0
        ):
            memory.mark_corpus_attached()

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

        # Step 5: Classify intent. Auth / rate-limit / service-unavailable /
        # transport / context-too-large failures from the classifier must
        # NOT silently drop through to the heuristic fallback or to an
        # empty dispatch — they need to surface as a structured RunError
        # so downstream tool wrappers can convert to ToolResult.create_error.
        # See ``classify_intent`` in ``kaos_agents/context/classify.py`` for
        # the "which exceptions re-raise" contract and probe 4b in
        # ``docs/design/skeptic-prod-ops-findings.md`` for the why.
        try:
            intent = await self._classify(message, memory, context_items)
        except Exception as exc:
            from kaos_agents.errors import classify_agent_failure

            failure = classify_agent_failure(exc)
            error_type = failure.kind if failure is not None else type(exc).__name__
            recovery_hint = (
                failure.recovery_hint
                if failure is not None
                else "Check logs for details. Try a simpler query."
            )
            logger.warning(
                "agent.run: intent classification failed (%s): %s; "
                "emitting RunError(error_type=%s)",
                type(exc).__name__,
                exc,
                error_type,
            )
            yield emitter.emit(
                RunError,
                error_type=error_type,
                message=str(exc),
                recovery_hint=recovery_hint,
            )
            # End the turn span before returning — otherwise the span
            # collector ends with a dangling parent and downstream OTel
            # traces show an unterminated turn.
            turn_duration_ms = (time.monotonic() - turn_t0) * 1000.0
            yield emitter.span_complete(
                SpanSubject.TURN,
                span_id=turn_span_id,
                name=turn_name,
                duration_ms=turn_duration_ms,
                attributes={"turn_number": turn_number, "error": error_type},
            )
            # Emit a TurnSummary so consumers that pattern-match on
            # ``TurnSummary`` (Runner.turn(), CLI, MCP tool wrapper) still
            # see a terminal frame instead of an unbounded stream.
            yield emitter.emit(
                TurnSummary,
                text="",
                intent="",
                tool_calls=(),
                tokens_used=0,
                cost_usd=0.0,
                input_tokens=0,
                output_tokens=0,
            )
            return

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

        # Step 6: Dispatch to streaming handler — yields events from the handler.
        #
        # Track-3 chunk A2 collapsed the parallel ``tool_calls`` /
        # ``tool_call_summaries`` lists into a single ``tool_executions``
        # list of :class:`ToolExecution` value records. The wire-side
        # :class:`ToolCallSummary` is derived via ``ToolExecution.to_summary()``
        # at TurnSummary emission time, with per-tool cost attribution
        # backfilled via dataclass replace.
        response_text = ""
        tool_executions: list[ToolExecution] = []
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
                # Track 3 chunk B2 — emit RDF triples for the events the
                # session knowledge graph cares about (tool calls, steps,
                # citations). emit_from_event is a no-op for events outside
                # the v1 vocabulary and never raises.
                from kaos_agents.memory.triples import emit_from_event

                emit_from_event(event, memory)
                yield event
                # Collect response data from terminal events for memory update
                if isinstance(event, TextDelta):
                    response_text += event.content
                elif isinstance(event, Span) and event.subject == SpanSubject.TOOL_CALL:
                    if event.phase == SpanPhase.COMPLETE:
                        attrs = event.attributes
                        tool_executions.append(
                            ToolExecution.from_dict_args(
                                tool_name=str(attrs.get("tool_name", "")),
                                arguments={},  # Args live on the START span; not threaded here
                                call_id=str(attrs.get("call_id", "")),
                                result_summary=str(attrs.get("result_summary", "")),
                                is_error=bool(attrs.get("is_error", False)),
                                duration_ms=event.duration_ms or 0.0,
                                plan_id=str(attrs.get("plan_id") or "") or None,
                                step_id=str(attrs.get("step_id") or "") or None,
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
            from kaos_agents.errors import classify_agent_failure

            failure = classify_agent_failure(exc)
            error_type = failure.kind if failure is not None else type(exc).__name__
            recovery_hint = (
                failure.recovery_hint
                if failure is not None
                else "Check logs for details. Try a simpler query."
            )
            logger.warning(
                "agent.run: dispatch failed (%s, error_type=%s): %s",
                type(exc).__name__,
                error_type,
                exc,
            )
            yield emitter.emit(
                RunError,
                error_type=error_type,
                message=str(exc),
                recovery_hint=recovery_hint,
            )

        # If no TextDelta events were yielded, get response from the non-streaming path.
        # This handles the case where _dispatch_streaming falls back to _dispatch.
        if not response_text:
            # The streaming handler may have set response_text via a different mechanism.
            # For BaseAgent's simple respond, we use the non-streaming path.
            pass

        # Auto-extract citations from the assistant's response text and
        # emit CitationFound events. Bug #5 of the workflow audit:
        # explain records reported citations=0 even when answers had
        # URLs / FR / statute cites. The streaming dispatch path
        # populates `response_text` via the `+=` accumulator above; the
        # non-streaming dispatch path further down does the same. Wire
        # citation extraction here so both paths fire it before the
        # TurnSummary is built. Helper is silent when kaos-citations is
        # not installed.
        if response_text:
            from kaos_agents.grounding import emit_citations_for_text

            for citation_event in emit_citations_for_text(emitter, response_text):
                yield citation_event

        logger.debug(
            "agent.step6_complete: session=%s response_len=%d tool_calls=%d",
            session_id,
            len(response_text),
            len(tool_executions),
        )

        # Step 7: Update memory.
        #
        # MESSAGES gets the rendered string content for prompt assembly.
        # ACTIONS items keep the human-readable rendered string as
        # ``content`` (still useful in prompts) AND carry the structured
        # ToolExecution under ``metadata['tool_execution']`` so
        # downstream consumers (graph triple emitter in chunk B2, audit
        # hook, MCP memory-query tool) can read typed data without
        # re-parsing the rendered string.
        #
        # The assistant write is also gated on is_internal_iteration:
        # iterations 2+ of an AgenticLoop produce intermediate drafts
        # that the critic may reject; only the post-loop memory-turn
        # endpoint writes the canonical final assistant text. Tool
        # executions (ACTIONS section) DO persist on every iteration so
        # the next iteration's classifier + planner see what was tried.
        if response_text and not is_internal_iteration:
            memory.add(MemoryType.MESSAGES, f"assistant: {response_text}")
            emit_memory_added(MemoryType.MESSAGES.value, item_count=1)
        if tool_executions:
            for te in tool_executions:
                summary = f"Tool: {te.tool_name}({te.arguments}) → {te.result_summary}"
                memory.add(
                    MemoryType.ACTIONS,
                    summary,
                    metadata={"tool_execution": te.to_dict()},
                )
            yield emitter.emit(
                MemoryEvent,
                kind=MemoryEventKind.ADDED,
                section=MemoryType.ACTIONS.value,
                item_count=len(tool_executions),
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

        # Backfill per-tool cost attribution onto the executions before
        # we project them to wire-side summaries for TurnSummary. Builds
        # a fresh list of frozen :class:`ToolExecution` records — we
        # ``dataclasses.replace`` each one with the per-tool slice of
        # usage on top of the base shape.
        from dataclasses import replace as _dc_replace

        attributed_executions: list[ToolExecution] = []
        for te in tool_executions:
            usage = per_tool_usage.get(te.tool_name, ZERO_USAGE)
            if usage is ZERO_USAGE:
                attributed_executions.append(te)
                continue
            attributed_executions.append(
                _dc_replace(
                    te,
                    cost_usd=usage.cost_usd,
                    input_tokens=usage.input_tokens,
                    output_tokens=usage.output_tokens,
                )
            )

        # Project to wire-side summaries for the event stream.
        attributed_summaries = tuple(te.to_summary() for te in attributed_executions)

        turn_duration_ms = (time.monotonic() - turn_t0) * 1000.0
        yield emitter.span_complete(
            SpanSubject.TURN,
            span_id=turn_span_id,
            name=turn_name,
            duration_ms=turn_duration_ms,
            attributes={"turn_number": turn_number},
        )
        yield emitter.emit(
            TurnSummary,
            text=response_text,
            intent=intent.intent.value,
            tool_calls=attributed_summaries,
            tokens_used=turn_usage.total_tokens,
            cost_usd=turn_usage.cost_usd,
            input_tokens=turn_usage.input_tokens,
            output_tokens=turn_usage.output_tokens,
        )

    # ``turn()`` is inherited from :class:`KaosAgent`. The default
    # collects events from :meth:`run` and converts via
    # :func:`kaos_agents.runtime.events_to_response.events_to_response`.

    # -- Streaming dispatch (override in subclasses) ---------------------------

    async def _dispatch_streaming(
        self,
        intent: IntentResult,
        message: str,
        memory: SessionMemory,
        context_items: dict[MemoryType, list[Any]],
        emitter: EventEmitter,
    ) -> AsyncIterator[KaosEvent]:
        """Dispatch to the appropriate streaming handler based on intent.

        Yields KaosEvent instances. Subclasses override to add streaming
        for tool_use, research, plan handlers.

        Default implementation falls back to _dispatch() and yields
        the result as a single TextDelta.

        Pattern-mismatch detection (0.1.0a10): when the per-turn intent
        demands ``_handle_plan`` or ``_handle_research`` but the
        running agent class hasn't overridden the base implementation
        (which silently degrades to ``_handle_respond``), emit a typed
        ``PatternMismatch`` event and redirect to ``_handle_tool_use``
        so at least ReAct fires instead of returning a confident
        training-data answer with zero tool calls. The most common
        trigger: a session opened with ``pattern="chat"`` (the API
        default) whose per-turn message classifies as ``PLAN`` or
        ``RESEARCH``. See ``kaos-modules/docs/plans/
        kaos-agents-autonomy-improvement-1.md`` for the diagnosis.
        """
        # RESEARCH intent: stream FindingsAgent dispatch directly so
        # ``CitationFound`` events reach SSE / Citations panel consumers.
        # The non-streaming path (``_handle_research`` → tuple) collapses
        # per-finding citations into a single TextDelta and loses them;
        # the streaming form preserves the per-finding granularity.
        # Subclasses that override ``_handle_research_streaming`` get
        # picked up automatically (ResearchAgent has its own override).
        if intent.intent == IntentType.RESEARCH:
            async for event in self._handle_research_streaming(
                message, memory, context_items, emitter
            ):
                yield event
            return

        # Pattern-mismatch redirect — before the silent-fall-through bug
        # can fire. The detector returns (handler, mismatch_event) so the
        # event flows through the generator and reaches downstream
        # consumers (SSE stream, OTel hook, test event lists). The old
        # path called ``emitter.emit(PatternMismatch, ...)`` inside the
        # detector and discarded the return — that only populated active
        # ``collect_events()`` collectors (unit tests) but never yielded
        # the event to the run() loop's ``async for`` iterator, so
        # production stream consumers saw zero ``PatternMismatch`` events
        # even when the redirect fired.
        redirect_handler, mismatch_event = self._detect_pattern_mismatch(intent, emitter)
        if redirect_handler is not None:
            if mismatch_event is not None:
                yield mismatch_event
            dispatched_redirect = await redirect_handler(message, memory, context_items)
            if len(dispatched_redirect) == 3:
                response_text, tool_calls, usage = dispatched_redirect
            else:  # pragma: no cover — defensive for pre-Phase-5.0 mocks
                response_text, tool_calls = dispatched_redirect
                usage = ZERO_USAGE
            # Surface the redirected handler's tool calls + text via the
            # same shape the default path uses below.
            async for ev in self._yield_dispatched_events(
                response_text, tool_calls, usage, emitter
            ):
                yield ev
            return

        # Default: use the non-streaming handlers and wrap the result.
        # Pre-Phase-5.0 callers (and test mocks) may return a 2-tuple
        # without usage — accept both shapes so we don't force every
        # downstream test fixture to be rewritten at once.
        dispatched = await self._dispatch(intent, message, memory, context_items)
        if len(dispatched) == 3:
            response_text, tool_calls, usage = dispatched
        else:
            response_text, tool_calls = dispatched
            usage = ZERO_USAGE

        async for ev in self._yield_dispatched_events(response_text, tool_calls, usage, emitter):
            yield ev

    def _detect_pattern_mismatch(
        self,
        intent: IntentResult,
        emitter: EventEmitter,
    ) -> tuple[Any, Any]:
        """Return ``(redirect_handler, mismatch_event)`` when the per-turn
        intent demands a handler the agent class hasn't overridden, or
        ``(None, None)`` otherwise.

        The mismatch event is constructed via ``emitter.emit`` so it gets
        a real sequence number + session/run id + collector push (for
        ``collect_events()`` consumers), BUT the actual yielding to the
        stream is the caller's responsibility — ``_dispatch_streaming``
        yields ``mismatch_event`` before invoking the redirect so the
        event reaches SSE / OTel / live-test consumers, not just
        in-process collectors.

        Kept as an instance method so subclasses that DO override
        ``_handle_plan`` / ``_handle_research`` (i.e. ``PlanExecuteAgent``,
        ``ResearchAgent``) get the unmodified dispatch path —
        ``_handler_is_default`` returns ``False`` for them and the
        function returns ``(None, None)``.

        Returns ``(None, None)`` when:
        * the intent is satisfied by an overridden handler, OR
        * the intent is not in {PLAN, RESEARCH} (other intents have
          ``BaseAgent``-level handlers that do not silently degrade —
          RESPOND, CLARIFY, TOOL_USE).
        """
        if intent.intent == IntentType.PLAN and self._handler_is_default("_handle_plan"):
            recommended = "plan"
            classified = "plan"
        else:
            # RESEARCH no longer falls through here: BaseAgent's
            # _handle_research is now a real FindingsAgent-backed
            # default (closes CS-B2 hallucination / CS-B3 give-up
            # cliff), so the silent-fall-through-to-respond bug it
            # used to redirect away from no longer exists.
            return None, None

        agent_pattern = type(self).metadata().pattern
        rationale = (
            f"Per-turn intent classified as '{classified}' but agent class "
            f"{type(self).__name__} (pattern={agent_pattern!r}) does not "
            f"override _handle_{classified}. Pre-0.1.0a10 BaseAgent would "
            "have silently degraded to _handle_respond (no tools, no plan, "
            "training-data answer). Redirecting to _handle_tool_use so "
            "ReAct fires at minimum. Recommendation: re-open the session "
            f"with pattern='{recommended}' for the full PlanExecuteAgent / "
            "ResearchAgent surface."
        )
        mismatch_event = emitter.emit(
            PatternMismatch,
            classified_intent=classified,
            agent_pattern=agent_pattern,
            recommended_pattern=recommended,
            fallback_handler="_handle_tool_use",
            rationale=rationale,
        )
        return self._handle_tool_use, mismatch_event

    def _handler_is_default(self, name: str) -> bool:
        """True when ``type(self).<name>`` is the unmodified
        :class:`BaseAgent` implementation (i.e., no subclass override).

        Used by :meth:`_detect_pattern_mismatch` — when a handler is
        the BaseAgent default, the silent-fall-through-to-_handle_respond
        bug is about to fire and we should redirect instead.
        """
        cls_method = getattr(type(self), name, None)
        base_method = getattr(BaseAgent, name, None)
        if cls_method is None or base_method is None:
            return False
        # Compare underlying functions to see through @classmethod /
        # bound-method wrappers.
        return getattr(cls_method, "__func__", cls_method) is getattr(
            base_method, "__func__", base_method
        )

    async def _yield_dispatched_events(
        self,
        response_text: str,
        tool_calls: list[ToolExecution],
        usage: InvocationUsage,
        emitter: EventEmitter,
    ) -> AsyncIterator[KaosEvent]:
        """Emit the standard tool-call / TextDelta / UsageObserved
        trio for a non-streaming dispatch result.

        Extracted from :meth:`_dispatch_streaming` so the
        :meth:`_detect_pattern_mismatch` redirect path emits the same
        event shape as the default path.
        """
        # Yield tool call events if any
        for tc in tool_calls:
            tc_span = emitter.span_start(
                SpanSubject.TOOL_CALL,
                name=f"tool.{tc.tool_name}",
                attributes={
                    "tool_name": tc.tool_name,
                    "call_id": tc.tool_name,  # Use tool_name as call_id for backward compat
                    "arguments": tc.arguments,
                },
            )
            yield tc_span
            yield emitter.span_complete(
                SpanSubject.TOOL_CALL,
                span_id=tc_span.span_id,
                name=f"tool.{tc.tool_name}",
                duration_ms=0.0,
                attributes={
                    "tool_name": tc.tool_name,
                    "call_id": tc.tool_name,
                    "result_summary": tc.result_summary,
                    "is_error": tc.is_error,
                },
            )

        # Yield the response text
        if response_text:
            yield emitter.emit(TextDelta, content=response_text)
        # Citation extraction lives in the outer `run()` loop above
        # (single emit point) — the outer loop's async-for accumulates
        # the TextDelta this method just yielded.

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

        # 0.1.0a17: surface the live tool-category catalog so the
        # classifier can reason about which category fits the user's
        # question instead of falling back to memory-only ``respond``
        # for verifiable real-world questions. Subclasses that hold a
        # KaosRuntime (ChatAgent, ResearchAgent, PlanExecuteAgent) read
        # it through ``self._runtime``; BaseAgent has no runtime
        # attribute so the helper returns ``""`` and the classifier
        # keeps its pre-fix behavior. See
        # :func:`render_tool_categories_for_classifier` for the
        # output shape and the senator-question regression note.
        from kaos_agents.context.tool_catalog import render_tool_categories_for_classifier

        runtime = getattr(self, "_runtime", None)
        available_tool_categories = render_tool_categories_for_classifier(runtime)

        return await classify_intent(
            message,
            memory,
            model=self._model_for_role("classify"),
            context_text=context_text,
            available_tool_categories=available_tool_categories,
        )

    async def _dispatch(
        self,
        intent: IntentResult,
        message: str,
        memory: SessionMemory,
        context_items: dict[MemoryType, list[Any]],
    ) -> tuple[str, list[ToolExecution], InvocationUsage]:
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
    ) -> tuple[str, list[ToolExecution], InvocationUsage]:
        """Handle simple conversational response. Uses a Call."""
        response, usage = await self._simple_respond(message, memory, context_items=context_items)
        return response, [], usage

    async def _handle_clarify(
        self,
        message: str,
        memory: SessionMemory,
        context_items: dict[MemoryType, list[Any]],
    ) -> tuple[str, list[ToolExecution], InvocationUsage]:
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
    ) -> tuple[str, list[ToolExecution], InvocationUsage]:
        """Handle tool-using request. Override in ChatAgent to use ReAct."""
        # BaseAgent falls back to simple response (no tools configured)
        return await self._handle_respond(message, memory, context_items)

    async def _handle_research(
        self,
        message: str,
        memory: SessionMemory,
        context_items: dict[MemoryType, list[Any]],
    ) -> tuple[str, list[ToolExecution], InvocationUsage]:
        """Handle research/document Q&A via the FindingsAgent default.

        Closes CS-B2 (hallucination) + CS-B3 (give-up cliff): when the
        session has attached documents, the in-memory DOCUMENTS section
        may have been redacted (corpus-stress fixtures) or summarized
        (large corpora) — the classifier reading that redacted text
        will hallucinate. The right move is to re-read the source
        bytes from the VFS, build a fresh ``DocumentView``, and let
        ``FindingsAgent`` ground the answer against the bytes.

        Non-streaming wrapper: drives :meth:`_run_findings_dispatch`
        and collapses its result to the legacy
        ``(answer, [], usage)`` tuple. Callers that want
        per-finding ``CitationFound`` events should use the streaming
        path (see :meth:`_handle_research_streaming`).

        Falls back to :meth:`_handle_respond` when no attached
        documents are resolvable (no DOCUMENTS items, or no VFS /
        runtime). The fallback preserves the historical behavior so
        agents without an attached corpus keep working.
        """
        findings_result, fallback_text, usage = await self._run_findings_dispatch(
            message, memory, context_items
        )
        if findings_result is None:
            if fallback_text is not None:
                return fallback_text, [], usage
            # No corpus to ground on — historical behavior.
            return await self._handle_respond(message, memory, context_items)
        answer = findings_result.answer
        if not answer:
            # FindingsAgent refused (no candidates / no relevant) —
            # surface the refusal reason so the user sees an honest
            # "I couldn't find that" rather than an empty string.
            refusal = findings_result.refusal
            if refusal is not None:
                answer = (
                    f"I couldn't ground that question in the attached documents ({refusal.reason})."
                )
            else:
                answer = "I couldn't ground that question in the attached documents."
        return answer, [], usage

    async def _handle_research_streaming(
        self,
        message: str,
        memory: SessionMemory,
        context_items: dict[MemoryType, list[Any]],
        emitter: EventEmitter,
    ) -> AsyncIterator[KaosEvent]:
        """Streaming variant of :meth:`_handle_research`.

        Yields:
        * ``Span(RESEARCH, START)`` / ``Span(RESEARCH, COMPLETE)`` boundary events
        * one ``CitationFound`` per surviving finding (with ``source_uri``
          set to the originating block_ref or filename when available)
        * a single ``TextDelta`` carrying the synthesized answer
        * ``UsageObserved`` with the filter+synthesis cost rollup

        On the no-corpus fallback path, emits a single ``TextDelta``
        + ``UsageObserved`` from :meth:`_simple_respond` (matching the
        pattern used by ``_handle_tool_use_streaming``'s fallback in
        :class:`ChatAgent`).
        """
        # SUBAGENT is the closest existing SpanSubject fit — FindingsAgent
        # genuinely is a sub-agent (selector → filter → synthesize) that the
        # outer agent delegates to. Telemetry parents land at the right
        # level in OTel without a new enum value.
        research_span = emitter.span_start(
            SpanSubject.SUBAGENT,
            name="research.findings_dispatch",
            attributes={"path": "findings_default"},
        )
        yield research_span

        findings_result, fallback_text, usage = await self._run_findings_dispatch(
            message, memory, context_items
        )

        if findings_result is None:
            text = fallback_text
            if text is None:
                # No DOCUMENTS / no VFS — simple respond, single TextDelta.
                text, usage = await self._simple_respond(
                    message, memory, context_items=context_items
                )
            if text:
                yield emitter.emit(TextDelta, content=text)
            yield emit_usage_observed(emitter, usage, source="research-fallback-respond")
            yield emitter.span_complete(
                SpanSubject.SUBAGENT,
                span_id=research_span.span_id,
                name="research.findings_dispatch",
                attributes={"findings": 0, "answer_chars": len(text or "")},
            )
            return

        # Surface the FindingsAgent run as an honest synthetic tool call
        # so ``response.tool_calls`` (and the Citations panel, and the
        # corpus-stress judge that grounds on tool trace) reflects that
        # retrieval actually happened. Without this, zero-tool-call
        # answers look like fabrications even when they're correctly
        # grounded by the dispatch.
        tc_span = emitter.span_start(
            SpanSubject.TOOL_CALL,
            name="tool.kaos-agent-findings-dispatch",
            attributes={
                "tool_name": "kaos-agent-findings-dispatch",
                "call_id": research_span.span_id,
                "arguments": (
                    ("question", message[:200]),
                    ("selector", "every_sentence_selector"),
                ),
            },
        )
        yield tc_span

        # Findings path — emit one CitationFound per surviving finding.
        # The finding's ``block_ref`` plus the ``source_uri`` lookup
        # together give the SPA Citations panel everything it needs to
        # back-link to the source AST.
        source_uri_for_block: dict[str, str] = {}
        for item in memory.get(MemoryType.DOCUMENTS):
            meta_uri = item.metadata.get("uri") or item.metadata.get("filename") or ""
            if meta_uri:
                source_uri_for_block[meta_uri] = meta_uri

        for finding in findings_result.findings:
            candidate = finding.candidate
            block_ref = candidate.block_ref or ""
            # Best-effort: the block_ref is a JSON pointer like
            # /body/3, not a filename. We surface block_ref as the
            # citation key; consumers join it with the DOCUMENTS
            # metadata downstream.
            yield emitter.emit(
                CitationFound,
                claim=candidate.text,
                source_uri=block_ref or candidate.finding_id,
                confidence=float(finding.relevance),
                verified=True,
            )

        # Close the synthetic tool-call span — populates response.tool_calls
        # with a "kaos-agent-findings-dispatch" entry (corpus-stress judge
        # reads ``response.tool_calls`` via _render_tool_trace, and our
        # entry matches the "findings" family pattern in
        # ``_assert_retrieval_tool``).
        result_summary = (
            f"FindingsAgent: enumerated={findings_result.total_enumerated} "
            f"filtered={findings_result.total_filtered} "
            f"cost=${findings_result.total_cost_usd:.4f} "
            f"answer_chars={len(findings_result.answer or '')}"
        )
        yield emitter.span_complete(
            SpanSubject.TOOL_CALL,
            span_id=tc_span.span_id,
            name="tool.kaos-agent-findings-dispatch",
            attributes={
                "tool_name": "kaos-agent-findings-dispatch",
                "result_summary": result_summary,
                "is_error": False,
                "cost_usd": findings_result.total_cost_usd,
            },
        )

        if findings_result.answer:
            yield emitter.emit(TextDelta, content=findings_result.answer)

        yield emit_usage_observed(emitter, usage, source="research-findings")
        yield emitter.span_complete(
            SpanSubject.SUBAGENT,
            span_id=research_span.span_id,
            name="research.findings_dispatch",
            attributes={
                "findings": len(findings_result.findings),
                "enumerated": findings_result.total_enumerated,
                "filtered": findings_result.total_filtered,
                "answer_chars": len(findings_result.answer or ""),
                "cost_usd": findings_result.total_cost_usd,
            },
        )

    async def _run_findings_dispatch(
        self,
        message: str,
        memory: SessionMemory,
        context_items: dict[MemoryType, list[Any]],
    ) -> tuple[Any | None, str | None, InvocationUsage]:
        """Resolve attached documents + run the FindingsAgent pipeline.

        Returns a 3-tuple ``(findings_result, fallback_text, usage)``:

        * ``findings_result`` — a ``FindingsResult`` when the dispatch
          ran; ``None`` when no resolvable corpus was found OR the
          FindingsAgent failed.
        * ``fallback_text`` — pre-resolved fallback answer (only set
          when the corpus is unresolvable AND a simple-respond would
          be wasteful, e.g. settings forbid LLM calls). Usually
          ``None`` — callers should run their own simple-respond
          fallback to keep responsibility local.
        * ``usage`` — ``InvocationUsage`` reflecting filter+synthesis
          spend (``ZERO_USAGE`` when no LLM call was made).

        Subclasses can override to inject a different selector /
        model / corpus assembly strategy. The default selects
        every-sentence on the merged DocumentView and uses the
        agent's configured ``_model`` for synthesis with the
        settings default for the cheap filter pass.
        """
        # Find DOCUMENTS in session memory — the SPA + corpus-stress
        # fixtures both populate this section on file upload.
        docs_items = memory.get(MemoryType.DOCUMENTS)
        if not docs_items:
            logger.debug("base_agent._run_findings_dispatch: no DOCUMENTS items, skipping")
            return None, None, ZERO_USAGE

        # Build the merged DocumentView + underlying ContentDocument +
        # sentence segmenter. Text-like items re-read VFS bytes for
        # ground truth; binary items hit the eager pre-flight parser
        # (kaos-pdf / kaos-office) when item.content is headline-only.
        bundle = await self._resolve_corpus_view_with_document(docs_items)
        if bundle is None:
            logger.debug(
                "base_agent._run_findings_dispatch: no resolvable DocumentView, "
                "falling through to simple respond"
            )
            return None, None, ZERO_USAGE
        full_view, full_document, segmenter = bundle

        # Retrieval planner: pick the narrowing strategy + typed probes.
        # Skip the LLM call for tiny corpora (the planner would pick
        # NONE anyway — see retrieval_plan_floor setting).
        from kaos_agents.patterns.retrieval import (
            LLMRetrievalPlanner,
            RetrievalPlanResult,
            RetrievalStrategy,
            apply_retrieval_plan,
        )

        plan_floor = int(getattr(self._settings, "retrieval_plan_floor", 5) or 5)
        planner_usage_cost = 0.0
        planner_usage_tokens = 0
        if len(docs_items) < plan_floor:
            plan = RetrievalPlanResult(
                strategy=RetrievalStrategy.NONE,
                reasoning=(
                    f"skip-floor: {len(docs_items)} docs < floor={plan_floor}, "
                    "no LLM planner call needed"
                ),
            )
        else:
            plan_model = (
                getattr(self._settings, "retrieval_plan_model", None)
                or getattr(self._settings, "default_llm_model", None)
                or self._model
            )
            corpus_summary = self._summarize_corpus_for_planner(docs_items)
            try:
                planner = LLMRetrievalPlanner(model=plan_model)
                plan = await planner.plan(question=message, corpus_summary=corpus_summary)
                if plan.usage is not None:
                    planner_usage_cost = float(getattr(plan.usage, "cost_usd", 0.0) or 0.0)
                    planner_usage_tokens = int(getattr(plan.usage, "total_tokens", 0) or 0)
            except Exception:
                logger.exception(
                    "base_agent._run_findings_dispatch: planner failed — "
                    "falling back to strategy=NONE"
                )
                plan = RetrievalPlanResult(
                    strategy=RetrievalStrategy.NONE,
                    reasoning="planner Call raised; degraded to NONE",
                )

        # Apply the plan — mechanical narrowing, no LLM.
        view, apply_result = await apply_retrieval_plan(
            plan,
            full_view=full_view,
            full_document=full_document,
            docs_items=docs_items,
            memory=memory,
            sentence_segmenter=segmenter,
        )
        logger.info(
            "base_agent._run_findings_dispatch: "
            "strategy=%s applied=%s kept=%d/%d planner_cost=$%.4f",
            plan.strategy.value,
            apply_result.strategy.value,
            apply_result.kept,
            len(docs_items),
            planner_usage_cost,
        )

        # Run FindingsAgent on the narrowed view.
        from kaos_agents._llm_imports import require_llm_core

        require_llm_core()
        from kaos_agents.patterns.findings import (
            FindingsAgent,
            every_sentence_selector,
        )

        filter_model = getattr(self._settings, "default_llm_model", None) or self._model
        synthesis_model = self._model or filter_model

        agent = FindingsAgent(
            selector=every_sentence_selector,
            filter_model=filter_model,
            synthesis_model=synthesis_model,
            chunk_size=20,
            num_parallel=3,
            relevance_threshold=0.4,
        )

        try:
            result = await agent.run(message, view)
        except Exception:
            logger.exception(
                "base_agent._run_findings_dispatch: FindingsAgent.run failed; "
                "falling through to simple respond"
            )
            return None, None, ZERO_USAGE

        total_cost = result.filter_cost_usd + result.synthesis_cost_usd + planner_usage_cost
        total_tokens = result.filter_tokens + result.synthesis_tokens + planner_usage_tokens
        usage = InvocationUsage(
            input_tokens=0,
            output_tokens=0,
            total_tokens=total_tokens,
            cost_usd=total_cost,
        )
        return result, None, usage

    def _summarize_corpus_for_planner(self, docs_items: list[Any]) -> str:
        """One-line corpus summary for the retrieval planner Signature.

        Counts by mime family + total docs. Cheap; no LLM call. The
        planner uses this to choose strategy + top_k — e.g. 50 docs
        of mixed mime → BM25 strategy at top_k=15.
        """
        from collections import Counter

        mime_groups: Counter[str] = Counter()
        for item in docs_items:
            mime = (item.metadata or {}).get("mime_type", "") or ""
            if "pdf" in mime:
                mime_groups["PDF"] += 1
            elif "wordprocessing" in mime or "docx" in mime:
                mime_groups["DOCX"] += 1
            elif "spreadsheet" in mime or "xlsx" in mime:
                mime_groups["XLSX"] += 1
            elif "presentation" in mime or "pptx" in mime:
                mime_groups["PPTX"] += 1
            elif "html" in mime:
                mime_groups["HTML"] += 1
            elif "json" in mime:
                mime_groups["JSON"] += 1
            elif mime.startswith("text/"):
                mime_groups["TEXT"] += 1
            else:
                mime_groups["OTHER"] += 1

        parts = ", ".join(f"{count} {fmt}" for fmt, count in mime_groups.most_common())
        return f"{len(docs_items)} docs ({parts})" if parts else f"{len(docs_items)} docs"

    @staticmethod
    def _looks_like_headline_only(content: str) -> bool:
        """Detect the SPA + corpus-stress headline-only shape for binary docs.

        Production SPA pre-extracts DOCX / PDF text at upload time so
        ``item.content`` is a multi-KB blob of real text. The
        corpus-stress fixture (and any caller that uploads bytes
        without pre-extracting) leaves ``item.content`` as a short
        header line of the form
        ``"filename: ... | path: ... | size_bytes: ... | content_type: ..."``.
        That signal is precise enough to trigger eager parsing on the
        FindingsAgent dispatch path without false-positiving on real
        body text.
        """
        if not content:
            return True
        # The exact shape the SPA upload pipeline + corpus_fixtures emits.
        return "size_bytes:" in content and len(content) < 500

    @staticmethod
    def _ocr_pdf_bytes_to_content_document(
        filename: str,
        body: bytes,
    ) -> Any | None:
        """OCR fallback for image-only PDFs that yield no text layer.

        Used when ``parse_pdf_bytes`` returns a ContentDocument with an
        empty body — i.e. the PDF is scanned and has no extractable
        text. Renders each page via kaos-pdf + Tesseract and emits one
        Paragraph per OCR line plus a ``[page N]`` marker between pages
        so the FindingsAgent dispatch has real text to enumerate over.

        Returns ``None`` when (a) kaos-pdf isn't installed, (b)
        Tesseract isn't on the host, or (c) OCR yields no text. The
        caller falls back to the empty parse result.
        """
        import tempfile
        from pathlib import Path

        try:
            from kaos_content.model.blocks import Paragraph
            from kaos_content.model.document import ContentDocument
            from kaos_content.model.inlines import Text
            from kaos_pdf import (
                TesseractEngine,
                TesseractNotInstalledError,
                get_page_count,
                render_page,
            )
        except ImportError:
            return None

        try:
            engine = TesseractEngine()
        except TesseractNotInstalledError:
            logger.debug(
                "base_agent._ocr_pdf_bytes: tesseract not installed; "
                "skipping OCR fallback for filename=%r",
                filename,
            )
            return None

        blocks: list[Any] = []
        try:
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=True) as tmp:
                tmp.write(body)
                tmp.flush()
                path = Path(tmp.name)
                n_pages = get_page_count(path)
                for page_idx in range(n_pages):
                    image = render_page(path, page_idx, dpi=300)
                    ocr = engine.extract_sync(image)
                    page_text = (ocr.text or "").strip()
                    if not page_text:
                        continue
                    blocks.append(Paragraph(children=(Text(value=f"[page {page_idx + 1}]"),)))
                    for raw_line in page_text.splitlines():
                        line = raw_line.strip()
                        if line:
                            blocks.append(Paragraph(children=(Text(value=line),)))
        except TesseractNotInstalledError:
            # pytesseract / system tesseract missing — install via
            # `kaos-pdf[ocr]` to enable this fallback. Treat as a
            # configuration miss rather than an exception.
            logger.debug(
                "base_agent._ocr_pdf_bytes: pytesseract not installed; "
                "skipping OCR fallback for filename=%r (install kaos-pdf[ocr])",
                filename,
            )
            return None
        except Exception:
            logger.exception(
                "base_agent._ocr_pdf_bytes: OCR fallback raised on filename=%r — skipping",
                filename,
            )
            return None

        if not blocks:
            return None
        return ContentDocument(body=tuple(blocks))

    @staticmethod
    def _parse_binary_bytes_to_content_document(
        filename: str,
        mime: str,
        body: bytes,
    ) -> Any | None:
        """Best-effort re-parse of binary VFS bytes into a ContentDocument.

        Option A (eager pre-flight extraction) — when the agent finds a
        DOCUMENTS item whose ``item.content`` is headline-only (the
        SPA's pre-upload state OR the corpus-stress fixture shape) and
        whose mime is a binary office / PDF format, parse the bytes
        in-process so :class:`FindingsAgent` can ground on the real
        document body rather than the headline metadata.

        PDFs use the bytes-native ``parse_pdf_bytes`` (no temp file
        round-trip). DOCX / PPTX / XLSX go through a ``NamedTemporaryFile``
        because their parsers are path-only today; the temp file is
        immediately deleted via the context manager. Returns ``None``
        when (a) the parser dep isn't installed, (b) the parse raises,
        or (c) the mime isn't one we handle here — the caller falls
        back to ``item.content`` so the dispatch keeps going.
        """
        import tempfile
        from pathlib import Path

        try:
            if "pdf" in mime:
                from kaos_pdf import parse_pdf_bytes

                doc = parse_pdf_bytes(body, filename=filename)
                # Scanned PDFs (no text layer) yield an empty body.
                # Without an OCR fallback the FindingsAgent dispatch
                # enumerates 0 candidates and synthesis emits "(empty)"
                # (corpus-stress S03). Try OCR before returning.
                if not getattr(doc, "body", None):
                    ocr_doc = BaseAgent._ocr_pdf_bytes_to_content_document(
                        filename=filename,
                        body=body,
                    )
                    if ocr_doc is not None:
                        return ocr_doc
                return doc
            if (
                "wordprocessingml" in mime
                or "officedocument.wordprocessingml" in mime
                or filename.lower().endswith(".docx")
            ):
                from kaos_office import parse_docx

                with tempfile.NamedTemporaryFile(suffix=".docx", delete=True) as tmp:
                    tmp.write(body)
                    tmp.flush()
                    return parse_docx(Path(tmp.name))
            if "presentationml" in mime or filename.lower().endswith(".pptx"):
                from kaos_office import parse_pptx

                with tempfile.NamedTemporaryFile(suffix=".pptx", delete=True) as tmp:
                    tmp.write(body)
                    tmp.flush()
                    return parse_pptx(Path(tmp.name))
            if "spreadsheetml" in mime or filename.lower().endswith(".xlsx"):
                # XLSX returns a TabularDocument — different shape from
                # ContentDocument. We coerce to ContentDocument by
                # serialising its rows as one paragraph per row so
                # FindingsAgent's sentence selector has a non-trivial
                # candidate set. Native TabularDocument handling is a
                # follow-up if richer cell-level grounding is needed.
                from kaos_content.model.blocks import Paragraph
                from kaos_content.model.document import ContentDocument
                from kaos_content.model.inlines import Text
                from kaos_office import parse_xlsx

                with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=True) as tmp:
                    tmp.write(body)
                    tmp.flush()
                    tabular = parse_xlsx(Path(tmp.name))

                blocks: list[Paragraph] = []
                # TabularDocument exposes `.tables` (a sequence of
                # Table value-types); each Table has `.rows` (sequence
                # of tuples). Render each row as a tab-joined paragraph.
                for table in getattr(tabular, "tables", []) or []:
                    title = getattr(table, "name", "") or ""
                    if title:
                        blocks.append(Paragraph(children=(Text(value=f"[sheet: {title}]"),)))
                    for row in getattr(table, "rows", []) or []:
                        cells = "\t".join(str(c) for c in row if c is not None)
                        if cells.strip():
                            blocks.append(Paragraph(children=(Text(value=cells),)))
                if not blocks:
                    return None
                return ContentDocument(body=tuple(blocks))
        except ImportError:
            logger.debug(
                "base_agent._parse_binary_bytes: parser dep missing for "
                "mime=%r filename=%r — falling back to item.content",
                mime,
                filename,
            )
            return None
        except Exception:
            logger.exception(
                "base_agent._parse_binary_bytes: parser raised on "
                "mime=%r filename=%r — falling back to item.content",
                mime,
                filename,
            )
            return None
        return None

    async def _build_corpus_view_from_documents(
        self,
        docs_items: list[Any],
    ) -> Any | None:
        """Merge ``DOCUMENTS`` section items into one :class:`DocumentView`.

        Thin compatibility wrapper around
        :meth:`_resolve_corpus_view_with_document`. Returns only the view
        (legacy callers don't need the underlying document). The applier
        path uses the richer accessor to get all three artifacts at once.

        Returns ``None`` when no usable content was assembled.
        """
        bundle = await self._resolve_corpus_view_with_document(docs_items)
        return bundle[0] if bundle is not None else None

    async def _resolve_corpus_view_with_document(
        self,
        docs_items: list[Any],
    ) -> tuple[Any, Any, Any] | None:
        """Merge ``DOCUMENTS`` items into ``(view, document, segmenter)``.

        Single-pass builder used by both the legacy view-only accessor
        and the retrieval-planner applier (which needs the document
        AST for ``kaos_content.search.search_document`` and the
        segmenter to rebuild a narrowed view).

        Text-like items (``mime_type`` starts with ``text/`` or is
        ``application/json``) are re-parsed from their VFS bytes
        when possible. Binary items fall back to the already-extracted
        ``item.content`` unless they look headline-only, in which case
        kaos-pdf / kaos-office parse the VFS bytes (Option A eager
        pre-flight extraction).

        Returns ``None`` when no usable content was assembled.
        """
        try:
            from kaos_content import parse_plain_text
            from kaos_content.model.blocks import Paragraph
            from kaos_content.model.document import ContentDocument
            from kaos_content.model.inlines import Text
            from kaos_content.parsers.html import parse_html
            from kaos_content.views.document_view import DocumentView
            from kaos_nlp_core._defaults import get_default_punkt_tokenizer
        except ImportError:
            logger.warning(
                "base_agent._build_corpus_view_from_documents: kaos_content / "
                "kaos_nlp_core not available; skipping findings dispatch"
            )
            return None

        # Annotated as ``list[Any]`` because parsed binary / HTML / plain-text
        # documents emit a mix of block types (Paragraph, Heading, BlockQuote
        # for HTML; Paragraph + List for plain text). The downstream
        # ContentDocument constructor and DocumentView accept any block type.
        blocks: list[Any] = []
        for item in docs_items:
            meta = item.metadata
            filename = meta.get("filename") or meta.get("uri") or "(unnamed)"
            mime = meta.get("mime_type", "")
            vfs_path = meta.get("vfs_path", "")

            text: str | None = None
            binary_doc: Any | None = None
            # ``_runtime`` is assigned by subclasses (ChatAgent etc.) — use
            # getattr so BaseAgent itself remains usable when no agent
            # was constructed with a runtime (e.g. pure-LLM smoke runs).
            runtime = getattr(self, "_runtime", None)
            runtime_vfs = getattr(runtime, "vfs", None) if runtime is not None else None
            if vfs_path and runtime_vfs is not None:
                # Read once + content-sniff. The manifest mime may LIE
                # (extension spoof S02, manifest-mime spoof S17) so we
                # trust the byte signature over the declared mime. Falls
                # through to declared mime when the detector is missing.
                try:
                    raw = await runtime_vfs.read(vfs_path)
                except Exception:
                    logger.debug(
                        "base_agent: VFS read failed for %s; using item.content",
                        vfs_path,
                    )
                    raw = None

                sniffed_mime = mime
                sniffed_group = ""
                if raw is not None:
                    try:
                        from kaos_nlp_core.content_type import detect as _ct_detect

                        ct = _ct_detect(raw[:65536])
                        sniffed_mime = ct.mime_type or mime
                        sniffed_group = ct.group or ""
                    except Exception:
                        # Detector unavailable / failed — fall through
                        # to declared mime + filename-based heuristics.
                        pass

                # Decide path on sniffed mime/group.
                is_text_format = sniffed_mime.startswith("text/") or sniffed_mime in (
                    "application/json",
                    "application/xml",
                )
                is_binary_doc = sniffed_group in {"pdf", "office-docx", "office-pptx"} or any(
                    tag in sniffed_mime
                    for tag in (
                        "pdf",
                        "wordprocessingml",
                        "presentationml",
                        "spreadsheetml",
                    )
                )

                if is_text_format and raw is not None:
                    # Text-like format: decode for the format-aware parsers below.
                    text = raw.decode("utf-8", errors="replace")
                    # Update mime so the downstream "html" / "json" branches
                    # see the sniffed type, not the lying manifest.
                    mime = sniffed_mime
                elif is_binary_doc and raw is not None:
                    # Binary format (signal from bytes regardless of manifest):
                    # eager-parse through kaos-pdf / kaos-office. Skips the
                    # headline-only gate because the manifest may have already
                    # mislead the SPA upload pipeline into NOT pre-extracting.
                    binary_doc = self._parse_binary_bytes_to_content_document(
                        filename=filename,
                        mime=sniffed_mime,
                        body=raw,
                    )
                elif self._looks_like_headline_only(item.content) and raw is not None:
                    # Declared binary mime (sniffer didn't recognize as
                    # PDF / office / text) + no pre-extracted body. Try
                    # the parser anyway based on declared mime — covers
                    # XLSX (sniffer reports the generic OPC group) and
                    # any other format the sniffer doesn't classify but
                    # the parser may still handle.
                    binary_doc = self._parse_binary_bytes_to_content_document(
                        filename=filename,
                        mime=mime,
                        body=raw,
                    )

            # Headline / label so the synthesis output can refer to
            # which document a finding came from.
            blocks.append(Paragraph(children=(Text(value=f"=== {filename} ==="),)))

            if binary_doc is not None:
                # Parsed binary → extend blocks with the real body.
                blocks.extend(binary_doc.body)
                continue

            if text is not None:
                # Format-aware parse: HTML → ContentDocument, JSON → one
                # line per key/value pair, plain text → kaos_content
                # plain-text parser.
                if "html" in mime:
                    try:
                        parsed_doc = parse_html(text)
                        blocks.extend(parsed_doc.body)
                        continue
                    except Exception:
                        logger.debug("base_agent: parse_html failed for %s", filename)
                if "json" in mime:
                    try:
                        import json as _json

                        obj = _json.loads(text)
                        pretty = _json.dumps(obj, indent=2)
                    except Exception:
                        pretty = text
                    for line in pretty.splitlines():
                        if line.strip():
                            blocks.append(Paragraph(children=(Text(value=line),)))
                    continue
                try:
                    parsed_doc = parse_plain_text(text)
                    blocks.extend(parsed_doc.body)
                    continue
                except Exception:
                    blocks.append(Paragraph(children=(Text(value=text),)))
                    continue

            # Fallback: use the already-extracted item.content as-is.
            body_text = item.content or ""
            if body_text:
                # Split on lines so the segmenter has paragraph boundaries.
                for line in body_text.splitlines():
                    if line.strip():
                        blocks.append(Paragraph(children=(Text(value=line),)))

        if not blocks:
            return None
        document = ContentDocument(body=tuple(blocks))
        segmenter = get_default_punkt_tokenizer()
        view = DocumentView(document, sentence_segmenter=segmenter)
        return view, document, segmenter

    async def _handle_plan(
        self,
        message: str,
        memory: SessionMemory,
        context_items: dict[MemoryType, list[Any]],
    ) -> tuple[str, list[ToolExecution], InvocationUsage]:
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

        instructions = self.instructions or _DEFAULT_RESPOND_INSTRUCTION
        if extra_instruction:
            instructions = f"{instructions} {extra_instruction}"

        # Use the default JSONCodec (native structured output via the
        # provider's JSON-schema / function-calling path). The earlier
        # override to ChatCodec — meant to dodge a 30K → 3K truncation
        # observed on Sonnet 4.6 with JSON-wrapped long outputs —
        # leaked `[response]` openers and hallucinated `[/response]`
        # closers from instruction-tuned models (Haiku 4.5 reliably,
        # Sonnet/Opus occasionally) into the UI. Modern Claude 4.x /
        # GPT-5.x / Gemini 2.5 have first-class structured-output
        # paths that don't truncate this way; if a JSON-side truncation
        # regresses, file it as a JSONCodec bug in kaos-llm-core rather
        # than working around it with text scaffolding here.
        from kaos_agents._examples import load_examples

        call = Call(
            RespondSignature,
            model=self._model_for_role("respond"),
            instructions=instructions,
            examples=load_examples("respond"),
        )
        # ``.invoke()`` returns the full Invocation so we can read
        # ``invocation.usage`` — the bare ``await call(...)`` path is
        # slightly cheaper but throws the usage record on the floor.
        invocation = await call.invoke(message=message, conversation_history=history)
        text = str(getattr(invocation.output, "response", "")) if invocation.output else ""
        # Defense-in-depth: strip any scratchpad-tag closers that a
        # non-JSON codec (or a model that hallucinated them inside a
        # JSON string field) left in the response body. ChatCodec
        # historically anchored only openers; this guards against the
        # symptom even if the codec gets reverted.
        text = _STRIP_SCRATCHPAD_RE.sub("", text).strip()
        return text, InvocationUsage.from_invocation(invocation)


# ---------------------------------------------------------------------------
# Event-to-response conversion (used by turn() backward compat)
# ---------------------------------------------------------------------------


# Event-stream → AgentResponse conversion lives in
# :mod:`kaos_agents.runtime.events_to_response` so :class:`KaosAgent`'s
# default :meth:`KaosAgent.turn` can use it without depending on
# this module. The previous private ``_events_to_response`` helper
# was inlined here pre-refactor; it's now the canonical
# :func:`kaos_agents.runtime.events_to_response.events_to_response`.
