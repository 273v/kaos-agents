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
        with collect_events():
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
        elif intent.intent == IntentType.RESEARCH and self._handler_is_default("_handle_research"):
            recommended = "research"
            classified = "research"
        else:
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
        """Handle research/document Q&A. Override to use RAG."""
        return await self._handle_respond(message, memory, context_items)

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
