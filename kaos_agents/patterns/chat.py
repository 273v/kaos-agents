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
    BudgetExceeded,
    EventEmitter,
    GroundingRefusalTriggered,
    KaosEvent,
    Span,
    SpanPhase,
    SpanSubject,
    TextDelta,
    UsageObserved,
    emit_usage_observed,
)
from kaos_agents.grounding.no_evidence_gate import (
    ToolObservationSummary,
    evaluate_no_evidence_gate,
    render_refusal_text,
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


def _collect_attached_document_names(memory: SessionMemory) -> tuple[str, ...]:
    """Names of documents attached to the session for the no-evidence gate.

    Pulls from :attr:`MemoryType.DOCUMENTS` items' metadata: each
    item carries ``manifest`` info (name, body_uri, mime_type) injected
    by ``runtime.artifact_hydration``. We extract any string-typed
    ``name`` / ``filename`` field — the gate uses these to decide
    whether the user's question was specifically about files (and
    therefore a "zero evidence" outcome should refuse with a clear
    error rather than fabricate an answer).

    Returns the empty tuple when DOCUMENTS section is absent, empty,
    or carries no name-bearing metadata.
    """
    if not memory.has_section(MemoryType.DOCUMENTS):
        return ()
    names: list[str] = []
    seen: set[str] = set()
    for item in memory.get(MemoryType.DOCUMENTS):
        meta = item.metadata or {}
        for key in ("name", "filename", "title"):
            value = meta.get(key)
            if isinstance(value, str) and value and value.lower() not in seen:
                seen.add(value.lower())
                names.append(value)
    return tuple(names)


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

    def _derive_capability_kinds_for_tools(
        self,
        tools: list[Any],
    ) -> dict[str, str]:
        """Best-effort map of ``tool_name -> CapabilityKind.value``.

        Looks up each bridged Tool's original ``KaosTool`` on
        ``self._runtime.tools`` and derives a ``Capability`` from its
        metadata via the auto-classifier. Returns an empty mapping when
        the runtime is unavailable or no metadata can be resolved —
        callers must tolerate that.

        This is the Phase 1.4 hook into the Capability registry. When
        the registry holds an explicit Capability for a tool (e.g., the
        future unified ``retrieve`` capability that aggregates several
        tools), the registry lookup wins over the auto-derivation.
        """
        result: dict[str, str] = {}
        runtime = self._runtime
        if runtime is None:
            return result
        try:
            from kaos_agents.registry import default_capability_registry
            from kaos_agents.registry.capability_classifier import derive_capability
        except ImportError:
            return result
        for t in tools:
            name = getattr(t, "name", None)
            if not name:
                continue
            name_str = str(name)
            # Prefer explicit Capability registrations over auto-derivation.
            cap_names = default_capability_registry.capabilities_for_tool(name_str)
            if cap_names:
                cap = default_capability_registry.get(cap_names[0])
                if cap is not None:
                    result[name_str] = str(cap.kind)
                    continue
            # Fall back to runtime tool metadata + classifier.
            kaos_tool = None
            try:
                kaos_tool = runtime.tools.get_tool(name_str)
            except Exception:
                kaos_tool = None
            if kaos_tool is None or not hasattr(kaos_tool, "metadata"):
                continue
            try:
                cap = derive_capability(kaos_tool.metadata)
            except ValueError:
                continue
            result[name_str] = str(cap.kind)
        return result

    async def _maybe_narrow_tools_via_fitness_ranker(
        self,
        *,
        tools: list[Any],
        query: str,
        memory: SessionMemory | None = None,
    ) -> list[Any]:
        """M1 — narrow the bridged ReAct catalog via the fitness ranker.

        Returns the (possibly-narrowed) tool list. Always preserves
        ``self._extra_llm_tools`` (delegation / handoff) even when the
        ranker doesn't pick them — those are framework-mandatory.

        Bypass conditions (all return ``tools`` unchanged):
        * ``settings.tool_fitness_enabled`` is False
        * ``len(tools) <= settings.tool_fitness_bypass_threshold``
        * ``query`` is empty / whitespace
        * the ranker raises or emits an unusable result
        """
        if not getattr(self._settings, "tool_fitness_enabled", True):
            return tools
        bypass_threshold = int(getattr(self._settings, "tool_fitness_bypass_threshold", 10))
        if len(tools) <= bypass_threshold:
            return tools
        clean_query = (query or "").strip()
        if not clean_query:
            return tools

        # Build a (name, description) catalog from the bridged Tool list.
        # Tool objects from kaos-llm-core expose .name and .description
        # as properties; falling back to the raw attribute keeps the
        # narrowing agnostic to future Tool refactors.
        #
        # Phase 1.4 of the Capability-layer refactor: when the runtime
        # is available we enrich each description with the tool's
        # derived ``CapabilityKind`` (e.g., ``[SEARCH]``, ``[READ]``)
        # so the planner sees the higher-level abstraction without
        # the ToolFitnessSignature contract changing. See
        # ``kaos-modules/docs/plans/2026-05-19-lateral-redesign-
        # capability-layer.md`` §16 and the progress doc.
        kind_by_tool = self._derive_capability_kinds_for_tools(tools)
        catalog: list[tuple[str, str]] = []
        for t in tools:
            name = getattr(t, "name", None) or ""
            description = getattr(t, "description", None) or ""
            if not name:
                continue
            kind = kind_by_tool.get(str(name))
            if kind is not None and description:
                description = f"[{kind.upper()}] {description}"
            elif kind is not None:
                description = f"[{kind.upper()}]"
            catalog.append((str(name), str(description)))
        if not catalog:
            return tools

        extra_names = {getattr(et, "name", None) for et in self._extra_llm_tools}
        extra_names.discard(None)

        # P0.2 (#549.F) — corpus scale signal for the ranker. When
        # ``memory`` is provided and the DOCUMENTS section is populated,
        # pass the document count to ToolFitnessSignature so Rule 10
        # can prefer corpus-aggregating tools (findings / corpus-filter
        # / search-document) over per-doc parsers at scale. The signal
        # is typed (an int the LLM reasons about), not a regex bias —
        # the docstring rule is load-bearing.
        corpus_size = 0
        if memory is not None and memory.has_section(MemoryType.DOCUMENTS):
            corpus_size = memory.section_item_count(MemoryType.DOCUMENTS)

        try:
            from kaos_agents.planning.tool_fitness import rank_tools_for_query

            top_k = int(getattr(self._settings, "tool_fitness_top_k", 8))
            ranker_model = self._model_for_role("classify")
            result = await rank_tools_for_query(
                query=clean_query,
                catalog=catalog,
                model=ranker_model,
                top_k=top_k,
                corpus_size=corpus_size,
            )
        except Exception as exc:
            logger.warning(
                "tool_fitness_ranker: invocation failed; falling back to "
                "unfiltered catalog. err=%s",
                exc,
            )
            return tools

        if result.fell_back or not result.valid_picks:
            logger.debug(
                "tool_fitness_ranker: no usable picks (fell_back=%s, "
                "picks=%r, valid=%r) — falling back to unfiltered "
                "catalog of %d tools",
                result.fell_back,
                result.picks,
                result.valid_picks,
                len(tools),
            )
            return tools

        # Preserve mandatory extras + tools the ranker selected. Keep
        # the original tool-list order within each group so observers /
        # tests see a stable arrangement.
        keep_names = set(result.valid_picks) | extra_names
        narrowed = [t for t in tools if getattr(t, "name", None) in keep_names]
        if not narrowed:
            # No overlap — pathological. Keep the full catalog so the
            # worker has *something* to work with.
            logger.warning(
                "tool_fitness_ranker: picks %r matched zero bridged "
                "tools — keeping full catalog of %d tools",
                result.valid_picks,
                len(tools),
            )
            return tools

        logger.info(
            "tool_fitness_ranker: narrowed catalog %d → %d for query (model=%s rationale=%r)",
            len(tools),
            len(narrowed),
            self._model_for_role("classify"),
            result.rationale[:80],
        )
        return narrowed

    # ─── Classifier promotion for the attached-corpus case ─────────────
    #
    # When the SPA uploads files, they land in ``SessionMemory.DOCUMENTS``
    # AND the per-turn assembled context. The base classifier reads that
    # text and — when the redacted / summarized preview happens to look
    # like a direct answer — picks RESPOND with high confidence. The
    # SPA's "what is X in the attached file?" question then gets a
    # hallucinated answer from the preview rather than a grounded answer
    # from the bytes. (See CS-B2 / CS-B3 corpus-stress regressions.)
    #
    # The promotion: when the LLM picked RESPOND or TOOL_USE while
    # DOCUMENTS are attached AND the message contains fact-lookup
    # keywords, override to RESEARCH so the FindingsAgent default
    # (kaos_agents.runtime.agent.BaseAgent._handle_research_streaming)
    # grounds the answer on the raw bytes via VFS.
    #
    # The keyword set is intentionally conservative — "what", "value of",
    # "find", "json", "file" etc. — to avoid hijacking greetings or
    # follow-up chitchat. Tune via subclassing if a specific domain has
    # different signals.

    _CORPUS_FACT_LOOKUP_TOKENS: tuple[str, ...] = (
        "file",
        "files",
        "attached",
        "attachment",
        "attachments",
        "document",
        "documents",
        "uploaded",
        "config",
        "json",
        "value of",
        "value for",
        "what is",
        "what's",
        "what does",
        "find",
        "look up",
        "lookup",
        "quote",
        "verbatim",
        "exact",
    )

    @staticmethod
    def _message_looks_like_corpus_lookup(
        message: str,
        tokens: tuple[str, ...],
    ) -> bool:
        """Cheap keyword screen — does the message look like a doc-grounded ask?"""
        lowered = message.lower()
        return any(tok in lowered for tok in tokens)

    async def _classify(
        self,
        message: str,
        memory: SessionMemory,
        context_items: dict[MemoryType, list[Any]] | None = None,
    ) -> IntentResult:
        """Promote RESPOND/TOOL_USE → RESEARCH when DOCUMENTS attached
        AND the prompt asks for a file-grounded fact.

        See module-level commentary for rationale (CS-B2 / CS-B3).
        Leaves RESEARCH / PLAN / CLARIFY verdicts alone — only
        promotes the two intents that don't grant the FindingsAgent
        path a chance to run.

        Subclasses can broaden / narrow the fact-lookup vocabulary by
        overriding ``_CORPUS_FACT_LOOKUP_TOKENS``.
        """
        result = await super()._classify(message, memory, context_items)

        if result.intent not in (IntentType.RESPOND, IntentType.TOOL_USE):
            return result

        # Count attached DOCUMENTS in this turn's context, falling back
        # to the section count when the per-turn assembler trimmed it.
        docs_in_context = 0
        if context_items is not None:
            docs_in_context = len(context_items.get(MemoryType.DOCUMENTS, []) or [])
        if docs_in_context == 0 and memory.has_section(MemoryType.DOCUMENTS):
            docs_in_context = memory.section_item_count(MemoryType.DOCUMENTS)

        if docs_in_context == 0:
            return result
        if not self._message_looks_like_corpus_lookup(message, self._CORPUS_FACT_LOOKUP_TOKENS):
            return result

        promoted_reason = (
            f"corpus_attached_promotion: {docs_in_context} DOCUMENTS attached "
            f"+ fact-lookup keywords; classifier said "
            f"{result.intent.value} ({result.confidence:.2f}); promoted to "
            "RESEARCH so FindingsAgent grounds the answer on bytes "
            "rather than redacted context."
        )
        logger.debug("chat_agent._classify: %s", promoted_reason)
        return IntentResult(
            intent=IntentType.RESEARCH,
            confidence=1.0,
            reasoning=promoted_reason,
            usage=result.usage,
        )

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

        # M1 — pre-ReAct tool-catalog narrowing (KFM-B05 mitigation).
        # When the registered catalog is large, weaker reasoners
        # (e.g. gpt-5.4-mini) short-circuit ReAct into a memory-only
        # text response. The fitness ranker scores tool fit against
        # the user's message and narrows the bridged catalog before
        # ReAct sees it. Bypass-if-small keeps tiny catalogs free of
        # ranker overhead; fall-back-on-error keeps the loop running
        # if the ranker provider is down or the model emits garbage.
        # Mandatory tools (delegation / handoff via ``_extra_llm_tools``)
        # are always preserved.
        tools = await self._maybe_narrow_tools_via_fitness_ranker(
            tools=tools,
            query=message,
            memory=memory,
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
            f"{self.instructions}\n\n{_REACT_INSTRUCTION}"
            if self.instructions
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

            # P0.3 (#549.G) — chat-level cost cap. After ReAct returns
            # we know the full cumulative cost of this dispatch (LLM
            # invocations + tool calls). When the operator has set
            # ``chat_max_cost_usd``, an exceeding spend short-circuits
            # the turn with a typed BudgetExceeded event + an honest
            # refusal that names the cap, the observed spend, and the
            # number of tool calls already burned. Mirrors the
            # AgenticLoop's cost-exceeded path (events/budget.py).
            # Disabled by default (``None``) for backward compatibility.
            chat_cost_cap = getattr(self._settings, "chat_max_cost_usd", None)
            if chat_cost_cap is not None:
                react_usage = InvocationUsage.from_invocation(invocation)
                observed_cost = float(react_usage.cost_usd)
                if observed_cost > float(chat_cost_cap):
                    # Count tool calls already burned to surface in the
                    # refusal — the user deserves to know how much work
                    # the agent attempted before stopping.
                    n_tool_calls = sum(len(it.tool_results) for it in result.trajectory)
                    logger.warning(
                        "chat_agent.chat_max_cost_usd: exceeded cap=$%.4f "
                        "observed=$%.4f tool_calls=%d model=%s",
                        chat_cost_cap,
                        observed_cost,
                        n_tool_calls,
                        self._model_for_role("respond"),
                    )
                    yield emitter.emit(
                        BudgetExceeded,
                        kind="chat_cost",
                        limit=float(chat_cost_cap),
                        actual=observed_cost,
                        reason=(
                            f"chat_max_cost_usd cap of ${chat_cost_cap:.4f} "
                            f"exceeded after {n_tool_calls} tool call(s); "
                            f"observed spend ${observed_cost:.4f}"
                        ),
                    )
                    # Emit the rolled-up usage so transparency surfaces
                    # (TurnSummary cost_usd, OTel, CostTrackingHook) see
                    # the spend that triggered the cap — not zero.
                    yield emit_usage_observed(emitter, react_usage, source="react-cost-cap")
                    # Honest refusal — names the cap, the spend, and what
                    # we managed to do. NOT silent termination.
                    refusal_text = (
                        f"I stopped after {n_tool_calls} tool call(s) "
                        f"because the configured cost cap of "
                        f"${chat_cost_cap:.4f} was exceeded "
                        f"(observed spend: ${observed_cost:.4f}). "
                        "Partial results from the work I completed are "
                        "available above, but no synthesis was performed. "
                        "Re-run with a higher chat_max_cost_usd if you "
                        "need the full answer."
                    )
                    yield emitter.emit(TextDelta, content=refusal_text)
                    return

            # Surface model reasoning blocks (Anthropic extended thinking,
            # OpenAI o1 reasoning, Google thinkingConfig) when the
            # underlying provider exposed them. Helper is a no-op when
            # `invocation.extras["native_thinking"]` is empty.
            from kaos_agents.events import emit_thinking_from_invocation

            thinking_event = emit_thinking_from_invocation(emitter, invocation)
            if thinking_event is not None:
                yield thinking_event

            # Emit tool call events from the trajectory.
            # No default-on truncation: the span's ``result_summary``
            # carries the FULL tool result text. Wire consumers (SPA
            # tool-chip, OTel, MCP host UIs) get the full value and
            # can choose their own display cap. Previously chopped to
            # 200 chars, which (a) hid late error text from the
            # no-evidence gate and (b) lied to the operator about what
            # the tool actually returned. See task #438.
            total_tool_calls = 0
            for iteration in result.trajectory:
                for obs in iteration.tool_results:
                    total_tool_calls += 1
                    result_full = str(obs.result) if obs.result else ""
                    call_id = obs.tool_call_id or obs.tool_name
                    args_tuple = tuple(sorted(obs.arguments.items())) if obs.arguments else ()
                    structured_content = _extract_structured_content(obs.result)
                    # Log line gets a short head for readability — a
                    # log line is not a load-bearing surface; the full
                    # text is on the span attribute.
                    log_preview = (
                        result_full if len(result_full) <= 160 else result_full[:160] + "..."
                    )
                    logger.debug(
                        "chat_agent.tool_call: tool=%s, is_error=%s, result_preview=%r",
                        obs.tool_name,
                        obs.is_error,
                        log_preview,
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
                        "result_summary": result_full,
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

            # Refuse to ship a fabricated final answer when every
            # tool call this turn returned an error AND the user
            # referenced files. See no_evidence_gate for the
            # production trigger (NDA hallucination incident,
            # session 01KRVYAEA3B1HG95DBAG6H0DJ3). Critical for
            # kaos-* legal-research bar — "confidently wrong" is the
            # worst class of failure; we'd rather refuse loudly than
            # synthesize a plausible-sounding answer the LLM has zero
            # evidence for. Pass FULL text to the gate so a late
            # error message (e.g. char 301+) can't slip past
            # truncation. See task #438 + the truncation audit's
            # P1 #10.
            obs_summaries = [
                ToolObservationSummary(
                    tool_name=str(getattr(obs, "tool_name", "") or ""),
                    is_error=bool(getattr(obs, "is_error", False)),
                    result_preview=(str(obs.result) if obs.result else ""),
                    arguments_preview=(str(getattr(obs, "arguments", "") or "")),
                )
                for iteration in result.trajectory
                for obs in iteration.tool_results
            ]
            attached_docs = _collect_attached_document_names(memory)
            no_evidence = evaluate_no_evidence_gate(
                observations=obs_summaries,
                user_message=message,
                attached_documents=attached_docs,
            )

            # Emit the final response
            response_text = str(result.answer) if result.outputs and result.answer else ""
            if not response_text and result.trajectory:
                response_text = result.trajectory[-1].text

            if no_evidence.refuse:
                logger.warning(
                    "chat_agent.no_evidence_refusal: tools=%d files=%d",
                    no_evidence.failed_tool_count,
                    len(no_evidence.referenced_files),
                )
                yield emitter.emit(
                    GroundingRefusalTriggered,
                    original_confidence=0.0,
                    min_confidence=0.0,
                    reason=no_evidence.reason,
                )
                response_text = render_refusal_text(no_evidence)

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
