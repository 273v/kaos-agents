"""ResearchAgent — document Q&A pattern via RAG.

Extends ChatAgent with RAG-backed research for the RESEARCH intent.
When documents are loaded in memory and the user asks a question,
dispatches to kaos-llm-core's RAG program for retrieval-grounded
generation with citation verification.

The pipeline:
1. Collect documents from the DOCUMENTS memory section
2. Build a corpus dict (URI → text)
3. Call RAG.query(question, documents) for grounded answer
4. Store verified claims in FINDINGS section with provenance
5. Return the answer (or "insufficient evidence" refusal)

Streaming: yields CitationFound or EvidenceInsufficient events.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any

from kaos_core.logging import get_logger

from kaos_agents.events import (
    AgentEvent,
    CitationFound,
    EventEmitter,
    EvidenceInsufficient,
    TextDelta,
    ToolCallResult,
    ToolCallStart,
    UsageObserved,
)
from kaos_agents.memory.types import MemoryType
from kaos_agents.models import IntentResult, IntentType, ToolCallRecord
from kaos_agents.patterns.chat import ChatAgent
from kaos_agents.usage import ZERO_USAGE, InvocationUsage, emit_usage_observed

if TYPE_CHECKING:
    from kaos_core.base.context import KaosContext
    from kaos_core.registry.container import KaosRuntime
    from kaos_core.vfs.core import VirtualFileSystem

    from kaos_agents.memory.session import SessionMemory
    from kaos_agents.memory.types import MemoryItem
    from kaos_agents.providers import ProviderConfig
    from kaos_agents.settings import KaosAgentSettings

logger = get_logger(__name__)

_RESEARCH_REACT_INSTRUCTION = """\
You are answering a question about a document corpus using retrieval tools.

STRATEGY:
1. Search with kaos-retrieval-bm25 using key terms from the question.
2. Look at the results. Check the expansion_assessment signal.
3. If results look relevant and cover the question:
   → Call kaos-retrieval-answer with the question and the relevant passage texts.
4. If results are noisy or miss the topic:
   → Try more specific terms, or use kaos-retrieval-hyde for vocabulary bridging.
5. If kaos-retrieval-answer says "insufficient evidence":
   → Use the what_would_resolve hint to search again with different terms.
6. After 2-3 search attempts, if you still can't find evidence:
   → State clearly that the corpus doesn't contain the answer.

IMPORTANT:
- Always search BEFORE answering. Never answer from your training data.
- Include passage previews when calling kaos-retrieval-answer.
- If the question references a specific document (e.g., "RFC 2119"), search for that name.
- Do NOT hallucinate. If the passages don't support the answer, refuse.
"""


class ResearchAgent(ChatAgent):
    """Agent with RAG-backed document Q&A for RESEARCH intent.

    Extends ChatAgent (which handles RESPOND, CLARIFY, TOOL_USE) with
    retrieval-augmented generation over documents loaded in memory.

    Documents are loaded into the DOCUMENTS section via ``load_document()``.
    When the intent classifier detects RESEARCH (question + documents present),
    the agent dispatches to kaos-llm-core's RAG program.

    Usage:
        agent = ResearchAgent(vfs=vfs, model="anthropic:claude-sonnet-4-6")
        agent.load_document(memory, "doc:report-1", document_text)
        response = await agent.turn("What is the effective date?", session_id="abc")
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
        rag_top_k: int = 10,
        rag_max_retries: int = 2,
        settings: KaosAgentSettings | None = None,
        provider: ProviderConfig | None = None,
        extra_llm_tools: tuple[Any, ...] = (),
        corpus: Any = None,
        permission_policy: Any = None,
        instructions: str | None = None,
    ) -> None:
        """Construct a ResearchAgent.

        Args:
            corpus: Optional pre-built corpus. Accepts anything satisfying
                the ``kaos_content.corpus.Corpus`` Protocol (``kaos_ml_core.Corpus``,
                ``ContentDocumentCorpus``) or a ``kaos_ml_core.CorpusIndex``
                (whose ``.corpus`` is unwrapped). When set, RAG queries use
                this corpus directly instead of rebuilding from
                ``MemoryType.DOCUMENTS`` on every turn — lets agents reuse
                a persistent, pre-indexed corpus across sessions.
                The memory-based flow is still available when ``corpus=None``.
        """
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
        self._rag_top_k = rag_top_k
        self._rag_max_retries = rag_max_retries
        self._corpus = _unwrap_corpus_arg(corpus)

    @property
    def corpus(self) -> Any:
        """The pre-built corpus passed to ``__init__``, or ``None``.

        When set, ``_handle_research_streaming`` hands this to
        ``RAG.query(documents=...)`` directly and bypasses
        ``_build_corpus(memory)``.
        """
        return self._corpus

    # Question words used by _classify to detect document-oriented queries
    # when a pre-bound corpus is present.
    _QUESTION_PREFIXES: tuple[str, ...] = (
        "what",
        "who",
        "where",
        "when",
        "why",
        "how",
        "which",
        "does",
        "is",
        "are",
        "can",
        "do",
        "tell",
        "explain",
        "describe",
        "summarize",
        "find",
        "list",
        "compare",
    )

    async def _classify(
        self,
        message: str,
        memory: SessionMemory,
        context_items: dict[MemoryType, list[Any]] | None = None,
    ) -> IntentResult:
        """Force RESEARCH intent when an explicit corpus is bound AND the
        message looks like a document question.

        Without this override, the base classifier reads ``memory.DOCUMENTS``
        and reports TOOL_USE / RESPOND when the section is empty — even
        though the agent has a fully-populated external corpus. Routing
        would then skip ``_handle_research_streaming`` and hand an
        answerable question to the hallucination-prone simple-respond path
        (observed in an early WS-3.6 live run: "$25/$50/$100" fabricated
        Delaware fees when the corpus said "$89").

        The heuristic: force RESEARCH only when the message contains a ``?``
        or starts with a common question/command word (what, who, summarize,
        find, etc.). Greetings, off-topic chat, and non-document commands
        fall through to the base classifier so they are handled normally.

        When ``self._corpus`` is ``None`` the base behavior applies — the
        memory-based routing must keep working for legacy callers.
        """
        if self._corpus is not None and _looks_like_question(message, self._QUESTION_PREFIXES):
            has_question_mark = "?" in message
            first_word = (
                message.lstrip().split(maxsplit=1)[0].lower().rstrip(".,!?:;")
                if message.strip()
                else ""
            )
            logger.debug(
                "research_agent._classify: forcing RESEARCH — corpus bound, "
                "question_mark=%s, first_word=%r",
                has_question_mark,
                first_word,
            )
            return IntentResult(
                intent=IntentType.RESEARCH,
                confidence=1.0,
                reasoning="ResearchAgent has a pre-bound corpus; routing directly to RAG.",
            )
        result = await super()._classify(message, memory, context_items)
        logger.debug(
            "research_agent._classify: base classifier returned intent=%s confidence=%.2f",
            result.intent.value,
            result.confidence,
        )
        return result

    def load_document(
        self,
        memory: SessionMemory,
        uri: str,
        text: str,
    ) -> None:
        """Load a document into the DOCUMENTS section for RAG retrieval.

        Documents are stored as "URI: <uri>\\n<text>" items. The URI is used
        as the corpus key for citation verification.

        Args:
            memory: The session memory to load into.
            uri: Document identifier (e.g., "doc:report-1", "ecfr:title26").
            text: Full document text.
        """
        memory.add(
            MemoryType.DOCUMENTS,
            f"URI: {uri}\n{text}",
            metadata={"uri": uri, "type": "document"},
        )

    # Threshold for using ReAct (agent-driven) vs one-shot RAG
    _REACT_CORPUS_THRESHOLD = 20

    async def _dispatch_streaming(
        self,
        intent: IntentResult,
        message: str,
        memory: SessionMemory,
        context_items: dict[MemoryType, list[Any]],
        emitter: EventEmitter,
    ) -> AsyncIterator[AgentEvent]:
        """Override dispatch: adaptive one-shot → ReAct escalation.

        Strategy:
        1. Try one-shot RAG first (fast, good refusals, cheap).
        2. If one-shot returns InsufficientEvidence AND we have a large corpus,
           escalate to ReAct with retrieval tools (the agent iterates).
        3. Small corpus (<20 passages) always uses one-shot (no escalation needed).

        This combines the strengths of both paths:
        - One-shot: fast, reliable refusals, low cost
        - ReAct: can iterate on hard queries, try HyDE, refine search
        """
        if intent.intent != IntentType.RESEARCH:
            logger.debug(
                "research_agent._dispatch_streaming: non-RESEARCH intent=%s, delegating to super",
                intent.intent.value,
            )
            async for event in super()._dispatch_streaming(
                intent, message, memory, context_items, emitter
            ):
                yield event
            return

        # Determine if we have a large corpus that could benefit from ReAct escalation
        can_escalate = False
        corpus_size = 0
        if self._corpus is not None:
            corpus_size = getattr(self._corpus, "size", 0)
            # Always thread corpus into context so retrieval tools can access it
            if hasattr(self, "_context") and self._context is not None:
                self._context._config["_corpus"] = self._corpus
            if corpus_size >= self._REACT_CORPUS_THRESHOLD:
                can_escalate = True

        logger.debug(
            "research_agent._dispatch_streaming: path=%s, corpus_size=%d, "
            "can_escalate=%s, threshold=%d",
            "one-shot-first" if can_escalate else "one-shot-only",
            corpus_size,
            can_escalate,
            self._REACT_CORPUS_THRESHOLD,
        )

        # Step 1: Try one-shot RAG first (fast path)
        got_insufficient = False
        async for event in self._handle_research_streaming(message, memory, context_items, emitter):
            if isinstance(event, EvidenceInsufficient):
                got_insufficient = True
                # Don't yield the InsufficientEvidence yet — we might escalate
                if can_escalate:
                    logger.debug(
                        "research_agent: one-shot insufficient, escalating to ReAct "
                        "(corpus=%d passages)",
                        getattr(self._corpus, "size", 0),
                    )
                    continue
            yield event

        if not got_insufficient or not can_escalate:
            if not got_insufficient:
                logger.debug(
                    "research_agent._dispatch_streaming: one-shot succeeded, no escalation needed"
                )
            return

        # Step 2: Escalate to ReAct with retrieval tools
        logger.info(
            "research_agent._dispatch_streaming: escalating to ReAct — "
            "one-shot returned insufficient evidence, corpus_size=%d",
            corpus_size,
        )
        yield emitter.emit(
            ToolCallStart,
            call_id="react-escalation",
            tool_name="react-escalation",
            arguments=(("reason", "one-shot RAG returned insufficient evidence"),),
        )

        saved_instructions = self._instructions
        self._instructions = (
            saved_instructions + "\n\n" if saved_instructions else ""
        ) + _RESEARCH_REACT_INSTRUCTION

        react_intent = IntentResult(
            intent=IntentType.TOOL_USE,
            confidence=intent.confidence,
            reasoning="Escalating to ReAct after one-shot RAG returned insufficient evidence.",
        )
        try:
            async for event in super()._dispatch_streaming(
                react_intent, message, memory, context_items, emitter
            ):
                yield event
        finally:
            self._instructions = saved_instructions

    async def _handle_research_streaming(
        self,
        message: str,
        memory: SessionMemory,
        context_items: dict[MemoryType, list[MemoryItem]],
        emitter: EventEmitter,
    ) -> AsyncIterator[AgentEvent]:
        """Handle document Q&A via RAG pipeline, yielding citation events.

        Corpus source of truth (checked in order):
        1. ``self._corpus`` — an explicit Corpus Protocol instance passed
           to ``__init__``. Preferred for persistent / pre-indexed corpora.
        2. ``MemoryType.DOCUMENTS`` — the legacy per-turn corpus rebuilt
           from documents loaded via :meth:`load_document`.
        """
        corpus: Any
        if self._corpus is not None:
            corpus = self._corpus
            n_docs_label = getattr(self._corpus, "size", None)
            if n_docs_label is None:
                n_docs_label = len(getattr(self._corpus, "_passages", ()) or [])
            logger.debug(
                "research_agent._handle_research: using pre-bound corpus, n_docs=%s",
                n_docs_label,
            )
        else:
            corpus = _build_corpus_bm25(memory, message, self._settings)
            n_docs_label = len(corpus) if isinstance(corpus, dict) else 0
            logger.debug(
                "research_agent._handle_research: built corpus from memory, n_docs=%d",
                n_docs_label,
            )

        if not corpus:
            logger.warning("research_agent: no documents loaded — falling back to simple response")
            response, usage = await self._simple_respond(
                message,
                memory,
                extra_instruction=(
                    "The user asked a research question but no documents are loaded. "
                    "Explain that documents need to be loaded first, and suggest how."
                ),
            )
            if response:
                yield emitter.emit(TextDelta, content=response)
            yield emit_usage_observed(emitter, usage, source="research-no-corpus")
            return

        try:
            from kaos_agents._llm_imports import require_llm_core

            require_llm_core()
            from kaos_llm_core.programs.rag import RAG
            from kaos_llm_core.signatures.grounding import (
                Answer,
                InsufficientEvidence,
                MatchStrategy,
            )

            rag = RAG(
                model=self._model_for_role("research"),
                top_k=self._rag_top_k,
                max_retries=self._rag_max_retries,
                match_strategies=(
                    MatchStrategy.STRICT,
                    MatchStrategy.SUBSTRING,
                    MatchStrategy.CASE_INSENSITIVE,
                    MatchStrategy.NORMALIZED_TOKEN,
                ),
            )

            # Emit a tool call event for the RAG query
            yield emitter.emit(
                ToolCallStart,
                call_id="rag-query",
                tool_name="rag-query",
                arguments=(("question", message), ("n_documents", str(n_docs_label))),
            )

            # ``.invoke()`` returns the full Invocation so we can emit
            # real usage. ``rag.query()``/``rag(...)`` are thin unwrappers
            # around the same pipeline that throw usage on the floor.
            rag_invocation = await rag.invoke(question=message, documents=corpus)
            result = rag_invocation.output
            yield emit_usage_observed(
                emitter,
                InvocationUsage.from_invocation(rag_invocation),
                source="rag",
            )

            if isinstance(result.grounded_answer, Answer):
                answer = result.grounded_answer
                response_text = str(answer.value)
                logger.debug(
                    "research_agent._handle_research: one-shot succeeded — "
                    "claims=%d, spans=%d, verified=%s",
                    len(answer.claims),
                    len(answer.spans),
                    result.is_verified,
                )

                # Emit citation events for each verified claim
                for claim in answer.claims:
                    for span in claim.supporting_spans:
                        yield emitter.emit(
                            CitationFound,
                            claim=claim.statement,
                            source_uri=span.source_uri,
                            confidence=claim.confidence,
                            verified=result.is_verified,
                        )

                    # Also store in memory (side effect)
                    sources = ", ".join(span.source_uri for span in claim.supporting_spans)
                    finding = (
                        f"[{claim.claim_type}] {claim.statement} "
                        f"(confidence={claim.confidence:.2f}, sources={sources})"
                    )
                    memory.add(
                        MemoryType.FINDINGS,
                        finding,
                        metadata={
                            "claim_type": str(claim.claim_type),
                            "statement": claim.statement,
                            "confidence": claim.confidence,
                            "verified": result.is_verified,
                            "sources": [s.source_uri for s in claim.supporting_spans],
                            "spans": [
                                {
                                    "source_uri": s.source_uri,
                                    "quote": s.quote,
                                    "char_span": list(s.char_span),
                                    "page": s.page,
                                }
                                for s in claim.supporting_spans
                            ],
                        },
                    )

                if result.is_verified:
                    response_text += (
                        f"\n\n[Verified: {len(answer.claims)} claim(s), "
                        f"{len(answer.spans)} citation(s)]"
                    )
                elif result.verification_errors:
                    n_errors = len(result.verification_errors)
                    response_text += f"\n\n[Warning: {n_errors} citation(s) could not be verified]"

                yield emitter.emit(
                    ToolCallResult,
                    call_id="rag-query",
                    tool_name="rag-query",
                    result_summary=f"{len(answer.claims)} claims, verified={result.is_verified}",
                    is_error=False,
                    duration_ms=0.0,
                )

                if response_text:
                    yield emitter.emit(TextDelta, content=response_text)

                logger.debug(
                    "research_agent: RAG completed — %d claims, verified=%s",
                    len(answer.claims),
                    result.is_verified,
                )

                # Write success to REFLECTION for cross-turn learning
                n_sources = len({s.source_uri for c in answer.claims for s in c.supporting_spans})
                reflection_text = (
                    f"RAG answered '{message[:60]}' with {len(answer.claims)} claims "
                    f"from {n_sources} source(s), verified={result.is_verified}. "
                    f"Used plain BM25 on {n_docs_label} docs."
                )
                memory.add(MemoryType.REFLECTION, reflection_text)
                logger.debug(
                    "research_agent._handle_research: wrote REFLECTION: %s",
                    reflection_text[:120],
                )

            elif isinstance(result.grounded_answer, InsufficientEvidence):
                refusal = result.grounded_answer
                logger.debug(
                    "research_agent._handle_research: one-shot insufficient — "
                    "reason=%r, what_would_resolve=%r",
                    refusal.reason[:100],
                    (refusal.what_would_resolve or "")[:100],
                )

                # Retry: use what_would_resolve to drive a second retrieval attempt
                retried = False
                if refusal.what_would_resolve and isinstance(corpus, dict) and self._corpus is None:
                    retry_query = refusal.what_would_resolve
                    logger.debug(
                        "research_agent.retry: retrying with query=%r, current_corpus_size=%d",
                        retry_query[:80],
                        len(corpus),
                    )
                    yield emitter.emit(
                        ToolCallStart,
                        call_id="rag-retry",
                        tool_name="rag-retry",
                        arguments=(
                            ("retry_query", retry_query),
                            ("reason", refusal.reason[:100]),
                        ),
                    )

                    # Search for additional documents using the hint
                    from kaos_agents.memory.search import search_memory

                    retry_results = search_memory(
                        memory,
                        retry_query,
                        sections=[MemoryType.DOCUMENTS],
                        top_k=self._settings.retrieval_top_k,
                        expand_relations=[],
                    )
                    if retry_results:
                        # Merge new documents into the corpus
                        new_items = memory.get_by_ids(
                            MemoryType.DOCUMENTS,
                            {r.item_id for r in retry_results},
                        )
                        for item in new_items:
                            uri, content = _extract_uri_and_content(item)
                            if uri not in corpus:
                                corpus[uri] = content

                        logger.debug(
                            "research_agent: retry found %d new docs, corpus now %d",
                            len(new_items),
                            len(corpus),
                        )

                        # Re-query RAG with expanded corpus. Use .invoke()
                        # so the retry's tokens/cost roll into the turn total.
                        retry_invocation = await rag.invoke(question=message, documents=corpus)
                        retry_result = retry_invocation.output
                        yield emit_usage_observed(
                            emitter,
                            InvocationUsage.from_invocation(retry_invocation),
                            source="rag-retry",
                        )

                        if isinstance(retry_result.grounded_answer, Answer):
                            retried = True
                            answer = retry_result.grounded_answer
                            logger.debug(
                                "research_agent.retry: succeeded — claims=%d, verified=%s",
                                len(answer.claims),
                                retry_result.is_verified,
                            )
                            response_text = str(answer.value)
                            for claim in answer.claims:
                                for span in claim.supporting_spans:
                                    yield emitter.emit(
                                        CitationFound,
                                        claim=claim.statement,
                                        source_uri=span.source_uri,
                                        confidence=claim.confidence,
                                        verified=retry_result.is_verified,
                                    )
                            response_text += (
                                f"\n\n[Retry succeeded: {len(answer.claims)} claim(s) "
                                f"after expanding search with: {retry_query[:60]}]"
                            )
                            yield emitter.emit(
                                ToolCallResult,
                                call_id="rag-retry",
                                tool_name="rag-retry",
                                result_summary=f"Retry succeeded: {len(answer.claims)} claims",
                                is_error=False,
                                duration_ms=0.0,
                            )
                            yield emitter.emit(TextDelta, content=response_text)

                            # Write success to REFLECTION for cross-turn learning
                            retry_reflection = (
                                f"RAG retry succeeded on '{message[:60]}' by searching for "
                                f"'{retry_query[:60]}'. Found {len(new_items)} additional docs."
                            )
                            memory.add(MemoryType.REFLECTION, retry_reflection)
                            logger.debug(
                                "research_agent.retry: wrote REFLECTION: %s",
                                retry_reflection[:120],
                            )

                if not retried:
                    logger.debug(
                        "research_agent._handle_research: final insufficient evidence "
                        "(retry_attempted=%s)",
                        bool(
                            refusal.what_would_resolve
                            and isinstance(corpus, dict)
                            and self._corpus is None
                        ),
                    )
                    yield emitter.emit(
                        EvidenceInsufficient,
                        reason=refusal.reason,
                        what_would_resolve=refusal.what_would_resolve or "",
                    )
                    yield emitter.emit(
                        ToolCallResult,
                        call_id="rag-query",
                        tool_name="rag-query",
                        result_summary=f"Insufficient evidence: {refusal.reason[:100]}",
                        is_error=False,
                        duration_ms=0.0,
                    )

                    response_text = (
                        f"I don't have sufficient evidence to answer this question.\n\n"
                        f"Reason: {refusal.reason}"
                    )
                    if refusal.what_would_resolve:
                        response_text += f"\n\nWhat would help: {refusal.what_would_resolve}"

                    yield emitter.emit(TextDelta, content=response_text)

                    # Write failure to REFLECTION for cross-turn learning
                    failure_reflection = (
                        f"RAG insufficient evidence on '{message[:60]}': {refusal.reason[:80]}. "
                        f"Would resolve: {(refusal.what_would_resolve or 'unknown')[:80]}"
                    )
                    memory.add(MemoryType.REFLECTION, failure_reflection)
                    logger.debug(
                        "research_agent._handle_research: wrote REFLECTION: %s",
                        failure_reflection[:120],
                    )

            else:
                yield emitter.emit(TextDelta, content=str(result.grounded_answer))

        except Exception as exc:
            logger.warning("research_agent: RAG failed: %s", exc)
            response, usage = await self._simple_respond(
                message,
                memory,
                extra_instruction=(
                    f"A document Q&A attempt failed: {exc}. "
                    "Answer the question using what you know, but note that you couldn't "
                    "verify your answer against the source documents."
                ),
            )
            if response:
                yield emitter.emit(TextDelta, content=response)
            yield emit_usage_observed(emitter, usage, source="rag-fallback")

    async def _handle_research(
        self,
        message: str,
        memory: SessionMemory,
        context_items: dict[MemoryType, list[MemoryItem]],
    ) -> tuple[str, list[ToolCallRecord], InvocationUsage]:
        """Handle document Q&A (non-streaming, backward compat).

        Delegates to _handle_research_streaming and collects events,
        avoiding logic duplication. Aggregates UsageObserved into the
        returned InvocationUsage.
        """
        emitter = EventEmitter(session_id="internal", run_id="internal")

        response_text = ""
        tool_calls: list[ToolCallRecord] = []
        usage_total = ZERO_USAGE
        async for event in self._handle_research_streaming(message, memory, context_items, emitter):
            if isinstance(event, TextDelta):
                response_text += event.content
            elif isinstance(event, ToolCallResult):
                tool_calls.append(
                    ToolCallRecord.from_dict_args(
                        tool_name=event.tool_name,
                        arguments={},
                        result_summary=event.result_summary,
                        is_error=event.is_error,
                    )
                )
            elif isinstance(event, UsageObserved):
                usage_total = usage_total + InvocationUsage.from_llm_usage(event)

        return response_text, tool_calls, usage_total


def _looks_like_question(message: str, prefixes: tuple[str, ...]) -> bool:
    """Return True if *message* looks like a document-oriented question.

    Checks two lightweight signals:
    1. The message contains a question mark (``?``).
    2. The first word (lowercased) matches a known question/command prefix
       such as "what", "summarize", "find", etc.

    This keeps greetings ("hello", "hi there"), off-topic chat, and
    navigation commands from being force-routed to RAG when a pre-bound
    corpus is present.
    """
    if "?" in message:
        return True
    first_word = (
        message.lstrip().split(maxsplit=1)[0].lower().rstrip(".,!?:;") if message.strip() else ""
    )
    return first_word in prefixes


def _unwrap_corpus_arg(corpus: Any) -> Any:
    """Normalize the ``corpus`` ctor argument.

    Accepts:
    - ``None`` → returned as-is (agent falls back to memory DOCUMENTS).
    - A ``kaos_ml_core.CorpusIndex`` → unwrap to ``.corpus`` because
      ``RAG.forward`` expects a Corpus Protocol instance, not an index.
    - Anything else (must satisfy the Corpus Protocol) → returned as-is.

    The Corpus Protocol check is NOT enforced here; the downstream
    ``RAG._is_corpus`` does it. Keeping this helper lenient lets tests
    inject mocks without importing kaos-content solely to satisfy the
    isinstance check.
    """
    if corpus is None:
        return None
    # Unwrap CorpusIndex without a hard import — avoid pulling kaos-ml-core
    # into kaos-agents' mandatory dependency list.
    if type(corpus).__name__ == "CorpusIndex" and hasattr(corpus, "corpus"):
        return corpus.corpus
    return corpus


def _build_corpus_bm25(
    memory: SessionMemory,
    query: str,
    settings: Any,
) -> dict[str, str]:
    """Build a corpus dict using plain BM25 retrieval for large corpora.

    For small corpora (< threshold), returns all documents (FIFO).
    For large corpora, uses BM25 search to select the most relevant
    documents. This is the production retrieval path — plain BM25 has been
    proven to outperform the hardcoded adaptive pipeline on cross-domain
    BEIR benchmarks (0.296 vs 0.231 NDCG@10 on NFCorpus).

    For more sophisticated retrieval (synonym expansion, HyDE, iterative
    search), use the RetrievalAgent via ``kaos_agents.retrieval_agent``.
    """
    n_docs = (
        memory.section_item_count(MemoryType.DOCUMENTS)
        if memory.has_section(MemoryType.DOCUMENTS)
        else 0
    )

    if n_docs < settings.retrieval_threshold:
        logger.debug(
            "research_agent._build_corpus_bm25: small corpus (%d < %d), returning all docs",
            n_docs,
            settings.retrieval_threshold,
        )
        return _build_corpus(memory)

    from kaos_agents.memory.search import search_memory

    results = search_memory(
        memory,
        query,
        sections=[MemoryType.DOCUMENTS],
        top_k=settings.retrieval_top_k,
        expand_relations=[],
    )

    if not results:
        logger.debug(
            "research_agent._build_corpus_bm25: BM25 returned no results for query=%r, "
            "falling back to all %d docs",
            query[:50],
            n_docs,
        )
        return _build_corpus(memory)

    selected_ids = {r.item_id for r in results}
    items = memory.get_by_ids(MemoryType.DOCUMENTS, selected_ids)

    logger.debug(
        "research.bm25_corpus: query=%r total=%d selected=%d",
        query[:50],
        n_docs,
        len(items),
    )

    return dict(_extract_uri_and_content(item) for item in items)


def _build_corpus_triaged(
    memory: SessionMemory,
    query: str,
    threshold: int = 20,
) -> dict[str, str]:
    """Build a corpus dict, narrowed by BM25 when the section is large.

    When the DOCUMENTS section has >= ``threshold`` items, uses
    ``triage_corpus()`` to select the most relevant subset. Below
    the threshold, returns all documents (identical to ``_build_corpus``).
    """
    from kaos_agents.context.triage import triage_corpus

    triage = triage_corpus(memory, query, threshold=threshold)
    if triage is not None:
        selected_ids = set(triage.selected_item_ids)
        items = memory.get_by_ids(MemoryType.DOCUMENTS, selected_ids)
        logger.debug(
            "research: triaged %d → %d documents for query=%r",
            triage.total_documents,
            len(items),
            query[:50],
        )
    else:
        items = memory.get(MemoryType.DOCUMENTS) if memory.has_section(MemoryType.DOCUMENTS) else []

    return dict(_extract_uri_and_content(item) for item in items)


def _extract_uri_and_content(item: Any) -> tuple[str, str]:
    """Extract URI and content text from a memory item.

    Parses URI from metadata first, then falls back to "URI: <uri>\\n<text>"
    prefix format. Returns (uri, content) where uri may be "mem:<item_id>"
    if no URI is found.
    """
    content = item.content
    uri = (item.metadata or {}).get("uri", "")
    if not uri and content.startswith("URI: "):
        first_line, _, rest = content.partition("\n")
        uri = first_line.removeprefix("URI: ").strip()
        content = rest
    elif content.startswith("URI: "):
        _, _, content = content.partition("\n")
    if not uri:
        uri = f"mem:{item.id}"
    return uri, content


def _build_corpus(memory: SessionMemory) -> dict[str, str]:
    """Build a corpus dict from the DOCUMENTS section.

    Parses "URI: <uri>\\n<text>" items into {uri: text} pairs.
    """
    if not memory.has_section(MemoryType.DOCUMENTS):
        return {}

    return dict(_extract_uri_and_content(item) for item in memory.get(MemoryType.DOCUMENTS))
