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
)
from kaos_agents.memory.types import MemoryType
from kaos_agents.models import IntentResult, IntentType, ToolCallRecord
from kaos_agents.patterns.chat import ChatAgent

if TYPE_CHECKING:
    from kaos_core.base.context import KaosContext
    from kaos_core.registry.container import KaosRuntime
    from kaos_core.vfs.core import VirtualFileSystem

    from kaos_agents.memory.session import SessionMemory
    from kaos_agents.memory.types import MemoryItem
    from kaos_agents.providers import ProviderConfig
    from kaos_agents.settings import KaosAgentSettings

logger = get_logger(__name__)


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

    async def _classify(
        self,
        message: str,
        memory: SessionMemory,
        context_items: dict[MemoryType, list[Any]] | None = None,
    ) -> IntentResult:
        """Force RESEARCH intent when an explicit corpus is bound.

        Without this override, the base classifier reads ``memory.DOCUMENTS``
        and reports TOOL_USE / RESPOND when the section is empty — even
        though the agent has a fully-populated external corpus. Routing
        would then skip ``_handle_research_streaming`` and hand an
        answerable question to the hallucination-prone simple-respond path
        (observed in an early WS-3.6 live run: "$25/$50/$100" fabricated
        Delaware fees when the corpus said "$89").

        When ``self._corpus`` is ``None`` the base behavior applies — the
        memory-based routing must keep working for legacy callers.
        """
        if self._corpus is not None:
            return IntentResult(
                intent=IntentType.RESEARCH,
                confidence=1.0,
                reasoning="ResearchAgent has a pre-bound corpus; routing directly to RAG.",
            )
        return await super()._classify(message, memory, context_items)

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

    async def _dispatch_streaming(
        self,
        intent: IntentResult,
        message: str,
        memory: SessionMemory,
        context_items: dict[MemoryType, list[Any]],
        emitter: EventEmitter,
    ) -> AsyncIterator[AgentEvent]:
        """Override dispatch to stream RAG citation events.

        For RESEARCH intent, yields CitationFound/EvidenceInsufficient.
        For other intents, delegates to ChatAgent (TOOL_USE) or BaseAgent.
        """
        if intent.intent != IntentType.RESEARCH:
            async for event in super()._dispatch_streaming(
                intent, message, memory, context_items, emitter
            ):
                yield event
            return

        async for event in self._handle_research_streaming(message, memory, context_items, emitter):
            yield event

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
        else:
            corpus = _build_corpus_bm25(memory, message, self._settings)
            n_docs_label = len(corpus) if isinstance(corpus, dict) else 0

        if not corpus:
            logger.warning("research_agent: no documents loaded — falling back to simple response")
            response = await self._simple_respond(
                message,
                memory,
                extra_instruction=(
                    "The user asked a research question but no documents are loaded. "
                    "Explain that documents need to be loaded first, and suggest how."
                ),
            )
            if response:
                yield emitter.emit(TextDelta, content=response)
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

            result = await rag.query(question=message, documents=corpus)

            if isinstance(result.grounded_answer, Answer):
                answer = result.grounded_answer
                response_text = str(answer.value)

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
                memory.add(
                    MemoryType.REFLECTION,
                    f"RAG answered '{message[:60]}' with {len(answer.claims)} claims "
                    f"from {n_sources} source(s), verified={result.is_verified}. "
                    f"Used plain BM25 on {n_docs_label} docs.",
                )

            elif isinstance(result.grounded_answer, InsufficientEvidence):
                refusal = result.grounded_answer

                # Retry: use what_would_resolve to drive a second retrieval attempt
                retried = False
                if (
                    refusal.what_would_resolve
                    and isinstance(corpus, dict)
                    and self._corpus is None
                ):
                    retry_query = refusal.what_would_resolve
                    logger.debug(
                        "research_agent: insufficient evidence — retrying with: %s",
                        retry_query[:80],
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

                        # Re-query RAG with expanded corpus
                        retry_result = await rag.query(
                            question=message, documents=corpus
                        )

                        if isinstance(retry_result.grounded_answer, Answer):
                            retried = True
                            answer = retry_result.grounded_answer
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
                            memory.add(
                                MemoryType.REFLECTION,
                                f"RAG retry succeeded on '{message[:60]}' by searching for "
                                f"'{retry_query[:60]}'. Found {len(new_items)} additional docs.",
                            )

                if not retried:
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
                        response_text += (
                            f"\n\nWhat would help: {refusal.what_would_resolve}"
                        )

                    yield emitter.emit(TextDelta, content=response_text)

                    # Write failure to REFLECTION for cross-turn learning
                    memory.add(
                        MemoryType.REFLECTION,
                        f"RAG insufficient evidence on '{message[:60]}': {refusal.reason[:80]}. "
                        f"Would resolve: {(refusal.what_would_resolve or 'unknown')[:80]}",
                    )

            else:
                yield emitter.emit(TextDelta, content=str(result.grounded_answer))

        except Exception as exc:
            logger.warning("research_agent: RAG failed: %s", exc)
            response = await self._simple_respond(
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

    async def _handle_research(
        self,
        message: str,
        memory: SessionMemory,
        context_items: dict[MemoryType, list[MemoryItem]],
    ) -> tuple[str, list[ToolCallRecord]]:
        """Handle document Q&A (non-streaming, backward compat).

        Delegates to _handle_research_streaming and collects events,
        avoiding logic duplication.
        """
        emitter = EventEmitter(session_id="internal", run_id="internal")

        response_text = ""
        tool_calls: list[ToolCallRecord] = []
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

        return response_text, tool_calls


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
