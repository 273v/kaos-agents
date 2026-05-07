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

from collections.abc import AsyncIterator, Sequence
from typing import TYPE_CHECKING, Any, Literal

from kaos_core.logging import get_logger

from kaos_agents.events import (
    CitationFound,
    EventEmitter,
    EvidenceInsufficient,
    KaosEvent,
    Span,
    SpanPhase,
    SpanSubject,
    TextDelta,
    UsageObserved,
    emit_usage_observed,
)
from kaos_agents.patterns.chat import ChatAgent
from kaos_agents.patterns.research.corpus import (
    _build_corpus_bm25,
    _extract_uri_and_content,
    _looks_like_question,
    _unwrap_corpus_arg,
)
from kaos_agents.patterns.research.outline import (
    _OUTLINE_MAX_DOCS_AUTO,
    _OUTLINE_MAX_TOTAL_CHARS,
    _build_outline_text,
)
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

if TYPE_CHECKING:
    from kaos_core.base.context import KaosContext
    from kaos_core.registry.container import KaosRuntime
    from kaos_core.vfs.core import VirtualFileSystem

    from kaos_agents.memory.session import SessionMemory
    from kaos_agents.types.memory import MemoryItem
    from kaos_agents.types.providers import ProviderConfig

# Runtime import — needed by ``_build_refusal_policy`` which calls
# ``KaosAgentSettings.resolve(...)`` to derive verifier_min_confidence.
# Was TYPE_CHECKING-only until N4 added the resolve call.
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

WHEN A CITATION SEEMS INCOMPLETE OR UNCLEAR:
The kaos-content-context-window tool expands around an AST node_ref —
think of it as ``grep -A N -B N`` over the document. Use it when:
- A retrieved passage mentions a defined term ("Effective Date",
  "Confidential Information") but the definition isn't in the passage.
- The passage trails off mid-clause and the answer might continue in
  the next paragraph.
- A list item references "the foregoing" or "Section 4" — those
  references resolve in nearby blocks.
- A footnote reference appears in the cited passage.
Pass the citation's ``block_ref`` as ``node_ref`` and ask for 2-5
blocks before/after; pass ``expand_to_section=true`` to clamp to the
enclosing heading-bounded section when crossing a heading boundary
would lose the answer.

IMPORTANT:
- Always search BEFORE answering. Never answer from your training data.
- Include passage previews when calling kaos-retrieval-answer.
- If the question references a specific document (e.g., "RFC 2119"), search for that name.
- Do NOT hallucinate. If the passages don't support the answer, refuse.
- When a citation feels incomplete, expand it with kaos-content-context-window
  before answering. A complete citation is better than a confident guess.
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

    @classmethod
    def metadata(cls) -> AgentMetadata:
        """Pattern identity — ``pattern="research"``, RAG-backed Q&A
        with citation verification."""
        return AgentMetadata(
            name="research-agent",
            description="RAG-backed document Q&A with citation verification.",
            pattern="research",
            tags=("research", "rag", "citation", "verified"),
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
        rag_top_k: int = 10,
        rag_max_retries: int = 2,
        settings: KaosAgentSettings | None = None,
        provider: ProviderConfig | None = None,
        extra_llm_tools: tuple[Any, ...] = (),
        corpus: Any = None,
        permission_policy: Any = None,
        instructions: str | None = None,
        documents: Sequence[Any] | None = None,
        show_outline: Literal["yes", "no", "auto"] = "auto",
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
            documents: Optional list of original ``ContentDocument`` instances
                used to build the corpus. When provided, the agent can compute
                a richer "table of contents" outline (with depth, page ranges)
                and inject it into the question prompt as a corpus preamble
                so the LLM sees the structure of the corpus before retrieval.
                When None, the agent falls back to extracting an outline from
                ``corpus`` passages' ``section_title`` (single-depth view).
            show_outline: Outline-injection policy.
                ``"auto"`` (default) — inject when ≤50 docs and the rendered
                outline fits in ~5000 chars. ``"yes"`` — always inject (truncated
                if needed). ``"no"`` — never inject (matches pre-P7 behavior).
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
        self._documents: tuple[Any, ...] = tuple(documents) if documents is not None else ()
        self._show_outline: Literal["yes", "no", "auto"] = show_outline
        # N4 — derive a verifier-confidence policy from settings if the
        # caller hasn't passed an explicit one. KaosAgentSettings.
        # verifier_min_confidence > 0 → build a default RefusalPolicy.
        # When 0 (default), this stays None and the legacy permissive
        # path runs. ``refuse_unverified_answers`` is a separate boolean
        # knob that targets ``result.is_verified`` directly (the
        # citation-spans-don't-verify-in-corpus failure mode).
        resolved_settings = settings or KaosAgentSettings.resolve(None)
        self._refusal_policy = self._build_refusal_policy(resolved_settings)
        self._refuse_unverified: bool = bool(
            getattr(resolved_settings, "refuse_unverified_answers", False)
        )
        # Cache the rendered outline string. Computed lazily on first turn
        # because rendering walks every document's heading tree (O(n_docs *
        # n_blocks)), and the outline is identical across turns of the same
        # corpus. Re-rendering per turn would burn 5-50 ms on a 50-doc deal
        # room for no benefit.
        self._outline_cache: str | None = None

    @property
    def corpus(self) -> Any:
        """The pre-built corpus passed to ``__init__``, or ``None``.

        When set, ``_handle_research_streaming`` hands this to
        ``RAG.query(documents=...)`` directly and bypasses
        ``_build_corpus(memory)``.
        """
        return self._corpus

    @property
    def documents(self) -> tuple[Any, ...]:
        """Original ``ContentDocument`` instances backing the corpus, or ``()``.

        Used by P7 outline injection to render a "table of contents"
        preamble so the LLM sees the corpus structure before retrieval.
        """
        return self._documents

    @staticmethod
    def _build_refusal_policy(settings: KaosAgentSettings) -> Any | None:
        """Construct a kaos-llm-core RefusalPolicy from settings.

        N4 — when ``settings.verifier_min_confidence > 0``, returns a
        default ``RefusalPolicy(min_confidence=that)``. Otherwise None
        (legacy permissive — no verifier-threshold enforcement). A
        partner deploys ``KAOS_AGENT_VERIFIER_MIN_CONFIDENCE=0.7`` and
        the agent will refuse low-confidence answers automatically; no
        code change required.

        Returns ``None`` when kaos-llm-core isn't installed (the [llm]
        extra is optional). The caller's check site already guards for
        ``policy is None`` so this keeps the agent functional without
        the LLM dep.
        """
        threshold = float(getattr(settings, "verifier_min_confidence", 0.0) or 0.0)
        if threshold <= 0.0:
            return None
        try:
            from kaos_llm_core.signatures.grounding import RefusalPolicy

            return RefusalPolicy(min_confidence=threshold)
        except ImportError:
            return None

    def _resolve_outline_prefix(self) -> str:
        """Return the rendered outline preamble, or ``""`` to skip injection.

        Honors ``self._show_outline``:
        - ``"no"``  → always skip (matches pre-P7 behavior).
        - ``"yes"`` → always render, truncating per-doc to fit ~5000 chars.
        - ``"auto"`` (default) → render only when the corpus has ≤50 docs
          AND the rendered outline is ≤5000 chars. Larger corpora would
          either overflow the prompt or contain so many headings that the
          LLM gains little structural signal beyond what BM25 already sees.

        Cached on the agent (``_outline_cache``) — re-rendering per turn
        is wasteful since the corpus is fixed across turns. Set the cache
        to ``None`` when the corpus changes (currently only on agent
        construction; future hot-swap would clear it explicitly).
        """
        if self._show_outline == "no":
            return ""
        if self._outline_cache is not None:
            return self._outline_cache
        try:
            outline = _build_outline_text(
                documents=self._documents,
                corpus=self._corpus,
                policy=self._show_outline,
                max_total_chars=_OUTLINE_MAX_TOTAL_CHARS,
                max_docs_for_auto=_OUTLINE_MAX_DOCS_AUTO,
            )
        except Exception as exc:  # pragma: no cover — outline is a hint, never fatal
            logger.debug("research_agent: outline build failed (%s); skipping injection", exc)
            outline = ""
        self._outline_cache = outline
        return outline

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
    ) -> AsyncIterator[KaosEvent]:
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
        yield emitter.span_start(
            SpanSubject.TOOL_CALL,
            name="tool.react-escalation",
            attributes={
                "tool_name": "react-escalation",
                "call_id": "react-escalation",
                "arguments": (("reason", "one-shot RAG returned insufficient evidence"),),
            },
        )

        saved_instructions = self._instructions
        # P7: thread the corpus outline into the ReAct system prompt too.
        # The escalation path drives an iterative search agent — knowing
        # what the corpus actually covers helps it pick search terms (and
        # know when to stop searching for non-existent topics).
        outline_prefix = self._resolve_outline_prefix()
        outline_block = f"\n\n{outline_prefix}" if outline_prefix else ""
        self._instructions = (
            (saved_instructions + "\n\n" if saved_instructions else "")
            + _RESEARCH_REACT_INSTRUCTION
            + outline_block
        )

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
    ) -> AsyncIterator[KaosEvent]:
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
            yield emit_usage_observed(emitter, usage, source="research")
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
            rag_query_span = emitter.span_start(
                SpanSubject.TOOL_CALL,
                name="tool.rag-query",
                attributes={
                    "tool_name": "rag-query",
                    "call_id": "rag-query",
                    "arguments": (
                        ("question", message),
                        ("n_documents", str(n_docs_label)),
                    ),
                },
            )
            yield rag_query_span

            # P7: prepend a corpus outline to the question so the LLM sees
            # the corpus's "table of contents" up-front and can reason
            # "this corpus has no NASA content, refuse" before BM25 finds
            # spurious lexical matches. ``decorated_question`` is the
            # question with an optional ``[CORPUS OUTLINE] ...`` preamble;
            # ``raw_question`` is the user's original wording (preserved
            # for memory + reflection so the outline doesn't pollute
            # conversation history).
            outline_prefix = self._resolve_outline_prefix()
            decorated_question = (
                f"{outline_prefix}\n\n[QUESTION]\n{message}" if outline_prefix else message
            )

            # ``.invoke()`` returns the full Invocation so we can emit
            # real usage. ``rag.query()``/``rag(...)`` are thin unwrappers
            # around the same pipeline that throw usage on the floor.
            rag_invocation = await rag.invoke(question=decorated_question, documents=corpus)
            result = rag_invocation.output
            yield emit_usage_observed(
                emitter,
                InvocationUsage.from_invocation(rag_invocation),
                source="rag-query",
            )

            # N4 — apply refusal policy (verifier_min_confidence threshold).
            # When the agent's effective_refusal_policy() is set and the
            # answer's confidence is below threshold, we collapse it to
            # InsufficientEvidence here so the rest of this branch sees
            # the refusal shape and the user gets an honest "I don't have
            # enough evidence" instead of a low-confidence claim.
            policy = self._refusal_policy
            if policy is not None and isinstance(result.grounded_answer, Answer):
                from kaos_agents.grounding import apply_refusal_policy

                collapsed, refusal_event = apply_refusal_policy(
                    result.grounded_answer,
                    policy,
                    sequence=getattr(emitter, "sequence", 0),
                )
                if refusal_event is not None:
                    yield refusal_event
                    try:
                        result = result.model_copy(update={"grounded_answer": collapsed})
                    except Exception:
                        result.grounded_answer = collapsed  # type: ignore[attr-defined]

            # N4 — when refuse_unverified_answers is enabled and the RAG
            # result reports ``is_verified=False`` (citation spans
            # didn't match in the source corpus), we refuse rather than
            # ship the answer with a "could not be verified" warning.
            # This is the mf09 Voyager fix: the LLM was confident, the
            # citations didn't actually verify, and the right move was a
            # refusal not a warning.
            if (
                self._refuse_unverified
                and not getattr(result, "is_verified", True)
                and isinstance(result.grounded_answer, Answer)
            ):
                try:
                    from kaos_llm_core.signatures.grounding import InsufficientEvidence

                    collapsed = InsufficientEvidence(
                        reason=(
                            "RAG returned an answer but its citation "
                            "spans could not be verified in the source "
                            "corpus. Refusing to present an unverified "
                            "answer (set "
                            "KAOS_AGENT_REFUSE_UNVERIFIED_ANSWERS=false to "
                            "downgrade to a warning instead)."
                        ),
                        attempted_claims=getattr(result.grounded_answer, "claims", []),
                    )
                    try:
                        result = result.model_copy(update={"grounded_answer": collapsed})
                    except Exception:
                        result.grounded_answer = collapsed  # type: ignore[attr-defined]
                    logger.info(
                        "research_agent: refused unverified answer (refuse_unverified_answers=True)"
                    )
                except ImportError:
                    pass

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

                    # Also store in memory (side effect). Use Pydantic's
                    # canonical JSON-mode serialisation rather than
                    # hand-flattening fields — manual `list(s.char_span)`
                    # erases the `tuple[int, int]` typing and
                    # `str(claim.claim_type)` erases the ClaimType enum.
                    # `model_dump(mode="json")` gives us a round-trippable
                    # representation of the typed grounding model.
                    sources = ", ".join(span.source_uri for span in claim.supporting_spans)
                    finding = (
                        f"[{claim.claim_type}] {claim.statement} "
                        f"(confidence={claim.confidence:.2f}, sources={sources})"
                    )
                    claim_payload = claim.model_dump(mode="json")
                    memory.add(
                        MemoryType.FINDINGS,
                        finding,
                        metadata={
                            "claim": claim_payload,
                            "verified": result.is_verified,
                            # Convenience accessors for callers that
                            # don't want to walk the full claim payload.
                            "claim_type": claim_payload["claim_type"],
                            "statement": claim_payload["statement"],
                            "confidence": claim_payload["confidence"],
                            "sources": [s["source_uri"] for s in claim_payload["supporting_spans"]],
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

                yield emitter.span_complete(
                    SpanSubject.TOOL_CALL,
                    span_id=rag_query_span.span_id,
                    name="tool.rag-query",
                    duration_ms=0.0,
                    attributes={
                        "tool_name": "rag-query",
                        "call_id": "rag-query",
                        "result_summary": (
                            f"{len(answer.claims)} claims, verified={result.is_verified}"
                        ),
                        "is_error": False,
                    },
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
                    rag_retry_span = emitter.span_start(
                        SpanSubject.TOOL_CALL,
                        name="tool.rag-retry",
                        attributes={
                            "tool_name": "rag-retry",
                            "call_id": "rag-retry",
                            "arguments": (
                                ("retry_query", retry_query),
                                ("reason", refusal.reason[:100]),
                            ),
                        },
                    )
                    yield rag_retry_span

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
                        # Re-decorate the question with the same outline preamble
                        # so the retry sees the same corpus context as the
                        # original attempt (cached on the agent — no extra cost).
                        retry_invocation = await rag.invoke(
                            question=decorated_question,
                            documents=corpus,
                        )
                        retry_result = retry_invocation.output
                        yield emit_usage_observed(
                            emitter,
                            InvocationUsage.from_invocation(retry_invocation),
                            source="rag-query",
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
                            yield emitter.span_complete(
                                SpanSubject.TOOL_CALL,
                                span_id=rag_retry_span.span_id,
                                name="tool.rag-retry",
                                duration_ms=0.0,
                                attributes={
                                    "tool_name": "rag-retry",
                                    "call_id": "rag-retry",
                                    "result_summary": (
                                        f"Retry succeeded: {len(answer.claims)} claims"
                                    ),
                                    "is_error": False,
                                },
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
                    yield emitter.span_complete(
                        SpanSubject.TOOL_CALL,
                        span_id=rag_query_span.span_id,
                        name="tool.rag-query",
                        duration_ms=0.0,
                        attributes={
                            "tool_name": "rag-query",
                            "call_id": "rag-query",
                            "result_summary": (f"Insufficient evidence: {refusal.reason[:100]}"),
                            "is_error": False,
                        },
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
            yield emit_usage_observed(emitter, usage, source="rag-query")

    async def _handle_research(
        self,
        message: str,
        memory: SessionMemory,
        context_items: dict[MemoryType, list[MemoryItem]],
    ) -> tuple[str, list[ToolExecution], InvocationUsage]:
        """Handle document Q&A (non-streaming, backward compat).

        Delegates to _handle_research_streaming and collects events,
        avoiding logic duplication. Aggregates UsageObserved into the
        returned InvocationUsage.
        """
        emitter = EventEmitter(session_id="internal", run_id="internal")

        response_text = ""
        tool_calls: list[ToolExecution] = []
        usage_total = ZERO_USAGE
        async for event in self._handle_research_streaming(message, memory, context_items, emitter):
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
