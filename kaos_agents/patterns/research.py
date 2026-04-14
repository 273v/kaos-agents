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
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from kaos_core.logging import get_logger

from kaos_agents.memory.types import MemoryType
from kaos_agents.models import ToolCallRecord
from kaos_agents.patterns.chat import ChatAgent

if TYPE_CHECKING:
    from kaos_core.base.context import KaosContext
    from kaos_core.registry.container import KaosRuntime
    from kaos_core.vfs.core import VirtualFileSystem

    from kaos_agents.memory.session import SessionMemory
    from kaos_agents.memory.types import MemoryItem
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
        agent.load_document(memory, "doc:contract-1", contract_text)
        response = await agent.turn("What is the termination clause?", session_id="abc")
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
        )
        self._rag_top_k = rag_top_k
        self._rag_max_retries = rag_max_retries

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
            uri: Document identifier (e.g., "doc:contract-1", "ecfr:title26").
            text: Full document text.
        """
        memory.add(
            MemoryType.DOCUMENTS,
            f"URI: {uri}\n{text}",
            metadata={"uri": uri, "type": "document"},
        )

    async def _handle_research(
        self,
        message: str,
        memory: SessionMemory,
        context_items: dict[MemoryType, list[MemoryItem]],
    ) -> tuple[str, list[ToolCallRecord]]:
        """Handle document Q&A via RAG pipeline."""
        # Collect documents from memory
        corpus = _build_corpus(memory)

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
            return response, []

        try:
            from kaos_agents._llm_imports import require_llm_core

            require_llm_core()
            from kaos_llm_core.programs.rag import RAG
            from kaos_llm_core.signatures.grounding import (
                Answer,
                InsufficientEvidence,
                MatchStrategy,
            )

            # Build RAG instance
            rag = RAG(
                model=self._model,
                top_k=self._rag_top_k,
                max_retries=self._rag_max_retries,
                match_strategies=(
                    MatchStrategy.STRICT,
                    MatchStrategy.SUBSTRING,
                    MatchStrategy.CASE_INSENSITIVE,
                    MatchStrategy.NORMALIZED_TOKEN,
                ),
            )

            # Query RAG
            result = await rag.query(question=message, documents=corpus)

            # Process the grounded answer
            tool_calls: list[ToolCallRecord] = []

            if isinstance(result.grounded_answer, Answer):
                answer = result.grounded_answer
                response_text = str(answer.value)

                # Store claims in FINDINGS with full structured provenance
                for claim in answer.claims:
                    sources = ", ".join(span.source_uri for span in claim.supporting_spans)
                    finding = (
                        f"[{claim.claim_type}] {claim.statement} "
                        f"(confidence={claim.confidence:.2f}, sources={sources})"
                    )
                    # Structured metadata preserves the full Claim data for
                    # downstream programmatic access (filtering, re-verification,
                    # audit). The content string is the human-readable summary.
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

                # Add verification status
                if result.is_verified:
                    response_text += (
                        f"\n\n[Verified: {len(answer.claims)} claim(s), "
                        f"{len(answer.spans)} citation(s)]"
                    )
                elif result.verification_errors:
                    n_errors = len(result.verification_errors)
                    response_text += f"\n\n[Warning: {n_errors} citation(s) could not be verified]"

                # Record as a tool call for provenance
                tool_calls.append(
                    ToolCallRecord.from_dict_args(
                        tool_name="rag-query",
                        arguments={"question": message, "n_documents": len(corpus)},
                        result_summary=f"{len(answer.claims)} claims, verified={result.is_verified}",
                        is_error=False,
                    )
                )

                logger.debug(
                    "research_agent: RAG completed — %d claims, %d spans, verified=%s, confidence=%.2f",
                    len(answer.claims),
                    len(answer.spans),
                    result.is_verified,
                    result.confidence,
                )

            elif isinstance(result.grounded_answer, InsufficientEvidence):
                refusal = result.grounded_answer
                response_text = (
                    f"I don't have sufficient evidence to answer this question.\n\n"
                    f"Reason: {refusal.reason}"
                )
                if refusal.what_would_resolve:
                    response_text += f"\n\nWhat would help: {refusal.what_would_resolve}"

                tool_calls.append(
                    ToolCallRecord.from_dict_args(
                        tool_name="rag-query",
                        arguments={"question": message, "n_documents": len(corpus)},
                        result_summary=f"Insufficient evidence: {refusal.reason[:100]}",
                        is_error=False,
                    )
                )

                logger.debug(
                    "research_agent: RAG refused — %s",
                    refusal.reason[:100],
                )

            else:
                response_text = str(result.grounded_answer)

            return response_text, tool_calls

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
            return response, []


def _build_corpus(memory: SessionMemory) -> dict[str, str]:
    """Build a corpus dict from the DOCUMENTS section.

    Parses "URI: <uri>\\n<text>" items into {uri: text} pairs.
    """
    corpus: dict[str, str] = {}
    if not memory.has_section(MemoryType.DOCUMENTS):
        return corpus

    items = memory.get(MemoryType.DOCUMENTS)
    for item in items:
        content = item.content
        # Parse URI from metadata or from the content prefix
        uri = item.metadata.get("uri", "")
        if not uri and content.startswith("URI: "):
            first_line, _, rest = content.partition("\n")
            uri = first_line.removeprefix("URI: ").strip()
            content = rest
        elif content.startswith("URI: "):
            _, _, content = content.partition("\n")

        if uri:
            corpus[uri] = content
        else:
            # Fall back to item ID as URI
            corpus[f"mem:{item.id}"] = content

    return corpus
