"""Retrieval tools for the RetrievalAgent sub-agent.

Each tool is a KaosTool that operates on the session's DOCUMENTS memory
section. The RetrievalAgent uses these via ReAct to dynamically choose
retrieval strategies based on the query and corpus.

Tools:
- kaos-retrieval-bm25: Plain BM25 keyword search
- kaos-retrieval-synonyms: Look up synonyms via OpenGloss Lexicon
- kaos-retrieval-hyde: Generate pseudo-document, then BM25 search
- kaos-retrieval-evaluate: Judge whether results cover the query

All tools return JSON with a consistent schema so the RetrievalAgent
can reason about results across multiple calls.
"""

from __future__ import annotations

from typing import Any

from kaos_core.base.tool import KaosTool
from kaos_core.logging import get_logger
from kaos_core.types.metadata import ToolAnnotations, ToolCapability, ToolCategory, ToolMetadata
from kaos_core.types.parameters import ParameterSchema
from kaos_core.types.results import ToolResult

logger = get_logger(__name__)

_MODULE = "kaos-agents"
_VERSION = "0.1.0"
_READ_ANNOTATIONS = ToolAnnotations(
    readOnlyHint=True,
    destructiveHint=False,
    idempotentHint=True,
    openWorldHint=False,
)


def _get_corpus_from_context(context: Any) -> Any | None:
    """Get the ContentDocumentCorpus from the execution context, if available."""
    if context is None:
        return None
    config = getattr(context, "_config", None)
    if config and isinstance(config, dict):
        return config.get("_corpus")
    return None


# Process-level cache: corpus → Searcher (avoid rebuilding per tool call)
_CORPUS_SEARCHER_CACHE: dict[int, Any] = {}


def _get_corpus_searcher(corpus: Any) -> Any:
    """Get or build a BM25 Searcher over the corpus passages."""
    corpus_id = id(corpus)
    if corpus_id in _CORPUS_SEARCHER_CACHE:
        return _CORPUS_SEARCHER_CACHE[corpus_id]

    from kaos_nlp_core.search import Searcher

    records = []
    passage_meta: list[dict[str, Any]] = []
    for passage in corpus.iter_passages():
        records.append({"id": len(records), "text": passage.text})
        passage_meta.append({
            "doc_uri": getattr(passage, "doc_uri", ""),
            "block_ref": passage.block_ref,
            "page": passage.page,
            "section_title": getattr(passage, "section_title", None),
        })

    searcher = Searcher.from_documents(records)
    _CORPUS_SEARCHER_CACHE[corpus_id] = (searcher, passage_meta)
    return searcher, passage_meta


def _get_memory(inputs: dict[str, Any], context: Any):
    """Load session memory from context."""
    from kaos_core.vfs.core import VirtualFileSystem

    from kaos_agents.memory.store import SessionStore

    session_id = inputs.get("session_id", "")
    if not session_id:
        return None, None, (
            "Missing 'session_id' parameter. "
            "Provide a valid session ID, e.g. {\"session_id\": \"my-session\"}. "
            "Alternative: use kaos-agent-chat which manages sessions automatically."
        )

    runtime = context.runtime if context else None
    vfs = runtime.vfs if runtime and hasattr(runtime, "vfs") else VirtualFileSystem()
    store = SessionStore(vfs)

    import asyncio

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor() as pool:
            memory = pool.submit(asyncio.run, store.load_or_create(session_id)).result(timeout=10)
    else:
        memory = asyncio.run(store.load_or_create(session_id))

    return memory, store, None


def _format_results(results: list, max_results: int = 10) -> list[dict]:
    """Format search results for tool output."""
    formatted = []
    for r in results[:max_results]:
        formatted.append(
            {
                "item_id": r.item_id,
                "score": round(r.score, 3),
                "preview": r.content[:300],
            }
        )
    return formatted


def _summarize_topics(results: list, max_results: int = 5) -> str:
    """Summarize the dominant topics across top results for self-assessment."""
    if not results:
        return "No results found."
    previews = [r.content[:150] for r in results[:max_results]]
    return " | ".join(previews)


_SPARSE_RESULT_THRESHOLD = 5  # Fewer results than this suggests vocabulary gap
_LOW_SCORE_THRESHOLD = 3.0  # BM25 scores below this are weak matches
_SCORE_DROP_RATIO = 3.0  # Top/5th score ratio above this suggests narrow match


def _assess_expansion_need(results: list, requested_k: int) -> dict[str, Any]:
    """Assess whether query expansion might help based on BM25 score distribution.

    Returns a dict with:
    - suggest_expansion: bool
    - reason: str explaining the assessment
    - recommendation: str for the agent
    - top_score, score_p50: float for context
    """
    if not results:
        return {
            "suggest_expansion": True,
            "reason": "No results found — likely a vocabulary mismatch",
            "recommendation": "Try synonym search or HyDE",
            "top_score": 0.0,
            "score_p50": 0.0,
        }

    scores = [r.score for r in results]
    top_score = scores[0]
    score_p50 = scores[len(scores) // 2] if len(scores) > 1 else top_score
    n_results = len(results)

    # Few results relative to what was requested
    if n_results < _SPARSE_RESULT_THRESHOLD:
        return {
            "suggest_expansion": True,
            "reason": f"Only {n_results} results (requested {requested_k}) — sparse coverage",
            "recommendation": "Try synonym search for alternative terminology",
            "top_score": round(top_score, 2),
            "score_p50": round(score_p50, 2),
        }

    # Very low scores across the board
    if top_score < _LOW_SCORE_THRESHOLD:
        return {
            "suggest_expansion": True,
            "reason": f"Top score only {top_score:.1f} — weak keyword match",
            "recommendation": "Try HyDE to bridge vocabulary gap",
            "top_score": round(top_score, 2),
            "score_p50": round(score_p50, 2),
        }

    # Sharp score drop — top results match well but coverage is narrow
    if len(scores) >= 5 and scores[0] / max(scores[4], 0.01) > _SCORE_DROP_RATIO:
        return {
            "suggest_expansion": False,
            "reason": "Strong top matches with sharp score drop — focused results",
            "recommendation": "Results look focused. Expansion not recommended unless coverage gaps are found",
            "top_score": round(top_score, 2),
            "score_p50": round(score_p50, 2),
        }

    # Default: good results, no expansion needed
    return {
        "suggest_expansion": False,
        "reason": f"{n_results} results with top score {top_score:.1f} — good coverage",
        "recommendation": "Results look adequate. Evaluate coverage before expanding",
        "top_score": round(top_score, 2),
        "score_p50": round(score_p50, 2),
    }


class BM25SearchTool(KaosTool):
    """Plain BM25 keyword search over session documents."""

    @property
    def metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name="kaos-retrieval-bm25",
            display_name="BM25 Document Search",
            description=(
                "Search documents using BM25 keyword matching. Returns the top-K "
                "most relevant documents for the query. Use this as the first search "
                "step, then evaluate results and decide if you need synonym expansion "
                "or other strategies."
            ),
            category=ToolCategory.DATA,
            capability=ToolCapability.QUERY,
            module_name=_MODULE,
            version=_VERSION,
            annotations=_READ_ANNOTATIONS,
            input_schema=[
                ParameterSchema(name="session_id", type="string", description="Session ID"),
                ParameterSchema(name="query", type="string", description="Search query"),
                ParameterSchema(
                    name="top_k",
                    type="integer",
                    description="Max results (default 20)",
                    required=False,
                    constraints={"min": 1, "max": 100},
                ),
            ],
        )

    async def execute(self, inputs: dict[str, Any], context: Any = None) -> ToolResult:
        query = inputs.get("query", "")
        if not query:
            return ToolResult.create_error(
                "Missing 'query' parameter. "
                "Provide a search query string, e.g. {\"query\": \"key terms\"}. "
                "Alternative: use kaos-retrieval-synonyms to look up terms first."
            )

        top_k = int(inputs.get("top_k", 20))

        # Check for ContentDocumentCorpus in context (passage-level search)
        corpus = _get_corpus_from_context(context)
        if corpus is not None:
            return await self._search_corpus(query, corpus, top_k)

        # Fall back to session memory search
        memory, _store, err = _get_memory(inputs, context)
        if err:
            return ToolResult.create_error(err)

        from kaos_agents.memory.search import search_memory
        from kaos_agents.memory.types import MemoryType

        results = search_memory(
            memory, query, sections=[MemoryType.DOCUMENTS], top_k=top_k, expand_relations=[]
        )

        formatted = _format_results(results)
        topic_summary = _summarize_topics(results)

        # Compute expansion signal from score distribution
        expansion_signal = _assess_expansion_need(results, top_k)

        return ToolResult.create_success(
            output={
                "query": query,
                "result_count": len(results),
                "results": formatted,
                "topic_summary": topic_summary,
                "expansion_assessment": expansion_signal,
            },
            summary=(
                f"BM25 found {len(results)} documents for '{query[:50]}'. "
                f"{expansion_signal['recommendation']}. "
                f"Topics: {topic_summary[:100]}"
            ),
        )


    async def _search_corpus(
        self, query: str, corpus: Any, top_k: int
    ) -> ToolResult:
        """Search a ContentDocumentCorpus at passage level."""
        searcher, passage_meta = _get_corpus_searcher(corpus)
        hits = searcher.search(query, top_k=top_k)

        formatted = []
        for h in hits:
            meta = passage_meta[int(h.doc_id)]
            formatted.append({
                "doc_uri": meta["doc_uri"],
                "block_ref": meta["block_ref"],
                "page": meta["page"],
                "section": meta["section_title"],
                "score": round(h.score, 3),
                "preview": h.text[:300] if hasattr(h, "text") and h.text else "",
            })

        # Build topic summary from top results
        topic_parts = [
            f"{r['doc_uri']}:{r['block_ref']} ({r['score']})"
            for r in formatted[:5]
        ]
        topic_summary = " | ".join(topic_parts)

        expansion_signal = _assess_expansion_need(
            [type("R", (), {"score": h.score, "content": ""})() for h in hits],
            top_k,
        )

        return ToolResult.create_success(
            output={
                "query": query,
                "result_count": len(formatted),
                "results": formatted,
                "topic_summary": topic_summary,
                "expansion_assessment": expansion_signal,
                "search_level": "passage",
                "corpus_size": corpus.size,
            },
            summary=(
                f"BM25 found {len(formatted)} passages for '{query[:50]}' "
                f"(from {corpus.size} total passages). "
                f"{expansion_signal['recommendation']}"
            ),
        )


class SynonymSearchTool(KaosTool):
    """Look up synonyms via OpenGloss Lexicon and search with expanded terms."""

    @property
    def metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name="kaos-retrieval-synonyms",
            display_name="Synonym-Expanded Search",
            description=(
                "Look up synonyms, hypernyms, and inflections for query terms using "
                "the OpenGloss dictionary (205K entries), then search with expanded "
                "terms. Use this when BM25 misses documents that use different "
                "terminology for the same concept."
            ),
            category=ToolCategory.DATA,
            capability=ToolCapability.QUERY,
            module_name=_MODULE,
            version=_VERSION,
            annotations=_READ_ANNOTATIONS,
            input_schema=[
                ParameterSchema(name="session_id", type="string", description="Session ID"),
                ParameterSchema(
                    name="word", type="string", description="Word to look up synonyms for"
                ),
                ParameterSchema(
                    name="also_search",
                    type="boolean",
                    description="Also run BM25 search with the synonyms (default true)",
                    required=False,
                ),
            ],
        )

    async def execute(self, inputs: dict[str, Any], context: Any = None) -> ToolResult:
        word = inputs.get("word", "").strip()
        if not word:
            return ToolResult.create_error(
                "Missing 'word' parameter. "
                "Provide a term to look up synonyms for, e.g. {\"word\": \"indemnification\"}. "
                "Alternative: use kaos-retrieval-bm25 to search directly by keyword."
            )

        from kaos_agents.memory.search import _load_lexicon

        lexicon = _load_lexicon()
        if lexicon is None:
            return ToolResult.create_error(
                "OpenGloss Lexicon not available. "
                "Fix: run scripts/build_opengloss_lexicon.py to build it."
            )

        synonyms = lexicon.synonyms(word) if lexicon.contains(word) else []
        hypernyms = lexicon.hypernyms(word) if lexicon.contains(word) else []
        inflections = lexicon.inflections(word) if lexicon.contains(word) else []

        # Also try without hyphens
        dehyphenated = word.replace("-", "")
        if dehyphenated != word and lexicon.contains(dehyphenated):
            synonyms = synonyms or lexicon.synonyms(dehyphenated)
            hypernyms = hypernyms or lexicon.hypernyms(dehyphenated)

        output: dict[str, Any] = {
            "word": word,
            "synonyms": synonyms[:10],
            "hypernyms": hypernyms[:5],
            "inflections": inflections[:5],
            "found": lexicon.contains(word) or lexicon.contains(dehyphenated),
        }

        also_search = inputs.get("also_search", True)
        if also_search and synonyms:
            memory, _store, err = _get_memory(inputs, context)
            if not err and memory:
                from kaos_agents.memory.search import search_memory
                from kaos_agents.memory.types import MemoryType

                expanded_query = " ".join(synonyms[:5])
                results = search_memory(
                    memory,
                    expanded_query,
                    sections=[MemoryType.DOCUMENTS],
                    top_k=20,
                    expand_relations=[],
                )
                output["search_results"] = _format_results(results)
                output["search_query"] = expanded_query

        found_in_dict = lexicon.contains(word) or lexicon.contains(dehyphenated)
        summary_parts = [f"'{word}' — {'FOUND' if found_in_dict else 'NOT FOUND'} in dictionary"]
        if synonyms:
            summary_parts.append(f"synonyms: {', '.join(synonyms[:5])}")
        if hypernyms:
            summary_parts.append(f"hypernyms: {', '.join(hypernyms[:3])}")
        if not found_in_dict:
            summary_parts.append(
                "Try breaking compound terms into parts, or use kaos-retrieval-hyde instead"
            )
        return ToolResult.create_success(output=output, summary=" — ".join(summary_parts))


class HyDESearchTool(KaosTool):
    """Generate a hypothetical document passage, then BM25 search against it."""

    @property
    def metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name="kaos-retrieval-hyde",
            display_name="HyDE Pseudo-Document Search",
            description=(
                "Generate a hypothetical document passage that would answer the query "
                "using domain-specific vocabulary, then search for real documents that "
                "match. Use this when the query uses different terminology than the "
                "documents (e.g., consumer language vs. technical language)."
            ),
            category=ToolCategory.DATA,
            capability=ToolCapability.QUERY,
            module_name=_MODULE,
            version=_VERSION,
            annotations=_READ_ANNOTATIONS,
            input_schema=[
                ParameterSchema(name="session_id", type="string", description="Session ID"),
                ParameterSchema(name="query", type="string", description="Original search query"),
            ],
        )

    async def execute(self, inputs: dict[str, Any], context: Any = None) -> ToolResult:
        query = inputs.get("query", "")
        if not query:
            return ToolResult.create_error(
                "Missing 'query' parameter. "
                "Provide the original search query to generate a pseudo-document from, "
                "e.g. {\"query\": \"indemnification obligations\"}. "
                "Alternative: use kaos-retrieval-bm25 for direct keyword search."
            )

        from kaos_agents.context.retrieval import _generate_pseudo_document

        pseudo_doc = _generate_pseudo_document(query)
        if not pseudo_doc:
            return ToolResult.create_error(
                "Failed to generate pseudo-document. The LLM call may have timed out. "
                "Alternative: use kaos-retrieval-bm25 with manually expanded query terms."
            )

        memory, _store, err = _get_memory(inputs, context)
        if err:
            return ToolResult.create_error(err)

        from kaos_agents.memory.search import search_memory
        from kaos_agents.memory.types import MemoryType

        results = search_memory(
            memory,
            pseudo_doc,
            sections=[MemoryType.DOCUMENTS],
            top_k=20,
            expand_relations=[],
        )

        formatted = _format_results(results)
        return ToolResult.create_success(
            output={
                "query": query,
                "pseudo_document": pseudo_doc,
                "pseudo_document_length": len(pseudo_doc),
                "result_count": len(results),
                "results": formatted,
            },
            summary=(
                f"HyDE generated {len(pseudo_doc)}-char pseudo-document, "
                f"found {len(results)} docs. "
                f"Pseudo-doc preview: {pseudo_doc[:150]}"
            ),
        )


class EvaluateCoverageTool(KaosTool):
    """Judge whether accumulated results cover the query. Suggest gaps."""

    @property
    def metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name="kaos-retrieval-evaluate",
            display_name="Evaluate Retrieval Coverage",
            description=(
                "Evaluate whether the documents found so far adequately cover all "
                "aspects of the search query. Returns a coverage assessment and "
                "suggests specific follow-up queries for any gaps. Use this after "
                "each search round to decide whether to continue searching."
            ),
            category=ToolCategory.DATA,
            capability=ToolCapability.QUERY,
            module_name=_MODULE,
            version=_VERSION,
            annotations=_READ_ANNOTATIONS,
            input_schema=[
                ParameterSchema(name="query", type="string", description="Original search query"),
                ParameterSchema(
                    name="document_summaries",
                    type="string",
                    description="Summaries of documents found so far (paste from previous search results)",
                ),
            ],
        )

    async def execute(self, inputs: dict[str, Any], context: Any = None) -> ToolResult:
        query = inputs.get("query", "")
        summaries = inputs.get("document_summaries", "")
        if not query:
            return ToolResult.create_error(
                "Missing 'query' parameter. "
                "Provide the original search query to evaluate coverage against, "
                "e.g. {\"query\": \"termination clauses\"}. "
                "Alternative: use kaos-retrieval-bm25 to run a search first."
            )
        if not summaries:
            return ToolResult.create_error(
                "Missing 'document_summaries' parameter. "
                "Paste the summaries from previous search results so coverage can be evaluated. "
                "Alternative: run kaos-retrieval-bm25 first, then pass its topic_summary here."
            )

        from kaos_agents.context.retrieval import _reflect_on_coverage
        from kaos_agents.memory.search import MemorySearchResult
        from kaos_agents.memory.types import MemoryType

        fake_results = [
            MemorySearchResult(
                content=summaries, section=MemoryType.DOCUMENTS, score=1.0, item_id="eval"
            )
        ]
        gap_queries = _reflect_on_coverage(query, fake_results)

        if gap_queries:
            return ToolResult.create_success(
                output={
                    "sufficient": False,
                    "gap_count": len(gap_queries),
                    "gap_queries": gap_queries,
                    "recommendation": (
                        "Try BM25 search with the suggested queries. "
                        "If vocabulary mismatch is the issue, try HyDE."
                    ),
                },
                summary=(
                    f"INSUFFICIENT: {len(gap_queries)} gap(s) found. "
                    f"Suggested queries: {', '.join(gap_queries[:3])}"
                ),
            )
        return ToolResult.create_success(
            output={
                "sufficient": True,
                "gap_count": 0,
                "gap_queries": [],
                "recommendation": "Coverage is sufficient. Proceed with the documents found.",
            },
            summary="SUFFICIENT: Coverage appears adequate. You can stop searching.",
        )


_RERANK_PASSAGE_TRUNCATE = 2000  # Max chars per passage for cross-encoder
_RERANK_MIN_CANDIDATES = 5  # Don't rerank fewer than this


class RerankTool(KaosTool):
    """Rerank BM25 results using a cross-encoder model for improved precision."""

    @property
    def metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name="kaos-retrieval-rerank",
            display_name="Cross-Encoder Reranker",
            description=(
                "Rerank previously retrieved documents using a cross-encoder model "
                "that reads each (query, passage) pair with full attention. Much more "
                "accurate than BM25 scoring alone — promotes truly relevant documents "
                "from deeper in the results. Use this AFTER running BM25 search when "
                "you have 10+ candidates and want to improve precision. Requires the "
                "sentence-transformers backend; falls back to BM25 score ordering if "
                "not available."
            ),
            category=ToolCategory.DATA,
            capability=ToolCapability.QUERY,
            module_name=_MODULE,
            version=_VERSION,
            annotations=_READ_ANNOTATIONS,
            input_schema=[
                ParameterSchema(name="session_id", type="string", description="Session ID"),
                ParameterSchema(name="query", type="string", description="Search query to rerank against"),
                ParameterSchema(
                    name="top_k",
                    type="integer",
                    description="Number of top results to return after reranking (default 20)",
                    required=False,
                    constraints={"min": 1, "max": 100},
                ),
            ],
        )

    async def execute(self, inputs: dict[str, Any], context: Any = None) -> ToolResult:
        query = inputs.get("query", "")
        if not query:
            return ToolResult.create_error(
                "Missing 'query' parameter. "
                "Provide the search query to rerank against, e.g. {\"query\": \"liability cap\"}. "
                "Alternative: use kaos-retrieval-bm25 for keyword search without reranking."
            )

        memory, _store, err = _get_memory(inputs, context)
        if err:
            return ToolResult.create_error(err)

        # First run BM25 to get candidates (wide net: 100 results)
        from kaos_agents.memory.search import search_memory
        from kaos_agents.memory.types import MemoryType

        candidates = search_memory(
            memory, query, sections=[MemoryType.DOCUMENTS], top_k=100, expand_relations=[]
        )

        if len(candidates) < _RERANK_MIN_CANDIDATES:
            formatted = _format_results(candidates)
            return ToolResult.create_success(
                output={
                    "query": query,
                    "reranked": False,
                    "reason": f"Too few candidates ({len(candidates)}) for reranking",
                    "result_count": len(candidates),
                    "results": formatted,
                },
                summary=f"Only {len(candidates)} candidates — returned BM25 order (reranking needs {_RERANK_MIN_CANDIDATES}+)",
            )

        # Try cross-encoder reranking
        try:
            from kaos_nlp_core.retrieval.protocol import RetrievalResult
            from kaos_nlp_transformers.reranker import CrossEncoderReranker

            reranker = CrossEncoderReranker.load()

            retrieval_results = [
                RetrievalResult(
                    text=c.content[:_RERANK_PASSAGE_TRUNCATE],
                    score=c.score,
                    doc_id=c.item_id,
                )
                for c in candidates
            ]

            top_k = int(inputs.get("top_k", 20))
            ranked = await reranker.rerank(query, retrieval_results, top_k=top_k)

            formatted = []
            for r in ranked:
                formatted.append({
                    "item_id": r.result.doc_id,
                    "bm25_score": round(r.result.score, 3),
                    "rerank_score": round(r.rerank_score, 3),
                    "preview": r.result.text[:300],
                })

            return ToolResult.create_success(
                output={
                    "query": query,
                    "reranked": True,
                    "method": "cross-encoder",
                    "model": reranker.model_id,
                    "candidates": len(candidates),
                    "result_count": len(ranked),
                    "results": formatted,
                },
                summary=(
                    f"Cross-encoder reranked {len(candidates)} candidates → top {len(ranked)}. "
                    f"Top score: {ranked[0].rerank_score:.3f}" if ranked else "No results"
                ),
            )

        except ImportError:
            # Fall back to BM25 order with a note
            top_k = int(inputs.get("top_k", 20))
            formatted = _format_results(candidates[:top_k])
            return ToolResult.create_success(
                output={
                    "query": query,
                    "reranked": False,
                    "reason": "Cross-encoder not available (install kaos-nlp-transformers[torch])",
                    "result_count": len(formatted),
                    "results": formatted,
                },
                summary=(
                    f"Cross-encoder not available — returning BM25 top-{top_k}. "
                    "Install kaos-nlp-transformers[torch] for cross-encoder reranking."
                ),
            )

        except Exception as exc:
            logger.warning("rerank_tool: cross-encoder failed: %s", exc)
            top_k = int(inputs.get("top_k", 20))
            formatted = _format_results(candidates[:top_k])
            return ToolResult.create_success(
                output={
                    "query": query,
                    "reranked": False,
                    "reason": f"Cross-encoder error: {exc}",
                    "result_count": len(formatted),
                    "results": formatted,
                },
                summary=f"Cross-encoder failed ({exc}) — returning BM25 order",
            )


_CORPUS_INFO_TOP_TERMS = 20
_CORPUS_INFO_MAX_URIS = 30


class CorpusInfoTool(KaosTool):
    """Inspect the document corpus — size, URIs, vocabulary, statistics."""

    @property
    def metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name="kaos-retrieval-corpus-info",
            display_name="Corpus Information",
            description=(
                "Inspect the loaded document corpus. Returns document count, total "
                "characters, average document length, top terms by frequency, and "
                "document URIs. Use this BEFORE searching to understand what kind of "
                "corpus you're working with — it helps you choose the right retrieval "
                "strategy (e.g., whether synonym expansion might help or hurt)."
            ),
            category=ToolCategory.DATA,
            capability=ToolCapability.QUERY,
            module_name=_MODULE,
            version=_VERSION,
            annotations=_READ_ANNOTATIONS,
            input_schema=[
                ParameterSchema(name="session_id", type="string", description="Session ID"),
            ],
        )

    async def execute(self, inputs: dict[str, Any], context: Any = None) -> ToolResult:
        memory, _store, err = _get_memory(inputs, context)
        if err:
            return ToolResult.create_error(err)

        from collections import Counter

        from kaos_agents.memory.types import MemoryType

        if not memory.has_section(MemoryType.DOCUMENTS):
            return ToolResult.create_success(
                output={"document_count": 0, "total_chars": 0},
                summary="No documents loaded. Use /load or kaos-agent-chat to add documents first.",
            )

        items = memory.get(MemoryType.DOCUMENTS)
        n_docs = len(items)
        total_chars = sum(len(item.content) for item in items)
        avg_chars = total_chars // n_docs if n_docs else 0

        # Extract URIs
        uris = []
        for item in items[:_CORPUS_INFO_MAX_URIS]:
            uri = (item.metadata or {}).get("uri", item.id[:8])
            uris.append(uri)

        # Compute top terms by frequency (simple word count)
        word_counts: Counter[str] = Counter()
        for item in items:
            words = item.content.lower().split()
            word_counts.update(w for w in words if len(w) >= 3)
        top_terms = [term for term, _ in word_counts.most_common(_CORPUS_INFO_TOP_TERMS)]

        output = {
            "document_count": n_docs,
            "total_chars": total_chars,
            "avg_chars_per_doc": avg_chars,
            "top_terms": top_terms,
            "document_uris": uris,
            "has_more_uris": n_docs > _CORPUS_INFO_MAX_URIS,
        }

        return ToolResult.create_success(
            output=output,
            summary=(
                f"{n_docs} documents, {total_chars:,} total chars "
                f"(avg {avg_chars:,}/doc). "
                f"Top terms: {', '.join(top_terms[:5])}"
            ),
        )


class GroundedAnswerTool(KaosTool):
    """Generate a grounded, cited answer from passages found by previous searches.

    Call this AFTER you have found relevant passages via BM25/HyDE/synonym search.
    Provide the question and the passage texts. The tool uses RAG to generate a
    verified, cited answer or explicitly refuses if evidence is insufficient.
    """

    @property
    def metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name="kaos-retrieval-answer",
            display_name="Generate Grounded Answer",
            description=(
                "Generate a cited answer from passages you have found. Provide the "
                "question and paste the relevant passage texts. The tool verifies "
                "citations against the source text and returns a grounded answer. "
                "If the passages don't contain enough evidence, it returns "
                "'insufficient evidence' with a hint about what to search for next. "
                "Use this AFTER searching — not as the first step."
            ),
            category=ToolCategory.DATA,
            capability=ToolCapability.QUERY,
            module_name=_MODULE,
            version=_VERSION,
            annotations=_READ_ANNOTATIONS,
            input_schema=[
                ParameterSchema(
                    name="question",
                    type="string",
                    description="The user's question to answer",
                ),
                ParameterSchema(
                    name="passages",
                    type="string",
                    description="The passage texts to answer from (paste from previous search results)",
                ),
            ],
        )

    async def execute(self, inputs: dict[str, Any], context: Any = None) -> ToolResult:
        question = inputs.get("question", "")
        passages_text = inputs.get("passages", "")
        if not question:
            return ToolResult.create_error(
                "Missing 'question'. Provide the question to answer. "
                "Alternative: use kaos-retrieval-bm25 to search for relevant passages first."
            )
        if not passages_text:
            return ToolResult.create_error(
                "Missing 'passages'. Paste the passage texts from your search results. "
                "Alternative: run kaos-retrieval-bm25 first, then pass the results here."
            )

        try:
            from kaos_agents._llm_imports import require_llm_core

            require_llm_core()
            from kaos_llm_core.programs.rag import RAG
            from kaos_llm_core.signatures.grounding import (
                Answer,
                InsufficientEvidence,
                MatchStrategy,
            )

            from kaos_agents.settings import DEFAULT_MODEL

            # Build a mini-corpus from the provided passages
            corpus = {"passage": passages_text}

            rag = RAG(
                model=DEFAULT_MODEL,
                top_k=5,
                max_retries=1,
                match_strategies=(
                    MatchStrategy.STRICT,
                    MatchStrategy.SUBSTRING,
                    MatchStrategy.CASE_INSENSITIVE,
                    MatchStrategy.NORMALIZED_TOKEN,
                ),
            )
            result = await rag.query(question=question, documents=corpus)

            if isinstance(result.grounded_answer, Answer):
                answer = result.grounded_answer
                claims = [
                    {
                        "statement": c.statement,
                        "confidence": c.confidence,
                        "sources": [s.source_uri for s in c.supporting_spans],
                    }
                    for c in answer.claims
                ]
                return ToolResult.create_success(
                    output={
                        "answered": True,
                        "answer": str(answer.value),
                        "claims": claims,
                        "verified": result.is_verified,
                    },
                    summary=f"Answer: {str(answer.value)[:200]}",
                )

            if isinstance(result.grounded_answer, InsufficientEvidence):
                refusal = result.grounded_answer
                return ToolResult.create_success(
                    output={
                        "answered": False,
                        "reason": refusal.reason,
                        "what_would_resolve": refusal.what_would_resolve or "",
                    },
                    summary=(
                        f"Insufficient evidence: {refusal.reason[:100]}. "
                        f"Try searching for: {refusal.what_would_resolve or 'more specific terms'}"
                    ),
                )

            return ToolResult.create_success(
                output={"answered": False, "reason": "Unexpected RAG result type"},
                summary="RAG returned unexpected result type",
            )

        except ImportError:
            return ToolResult.create_error(
                "kaos-llm-core not installed. "
                "Fix: install with pip install kaos-agents[llm]. "
                "Alternative: synthesize the answer manually from the passage texts."
            )
        except Exception as exc:
            logger.warning("grounded_answer_tool: RAG failed: %s", exc)
            return ToolResult.create_error(
                f"Answer generation failed: {exc}. "
                "Try providing fewer or more focused passages. "
                "Alternative: answer based on the passage texts directly."
            )


def register_retrieval_tools(runtime: Any) -> int:
    """Register all retrieval tools with a KaosRuntime."""
    tools: list[KaosTool] = [
        BM25SearchTool(),
        SynonymSearchTool(),
        HyDESearchTool(),
        EvaluateCoverageTool(),
        RerankTool(),
        CorpusInfoTool(),
        GroundedAnswerTool(),
    ]
    for tool in tools:
        runtime.tools.register_tool(tool)
    return len(tools)
