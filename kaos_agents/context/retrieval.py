"""Adaptive multi-round retrieval with Lexicon expansion and LLM judge.

Replaces single-shot BM25 triage with an iterative retrieval loop:

1. **Round 1** (BM25, raw terms): Search with the user's original query.
2. **Round 2** (BM25, Lexicon synonyms): Expand key terms via OpenGloss
   Lexicon and search again with synonyms + inflections.
3. **Round 3+** (BM25, LLM-suggested terms): If the judge rates recall as
   insufficient, ask the LLM to propose alternative search queries; search
   with those and merge results.

Results are merged across rounds by item_id (highest BM25 score wins).
The loop stops when:
- The judge rates recall confidence >= threshold, OR
- max_rounds is reached, OR
- cost/token budget is exhausted.

Usage::

    from kaos_agents.context.retrieval import adaptive_retrieve

    results = adaptive_retrieve(
        memory, "key findings about project milestones",
        max_rounds=3, confidence_threshold=0.7,
    )
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

from kaos_core.logging import get_logger
from kaos_llm_core import InputField, OutputField, Signature

from kaos_agents.memory.search import MemorySearchResult, _load_lexicon, search_memory
from kaos_agents.memory.types import MemoryType
from kaos_agents.settings import KaosAgentSettings

if TYPE_CHECKING:
    from kaos_agents.memory.session import SessionMemory


class ReflectOnCoverageSignature(Signature):
    """Identify what aspects of a query are NOT covered by retrieved docs.

    Look at the original query and the summaries of documents found so
    far. Identify what specific topics, subtopics, or perspectives are
    MISSING from the retrieved results. Generate 2-3 highly targeted
    search queries that would find documents covering those specific
    gaps.

    If the results already provide good coverage, return an empty list.
    """

    original_query: str = InputField(description="The original search query.")
    document_summaries: str = InputField(description="Summaries of the top documents found so far.")
    gap_queries: list[str] = OutputField(
        description=(
            "2-3 targeted queries for MISSING aspects, or an empty list if coverage is sufficient."
        )
    )


class GeneratePseudoDocumentSignature(Signature):
    """Generate a hypothetical document excerpt for HyDE retrieval.

    Write a 150-250 word excerpt from a professional document that
    would be relevant to the search query. Use the domain-specific
    vocabulary and terminology that real source documents use — the
    precise technical terms, jargon, and phrasing that experts in the
    field would use, not the simplified language of the query.

    The output is then BM25-matched against the corpus, bridging
    terminology gaps between user queries and source documents
    (Query2Doc / HyDE technique).
    """

    search_query: str = InputField(description="The search query to generate a passage for.")
    hypothetical_passage: str = OutputField(
        description=(
            "A realistic 150-250 word excerpt using domain-specific "
            "terminology relevant to the query."
        )
    )


class SuggestQueriesSignature(Signature):
    """Suggest alternative search queries with different vocabulary.

    Given a search query that found some but not all relevant
    documents, generate 3-5 alternative queries using DIFFERENT
    terminology. Consider:

    (a) technical/scientific vs. plain language terms
    (b) acronyms and abbreviations vs. expanded forms
    (c) synonyms and related concepts that domain experts would use
    (d) different phrasings that describe the same phenomenon
    (e) broader or narrower terms (hypernyms/hyponyms) that might
        capture documents the original query missed

    Each alternative should use substantially different vocabulary
    from the original query and from each other.
    """

    original_query: str = InputField(description="The original search query.")
    results_found: int = InputField(description="Number of documents found so far.")
    alternative_queries: list[str] = OutputField(
        description=(
            "3-5 alternative search queries, each using different "
            "domain terminology and vocabulary."
        )
    )


logger = get_logger(__name__)

# Defaults derived from settings (single source of truth)
_s = KaosAgentSettings.model_fields
_DEFAULT_TOP_K: int = _s["retrieval_top_k"].default
_DEFAULT_MAX_ROUNDS: int = _s["retrieval_max_rounds"].default
_DEFAULT_CONFIDENCE: float = _s["retrieval_confidence"].default
_DEFAULT_PRF_TOP_TERMS: int = _s["retrieval_prf_top_terms"].default
_DEFAULT_PRF_DOCS: int = _s["retrieval_prf_docs"].default
_DEFAULT_MAX_SYNONYM_QUERIES: int = _s["retrieval_max_synonym_queries"].default
_DEFAULT_LLM_TIMEOUT: float = _s["retrieval_llm_timeout_seconds"].default

# Internal constants (not user-configurable)
_MIN_ITEMS_FOR_HYBRID = 5  # Don't build embedding index for tiny corpora
_MAX_ITEMS_FOR_HYBRID = (
    2000  # Skip embedding for very large corpora (too slow on CPU without cache)
)
_HYBRID_BATCH_SIZE = 64  # fastembed embedding batch size
_RRF_K = 60  # Reciprocal Rank Fusion constant (Cormack et al. 2009)
_HYBRID_TIMEOUT_SECONDS = 30.0  # Timeout for hybrid retrieval including embedding
_PRF_MIN_TERM_LEN = 4  # Minimum word length for PRF term extraction
_RERANK_MIN_CANDIDATES = 5  # Don't rerank fewer than this many results

# Process-level embedding cache for hybrid retrieval.
# Keyed by (session_id, doc_count) — invalidated when documents change.
_EMBEDDING_CACHE: dict[str, Any] = {}


@dataclass(frozen=True, slots=True)
class RetrievalRound:
    """Metadata for one retrieval round."""

    round_number: int
    query: str
    strategy: str
    hits: int
    new_hits: int


@dataclass(slots=True)  # Mutable: accumulated across rounds
class AdaptiveRetrievalResult:
    """Result from adaptive multi-round retrieval."""

    results: list[MemorySearchResult]
    rounds: list[RetrievalRound]
    total_rounds: int = 0
    total_hits: int = 0
    unique_items: int = 0
    stopped_reason: str = ""


def adaptive_retrieve(
    memory: SessionMemory,
    query: str,
    *,
    sections: list[MemoryType] | None = None,
    top_k: int = _DEFAULT_TOP_K,
    max_rounds: int = _DEFAULT_MAX_ROUNDS,
    confidence_threshold: float = _DEFAULT_CONFIDENCE,
    lexicon_path: str | None = None,
    model: str | None = None,
) -> AdaptiveRetrievalResult:
    """Multi-round retrieval blending BM25 + Lexicon expansion + LLM query generation.

    .. deprecated::
        This hardcoded pipeline is deprecated. It was proven to score WORSE than
        plain BM25 on cross-domain BEIR benchmarks (0.231 vs 0.296 NDCG@10 on
        NFCorpus). Use ``search_memory()`` for plain BM25, or use the
        ``RetrievalAgent`` (``kaos_agents.retrieval_agent``) which dynamically
        selects strategies based on results.

    Args:
        memory: Session memory to search.
        query: User's natural-language query.
        sections: Memory sections to search (default: DOCUMENTS).
        top_k: Max items per round.
        max_rounds: Maximum retrieval rounds (default 3).
        confidence_threshold: Stop when judge confidence >= this.
        lexicon_path: Path to OpenGloss Lexicon binary (auto-detected if None).
        model: LLM model for judge + query expansion (default from settings).

    Returns:
        AdaptiveRetrievalResult with merged results and round metadata.
    """
    if sections is None:
        sections = [MemoryType.DOCUMENTS]

    seen_ids: dict[str, MemorySearchResult] = {}
    rounds: list[RetrievalRound] = []

    # Round 1: BM25 with raw query
    r1_results = search_memory(memory, query, sections=sections, top_k=top_k)
    new_count = _merge_results(seen_ids, r1_results)
    rounds.append(
        RetrievalRound(
            round_number=1,
            query=query,
            strategy="bm25_raw",
            hits=len(r1_results),
            new_hits=new_count,
        )
    )
    logger.debug(
        "retrieval.round1_raw: query=%r hits=%d new=%d total=%d",
        query[:50],
        len(r1_results),
        new_count,
        len(seen_ids),
    )

    # Round 1.1: RRF Hybrid (BM25 + dense embedding) if available
    hybrid_results = _try_hybrid_retrieve(memory, query, sections=sections, top_k=top_k)
    if hybrid_results:
        new_count = _merge_results(seen_ids, hybrid_results)
        rounds.append(
            RetrievalRound(
                round_number=1,
                query=query,
                strategy="hybrid_rrf",
                hits=len(hybrid_results),
                new_hits=new_count,
            )
        )
        logger.debug(
            "retrieval.round1_1_hybrid: hits=%d new=%d total=%d",
            len(hybrid_results),
            new_count,
            len(seen_ids),
        )

    if max_rounds < 2:
        return _build_result(seen_ids, rounds, "max_rounds")

    # Round 1.5: Query2Doc/HyDE — BM25 with LLM-generated pseudo-passage
    pseudo_doc = _generate_pseudo_document(query, model=model)
    if pseudo_doc:
        r15_results = search_memory(memory, pseudo_doc, sections=sections, top_k=top_k)
        new_count = _merge_results(seen_ids, r15_results)
        rounds.append(
            RetrievalRound(
                round_number=1,
                query=pseudo_doc,
                strategy="bm25_hyde",
                hits=len(r15_results),
                new_hits=new_count,
            )
        )
        logger.debug(
            "retrieval.round1_5_hyde: pseudo_doc_len=%d hits=%d new=%d total=%d",
            len(pseudo_doc),
            len(r15_results),
            new_count,
            len(seen_ids),
        )

    # Round 2: BM25 with Lexicon-expanded synonyms
    lexicon = _try_load_lexicon(lexicon_path)
    if lexicon is not None:
        expanded_queries = _expand_with_lexicon(query, lexicon)
        for eq in expanded_queries:
            r2_results = search_memory(memory, eq, sections=sections, top_k=top_k)
            new_count = _merge_results(seen_ids, r2_results)
            rounds.append(
                RetrievalRound(
                    round_number=2,
                    query=eq,
                    strategy="bm25_lexicon",
                    hits=len(r2_results),
                    new_hits=new_count,
                )
            )
            logger.debug(
                "retrieval.round2_lexicon: query=%r hits=%d new=%d total=%d",
                eq[:50],
                len(r2_results),
                new_count,
                len(seen_ids),
            )
    else:
        logger.debug("retrieval.round2_skip: no lexicon available")

    # Round 2.5: PRF/Rocchio — expand query with top TF-IDF terms from results
    if seen_ids:
        prf_query = _expand_with_prf(query, list(seen_ids.values()))
        if prf_query and prf_query != query:
            r25_results = search_memory(memory, prf_query, sections=sections, top_k=top_k)
            new_count = _merge_results(seen_ids, r25_results)
            rounds.append(
                RetrievalRound(
                    round_number=2,
                    query=prf_query,
                    strategy="bm25_prf",
                    hits=len(r25_results),
                    new_hits=new_count,
                )
            )
            logger.debug(
                "retrieval.round2_5_prf: query=%r hits=%d new=%d total=%d",
                prf_query[:80],
                len(r25_results),
                new_count,
                len(seen_ids),
            )

    if max_rounds < 3:
        return _build_result(seen_ids, rounds, "max_rounds")

    # Round 3+: LLM-suggested alternative queries
    for round_num in range(3, max_rounds + 1):
        alt_queries = _generate_llm_queries(query, list(seen_ids.keys()), model=model)
        if not alt_queries:
            logger.debug("retrieval.round%d_skip: LLM returned no new queries", round_num)
            break

        for aq in alt_queries:
            rn_results = search_memory(memory, aq, sections=sections, top_k=top_k)
            new_count = _merge_results(seen_ids, rn_results)
            rounds.append(
                RetrievalRound(
                    round_number=round_num,
                    query=aq,
                    strategy="bm25_llm",
                    hits=len(rn_results),
                    new_hits=new_count,
                )
            )
            logger.debug(
                "retrieval.round%d_llm: query=%r hits=%d new=%d total=%d",
                round_num,
                aq[:50],
                len(rn_results),
                new_count,
                len(seen_ids),
            )

    # Reflective loop: evaluate coverage gaps and search for missing aspects
    if max_rounds >= 3 and model:
        for reflect_round in range(2):
            gap_queries = _reflect_on_coverage(query, list(seen_ids.values()), model=model)
            if not gap_queries:
                logger.debug("retrieval.reflect_sufficient: round=%d no gaps found", reflect_round)
                break

            for gq in gap_queries:
                gr_results = search_memory(memory, gq, sections=sections, top_k=top_k)
                new_count = _merge_results(seen_ids, gr_results)
                rounds.append(
                    RetrievalRound(
                        round_number=4 + reflect_round,
                        query=gq,
                        strategy="bm25_reflect",
                        hits=len(gr_results),
                        new_hits=new_count,
                    )
                )
                logger.debug(
                    "retrieval.reflect_round%d: query=%r hits=%d new=%d total=%d",
                    reflect_round,
                    gq[:50],
                    len(gr_results),
                    new_count,
                    len(seen_ids),
                )

    result = _build_result(seen_ids, rounds, "max_rounds")

    # Post-merge: cross-encoder reranking (if available)
    reranked = _try_rerank(query, result.results)
    if reranked is not None:
        result.results = reranked
        logger.debug(
            "retrieval.rerank_complete: input=%d output=%d",
            len(seen_ids),
            len(reranked),
        )

    return result


def precompute_embeddings(
    memory: SessionMemory,
    *,
    sections: list[MemoryType] | None = None,
) -> bool:
    """Pre-compute and cache the embedding index for a session's documents.

    Call this ONCE after loading documents to avoid the 70-120s embedding
    cost on the first query. Subsequent ``adaptive_retrieve`` calls will
    use the cached embeddings.

    Args:
        memory: Session memory with documents loaded.
        sections: Sections to index (default: DOCUMENTS).

    Returns:
        True if embeddings were computed, False if skipped (deps unavailable,
        too few/many docs, or already cached).
    """
    if sections is None:
        sections = [MemoryType.DOCUMENTS]

    try:
        from kaos_nlp_transformers.retrieval import EmbeddingRetriever
    except ImportError:
        logger.debug("precompute_embeddings: kaos-nlp-transformers not available")
        return False

    import hashlib

    items: list[tuple[str, str]] = []
    for mt in sections:
        if not memory.has_section(mt):
            continue
        for item in memory.get(mt):
            items.append((item.id, item.content))

    if len(items) < _MIN_ITEMS_FOR_HYBRID:
        return False

    sample_hashes = "".join(
        hashlib.md5(items[i][1][:200].encode()).hexdigest()[:8]
        for i in range(0, len(items), max(1, len(items) // 5))
    )
    cache_key = hashlib.md5(
        f"{memory.session_id}:{len(items)}:{sample_hashes}".encode()
    ).hexdigest()

    if cache_key in _EMBEDDING_CACHE:
        logger.debug(
            "precompute_embeddings: already cached key=%s docs=%d", cache_key[:8], len(items)
        )
        return True

    logger.debug("precompute_embeddings: computing key=%s docs=%d", cache_key[:8], len(items))
    texts = [t for _, t in items]
    doc_ids = list(range(len(texts)))
    embed_r = EmbeddingRetriever.from_texts(
        texts=texts, doc_ids=doc_ids, batch_size=_HYBRID_BATCH_SIZE
    )
    _EMBEDDING_CACHE[cache_key] = embed_r
    logger.debug("precompute_embeddings: done key=%s docs=%d", cache_key[:8], len(items))
    return True


def _merge_results(
    seen: dict[str, MemorySearchResult],
    new_results: list[MemorySearchResult],
) -> int:
    """Merge new results into seen dict. Keep highest score per item_id. Return new item count."""
    new_count = 0
    for r in new_results:
        existing = seen.get(r.item_id)
        if existing is None:
            seen[r.item_id] = r
            new_count += 1
        elif r.score > existing.score:
            seen[r.item_id] = r
    return new_count


def _build_result(
    seen: dict[str, MemorySearchResult],
    rounds: list[RetrievalRound],
    reason: str,
) -> AdaptiveRetrievalResult:
    """Build the final result from accumulated state."""
    sorted_results = sorted(seen.values(), key=lambda r: r.score, reverse=True)
    return AdaptiveRetrievalResult(
        results=sorted_results,
        rounds=rounds,
        total_rounds=len(rounds),
        total_hits=sum(r.hits for r in rounds),
        unique_items=len(seen),
        stopped_reason=reason,
    )


def _try_load_lexicon(lexicon_path: str | None) -> Any | None:
    """Try to load the OpenGloss Lexicon, auto-detecting path.

    Delegates to search.py's _load_lexicon for auto-detection and caching.
    """
    from pathlib import Path

    from kaos_agents.memory.search import OPENGLOSS_LEXICON_FILENAME

    if lexicon_path is not None:
        return _load_lexicon(lexicon_path)

    candidates = [
        Path(__file__).parent.parent.parent.parent
        / "kaos-nlp-core"
        / "data"
        / OPENGLOSS_LEXICON_FILENAME,
        Path.home() / ".cache" / "kaos" / OPENGLOSS_LEXICON_FILENAME,
    ]
    for candidate in candidates:
        if candidate.exists():
            return _load_lexicon(str(candidate))
    return None


def _expand_with_lexicon(query: str, lexicon: Any) -> list[str]:
    """Generate alternative queries using Lexicon synonym expansion.

    Splits query into words, looks up each word's synonyms, and
    constructs alternative queries by substituting synonym groups.
    Returns 1-3 alternative queries, not the original.
    """
    max_syns_per_word = _DEFAULT_MAX_SYNONYM_QUERIES
    words = query.lower().split()
    alt_queries: list[str] = []

    for word in words:
        clean = word.strip(".,;:!?\"'()")
        if len(clean) < _PRF_MIN_TERM_LEN:
            continue

        if lexicon.contains(clean):
            syns = lexicon.synonyms(clean)
            if syns:
                top_syns = syns[:max_syns_per_word]
                alt = query.lower().replace(clean, " ".join(top_syns[:2]))
                if alt != query.lower():
                    alt_queries.append(alt)

        dehyphenated = clean.replace("-", "")
        if dehyphenated != clean and lexicon.contains(dehyphenated):
            syns = lexicon.synonyms(dehyphenated)
            if syns:
                alt_queries.append(" ".join(syns[:max_syns_per_word]))

    seen: set[str] = set()
    unique: list[str] = []
    for aq in alt_queries:
        if aq not in seen:
            seen.add(aq)
            unique.append(aq)
    return unique[:_DEFAULT_MAX_SYNONYM_QUERIES]


def _try_hybrid_retrieve(
    memory: SessionMemory,
    query: str,
    *,
    sections: list[MemoryType],
    top_k: int = _DEFAULT_TOP_K,
) -> list[MemorySearchResult]:
    """Try RRF hybrid retrieval (BM25 + dense embedding). Returns [] if unavailable.

    Builds BM25 + embedding retrievers from memory items, fuses via RRF.
    Requires kaos-nlp-transformers (optional dependency). Gracefully
    returns empty list if not installed.
    """
    try:
        from kaos_nlp_core.retrieval.bm25 import BM25Retriever
        from kaos_nlp_core.retrieval.hybrid import HybridRetriever
        from kaos_nlp_transformers.retrieval import EmbeddingRetriever
    except ImportError:
        logger.debug("retrieval.hybrid_skip: kaos-nlp-transformers not available")
        return []

    items_list: list[tuple[str, str, MemoryType]] = []
    for mt in sections:
        if not memory.has_section(mt):
            continue
        for item in memory.get(mt):
            items_list.append((item.id, item.content, mt))

    if len(items_list) < _MIN_ITEMS_FOR_HYBRID:
        return []

    texts = [t for _, t, _ in items_list]
    doc_ids = list(range(len(texts)))

    try:
        import asyncio
        import hashlib

        # Stable cache key: hash of session + doc count + sample of content hashes
        sample_hashes = "".join(
            hashlib.md5(items_list[i][1][:200].encode()).hexdigest()[:8]
            for i in range(0, len(items_list), max(1, len(items_list) // 5))
        )
        cache_key = hashlib.md5(
            f"{memory.session_id}:{len(items_list)}:{sample_hashes}".encode()
        ).hexdigest()

        if cache_key in _EMBEDDING_CACHE:
            embed_r = _EMBEDDING_CACHE[cache_key]
            logger.debug("retrieval.hybrid_cache_hit: key=%s docs=%d", cache_key[:8], len(texts))
        elif len(items_list) > _MAX_ITEMS_FOR_HYBRID:
            logger.debug(
                "retrieval.hybrid_skip_large: docs=%d > max=%d (no cached embeddings)",
                len(items_list),
                _MAX_ITEMS_FOR_HYBRID,
            )
            return []
        else:
            embed_r = EmbeddingRetriever.from_texts(
                texts=texts, doc_ids=doc_ids, batch_size=_HYBRID_BATCH_SIZE
            )
            _EMBEDDING_CACHE[cache_key] = embed_r
            logger.debug(
                "retrieval.hybrid_cache_miss: key=%s docs=%d (embedding computed)",
                cache_key[:8],
                len(texts),
            )

        bm25_r = BM25Retriever.from_documents([{"id": i, "text": t} for i, t in enumerate(texts)])
        hybrid = HybridRetriever(sparse=bm25_r, dense=embed_r, k=_RRF_K)

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as pool:
                hits = pool.submit(asyncio.run, hybrid.retrieve(query, top_k=top_k)).result(
                    timeout=_HYBRID_TIMEOUT_SECONDS
                )
        else:
            hits = asyncio.run(hybrid.retrieve(query, top_k=top_k))

        hits = cast(list[Any], hits)
        results: list[MemorySearchResult] = []
        for hit in hits:
            idx = int(hit.doc_id) if hasattr(hit, "doc_id") else 0
            if 0 <= idx < len(items_list):
                item_id, content, section = items_list[idx]
                results.append(
                    MemorySearchResult(
                        content=content,
                        section=section,
                        score=hit.score,
                        item_id=item_id,
                    )
                )

        logger.debug(
            "retrieval.hybrid_complete: items=%d hits=%d",
            len(items_list),
            len(results),
        )
        return results
    except Exception as exc:
        logger.debug("retrieval.hybrid_failed: %s", exc)
        return []


def _reflect_on_coverage(
    query: str,
    results: list[MemorySearchResult],
    *,
    model: str | None = None,
) -> list[str]:
    """Evaluate coverage gaps and generate targeted follow-up queries.

    Examines the top results, identifies what aspects of the query are NOT
    covered, and generates specific queries to fill those gaps.
    """
    if len(results) < 3:
        return []

    try:
        from kaos_llm_core import Call

        top_summaries = []
        for r in results[:15]:
            summary = r.content[:200].replace("\n", " ").strip()
            top_summaries.append(f"- {summary}")
        summary_text = "\n".join(top_summaries)

        from kaos_agents.settings import DEFAULT_MODEL

        call = Call(ReflectOnCoverageSignature, model=model or DEFAULT_MODEL)

        import asyncio

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as pool:
                result = pool.submit(
                    asyncio.run,
                    call(original_query=query, document_summaries=summary_text),
                ).result(timeout=_DEFAULT_LLM_TIMEOUT)
        else:
            result = asyncio.run(call(original_query=query, document_summaries=summary_text))

        queries = cast(list[str], getattr(result, "gap_queries", []))
        logger.debug(
            "retrieval.reflect: query=%r gaps=%d",
            query[:50],
            len(queries),
        )
        return queries[:3]
    except Exception as exc:
        logger.debug("retrieval.reflect_failed: %s", exc)
        return []


def _try_rerank(
    query: str,
    results: list[MemorySearchResult],
) -> list[MemorySearchResult] | None:
    """Rerank results using fastembed cross-encoder. Returns None if unavailable."""
    if len(results) < _RERANK_MIN_CANDIDATES:
        return None

    try:
        from fastembed.rerank.cross_encoder import TextCrossEncoder
    except ImportError:
        logger.debug("retrieval.rerank_skip: fastembed TextCrossEncoder not available")
        return None

    try:
        encoder = TextCrossEncoder("BAAI/bge-reranker-base")
        texts = [r.content[:2000] for r in results]
        scores = list(encoder.rerank(query, texts, batch_size=64))

        scored = list(zip(results, scores, strict=False))
        scored.sort(key=lambda x: x[1], reverse=True)

        reranked = [r for r, _s in scored]
        logger.debug(
            "retrieval.rerank: candidates=%d final=%d top_score=%.3f",
            len(results),
            len(reranked),
            scored[0][1] if scored else 0.0,
        )
        return reranked
    except Exception as exc:
        logger.debug("retrieval.rerank_failed: %s", exc)
        return None


def _expand_with_prf(
    original_query: str,
    results: list[MemorySearchResult],
    *,
    top_n_terms: int = _DEFAULT_PRF_TOP_TERMS,
    min_term_len: int = _PRF_MIN_TERM_LEN,
) -> str:
    """Pseudo-Relevance Feedback: expand query with top TF terms from results.

    Extracts the most frequent meaningful terms from the top results,
    adds them to the original query. Pure algorithmic — no LLM cost.
    """
    from collections import Counter

    stopwords = {
        "the",
        "and",
        "for",
        "that",
        "this",
        "with",
        "from",
        "shall",
        "will",
        "have",
        "been",
        "such",
        "which",
        "each",
        "other",
        "than",
        "under",
        "upon",
        "into",
        "between",
        "through",
        "during",
        "after",
        "before",
        "above",
        "below",
        "any",
        "all",
        "both",
        "more",
        "most",
        "some",
        "not",
        "only",
        "very",
        "can",
        "may",
        "must",
        "should",
        "would",
        "could",
        "its",
        "his",
        "her",
        "their",
        "our",
        "your",
        "also",
        "however",
        "therefore",
        "although",
        "because",
        "provided",
        "including",
        "without",
        "within",
        "among",
    }

    query_terms = set(original_query.lower().split())
    term_counts: Counter[str] = Counter()

    for r in results[:_DEFAULT_PRF_DOCS]:
        words = r.content.lower().split()
        for word in words:
            clean = word.strip(".,;:!?\"'()-/\\[]{}#$%&*+=<>@^_`~")
            if (
                len(clean) >= min_term_len
                and clean not in stopwords
                and clean not in query_terms
                and clean.isalpha()
            ):
                term_counts[clean] += 1

    top_terms = [term for term, _ in term_counts.most_common(top_n_terms)]
    if not top_terms:
        return ""

    expanded = f"{original_query} {' '.join(top_terms)}"
    logger.debug(
        "retrieval.prf_expand: original_terms=%d new_terms=%d expanded=%r",
        len(query_terms),
        len(top_terms),
        expanded[:100],
    )
    return expanded


def _generate_pseudo_document(
    query: str,
    *,
    model: str | None = None,
) -> str:
    """Generate a hypothetical document passage that would answer the query.

    Uses the Query2Doc/HyDE technique: the LLM writes a fake passage using
    the vocabulary that real documents would use. BM25 then matches this
    passage against the corpus, bridging terminology gaps.
    """
    try:
        from kaos_llm_core import Call

        from kaos_agents.settings import DEFAULT_MODEL

        call = Call(GeneratePseudoDocumentSignature, model=model or DEFAULT_MODEL)

        import asyncio

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as pool:
                result = pool.submit(asyncio.run, call(search_query=query)).result(
                    timeout=_DEFAULT_LLM_TIMEOUT
                )
        else:
            result = asyncio.run(call(search_query=query))

        passage = cast(str, getattr(result, "hypothetical_passage", ""))
        logger.debug("retrieval.hyde_generated: query=%r passage_len=%d", query[:50], len(passage))
        return passage
    except Exception as exc:
        logger.debug("retrieval.hyde_failed: %s", exc)
        return ""


def _generate_llm_queries(
    original_query: str,
    found_item_ids: list[str],
    *,
    model: str | None = None,
) -> list[str]:
    """Ask an LLM to suggest alternative search queries.

    Uses a lightweight Call to generate 2-3 alternative queries that
    might find documents the original query missed.
    """
    try:
        from kaos_llm_core import Call

        from kaos_agents.settings import DEFAULT_MODEL

        call = Call(SuggestQueriesSignature, model=model or DEFAULT_MODEL)

        import asyncio

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as pool:
                result = pool.submit(
                    asyncio.run,
                    call(original_query=original_query, results_found=len(found_item_ids)),
                ).result(timeout=_DEFAULT_LLM_TIMEOUT)
        else:
            result = asyncio.run(
                call(original_query=original_query, results_found=len(found_item_ids))
            )
        queries = cast(list[str], getattr(result, "alternative_queries", []))
        logger.debug(
            "retrieval.llm_expand: original=%r suggestions=%d",
            original_query[:50],
            len(queries),
        )
        return queries[:_DEFAULT_MAX_SYNONYM_QUERIES]
    except Exception as exc:
        logger.debug("retrieval.llm_expand_failed: %s", exc)
        return []
