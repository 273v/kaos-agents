"""Corpus + retrieval helpers for ``ResearchAgent``.

These helpers translate the DOCUMENTS section of a ``SessionMemory`` into a
``dict[uri, text]`` corpus that ``RAG.query`` consumes:

- ``_looks_like_question`` — lightweight heuristic for routing prebound-corpus
  agents to RESEARCH on document-shaped utterances only.
- ``_unwrap_corpus_arg`` — normalize the ``corpus`` ctor argument
  (``CorpusIndex`` → ``.corpus``).
- ``_build_corpus`` — flatten the entire DOCUMENTS section to a corpus dict.
- ``_build_corpus_bm25`` — BM25-narrowed corpus for large sections (the
  production retrieval path).
- ``_build_corpus_triaged`` — alternative narrowing via ``triage_corpus``.
- ``_extract_uri_and_content`` — pull (uri, content) out of a memory item,
  honoring metadata ``uri`` and the legacy ``"URI: <uri>\\n<text>"`` prefix.

Pure helpers — no agent state, no module-level configuration. The
``ResearchAgent`` class imports these from ``.corpus``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from kaos_core.logging import get_logger

from kaos_agents.types.memory import MemoryType

if TYPE_CHECKING:
    from kaos_agents.memory.session import SessionMemory

logger = get_logger(__name__)


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
