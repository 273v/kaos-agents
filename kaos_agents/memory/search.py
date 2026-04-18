"""BM25 search over memory sections with optional query expansion.

Indexes searchable memory sections (MESSAGES, ACTIONS, DOCUMENTS, FINDINGS)
using kaos-nlp-core's BM25 Searcher and provides ranked results.

When the OpenGloss Lexicon is available, queries are automatically expanded
with synonyms, hypernyms, and inflections to improve recall on terminology-
diverse corpora (e.g., "renewable energy" also finds "sustainable power",
"clean energy").
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from kaos_core.logging import get_logger

from kaos_agents.memory.types import MemoryType

if TYPE_CHECKING:
    from kaos_agents.memory.session import SessionMemory

logger = get_logger(__name__)

OPENGLOSS_LEXICON_FILENAME = "opengloss-v1.3.lexicon.bin"

_LEXICON_CACHE: dict[str, Any] = {}


def _load_lexicon(lexicon_path: str | None = None) -> Any | None:
    """Load the OpenGloss Lexicon, with process-level caching."""
    if lexicon_path is None:
        candidates = [
            Path(__file__).parent.parent.parent.parent
            / "kaos-nlp-core"
            / "data"
            / OPENGLOSS_LEXICON_FILENAME,
            Path.home() / ".cache" / "kaos" / "opengloss-v1.3.lexicon.bin",
        ]
        for candidate in candidates:
            if candidate.exists():
                lexicon_path = str(candidate)
                break

    if lexicon_path is None:
        return None

    if lexicon_path in _LEXICON_CACHE:
        return _LEXICON_CACHE[lexicon_path]

    try:
        from kaos_nlp_core.lexicon import Lexicon

        lex = Lexicon.load(lexicon_path)
        _LEXICON_CACHE[lexicon_path] = lex
        logger.debug("memory.search: loaded lexicon from %s (%d entries)", lexicon_path, len(lex))
        return lex
    except Exception as exc:
        logger.debug("memory.search: failed to load lexicon from %s: %s", lexicon_path, exc)
        _LEXICON_CACHE[lexicon_path] = None
        return None


@dataclass(frozen=True, slots=True)
class MemorySearchResult:
    """A single search result from memory.

    Attributes:
        content: The memory item's content text.
        section: Which section the item came from.
        score: BM25 relevance score.
        item_id: The MemoryItem ID.
    """

    content: str
    section: MemoryType
    score: float
    item_id: str


def search_memory(
    memory: SessionMemory,
    query: str,
    *,
    sections: list[MemoryType] | None = None,
    top_k: int = 10,
    lexicon_path: str | None = None,
    expand_relations: list[str] | None = None,
    expansion_depth: int = 1,
) -> list[MemorySearchResult]:
    """Search across memory sections using BM25 with optional query expansion.

    Indexes all items in the requested searchable sections, runs the
    query, and returns ranked results.

    When the OpenGloss Lexicon is available (auto-detected or via
    ``lexicon_path``), queries are expanded with synonyms, hypernyms,
    and inflections before searching.

    Args:
        memory: The session memory to search.
        query: Natural-language search query.
        sections: Which sections to search. Defaults to all sections
            with ``searchable=True`` (MESSAGES, ACTIONS, DOCUMENTS, FINDINGS).
        top_k: Maximum number of results to return.
        lexicon_path: Path to a Lexicon binary file. Auto-detected if None.
        expand_relations: Relation types for query expansion.
            Defaults to ``["synonym", "inflection"]``.
        expansion_depth: Maximum BFS depth for expansion (default 1).

    Returns:
        Ranked list of :class:`MemorySearchResult`, highest score first.
    """
    from kaos_nlp_core.search import Searcher

    if sections is None:
        sections = [
            mt
            for mt in memory.section_names
            if memory.has_section(mt)
            and hasattr(memory._sections.get(mt), "config")
            and memory._sections[mt].config.searchable
        ]

    records: list[dict[str, object]] = []
    index_to_item: dict[int, tuple[str, str, MemoryType]] = {}

    idx = 0
    for mt in sections:
        if not memory.has_section(mt):
            continue
        items = memory.get(mt)
        for item in items:
            records.append({"id": idx, "text": item.content})
            index_to_item[idx] = (item.id, item.content, mt)
            idx += 1

    if not records:
        return []

    lexicon = _load_lexicon(lexicon_path)
    if expand_relations is None:
        expand_relations = ["synonym", "inflection"]

    try:
        kwargs: dict = {}
        if lexicon is not None:
            kwargs["lexicon"] = lexicon
            kwargs["expand_relations"] = expand_relations
            kwargs["expansion_depth"] = expansion_depth
        searcher = Searcher.from_documents(records, **kwargs)
        hits = searcher.search(query, top_k=top_k)
    except Exception as exc:
        logger.debug("memory.search: failed: %s", exc, exc_info=True)
        return []

    results: list[MemorySearchResult] = []
    for hit in hits:
        hit_idx = int(hit.doc_id) if hasattr(hit, "doc_id") else 0
        item_id, content, section = index_to_item.get(hit_idx, ("", "", MemoryType.MESSAGES))
        results.append(
            MemorySearchResult(
                content=content,
                section=section,
                score=hit.score,
                item_id=item_id,
            )
        )

    logger.debug(
        "memory.search: query=%r sections=%d items=%d results=%d lexicon=%s",
        query[:50],
        len(sections),
        len(records),
        len(results),
        "yes" if lexicon else "no",
    )

    return results
