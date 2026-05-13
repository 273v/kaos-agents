"""ResearchAgent — document Q&A pattern via RAG.

The implementation is split across three sibling modules:

- ``agent`` — the :class:`ResearchAgent` class (RAG dispatch, ReAct
  escalation, citation/refusal handling).
- ``corpus`` — corpus + retrieval helpers that translate the DOCUMENTS
  memory section into a ``dict[uri, text]`` for RAG.query.
- ``outline`` — corpus-outline rendering (the ``[CORPUS OUTLINE]``
  preamble injected into research questions).

The package re-exports the public ``ResearchAgent`` class plus the
private corpus helpers that existing tests import directly from
``kaos_agents.patterns.research`` — preserving the pre-split import
surface so callers never see the carve.
"""

from __future__ import annotations

from kaos_agents.patterns.research.agent import ResearchAgent
from kaos_agents.patterns.research.corpus import (
    _build_corpus,
    _build_corpus_bm25,
    _build_corpus_triaged,
    _extract_uri_and_content,
    _looks_like_question,
    _unwrap_corpus_arg,
)

__all__ = [
    "ResearchAgent",
    "_build_corpus",
    "_build_corpus_bm25",
    "_build_corpus_triaged",
    "_extract_uri_and_content",
    "_looks_like_question",
    "_unwrap_corpus_arg",
]
