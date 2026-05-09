"""Perception subsystem — read-only fact-finding for kaos-agents.

Phase 1.B of the kaos-agents ground-up rewrite. Answers the rewrite
plan's Q3 ("How does the agent find things out?") by composing:

- The institutional KnowledgeBase (Phase 4 stub today),
- The session-memory BM25 search,
- A read-only-tool fan-out filtered by ``readOnlyHint=True``,
- A kaos-llm-core RAG wrapper that emits ``CitationFound`` events.

The package is purely additive in Phase 1.B — nothing in the existing
:mod:`kaos_agents.runtime` / :mod:`kaos_agents.patterns` consumes it
yet. Phase 2 will wire :class:`Perceiver` into the agent loop.
"""

from __future__ import annotations

from kaos_agents.perception.perceiver import Perceiver
from kaos_agents.perception.rag import PerceptionRAG
from kaos_agents.perception.registry import read_only_tools
from kaos_agents.perception.types import (
    PerceptionItem,
    PerceptionQuery,
    PerceptionQueryKind,
    PerceptionRefusal,
    PerceptionResult,
)

__all__ = [
    "Perceiver",
    "PerceptionItem",
    "PerceptionQuery",
    "PerceptionQueryKind",
    "PerceptionRAG",
    "PerceptionRefusal",
    "PerceptionResult",
    "read_only_tools",
]
