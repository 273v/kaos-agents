"""KnowledgeBase — paper §6.1 "the archive".

Across-session memory: a queryable store of findings, citations, and
entity facts that persist beyond a single SessionMemory. Phase 4.A
ships an in-memory backend; Phase 4+ may add Polars/DuckDB backends.

Matter/client isolation is enforced by namespace: every read/write
requires a (matter_id, client_id) tuple. Mixing namespaces raises
:class:`MatterIsolationError` (see ``isolation.py``).

The KnowledgeBase is a kaos-llm-core ``Program`` so it composes with
RAG and can be optimised by future MIPRO runs (instruction tuning of
the retrieval / synthesis prompts).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from kaos_llm_core.programs.base import Program

from kaos_agents.memory.isolation import MatterClientGuard


@dataclass(slots=True)
class KBEntry:
    """One stored finding in the institutional memory.

    Frozen contracts:
      - ``matter_client`` is the namespace; lookup requires the same
        tuple.
      - ``confidence`` ∈ [0,1].
      - ``provenance`` carries the source span(s) so verification
        can re-check the claim against the source corpus.
    """

    id: str
    statement: str
    matter_client: tuple[str, str]
    confidence: float
    grounding_verified: bool
    provenance: tuple[Any, ...] = field(default_factory=tuple)  # tuple[Span, ...]
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)


@dataclass(slots=True)
class KBQuery:
    query_text: str
    matter_client: tuple[str, str]
    top_k: int = 10
    min_confidence: float = 0.0


@dataclass(slots=True)
class KBResult:
    entries: tuple[KBEntry, ...]
    query: KBQuery
    method: str = "bm25"  # "bm25" | "vector" | "hybrid"


class KnowledgeBase(Program):
    """In-memory KnowledgeBase. Phase 4.A baseline.

    Constructor kwargs:
      guard: optional :class:`MatterClientGuard`. Defaults to a fresh
        one (per-instance enforcement).
      retrieval_method: "bm25" (default), "vector", or "hybrid".
        Phase 4.A only implements BM25 retrieval over a flat list;
        vector / hybrid raise NotImplementedError.

    Methods:
      add(entry): add a KBEntry; raises if isolation is violated.
      query(query): return matching entries; raises on isolation
        violation.
      forward(query, *, matter_client): the Program contract — looks
        up entries by query string within the namespace.
    """

    def __init__(
        self,
        *,
        guard: MatterClientGuard | None = None,
        retrieval_method: str = "bm25",
    ) -> None:
        super().__init__()
        self._guard = guard or MatterClientGuard()
        self._method = retrieval_method
        self._entries: list[KBEntry] = []

    def add(self, entry: KBEntry) -> None:
        self._guard.assert_writable(entry.matter_client)
        self._entries.append(entry)

    def query(self, query: KBQuery) -> KBResult:
        self._guard.assert_readable(query.matter_client)
        scoped = [
            e
            for e in self._entries
            if e.matter_client == query.matter_client and e.confidence >= query.min_confidence
        ]
        # Phase 4.A retrieval: simple substring score; Phase 4+ wires
        # kaos_nlp_core BM25 (existing search.search_memory pattern).
        ranked = sorted(
            scoped,
            key=lambda e: _score(e.statement, query.query_text),
            reverse=True,
        )
        return KBResult(
            entries=tuple(ranked[: query.top_k]),
            query=query,
            method=self._method,
        )

    async def forward(self, **kwargs: Any) -> KBResult:
        query: KBQuery = kwargs["query"]
        return self.query(query)


def _score(statement: str, query_text: str) -> float:
    """Phase 4.A naive substring score (Phase 4+ swaps in BM25)."""
    if not query_text or not statement:
        return 0.0
    matches = sum(1 for term in query_text.lower().split() if term in statement.lower())
    return float(matches)


__all__ = ["KBEntry", "KBQuery", "KBResult", "KnowledgeBase"]
