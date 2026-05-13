"""Perceiver — the read-only fact-finding Program.

Composes:

1. Institutional memory (KnowledgeBase) — Phase 4 stub returns nothing.
2. Session memory (BM25 search via :func:`kaos_agents.memory.search.search_memory`).
3. Read-only tool fan-out — only ``readOnlyHint=True`` tools matching
   :attr:`PerceptionQuery.required_capabilities`.
4. Optional RAG (:class:`PerceptionRAG`) when
   ``query.kind == PerceptionQueryKind.DOCUMENT_QA``.

Order is: KB -> session -> tools -> RAG. When ``first_sufficient`` is
True (the default), the perceiver short-circuits as soon as any source
returns at least one item AND the aggregate confidence is at or above
``min_confidence``. Returns a :class:`PerceptionResult` with
:attr:`sources_consulted` listing each consulted source in order.

Phase 1.B ships the Program with no wiring to Runner / Memory; the
constructor takes optional dependencies so unit tests can pass stubs.
Per the resolved-decisions in the rewrite plan, ``CitationFound`` and
``EvidenceInsufficient`` events emitted by RAG flow up through the
active event collector — they ARE in the value-events propagation set.
"""

from __future__ import annotations

import fnmatch
import inspect
from typing import Any

from kaos_llm_core.programs.base import Program

from kaos_agents.perception.registry import read_only_tools
from kaos_agents.perception.types import (
    PerceptionItem,
    PerceptionQuery,
    PerceptionQueryKind,
    PerceptionRefusal,
    PerceptionResult,
)


class Perceiver(Program):
    """Read-only fact-finding Program.

    Args:
        knowledge_base: Optional Phase 4 institutional memory. Must
            expose an awaitable ``query(query)`` method that returns
            an iterable of :class:`PerceptionItem`. ``None`` skips
            KB consultation.
        session_memory: Optional kaos-agents :class:`SessionMemory`
            instance. When provided, the perceiver runs
            :func:`kaos_agents.memory.search.search_memory` against it.
        tools: Full tool list. Filtered through :func:`read_only_tools`
            internally — destructive tools never reach the inner loop.
        rag: Optional :class:`PerceptionRAG` instance. Used only when
            ``query.kind == DOCUMENT_QA``.
        first_sufficient: When True (default), short-circuit on the
            first source that yields enough confident items. When
            False, every source is consulted before returning.
        min_confidence: Aggregate confidence threshold for the
            short-circuit decision and for issuing a refusal.
    """

    def __init__(
        self,
        *,
        knowledge_base: Any | None = None,
        session_memory: Any | None = None,
        tools: tuple[Any, ...] = (),
        rag: Any | None = None,
        first_sufficient: bool = True,
        min_confidence: float = 0.5,
    ) -> None:
        super().__init__()
        # Use private attributes so the Program graph auto-registration
        # in __setattr__ ignores them. Phase 1.B is purely additive — no
        # surprising children in the program graph.
        self._knowledge_base = knowledge_base
        self._session_memory = session_memory
        self._tools_unfiltered: tuple[Any, ...] = tuple(tools)
        self._read_only_tools: tuple[Any, ...] = read_only_tools(self._tools_unfiltered)
        self._rag = rag
        self._first_sufficient = first_sufficient
        self._min_confidence = min_confidence

    @property
    def read_only_tools(self) -> tuple[Any, ...]:
        """View of the perceiver's filtered (read-only) tool set."""
        return self._read_only_tools

    async def forward(self, query: PerceptionQuery) -> PerceptionResult:  # ty: ignore[invalid-method-override]
        """Consult sources in order; return items + refusal as appropriate."""
        sources_consulted: list[str] = []
        items: list[PerceptionItem] = []

        # ---- 1. Institutional memory (Phase 4 stub) -----------------
        if self._knowledge_base is not None:
            sources_consulted.append("kb")
            kb_items = await _await_iterable(self._knowledge_base.query(query))
            items.extend(_coerce_items(kb_items, default_source="kb"))
            if self._first_sufficient and _enough(items, self._min_confidence):
                return _ok(items, sources_consulted)

        # ---- 2. Session memory ---------------------------------------
        if self._session_memory is not None:
            sources_consulted.append("session_memory")
            mem_items = _search_session(self._session_memory, query)
            items.extend(mem_items)
            if self._first_sufficient and _enough(items, self._min_confidence):
                return _ok(items, sources_consulted)

        # ---- 3. Read-only tool fan-out ------------------------------
        if self._read_only_tools and query.required_capabilities:
            for tool in self._read_only_tools:
                tool_name = _tool_name(tool)
                if not _matches_capabilities(tool_name, query.required_capabilities):
                    continue
                source_label = f"tool:{tool_name}" if tool_name else "tool:?"
                sources_consulted.append(source_label)
                tool_items = await _invoke_tool(tool, query, source_label)
                items.extend(tool_items)
                if self._first_sufficient and _enough(items, self._min_confidence):
                    return _ok(items, sources_consulted)

        # ---- 4. RAG (DOCUMENT_QA only) ------------------------------
        if self._rag is not None and query.kind == PerceptionQueryKind.DOCUMENT_QA:
            sources_consulted.append("rag")
            rag_items = await _invoke_rag(self._rag, query)
            items.extend(rag_items)

        if not items:
            return PerceptionResult(
                items=(),
                confidence=0.0,
                refusal=PerceptionRefusal(
                    reason=(
                        f"No source could answer the query (consulted: "
                        f"{', '.join(sources_consulted) or 'none'}). "
                        "Add a knowledge_base, session_memory, or read-only tool, "
                        "or relax required_capabilities."
                    ),
                    kind="evidence_insufficient",
                    suggested_alternatives=("act", "ask_user", "broaden_query"),
                ),
                sources_consulted=tuple(sources_consulted),
            )

        return _ok(items, sources_consulted)


# --- helpers ----------------------------------------------------------


def _ok(items: list[PerceptionItem], sources: list[str]) -> PerceptionResult:
    return PerceptionResult(
        items=tuple(items),
        confidence=1.0 if items else 0.0,
        refusal=None,
        sources_consulted=tuple(sources),
    )


def _enough(items: list[PerceptionItem], min_confidence: float) -> bool:
    """Short-circuit gate: at least one item AND average score / 1.0 >= threshold.

    Items with no score count as fully-confident (score=1.0); items
    with a numeric score contribute the score itself. The threshold
    test is applied to the mean.
    """
    if not items:
        return False
    scores: list[float] = []
    for item in items:
        scores.append(item.score if item.score is not None else 1.0)
    return (sum(scores) / len(scores)) >= min_confidence


def _coerce_items(values: Any, *, default_source: str) -> list[PerceptionItem]:
    """Best-effort coercion of heterogeneous source outputs to PerceptionItem.

    Accepts an iterable of either:

    - :class:`PerceptionItem` instances (returned as-is).
    - dict-like with at least a ``content`` key.
    - objects with a ``content`` attribute.
    """
    out: list[PerceptionItem] = []
    if values is None:
        return out
    for v in values:
        if isinstance(v, PerceptionItem):
            out.append(v)
            continue
        if isinstance(v, dict):
            content = v.get("content", "")
            if not content:
                continue
            out.append(
                PerceptionItem(
                    content=str(content),
                    source=str(v.get("source", default_source)),
                    score=v.get("score"),
                )
            )
            continue
        content = getattr(v, "content", None)
        if content is None:
            continue
        out.append(
            PerceptionItem(
                content=str(content),
                source=str(getattr(v, "source", default_source)),
                score=getattr(v, "score", None),
            )
        )
    return out


async def _await_iterable(value: Any) -> Any:
    """Resolve a value that may be a coroutine or already an iterable."""
    if inspect.isawaitable(value):
        return await value
    return value


def _search_session(session_memory: Any, query: PerceptionQuery) -> list[PerceptionItem]:
    """Run BM25 search via the existing session-memory API and coerce results.

    Imports :func:`kaos_agents.memory.search.search_memory` lazily to
    keep the perception module's import cost low for callers that
    never use a SessionMemory.
    """
    # Allow stub SessionMemory in tests to expose a ``search`` method
    # directly (returning items already in PerceptionItem-like shape).
    direct_search = getattr(session_memory, "search", None)
    if callable(direct_search):
        results = direct_search(query.query_text, top_k=query.max_results)
        return _coerce_items(results, default_source="session_memory")

    # Production path — the canonical search_memory helper.
    try:
        from kaos_agents.memory.search import search_memory
    except Exception:
        return []
    try:
        results = search_memory(session_memory, query.query_text, top_k=query.max_results)
    except Exception:
        return []
    items: list[PerceptionItem] = []
    for hit in results:
        items.append(
            PerceptionItem(
                content=str(hit.content),
                source="session_memory",
                score=float(hit.score),
                metadata={"item_id": str(hit.item_id), "section": str(hit.section)},
            )
        )
    return items


def _tool_name(tool: Any) -> str:
    """Best-effort tool name extraction across the duck-typed shapes."""
    metadata = getattr(tool, "metadata", None)
    if metadata is not None:
        name = getattr(metadata, "name", None)
        if isinstance(name, str):
            return name
    name = getattr(tool, "name", None)
    if isinstance(name, str):
        return name
    if isinstance(tool, dict):
        return str(tool.get("name", ""))
    return ""


def _matches_capabilities(name: str, patterns: tuple[str, ...]) -> bool:
    """Glob match against the perceiver's required-capabilities list."""
    if not patterns:
        return False
    return any(fnmatch.fnmatch(name, pattern) for pattern in patterns)


async def _invoke_tool(
    tool: Any,
    query: PerceptionQuery,
    source_label: str,
) -> list[PerceptionItem]:
    """Invoke a read-only tool with a uniform ``query`` payload.

    The fan-out shape: pass ``{"query": query_text}`` to the tool's
    ``execute`` method (KaosTool) or its bare ``__call__``. Failures
    are swallowed and the tool contributes no items — Phase 2 wiring
    will add a structured error event; Phase 1.B is silent on the
    failure path so the test surface stays small.
    """
    payload = {"query": query.query_text, "max_results": query.max_results}
    try:
        execute = getattr(tool, "execute", None)
        if callable(execute):
            raw = execute(payload)
        elif callable(tool):
            raw = tool(payload)
        else:
            return []
        if inspect.isawaitable(raw):
            raw = await raw
    except Exception:
        return []

    return _coerce_items(_extract_tool_payload(raw), default_source=source_label)


def _extract_tool_payload(raw: Any) -> Any:
    """Pull the iterable-of-items out of a tool's heterogeneous return shape.

    KaosTool execute() returns a ``ToolResult`` with ``.structuredContent``
    or ``.content``; raw callables may return a list directly. Best-effort.
    """
    # ToolResult — pull structured content first, then text.
    structured = getattr(raw, "structuredContent", None)
    if isinstance(structured, dict):
        items = structured.get("items")
        if items is not None:
            return items
        if structured.get("content"):
            return [structured]
    # Raw list / tuple / set — pass through.
    if isinstance(raw, (list, tuple, set, frozenset)):
        return list(raw)
    # Dict with explicit "items" key.
    if isinstance(raw, dict) and "items" in raw:
        return raw["items"]
    return []


async def _invoke_rag(rag: Any, query: PerceptionQuery) -> list[PerceptionItem]:
    """Run the configured RAG and convert its output to PerceptionItems.

    Tolerant of the exact RAG surface — uses ``__call__`` with
    ``question=query_text`` when available, and accepts either an
    ``Answer[T]`` (extract value + spans) or a string. The DOCUMENT_QA
    contract requires the caller to wire a corpus into the RAG itself
    (e.g. via a pre-bound ``Retriever``); when no corpus is reachable
    the RAG either raises or returns an InsufficientEvidence — both
    paths produce zero items here.
    """
    try:
        # PerceptionRAG / RAG: pass kwargs through __call__. The rewrite
        # plan does not pin the kwarg name yet; ``question`` is the
        # canonical kaos-llm-core RAG input.
        output = await rag(question=query.query_text)
    except Exception:
        return []

    # Output may already be coerced to a PerceptionItem-shaped list by
    # tests; pass through when so.
    if isinstance(output, list):
        return _coerce_items(output, default_source="rag")

    # Answer[T] — pull value text and spans.
    spans = getattr(output, "spans", None)
    value = getattr(output, "value", None)
    if value is not None:
        from kaos_llm_core.signatures.grounding import Span as _Span

        citations = tuple(s for s in (spans or []) if isinstance(s, _Span))
        return [
            PerceptionItem(
                content=str(value),
                source="rag",
                citations=citations,
                score=getattr(output, "confidence", None),
            )
        ]
    return []


__all__ = ["Perceiver"]
