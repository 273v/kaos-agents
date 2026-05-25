"""Unit tests for :class:`kaos_agents.perception.perceiver.Perceiver`.

Phase 1.B contract:

- All-None deps + no items found → :class:`PerceptionRefusal`.
- Stub session_memory yielding items → :class:`PerceptionResult` with items.
- ``first_sufficient`` short-circuits before later sources are consulted.
- ``sources_consulted`` reflects the exact consultation order.
- Read-only-tool fan-out filters destructive tools and matches
  ``required_capabilities`` via glob.
"""

from __future__ import annotations

from typing import Any

import pytest
from kaos_core.base.context import KaosContext
from kaos_core.base.tool import KaosTool
from kaos_core.types.metadata import ToolAnnotations, ToolCapability, ToolCategory, ToolMetadata
from kaos_core.types.results import ToolResult

from kaos_agents.perception.perceiver import Perceiver
from kaos_agents.perception.types import (
    PerceptionItem,
    PerceptionQuery,
    PerceptionQueryKind,
)

# --- Stubs ------------------------------------------------------------


class _StubSession:
    """SessionMemory stub — exposes ``search`` returning PerceptionItems."""

    def __init__(self, items: list[PerceptionItem]) -> None:
        self._items = items
        self.queries: list[str] = []

    def search(self, query: str, *, top_k: int) -> list[PerceptionItem]:
        self.queries.append(query)
        return list(self._items[:top_k])


class _StubKB:
    """Knowledge-base stub — async query returning PerceptionItems."""

    def __init__(self, items: list[PerceptionItem]) -> None:
        self._items = items

    async def query(self, q: PerceptionQuery) -> list[PerceptionItem]:
        return list(self._items)


def _ro_tool(name: str, items: list[dict[str, Any]]) -> KaosTool:
    """Build a real KaosTool subclass that returns the given items."""

    class _Tool(KaosTool):
        @property
        def metadata(self) -> ToolMetadata:
            return ToolMetadata(
                name=name,
                display_name=name,
                description="x",
                category=ToolCategory.TEXT,
                capability=ToolCapability.TRANSFORM,
                module_name="test",
                version="0.0.1",
                annotations=ToolAnnotations(readOnlyHint=True),
            )

        async def execute(
            self, inputs: dict[str, Any], context: KaosContext | None = None
        ) -> ToolResult:
            return ToolResult.create_success(
                output={"items": list(items)}, summary=f"{len(items)} items"
            )

    return _Tool()


def _wr_tool(name: str) -> KaosTool:
    """Build a real KaosTool that is NOT read-only — must be filtered out."""

    class _Tool(KaosTool):
        @property
        def metadata(self) -> ToolMetadata:
            return ToolMetadata(
                name=name,
                display_name=name,
                description="x",
                category=ToolCategory.TEXT,
                capability=ToolCapability.TRANSFORM,
                module_name="test",
                version="0.0.1",
                annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=True),
            )

        async def execute(
            self, inputs: dict[str, Any], context: KaosContext | None = None
        ) -> ToolResult:  # pragma: no cover - must never be called
            raise AssertionError("destructive tool must not be invoked by the perceiver")

    return _Tool()


# --- Tests ------------------------------------------------------------


@pytest.mark.asyncio
async def test_all_none_deps_returns_refusal() -> None:
    p = Perceiver()
    result = await p(query=PerceptionQuery(query_text="anything"))
    assert result.refusal is not None
    assert result.refusal.kind == "evidence_insufficient"
    assert result.items == ()
    assert result.confidence == 0.0
    assert result.sources_consulted == ()


@pytest.mark.asyncio
async def test_session_memory_items_propagate() -> None:
    items = [
        PerceptionItem(content="alpha", source="session_memory", score=0.9),
        PerceptionItem(content="beta", source="session_memory", score=0.8),
    ]
    session = _StubSession(items)
    p = Perceiver(session_memory=session, first_sufficient=True)

    result = await p(query=PerceptionQuery(query_text="alpha"))

    assert result.refusal is None
    assert tuple(item.content for item in result.items) == ("alpha", "beta")
    assert result.sources_consulted == ("session_memory",)
    assert session.queries == ["alpha"]


@pytest.mark.asyncio
async def test_first_sufficient_short_circuits_before_tools() -> None:
    """High-confidence KB hit blocks session/tool consultation."""
    kb_items = [PerceptionItem(content="kb-hit", source="kb", score=0.95)]
    session_items = [PerceptionItem(content="should-not-appear", source="session_memory")]

    p = Perceiver(
        knowledge_base=_StubKB(kb_items),
        session_memory=_StubSession(session_items),
        tools=(_ro_tool("kaos-test-ro", [{"content": "tool-hit", "source": "tool:t"}]),),
        first_sufficient=True,
        min_confidence=0.5,
    )

    result = await p(
        query=PerceptionQuery(
            query_text="x",
            required_capabilities=("kaos-test-ro",),
        )
    )

    # KB short-circuits everything: only kb in sources_consulted.
    assert result.sources_consulted == ("kb",)
    assert tuple(i.content for i in result.items) == ("kb-hit",)


@pytest.mark.asyncio
async def test_first_sufficient_false_consults_every_source() -> None:
    kb_items = [PerceptionItem(content="kb-hit", source="kb", score=0.95)]
    session_items = [PerceptionItem(content="session-hit", source="session_memory", score=0.9)]

    p = Perceiver(
        knowledge_base=_StubKB(kb_items),
        session_memory=_StubSession(session_items),
        first_sufficient=False,
    )

    result = await p(query=PerceptionQuery(query_text="x"))

    assert result.sources_consulted == ("kb", "session_memory")
    assert {i.content for i in result.items} == {"kb-hit", "session-hit"}


@pytest.mark.asyncio
async def test_destructive_tool_filtered_at_construction() -> None:
    """Write tools never enter the perceiver's read-only set."""
    p = Perceiver(tools=(_wr_tool("kaos-test-write"), _ro_tool("kaos-test-ro", [])))
    assert len(p.read_only_tools) == 1
    assert p.read_only_tools[0].metadata.name == "kaos-test-ro"


@pytest.mark.asyncio
async def test_required_capabilities_glob_match() -> None:
    """Only tools whose name matches required_capabilities run."""
    matching = _ro_tool(
        "kaos-source-fr-search",
        [{"content": "fr-result", "source": "tool:fr"}],
    )
    other = _ro_tool(
        "kaos-web-fetch",
        [{"content": "web-result", "source": "tool:web"}],
    )
    p = Perceiver(tools=(matching, other), first_sufficient=False)

    result = await p(
        query=PerceptionQuery(
            query_text="x",
            required_capabilities=("kaos-source-fr-*",),
        )
    )

    assert "tool:kaos-source-fr-search" in result.sources_consulted
    assert "tool:kaos-web-fetch" not in result.sources_consulted
    assert any(i.content == "fr-result" for i in result.items)


@pytest.mark.asyncio
async def test_required_capabilities_empty_skips_tool_fanout() -> None:
    """No required_capabilities → tools are NOT invoked even when read-only."""
    tool = _ro_tool("kaos-test-ro", [{"content": "x", "source": "tool:x"}])
    p = Perceiver(tools=(tool,))

    result = await p(query=PerceptionQuery(query_text="anything"))

    # Refusal because no source was consulted.
    assert result.refusal is not None
    assert all(s != "tool:kaos-test-ro" for s in result.sources_consulted)


@pytest.mark.asyncio
async def test_rag_only_fires_for_document_qa_kind() -> None:
    """Non-DOCUMENT_QA queries skip the RAG branch entirely."""

    fired = []

    class _StubInvocation:
        def __init__(self, output: Any) -> None:
            self.output = output

    class _StubRAG:
        async def invoke(self, **kwargs: Any) -> _StubInvocation:
            # Iteration 8 of the kaos-llm-core design audit: perceiver
            # now calls ``rag.invoke()`` so ``Invocation.usage`` flows
            # through. Stub mirrors the canonical RAG return shape.
            fired.append(kwargs)
            return _StubInvocation(output=[PerceptionItem(content="rag-out", source="rag")])

    p = Perceiver(rag=_StubRAG(), first_sufficient=False)

    # GENERAL_RECALL: RAG must NOT be consulted.
    result = await p(query=PerceptionQuery(query_text="x", kind=PerceptionQueryKind.GENERAL_RECALL))
    assert "rag" not in result.sources_consulted
    assert fired == []

    # DOCUMENT_QA: RAG IS consulted.
    result = await p(query=PerceptionQuery(query_text="x", kind=PerceptionQueryKind.DOCUMENT_QA))
    assert result.sources_consulted == ("rag",)
    assert any(i.content == "rag-out" for i in result.items)
    assert len(fired) == 1
