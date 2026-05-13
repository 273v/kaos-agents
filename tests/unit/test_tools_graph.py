"""Tests for the session knowledge-graph MCP tools (Track 3 chunk B3).

Covers the 3 tools that expose the per-session graph populated by the
chunk B2 triple emitter:

- ``kaos-agent-graph-walk`` — N-hop ego subgraph
- ``kaos-agent-graph-sparql`` — SPARQL SELECT/ASK
- ``kaos-agent-graph-projection`` — pre-built typed views

For SPARQL paths, pyoxigraph is required; if missing, tests verify the
agent-friendly error pointing to the install command.
"""

from __future__ import annotations

import importlib.util

import pytest
from kaos_core.base.context import KaosContext
from kaos_core.registry.container import KaosRuntime
from kaos_core.types.enums import StorageBackend
from kaos_core.vfs.core import VirtualFileSystem
from kaos_core.vfs.models import VFSConfig

from kaos_agents.events import CitationFound, EventEmitter, SpanSubject
from kaos_agents.memory.session import SessionMemory
from kaos_agents.memory.store import SessionStore
from kaos_agents.memory.triples import (
    KAOS,
    emit_from_event,
    step_iri,
    tool_call_iri,
)
from kaos_agents.tools.graph import (
    AgentGraphProjectionTool,
    AgentGraphSparqlTool,
    AgentGraphWalkTool,
)

_HAS_PYOXIGRAPH = importlib.util.find_spec("pyoxigraph") is not None


def _data(res) -> dict:
    """Extract structuredContent and narrow the type for ty.

    ``ToolResult.structuredContent`` is typed ``dict | None``; success
    cases always populate it. The cast keeps tests readable while
    keeping ty happy.
    """
    sc = res.structuredContent
    assert isinstance(sc, dict)
    return sc


def _text(res) -> str:
    """Extract the first text-content block's text, narrowed for ty.

    Error results return a ``TextContent`` block; we only access ``text``
    after asserting that's the type we got.
    """
    block = res.content[0]
    text = getattr(block, "text", None)
    assert isinstance(text, str)
    return text


@pytest.fixture
def context_and_session():
    """Build a session populated with one tool call + one citation,
    return the matching KaosContext for tool execute() calls.

    The graph layout:
      <call:tc-1>     a kaos:ToolCall ;
                      prov:wasAssociatedWith <agent-runtime> ;
                      prov:wasInformedBy <step:s001> .
      <step:s001>     a kaos:Step .
      <finding:HASH>  a kaos:Finding ;
                      cito:cites <https://x.com/1> .
      <https://x.com/1> a kaos:Document .
    """
    import asyncio

    async def _setup():
        vfs = VirtualFileSystem(config=VFSConfig(default_backend=StorageBackend.MEMORY))
        store = SessionStore(vfs)

        mem = SessionMemory("graph-tools-test")
        em = EventEmitter(session_id="graph-tools-test", run_id="r1")

        emit_from_event(
            em.span_start(
                SpanSubject.TOOL_CALL,
                attributes={
                    "tool_name": "fr-search",
                    "call_id": "tc-1",
                    "step_id": "s001",
                },
            ),
            mem,
        )
        emit_from_event(
            em.emit(
                CitationFound,
                claim="EPA filed 3 enforcement actions",
                source_uri="https://www.epa.gov/abc",
                confidence=0.95,
                verified=True,
            ),
            mem,
        )

        await store.save(mem)

        rt = KaosRuntime()
        rt.vfs = vfs
        ctx = KaosContext(runtime=rt, session_id="graph-tools-test")
        return ctx, mem

    return asyncio.run(_setup())


# ---------------------------------------------------------------------------
# Walk tool
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestGraphWalk:
    @pytest.mark.asyncio
    async def test_walk_from_tool_call(self, context_and_session) -> None:
        ctx, _mem = context_and_session
        tool = AgentGraphWalkTool()

        res = await tool.execute(
            {
                "session_id": "graph-tools-test",
                "start_iri": tool_call_iri("tc-1"),
                "max_hops": 2,
            },
            ctx,
        )

        assert res.isError is False
        data = _data(res)
        # Walk reaches at least: call + step + agent (2 hops)
        assert data["node_count"] >= 3
        assert data["edge_count"] >= 2
        assert tool_call_iri("tc-1") in data["nodes"]
        assert step_iri("s001") in data["nodes"]

    @pytest.mark.asyncio
    async def test_walk_radius_one(self, context_and_session) -> None:
        ctx, _mem = context_and_session
        tool = AgentGraphWalkTool()

        res = await tool.execute(
            {
                "session_id": "graph-tools-test",
                "start_iri": tool_call_iri("tc-1"),
                "max_hops": 1,
            },
            ctx,
        )

        assert res.isError is False
        # 1 hop covers direct neighbors only
        assert _data(res)["max_hops"] == 1

    @pytest.mark.asyncio
    async def test_walk_unknown_node_returns_friendly_error(self, context_and_session) -> None:
        ctx, _mem = context_and_session
        tool = AgentGraphWalkTool()

        res = await tool.execute(
            {
                "session_id": "graph-tools-test",
                "start_iri": "https://no-such-node.example/x",
            },
            ctx,
        )

        assert res.isError is True
        msg = _text(res)
        assert "not in the session graph" in msg
        assert "kaos-agent-graph-projection" in msg  # pointer to discovery tool

    @pytest.mark.asyncio
    async def test_walk_missing_session_id(self, context_and_session) -> None:
        ctx, _mem = context_and_session
        tool = AgentGraphWalkTool()

        res = await tool.execute({"start_iri": tool_call_iri("tc-1"), "max_hops": 1}, ctx)

        assert res.isError is True
        assert "session_id" in _text(res).lower()

    @pytest.mark.asyncio
    async def test_walk_missing_start_iri(self, context_and_session) -> None:
        ctx, _mem = context_and_session
        tool = AgentGraphWalkTool()

        res = await tool.execute({"session_id": "graph-tools-test"}, ctx)

        assert res.isError is True
        assert "start_iri" in _text(res).lower()

    @pytest.mark.asyncio
    async def test_walk_metadata_is_read_only(self) -> None:
        """Walk is read-only — auto-approved by Claude Code."""
        meta = AgentGraphWalkTool().metadata
        assert meta.annotations is not None
        assert meta.annotations.readOnlyHint is True
        assert meta.annotations.destructiveHint is False
        assert meta.name == "kaos-agent-graph-walk"


# ---------------------------------------------------------------------------
# SPARQL tool — split into pyoxigraph-required and -graceful paths
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.skipif(not _HAS_PYOXIGRAPH, reason="requires kaos-graph[rdf] / pyoxigraph")
class TestGraphSparqlLive:
    @pytest.mark.asyncio
    async def test_select_returns_rows(self, context_and_session) -> None:
        ctx, _mem = context_and_session
        tool = AgentGraphSparqlTool()

        res = await tool.execute(
            {
                "session_id": "graph-tools-test",
                "query": "SELECT ?s ?p ?o WHERE { ?s ?p ?o } LIMIT 50",
            },
            ctx,
        )

        assert res.isError is False
        data = _data(res)
        assert data["query_type"] == "select"
        assert data["row_count"] > 0
        assert "s" in data["variables"]

    @pytest.mark.asyncio
    async def test_ask_returns_bool(self, context_and_session) -> None:
        ctx, _mem = context_and_session
        tool = AgentGraphSparqlTool()

        res = await tool.execute(
            {
                "session_id": "graph-tools-test",
                "query": "ASK { ?s <http://purl.org/spar/cito/cites> ?o }",
                "query_type": "ask",
            },
            ctx,
        )

        assert res.isError is False
        assert _data(res)["result"] is True

    @pytest.mark.asyncio
    async def test_invalid_sparql_returns_friendly_error(self, context_and_session) -> None:
        ctx, _mem = context_and_session
        tool = AgentGraphSparqlTool()

        res = await tool.execute(
            {
                "session_id": "graph-tools-test",
                "query": "this is not valid sparql at all",
            },
            ctx,
        )

        assert res.isError is True
        msg = _text(res)
        assert "SPARQL query failed" in msg
        # Error includes hints for common predicates
        assert "<http" in msg


@pytest.mark.unit
@pytest.mark.skipif(_HAS_PYOXIGRAPH, reason="only run when pyoxigraph is missing")
class TestGraphSparqlMissingDep:
    @pytest.mark.asyncio
    async def test_sparql_missing_pyoxigraph_returns_install_hint(
        self, context_and_session
    ) -> None:
        ctx, _mem = context_and_session
        tool = AgentGraphSparqlTool()

        res = await tool.execute(
            {
                "session_id": "graph-tools-test",
                "query": "SELECT ?s WHERE { ?s ?p ?o }",
            },
            ctx,
        )

        assert res.isError is True
        msg = _text(res)
        assert "kaos-graph[rdf]" in msg


# ---------------------------------------------------------------------------
# Projection tool
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestGraphProjection:
    @pytest.mark.asyncio
    async def test_all_nodes_works_without_pyoxigraph(self, context_and_session) -> None:
        """all_nodes is the SPARQL-free escape hatch for inventorying the graph."""
        ctx, _mem = context_and_session
        tool = AgentGraphProjectionTool()

        res = await tool.execute(
            {"session_id": "graph-tools-test", "projection_name": "all_nodes"}, ctx
        )

        assert res.isError is False
        data = _data(res)
        assert data["row_count"] >= 5  # call + step + agent + finding + doc + class IRIs
        # Class IRIs are also nodes (created by _ensure_type_edge)
        assert any(n.startswith(f"{KAOS}finding/") for n in data["nodes"])
        assert any(n.startswith(f"{KAOS}call/") for n in data["nodes"])

    @pytest.mark.asyncio
    async def test_unknown_projection_returns_friendly_error(self, context_and_session) -> None:
        ctx, _mem = context_and_session
        tool = AgentGraphProjectionTool()

        res = await tool.execute(
            {"session_id": "graph-tools-test", "projection_name": "no_such_view"}, ctx
        )

        assert res.isError is True
        msg = _text(res)
        assert "Unknown projection" in msg
        assert "all_nodes" in msg  # available list shown

    @pytest.mark.asyncio
    async def test_projection_metadata_lists_all_views(self) -> None:
        meta = AgentGraphProjectionTool().metadata
        # The description enumerates the available views — agents read the
        # description to pick a projection without invoking the tool first.
        for name in (
            "findings_with_citations",
            "tool_calls_by_step",
            "step_timeline",
            "all_nodes",
        ):
            assert name in meta.description


@pytest.mark.unit
@pytest.mark.skipif(not _HAS_PYOXIGRAPH, reason="requires kaos-graph[rdf] / pyoxigraph")
class TestGraphProjectionLive:
    @pytest.mark.asyncio
    async def test_findings_with_citations(self, context_and_session) -> None:
        ctx, _mem = context_and_session
        tool = AgentGraphProjectionTool()

        res = await tool.execute(
            {
                "session_id": "graph-tools-test",
                "projection_name": "findings_with_citations",
            },
            ctx,
        )

        assert res.isError is False
        data = _data(res)
        # We populated the session with one CitationFound — exactly one finding
        assert data["row_count"] == 1
        row = data["rows"][0]
        assert row["finding"].startswith(f"{KAOS}finding/")
        assert row["doc"] == "https://www.epa.gov/abc"

    @pytest.mark.asyncio
    async def test_tool_calls_by_step(self, context_and_session) -> None:
        ctx, _mem = context_and_session
        tool = AgentGraphProjectionTool()

        res = await tool.execute(
            {"session_id": "graph-tools-test", "projection_name": "tool_calls_by_step"},
            ctx,
        )

        assert res.isError is False
        data = _data(res)
        # One tool call, linked to one step via prov:wasInformedBy
        assert data["row_count"] >= 1
        row = data["rows"][0]
        assert row["call"].startswith(f"{KAOS}call/")
        assert row["step"].startswith(f"{KAOS}step/")


# ---------------------------------------------------------------------------
# register_agent_tools wires the 3 graph tools
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRegistration:
    def test_graph_tools_in_register_agent_tools(self) -> None:
        from unittest.mock import MagicMock

        from kaos_agents.tools import register_agent_tools

        runtime = MagicMock()
        runtime.module_settings = {}
        runtime.tools = MagicMock()
        registered: list[str] = []
        runtime.tools.register_tool = lambda tool: registered.append(tool.metadata.name)

        register_agent_tools(runtime)

        assert "kaos-agent-graph-walk" in registered
        assert "kaos-agent-graph-sparql" in registered
        assert "kaos-agent-graph-projection" in registered
