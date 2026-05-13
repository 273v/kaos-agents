"""Tests for the per-session knowledge graph (Track 3 chunk B1).

Confirms:
- ``MemoryType.GRAPH`` is in the enum + DEFAULT_SECTIONS
- ``SessionMemory.graph`` is lazy — None until first access
- Lazy access constructs a directed multi ``kaos_graph.Graph`` named
  for the session
- ``SessionMemory.graph`` setter replaces the graph wholesale (used
  by SessionStore hydration)
- Persistence: SessionStore.save writes Turtle when the session has
  triples; skips when empty
- Hydration: SessionStore.load reads Turtle back and restores triples
- A fresh-session round-trip with an empty graph leaves no graph file
"""

from __future__ import annotations

import pytest
from kaos_core.types.enums import StorageBackend
from kaos_core.vfs.core import VirtualFileSystem
from kaos_core.vfs.models import VFSConfig

from kaos_agents.memory.session import SessionMemory
from kaos_agents.memory.store import SessionStore, _session_graph_path
from kaos_agents.types import DEFAULT_SECTIONS, MemoryType


def _memory_vfs() -> VirtualFileSystem:
    return VirtualFileSystem(config=VFSConfig(default_backend=StorageBackend.MEMORY))


# Use full IRIs because kaos_graph.rdf.to_turtle requires absolute IRIs.
_FINDING = "https://kaos.273v.com/finding/f1"
_DOC = "https://kaos.273v.com/doc/d1"
_CITES = "http://purl.org/spar/cito/cites"


@pytest.mark.unit
class TestMemoryTypeEnum:
    def test_graph_in_enum(self) -> None:
        assert MemoryType.GRAPH.value == "graph"

    def test_graph_in_default_sections(self) -> None:
        graph_configs = [s for s in DEFAULT_SECTIONS if s.memory_type == MemoryType.GRAPH]
        assert len(graph_configs) == 1
        config = graph_configs[0]
        assert config.budget_tokens == 0  # unbounded
        assert config.eviction_policy.value == "none"
        assert config.persistence_mode.value == "snapshot"


@pytest.mark.unit
class TestSessionMemoryGraph:
    def test_graph_lazy_initial_state(self) -> None:
        mem = SessionMemory("session-1")
        # Internal state shows lazy
        assert mem._graph is None

    def test_graph_constructs_on_access(self) -> None:
        mem = SessionMemory("session-1")
        g = mem.graph
        assert g is not None
        # Same instance on subsequent access (no re-init)
        assert mem.graph is g

    def test_graph_is_directed_multi(self) -> None:
        mem = SessionMemory("session-1")
        g = mem.graph
        assert g.is_directed is True
        assert g.is_multi is True

    def test_graph_named_for_session(self) -> None:
        mem = SessionMemory("session-abc")
        assert mem.graph.name == "session-abc"

    def test_graph_setter_replaces_wholesale(self) -> None:
        from kaos_graph import Graph

        mem = SessionMemory("session-1")
        original = mem.graph
        replacement = Graph(directed=True, multi=True, name="replacement")
        mem.graph = replacement
        assert mem.graph is replacement
        assert mem.graph is not original


@pytest.mark.unit
class TestSessionStorePersistence:
    @pytest.mark.asyncio
    async def test_empty_graph_skips_turtle_write(self) -> None:
        """Sessions that never built a graph leave no graph.ttl in VFS."""
        vfs = _memory_vfs()
        store = SessionStore(vfs)

        mem = SessionMemory("empty-session")
        await store.save(mem)

        # No graph.ttl was written
        graph_path = _session_graph_path("empty-session")
        assert not await vfs.exists(graph_path)

    @pytest.mark.asyncio
    async def test_lazy_access_with_no_triples_skips_write(self) -> None:
        """Even if the graph was lazy-initialized, save skips writing
        when there are zero edges (no triples to persist)."""
        vfs = _memory_vfs()
        store = SessionStore(vfs)

        mem = SessionMemory("touched-but-empty")
        _ = mem.graph  # Force lazy init, but don't add any triples
        await store.save(mem)

        graph_path = _session_graph_path("touched-but-empty")
        assert not await vfs.exists(graph_path)

    @pytest.mark.asyncio
    async def test_round_trip_with_triples(self) -> None:
        """A graph with triples persists as Turtle and hydrates back."""
        vfs = _memory_vfs()
        store = SessionStore(vfs)

        # Build a session with one cito:cites triple
        mem = SessionMemory("with-triples")
        g = mem.graph
        g.add_node(_FINDING)
        g.add_node(_DOC)
        g.add_edge(_FINDING, _DOC, predicate=_CITES)

        # Save
        await store.save(mem)

        # Confirm Turtle was written
        graph_path = _session_graph_path("with-triples")
        assert await vfs.exists(graph_path)

        # Hydrate via load
        loaded = await store.load("with-triples")
        assert loaded.graph.n_nodes == 2
        assert loaded.graph.n_edges == 1

        # Verify the specific edge exists
        assert loaded.graph.has_node(_FINDING)
        assert loaded.graph.has_node(_DOC)
        assert loaded.graph.has_edge(_FINDING, _DOC)

    @pytest.mark.asyncio
    async def test_corrupt_turtle_does_not_break_load(self) -> None:
        """Hydration failures on graph.ttl are logged + non-fatal —
        the rest of the session loads with a fresh empty graph."""
        vfs = _memory_vfs()
        store = SessionStore(vfs)

        mem = SessionMemory("session-with-bad-ttl")
        await store.save(mem)
        # Hand-corrupt the graph.ttl after save
        await vfs.write(
            _session_graph_path("session-with-bad-ttl"),
            b"@@@ not valid turtle @@@",
        )

        # Load should NOT raise — falls back to empty graph
        loaded = await store.load("session-with-bad-ttl")
        # The lazy-init default is None; accessing .graph creates a fresh empty
        assert loaded.graph.n_edges == 0
