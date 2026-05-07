"""SessionStore — VFS-backed persistence for SessionMemory.

Handles save/load of memory state. STREAMING sections are appended as JSONL.
SNAPSHOT sections are written as full JSON at checkpoints. The store is
async to match the VFS API.

VFS layout per session:
    kaos-agents/sessions/{session_id}/memory.json

For Phase 1, we use a single JSON file per session. Phase 2 will add
per-section JSONL streaming for high-write sections.
"""

from __future__ import annotations

import json

from kaos_core.logging import get_logger
from kaos_core.vfs.core import VirtualFileSystem

from kaos_agents.errors import SessionCorruptedError, SessionNotFoundError
from kaos_agents.memory.session import SessionMemory
from kaos_agents.types.memory import DEFAULT_SECTIONS, MemoryType, SectionConfig

logger = get_logger(__name__)

# VFS path prefix for all agent sessions
_SESSION_PREFIX = "kaos-agents/sessions"


def _session_path(session_id: str) -> str:
    """VFS path for a session's memory snapshot."""
    return f"{_SESSION_PREFIX}/{session_id}/memory.json"


def _session_graph_path(session_id: str) -> str:
    """VFS path for a session's RDF knowledge graph (Turtle).

    Track 3 chunk B1 — the per-session knowledge graph persists alongside
    the JSON memory snapshot. Both files live under
    ``{_SESSION_PREFIX}/{session_id}/`` so a session is one directory in
    VFS that carries everything (memory + graph).
    """
    return f"{_SESSION_PREFIX}/{session_id}/graph.ttl"


class SessionStore:
    """VFS-backed persistence for SessionMemory.

    Saves and loads complete memory snapshots as JSON. Each session is stored
    under a deterministic VFS path keyed by session_id.

    The store is stateless — it does not cache sessions in memory. Each
    save/load is a complete round-trip to the VFS.
    """

    __slots__ = ("_sections", "_vfs")

    def __init__(
        self,
        vfs: VirtualFileSystem,
        *,
        sections: tuple[SectionConfig, ...] = DEFAULT_SECTIONS,
    ) -> None:
        self._vfs = vfs
        self._sections = sections

    async def save(self, memory: SessionMemory) -> str:
        """Save memory state to VFS. Returns the VFS path of the JSON snapshot.

        Track 3 chunk B1: also persists the per-session knowledge graph
        as Turtle under ``{_SESSION_PREFIX}/{session_id}/graph.ttl`` if
        the session has touched its ``.graph`` (i.e. one or more triples
        emitted). Sessions that never built a graph skip the write —
        no empty Turtle files in VFS.
        """
        path = _session_path(memory.session_id)
        data = memory.to_dict()
        payload = json.dumps(data, separators=(",", ":"), default=str).encode()
        await self._vfs.write(path, payload)

        # Persist the knowledge graph as Turtle, if it exists and has
        # any triples. We touch the lazy ``.graph`` property only if it
        # was already constructed — accessing it here would force-init
        # an empty graph for every session, defeating the lazy design.
        if memory._graph is not None:
            graph = memory._graph
            if graph.n_edges > 0:
                from kaos_graph.rdf import to_turtle

                turtle_path = _session_graph_path(memory.session_id)
                await self._vfs.write(turtle_path, to_turtle(graph).encode())
                logger.debug(
                    "store.save: session=%s graph_path=%s edges=%d",
                    memory.session_id,
                    turtle_path,
                    graph.n_edges,
                )

        logger.debug(
            "store.save: session=%s path=%s bytes=%d",
            memory.session_id,
            path,
            len(payload),
        )
        return path

    async def load(self, session_id: str) -> SessionMemory:
        """Load memory state from VFS.

        Raises SessionNotFoundError if no saved state exists.
        Raises SessionCorruptedError if the saved state cannot be deserialized.
        """
        path = _session_path(session_id)
        if not await self._vfs.exists(path):
            raise SessionNotFoundError(
                f"No saved session found for session_id={session_id!r}. "
                f"Create a new session with SessionMemory(session_id=...).",
            )

        try:
            payload = await self._vfs.read(path)
            data = json.loads(payload)
            memory = SessionMemory.from_dict(data, sections=self._sections)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise SessionCorruptedError(
                f"Session {session_id!r} exists but cannot be deserialized: {exc}. "
                f"The session file may be corrupted. Delete and recreate the session.",
            ) from exc
        except (KeyError, TypeError, ValueError) as exc:
            raise SessionCorruptedError(
                f"Session {session_id!r} has invalid structure: {exc}. "
                f"The snapshot may be from an incompatible version. "
                f"Delete with SessionStore.delete() and recreate the session.",
            ) from exc

        # Track 3 chunk B1: hydrate the knowledge graph from Turtle if a
        # snapshot exists. Sessions that never built a graph have no
        # graph.ttl — leave memory.graph as the lazy-empty default.
        graph_path = _session_graph_path(session_id)
        if await self._vfs.exists(graph_path):
            try:
                from kaos_graph.rdf import load_rdf

                turtle_payload = await self._vfs.read(graph_path)
                graph, _stats = load_rdf(
                    turtle_payload.decode("utf-8"),
                    format="turtle",
                )
                memory.graph = graph
                logger.debug(
                    "store.load: session=%s graph_path=%s triples=%d",
                    session_id,
                    graph_path,
                    _stats.total_triples,
                )
            except Exception as exc:
                # Graph hydration failures are non-fatal — the session
                # still loads with an empty (lazy) graph. Log and proceed.
                logger.warning(
                    "store.load: session=%s graph hydration failed: %s",
                    session_id,
                    exc,
                )

        logger.debug(
            "store.load: session=%s turns=%d tokens=%d",
            session_id,
            memory.turn_count,
            memory.total_tokens,
        )
        return memory

    async def exists(self, session_id: str) -> bool:
        """Check if a saved session exists."""
        return await self._vfs.exists(_session_path(session_id))

    async def delete(self, session_id: str) -> bool:
        """Delete a saved session. Returns True if it existed."""
        path = _session_path(session_id)
        if not await self._vfs.exists(path):
            return False
        await self._vfs.delete(path)
        logger.debug("store.delete: session=%s", session_id)
        return True

    async def list_sessions(self) -> list[str]:
        """List all saved session IDs."""
        paths = await self._vfs.list(f"{_SESSION_PREFIX}/")
        # Extract session_id from paths like "kaos-agents/sessions/{id}/memory.json"
        session_ids = []
        for path in paths:
            parts = path.strip("/").split("/")
            if len(parts) >= 3 and parts[-1] == "memory.json":
                session_ids.append(parts[-2])
        return session_ids

    async def load_or_create(
        self,
        session_id: str,
        *,
        load_recipes: bool = True,
    ) -> SessionMemory:
        """Load an existing session or create a fresh one.

        When creating a new session, automatically loads built-in recipes
        into the PLAN_EXAMPLES memory section (unless ``load_recipes=False``).
        Existing sessions already have their own persisted state and are
        not modified.

        Args:
            session_id: Session identifier.
            load_recipes: Whether to load built-in recipes into PLAN_EXAMPLES
                for new sessions. Default True.

        Returns:
            A SessionMemory — either restored from VFS or freshly created.
        """
        if await self.exists(session_id):
            return await self.load(session_id)

        logger.debug("store.load_or_create: new session %s", session_id)
        memory = SessionMemory(
            session_id=session_id,
            sections=self._sections,
        )

        if load_recipes:
            self._load_default_recipes(memory)

        return memory

    @staticmethod
    def _load_default_recipes(memory: SessionMemory) -> None:
        """Load built-in recipes into PLAN_EXAMPLES for a new session."""
        try:
            from kaos_agents.recipes import format_recipe_for_memory, load_builtin_recipes

            recipes = load_builtin_recipes()
            if recipes:
                formatted = [format_recipe_for_memory(r) for r in recipes]
                n = memory.load_explicit(MemoryType.PLAN_EXAMPLES, formatted)
                logger.debug("store: loaded %d recipes into PLAN_EXAMPLES", n)
        except Exception as exc:
            logger.debug("store: failed to load recipes (non-fatal): %s", exc)
