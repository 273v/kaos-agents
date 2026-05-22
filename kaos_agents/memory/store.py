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

import asyncio
import json
import os
import urllib.parse
from pathlib import Path

from kaos_core.logging import get_logger
from kaos_core.vfs.core import VirtualFileSystem

from kaos_agents.errors import SessionCorruptedError, SessionNotFoundError
from kaos_agents.memory.session import SessionMemory
from kaos_agents.types.memory import DEFAULT_SECTIONS, MemoryType, SectionConfig

logger = get_logger(__name__)

# VFS path prefix for all agent sessions
_SESSION_PREFIX = "kaos-agents/sessions"

# Characters that are safe in a single path component on every OS we
# target (Linux/macOS/Windows). Anything outside this set gets percent-
# encoded by `_safe_component` so the disk-backed VFS can materialise the
# directory on Windows, where ``:`` / ``<`` / ``>`` / ``"`` / ``|`` /
# ``?`` / ``*`` are reserved. Hex digits in percent-escapes are uppercase
# (urllib default) so the encoding is canonical and round-trips cleanly.
_SAFE_PATH_CHARS = "-_.~"


def _safe_component(name: str) -> str:
    """Encode ``name`` so it can be used as one VFS path component on any OS.

    Tenant-scoped session ids use ``<tenant>:<session>`` (see
    ``scope_session_id``), and the ``:`` is reserved on the Windows
    filesystem (alternate data streams). The disk backend would happily
    write ``./.kaos-vfs/.../sessions/<tenant>:<session>/memory.json`` on
    POSIX and fail with ``NotADirectoryError`` on Windows. Percent-
    encoding via :func:`urllib.parse.quote` keeps the disk path valid
    everywhere; the encoding is bijective so ``_unsafe_component``
    recovers the original ``session_id`` for ``list_sessions``.
    """
    return urllib.parse.quote(name, safe=_SAFE_PATH_CHARS)


def _unsafe_component(encoded: str) -> str:
    """Inverse of :func:`_safe_component` — recover the original id."""
    return urllib.parse.unquote(encoded)


def _session_path(session_id: str) -> str:
    """VFS path for a session's memory snapshot."""
    return f"{_SESSION_PREFIX}/{_safe_component(session_id)}/memory.json"


def _session_graph_path(session_id: str) -> str:
    """VFS path for a session's RDF knowledge graph (Turtle).

    Track 3 chunk B1 — the per-session knowledge graph persists alongside
    the JSON memory snapshot. Both files live under
    ``{_SESSION_PREFIX}/{_safe_component(session_id)}/`` so a session is
    one directory in VFS that carries everything (memory + graph).
    """
    return f"{_SESSION_PREFIX}/{_safe_component(session_id)}/graph.ttl"


async def _atomic_write(vfs: VirtualFileSystem, path: str, data: bytes) -> None:
    """Write ``data`` to ``path`` atomically (KC17-P1-3).

    Pre-KC17 ``save()`` called ``vfs.write(memory.json)`` followed by
    ``vfs.write(graph.ttl)`` as two non-atomic operations. A SIGTERM
    between the writes left a torn on-disk state that the next ``load()``
    consumed. Even within a single file, ``vfs.write(path, data)`` on the
    disk backend uses ``Path.write_bytes`` which truncates the file before
    writing the new bytes — so SIGTERM mid-write left an empty file the
    next loader treated as corrupt JSON.

    This helper writes to ``{path}.tmp`` first, fsyncs the fd, then
    ``os.replace``s into place. ``os.replace`` is POSIX-atomic on the
    same filesystem so the next ``load()`` sees either the OLD bytes or
    the NEW bytes — never partial bytes.

    Behaviour by backend:

    - DiskBackend: temp+fsync+os.replace via the resolved disk path; also
      fsyncs the parent directory (Linux durability). The VFS abstraction
      doesn't expose temp+rename so we drop to raw Path ops on the
      resolved disk path. The path resolver still enforces the safe-join
      boundary so this is not an escape from VFS sandboxing.
    - Memory / other backends: a torn state isn't reachable (in-process
      bytes) so we fall back to ``vfs.write`` and rely on the backend
      being internally coherent.
    """
    disk_path = vfs.resolve_disk_path(path)
    if disk_path is None:
        # Non-disk backend — no kernel-level rename available. The
        # memory backend's write is internally atomic (single dict
        # assignment), so torn states aren't a concern here.
        await vfs.write(path, data)
        return

    # Disk-backed path: temp+fsync+replace. We can't reuse vfs.write
    # because it doesn't expose the temp/rename split, but we WANT to
    # respect the resolved sandbox path.
    tmp_path: Path = disk_path.with_suffix(disk_path.suffix + ".tmp")

    def _do_write() -> None:
        import contextlib

        disk_path.parent.mkdir(parents=True, exist_ok=True)
        # Write to .tmp then atomically replace.
        with tmp_path.open("wb") as fp:
            fp.write(data)
            fp.flush()
            # fsync can fail on some filesystems (e.g. tmpfs); the
            # os.replace below is still atomic so we proceed.
            with contextlib.suppress(OSError):
                os.fsync(fp.fileno())
        # os.replace is the POSIX-atomic rename; Path.replace wraps it,
        # but we use the os.replace form directly to make the atomicity
        # contract explicit at the call site (and to mirror the standard
        # tmp+rename idiom in the Python docs).
        os.replace(tmp_path, disk_path)  # noqa: PTH105 — explicit POSIX rename
        # Best-effort directory fsync (Linux durability — the rename
        # is only persistent after the parent dir's metadata is on
        # disk). Ignored on platforms that don't support it.
        try:
            fd = os.open(disk_path.parent, os.O_RDONLY)
        except OSError:
            return
        try:
            with contextlib.suppress(OSError):
                os.fsync(fd)
        finally:
            os.close(fd)

    await asyncio.to_thread(_do_write)


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

        KC17-P1-3: both writes go through :func:`_atomic_write` which
        does temp+fsync+os.replace on disk-backed VFS so a SIGTERM
        between the two writes leaves either both-old or both-new bytes
        on disk — never a torn state that ``load()`` reads as corrupt.
        """
        path = _session_path(memory.session_id)
        data = memory.to_dict()
        payload = json.dumps(data, separators=(",", ":"), default=str).encode()
        await _atomic_write(self._vfs, path, payload)

        # Persist the knowledge graph as Turtle, if it exists and has
        # any triples. We touch the lazy ``.graph`` property only if it
        # was already constructed — accessing it here would force-init
        # an empty graph for every session, defeating the lazy design.
        if memory._graph is not None:
            graph = memory._graph
            if graph.n_edges > 0:
                from kaos_graph.rdf import to_turtle

                turtle_path = _session_graph_path(memory.session_id)
                await _atomic_write(self._vfs, turtle_path, to_turtle(graph).encode())
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
        """Delete a saved session and all sibling files. Returns True if it existed.

        KC17-P1-1: removes ALL files under ``{_SESSION_PREFIX}/{session_id}/``,
        not just ``memory.json``. Pre-KC17 this left ``graph.ttl`` (and any
        future per-session files) on disk so the next ``exists()`` returned
        True for the session even after a successful DELETE — a privacy /
        right-to-delete defect for regulated deployments.

        Returns ``True`` when at least one session-owned file was removed;
        ``False`` when the session directory was already empty (idempotent).
        """
        memory_path = _session_path(session_id)
        graph_path = _session_graph_path(session_id)

        removed = 0
        for path in (memory_path, graph_path):
            if await self._vfs.exists(path):
                try:
                    await self._vfs.delete(path)
                    removed += 1
                except Exception as exc:
                    # Best-effort: keep going so a partial failure
                    # doesn't leak the other files. The audit trail
                    # carries the warning.
                    logger.warning(
                        "store.delete: failed to delete %s (continuing): %s",
                        path,
                        exc,
                    )

        if removed == 0:
            return False
        logger.debug("store.delete: session=%s files_removed=%d", session_id, removed)
        return True

    async def list_sessions(self) -> list[str]:
        """List all saved session IDs."""
        paths = await self._vfs.list(f"{_SESSION_PREFIX}/")
        # Extract session_id from paths like
        # "kaos-agents/sessions/{encoded_id}/memory.json". The directory
        # name is percent-encoded by _safe_component so the disk-backed
        # VFS can land it on Windows; we decode here to return the
        # caller-supplied session_id verbatim.
        session_ids = []
        for path in paths:
            parts = path.strip("/").split("/")
            if len(parts) >= 3 and parts[-1] == "memory.json":
                session_ids.append(_unsafe_component(parts[-2]))
        return session_ids

    async def load_or_create(
        self,
        session_id: str,
        *,
        load_recipes: bool = True,
        matter_id: str | None = None,
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
            matter_id: Plan §Issue 2 — optional firm-side ethical-wall
                identifier (e.g. ``"ABC-2026-0042"``). Only applied to
                newly-created sessions; existing sessions keep their
                persisted ``matter_id``. ``None`` (default) leaves the
                session unscoped.

        Returns:
            A SessionMemory — either restored from VFS or freshly created.
        """
        if await self.exists(session_id):
            return await self.load(session_id)

        logger.debug("store.load_or_create: new session %s", session_id)
        memory = SessionMemory(
            session_id=session_id,
            sections=self._sections,
            matter_id=matter_id,
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
