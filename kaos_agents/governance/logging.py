"""JSONLAuditLogger — durable JSONL audit log of every KaosEvent.

Paper §Q10: "the KaosEvent stream IS the audit log; durable JSONL
append to VFS at every event." A :class:`KaosHook` subclass that
appends one JSON-serialized line per observed event to a VFS path,
producing a tamper-evident replayable record of the whole turn.

Design:

- One audit line per event. The event is serialized via
  :func:`kaos_agents.events.serialize_event_json` (registry-backed,
  round-trippable through :func:`deserialize_event_json`).
- Append-only — never rewrites existing lines. Caller is responsible
  for log rotation / retention.
- Per-session log files by default
  (``{root}/audit/session-{session_id}.jsonl``); the path resolver is
  pluggable via the ``path_for_session`` constructor kwarg.
- Failures are logged and swallowed — the audit logger must never
  crash the agent loop. (CLAUDE.md discipline: agents tolerate
  observability failures.)

The VFS we ship in ``kaos_core.vfs.core`` exposes async ``read`` and
``write`` (bytes-in, bytes-out) but no native ``append``. We use the
read-modify-write pattern below; production backends that gain a
true append API can subclass and override :meth:`_append_line`.

Usage::

    from kaos_agents.governance import JSONLAuditLogger
    from kaos_core.vfs.core import VirtualFileSystem

    vfs = VirtualFileSystem()
    logger = JSONLAuditLogger(vfs=vfs)
    runner = Runner(agent, hooks=(logger,))
    # Every event the runner emits lands in
    # ``audit/session-<id>.jsonl`` as a JSON line.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from kaos_core.logging import get_logger

from kaos_agents.events import serialize_event_json
from kaos_agents.hooks.base import KaosHook

if TYPE_CHECKING:
    from kaos_agents.events import KaosEvent

logger = get_logger(__name__)

PathResolver = Callable[[str], str]


def _default_path_resolver(session_id: str) -> str:
    """Default audit-log path: ``audit/session-{session_id}.jsonl``."""
    safe = session_id.replace("/", "_") if session_id else "unscoped"
    return f"audit/session-{safe}.jsonl"


class JSONLAuditLogger(KaosHook):
    """KaosHook that appends every observed event to a VFS JSONL log.

    Constructor kwargs:
      vfs: A :class:`kaos_core.vfs.core.VirtualFileSystem` instance.
        When ``None`` (default), the hook is a no-op — useful for
        tests that wire the hook through the construction graph but
        don't want to exercise persistence.
      path_for_session: Callable returning the VFS path for a given
        session_id. Default produces
        ``audit/session-<session_id>.jsonl``.
      include_event_kinds: Optional tuple of class names to filter on.
        ``None`` (default) means log every KaosEvent. When non-None,
        only events whose ``type(event).__name__`` is in the tuple
        are logged.
      context_id: Optional VFS context id forwarded to ``vfs.read`` /
        ``vfs.write`` for namespace / context-scoped backends. ``None``
        falls back to the VFS's default context handling.

    Override :meth:`on_event` if you need a synchronous append path
    (the default catch-all already handles every KaosEvent subclass
    via the dispatch table).
    """

    __slots__ = ("_context_id", "_include", "_resolver", "_vfs")

    def __init__(
        self,
        *,
        vfs: Any | None = None,
        path_for_session: PathResolver | None = None,
        include_event_kinds: tuple[str, ...] | None = None,
        context_id: str | None = None,
    ) -> None:
        self._vfs = vfs
        self._resolver = path_for_session or _default_path_resolver
        self._include = include_event_kinds
        self._context_id = context_id

    @classmethod
    def metadata(cls) -> Any:
        from kaos_agents.types.metadata import HookMetadata

        return HookMetadata(
            name="kaos-agents-jsonl-audit-logger",
            description="Durable JSONL audit log of every KaosEvent.",
            listens_to=(),
        )

    async def on_event(self, event: KaosEvent) -> None:
        """Catch-all: every event flows through here.

        :class:`KaosHook` already dispatches typed callbacks
        (``on_turn_start`` etc.); :meth:`on_event` is the synthesised
        catch-all the dispatch fires for non-specific events. We
        override it so a single hook covers every event type without
        reimplementing the dispatch table.
        """
        if self._vfs is None:
            return  # configured no-op
        kind = type(event).__name__
        if self._include is not None and kind not in self._include:
            return
        try:
            session_id = getattr(event, "session_id", None) or "unscoped"
            path = self._resolver(session_id)
            line = serialize_event_json(event)
            await self._append_line(path, line)
        except Exception as exc:
            # Audit logger must never crash the agent loop — observability
            # failures are warnings, not fatal errors.
            logger.warning(
                "JSONLAuditLogger: failed to append %s for session %s: %s",
                kind,
                getattr(event, "session_id", "?"),
                exc,
            )

    async def _append_line(self, path: str, line: str) -> None:
        """Append one JSON line + newline to ``path`` via the VFS.

        Uses the VFS read-modify-write pattern. Production VFS
        backends that grow a real append API should subclass and
        override this method.
        """
        vfs = self._vfs
        if vfs is None:
            # on_event() guards against this, but be defensive for
            # subclasses that may bypass that guard.
            return
        existing = b""
        try:
            existing = await vfs.read(path, context_id=self._context_id)
        except Exception:
            # First-write path: VFS backends raise heterogeneously on
            # missing paths (FileNotFoundError, KeyError, OSError, etc.).
            existing = b""
        if existing and not existing.endswith(b"\n"):
            existing = existing + b"\n"
        new_content = existing + line.encode("utf-8") + b"\n"
        await vfs.write(path, new_content, context_id=self._context_id)


__all__ = ["JSONLAuditLogger", "PathResolver"]
