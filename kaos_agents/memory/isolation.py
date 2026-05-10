"""MatterClientGuard — namespace enforcement for institutional memory.

Every KB read/write requires a ``(matter_id, client_id)`` tuple.
Crossing namespaces raises :class:`MatterIsolationError` with an
agent-friendly message (what + how-to-fix + alternative).

Phase 4.A guard is a passive checker — the caller passes the
namespace explicitly per call. Phase 4+ may bind a guard to a
session at construction.
"""

from __future__ import annotations

from kaos_agents.errors import KaosAgentError


class MatterIsolationError(KaosAgentError):
    """Raised when a KB read/write crosses matter/client namespaces."""


class MatterClientGuard:
    """Phase 4.A guard. Stateless by default; bind a fixed namespace
    via :meth:`bind` to enforce a single namespace for the guard's
    lifetime (used by per-session KB instances).
    """

    def __init__(self, *, bound: tuple[str, str] | None = None) -> None:
        self._bound = bound

    def bind(self, matter_client: tuple[str, str]) -> MatterClientGuard:
        """Return a new guard pinned to ``matter_client``."""
        return MatterClientGuard(bound=matter_client)

    def assert_readable(self, matter_client: tuple[str, str]) -> None:
        self._assert(matter_client, op="read")

    def assert_writable(self, matter_client: tuple[str, str]) -> None:
        self._assert(matter_client, op="write")

    def _assert(self, namespace: tuple[str, str], *, op: str) -> None:
        if not isinstance(namespace, tuple) or len(namespace) != 2:
            raise MatterIsolationError(
                f"Invalid matter_client namespace for {op}: "
                f"got {namespace!r}; expected a (matter_id, client_id) tuple. "
                f"Fix: pass a 2-tuple of strings. "
                f"Alternative: use the unbound default namespace ('default','default') "
                f"for non-isolated workloads."
            )
        if self._bound is not None and namespace != self._bound:
            raise MatterIsolationError(
                f"Cross-namespace {op} blocked: "
                f"guard is bound to {self._bound!r} but caller passed {namespace!r}. "
                f"Fix: use the bound namespace, or rebind the guard via "
                f"``MatterClientGuard().bind({namespace!r})``. "
                f"Alternative: construct a new KnowledgeBase per namespace."
            )


__all__ = ["MatterClientGuard", "MatterIsolationError"]
