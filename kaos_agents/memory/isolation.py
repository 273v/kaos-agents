"""MatterClientGuard — namespace enforcement for institutional memory.

Every KB read/write requires a ``(matter_id, client_id)`` tuple.
Crossing namespaces raises :class:`MatterIsolationError` with an
agent-friendly message (what + how-to-fix + alternative).

Phase 4.A guard is a passive checker — the caller passes the
namespace explicitly per call. Phase 4+ may bind a guard to a
session at construction.

This module also exposes :class:`MatterIsolationHook` — a Runner-
level :class:`KaosHook` that detects cross-matter tool-call args
(file paths, session URIs, namespace strings) and refuses the call
when the resolved namespace doesn't match the session's bound
``matter_id``. See plan
``2026-05-22-launch-blocker-top-10.md`` §Issue 2.
"""

from __future__ import annotations

from typing import Any

from kaos_agents.errors import KaosAgentError
from kaos_agents.events.spans import Span, SpanSubject
from kaos_agents.hooks.base import HookAction, KaosHook


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


# ── Runner-installable Hook (plan §Issue 2) ─────────────────────────


def _scan_value_for_matter_ids(value: Any) -> set[str]:
    """Best-effort scan of a tool-call argument value for matter id
    fragments. Returns the set of distinct matter ids found.

    Looks for:

    * literal ``matter_id`` keys in dicts;
    * VFS paths shaped ``matters/<id>/...`` or ``sessions/<sid>/...``
      with an explicit ``matter_id=...`` query fragment;
    * any string starting with ``matter:<id>`` (canonical resource
      URI form).

    Returns an empty set when no matter id is found — the hook treats
    "no matter mentioned" as "no cross-matter access", which is the
    safe default for free-text tool args (search queries, prompts).
    """
    out: set[str] = set()
    if isinstance(value, str):
        s = value
        # ``matter:<id>`` URI form.
        if s.startswith("matter:"):
            tail = s.split("matter:", 1)[1]
            if tail:
                out.add(tail.split("/", 1)[0].strip())
        # ``matters/<id>/`` path form.
        idx = s.find("matters/")
        if idx >= 0:
            tail = s[idx + len("matters/") :]
            mid = tail.split("/", 1)[0].strip()
            if mid:
                out.add(mid)
        # ``matter_id=<id>`` query-fragment form.
        idx = s.find("matter_id=")
        if idx >= 0:
            tail = s[idx + len("matter_id=") :]
            mid = tail.split("&", 1)[0].split("/", 1)[0].strip()
            if mid:
                out.add(mid)
    elif isinstance(value, dict):
        # Literal kwarg.
        mid = value.get("matter_id")
        if isinstance(mid, str) and mid:
            out.add(mid)
        # Recurse one level into nested dicts / lists.
        for v in value.values():
            out.update(_scan_value_for_matter_ids(v))
    elif isinstance(value, (list, tuple)):
        for v in value:
            out.update(_scan_value_for_matter_ids(v))
    return out


class MatterIsolationHook(KaosHook):
    """Runner-installable hook that refuses cross-matter tool calls.

    Construct with the session's bound ``matter_id``; the hook then
    inspects ``on_tool_call_start`` events for matter-id references
    inside the tool's args. Any reference that doesn't match the
    bound matter raises :class:`MatterIsolationError` BEFORE the
    tool executes.

    Conservative default: if the bound matter is ``None`` (legacy
    session created before the field shipped), the hook is a no-op.
    Operators MUST set a matter_id on the session to engage the
    isolation guarantee — defaulting to "always block" would break
    every pre-existing session.

    Plan: §Issue 2 "Tenancy is per-token, legal model is per-matter".
    Companion to the SPA-side
    ``app/services/baa_gate.assert_session_baa_compliance`` enforcer
    that fires at HTTP boundary; this hook fires at the tool-call
    boundary inside the agent loop.
    """

    def __init__(self, *, matter_id: str | None) -> None:
        self._matter_id = matter_id

    @property
    def matter_id(self) -> str | None:
        """The bound matter id, or ``None`` if the hook is a no-op."""
        return self._matter_id

    async def on_tool_call_start(self, event: Span) -> HookAction:
        """Scan the tool's args for a cross-matter reference.

        Args are looked up from ``event.attributes`` under the
        ``"args"`` key — the dispatch layer puts them there. When the
        Span lacks args (e.g. an in-test stub), the hook returns
        ``CONTINUE`` defensively.
        """
        if self._matter_id is None:
            return HookAction.CONTINUE
        if event.subject != SpanSubject.TOOL_CALL:
            return HookAction.CONTINUE

        attrs = event.attributes if isinstance(event.attributes, dict) else {}
        args = attrs.get("args")
        if args is None:
            return HookAction.CONTINUE

        found = _scan_value_for_matter_ids(args)
        if not found:
            return HookAction.CONTINUE
        cross_matter = {m for m in found if m != self._matter_id}
        if not cross_matter:
            return HookAction.CONTINUE

        # Refuse — raise so the dispatch layer surfaces a typed error
        # to the agent (with what + fix + alternative).
        tool_name = attrs.get("tool_name", "<unknown-tool>")
        raise MatterIsolationError(
            f"Cross-matter tool call refused on tool {tool_name!r}: "
            f"session is bound to matter_id={self._matter_id!r} but "
            f"args reference matter id(s) {sorted(cross_matter)!r}. "
            f"Fix: only use file paths / URIs scoped to the bound "
            f"matter. "
            f"Alternative: spawn a new session bound to the target "
            f"matter via POST /v1/sessions with matter_id=<target>."
        )


__all__ = [
    "MatterClientGuard",
    "MatterIsolationError",
    "MatterIsolationHook",
]
