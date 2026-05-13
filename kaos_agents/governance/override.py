"""OverrideHook — admin event injection / force-stop / replay primitives.

Paper §Q10: "admin can inject events (force-stop, clear-section,
replay-from-snapshot)". A :class:`KaosHook` whose primary purpose is
to surface admin actions through the same event stream so every
override is itself audit-logged via the JSONLAuditLogger.

Phase 5.F ships the value types + a minimal hook that emits
:class:`Span(STEP, ERROR)` events for force-stop / clear-section. The
broader replay-from-snapshot pathway is Phase 6 (requires AgentLoop's
resume-from-snapshot wiring).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum, unique
from typing import Any

from kaos_agents.events.emitter import EventEmitter
from kaos_agents.events.spans import SpanPhase, SpanSubject
from kaos_agents.hooks.base import KaosHook


@unique
class OverrideKind(StrEnum):
    """Admin override kinds — paper §Q10 vocabulary."""

    FORCE_STOP = "force_stop"  # admin demands the run terminate
    CLEAR_SECTION = "clear_section"  # admin clears a memory section
    REPLAY_SNAPSHOT = "replay_snapshot"  # restore a StateSnapshot
    INJECT_EVENT = "inject_event"  # add a synthetic event mid-run


@dataclass(frozen=True, slots=True)
class OverrideRecord:
    """One admin override action — fully auditable."""

    kind: OverrideKind
    issued_by: str  # admin id / user id
    reason: str
    issued_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    payload: dict[str, Any] = field(default_factory=dict)


class OverrideHook(KaosHook):
    """KaosHook that records and surfaces admin override actions.

    Constructor kwargs:
      authorized_admins: Optional tuple of admin id strings; when
        provided, ``record()`` rejects unauthorized issuers. ``None``
        (default) accepts any issuer (suitable for single-tenant
        deployments).

    The hook itself doesn't intercept events at runtime — it provides
    a typed entry point for admins to issue overrides. The recorded
    OverrideRecord lands in ``hook.records`` and (when an emitter is
    provided to :meth:`record`) emits a :class:`Span(STEP, ERROR)`
    event so the override is visible in the live event stream.

    Phase 6 will wire the runner to consume these (e.g. force-stop
    raises a typed exception that propagates through the Runner).
    """

    __slots__ = ("_authorized", "_records")

    def __init__(
        self,
        *,
        authorized_admins: tuple[str, ...] | None = None,
    ) -> None:
        self._authorized = authorized_admins
        self._records: list[OverrideRecord] = []

    @classmethod
    def metadata(cls) -> Any:
        from kaos_agents.types.metadata import HookMetadata

        return HookMetadata(
            name="kaos-agents-override-hook",
            description="Admin event injection / force-stop / replay.",
            listens_to=(),
        )

    @property
    def records(self) -> tuple[OverrideRecord, ...]:
        """All overrides recorded so far (immutable view)."""
        return tuple(self._records)

    def record(
        self,
        *,
        kind: OverrideKind,
        issued_by: str,
        reason: str,
        emitter: EventEmitter | None = None,
        payload: dict[str, Any] | None = None,
    ) -> OverrideRecord:
        """Record an override. Optionally emits a Span via ``emitter``.

        Raises :class:`PermissionError` when ``authorized_admins`` is
        configured and ``issued_by`` is not in the list.
        """
        if self._authorized is not None and issued_by not in self._authorized:
            raise PermissionError(
                f"Override rejected: issuer {issued_by!r} is not in "
                f"authorized_admins. Add the issuer to OverrideHook's "
                f"authorized_admins tuple, or use a hook without an "
                f"authorized list for unrestricted overrides."
            )
        record = OverrideRecord(
            kind=kind,
            issued_by=issued_by,
            reason=reason,
            payload=dict(payload or {}),
        )
        self._records.append(record)
        if emitter is not None:
            # Surface the override through the event stream so it lands
            # in the audit log alongside the run's events.
            emitter.span(
                SpanSubject.STEP,
                SpanPhase.ERROR,
                name=f"override.{kind.value}",
                error_type="AdminOverride",
                error_message=f"{issued_by}: {reason}",
                attributes={
                    "override_kind": kind.value,
                    "issued_by": issued_by,
                    "reason": reason,
                    **record.payload,
                },
            )
        return record


__all__ = ["OverrideHook", "OverrideKind", "OverrideRecord"]
