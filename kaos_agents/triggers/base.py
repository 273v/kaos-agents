"""Trigger taxonomy — paper §Q1: how does the agent know it has work?

A :class:`Trigger` is the typed event that opens a turn. The agent
loop is trigger-source agnostic: MCP tool calls, HTTP requests, CLI
prompts, scheduled jobs, sub-agent escalations, webhooks, filesystem
changes, and parent → sub-agent delegations all produce the same
value type with a ``kind`` discriminator.

Phase 2.A ships:

- :class:`TriggerKind` — the 8-member ``StrEnum`` discriminator.
- :class:`Trigger` — the frozen pydantic event; subclasses
  :class:`kaos_agents.base.event.KaosEvent` so triggers flow through
  the existing event-stream / serializer machinery.
- Eight source-tagged ``classmethod`` factories on ``Trigger``
  (``.mcp(...)``, ``.http(...)``, ``.cli(...)``, ``.scheduled(...)``,
  ``.escalation(...)``, ``.webhook(...)``, ``.filesystem(...)``,
  ``.delegation(...)``).
- :class:`TriggerSource` — the async-iterator ABC for pull-based
  sources (cron, filesystem watch, message-queue consumers). Concrete
  implementations land in Phase 4+.

The ``payload`` is intentionally a free-form ``Mapping[str, Any]`` so
each source can carry its own discriminated fields without forcing
the AgentLoop to know each shape. Convention by kind:

    MCP:        {"message": str, "tool_name": str | None}
    HTTP:       {"message": str, "request_id": str | None,
                 "headers": Mapping[str, str]}
    CLI:        {"message": str, "tty": bool}
    SCHEDULED:  {"job_name": str, "fired_at": str (ISO-8601),
                 "schedule": str | None}
    ESCALATION: {"reason": str, "kind": str (an EscalationKind),
                 "parent_turn_id": str | None,
                 "details": Mapping[str, Any]}
    WEBHOOK:    {"event_type": str, "body": Any,
                 "signature": str | None}
    FILESYSTEM: {"path": str, "event": str, "stat": Mapping[str, Any]}
    DELEGATION: {"goal": str, "parent_turn_id": str,
                 "sub_agent_hash": str | None}

Triggers fire *before* a turn is bound to a session/run, so the
inherited ``KaosEvent`` base fields are filled with pre-turn defaults:
``timestamp = time.monotonic()``, ``sequence = 0``, ``session_id =
""``, ``run_id = ""``, ``agent_id = None``. The :class:`EventEmitter`
re-broadcasts a ``Span(TURN, START)`` keyed by the loop's ``run_id``
once the turn opens.
"""

from __future__ import annotations

import time
import warnings
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Mapping
from datetime import UTC, datetime
from enum import StrEnum, unique
from typing import Any

from pydantic import ConfigDict, Field

from kaos_agents.base.event import KaosEvent

# The ``metadata`` field on :class:`Trigger` (declared below) intentionally
# shadows the :meth:`KaosEvent.metadata` *classmethod* (which describes
# the event subclass — name/description/category, not per-instance
# attributes). At the instance level ``trigger.metadata`` is the
# per-trigger string mapping; the classmethod stays accessible as
# ``Trigger.metadata()``. Pydantic emits a ``UserWarning`` about the
# name collision; suppress it locally so the public surface stays
# quiet without globally silencing genuine pydantic shadowing in other
# code.
warnings.filterwarnings(
    "ignore",
    message=(
        r'Field name "metadata" in "Trigger" '
        r'shadows an attribute in parent "KaosEvent"'
    ),
    category=UserWarning,
)


@unique
class TriggerKind(StrEnum):
    """How a turn was kicked off.

    Eight members, mirroring paper §Q1 / plan §4 Q1. Subclasses do
    not extend this enum — new sources (e.g. message-queue consumers)
    should fit into one of the existing kinds via ``payload`` shape,
    or warrant a deliberate enum addition with a paper §Q1 update.
    """

    MCP = "mcp"  # inbound MCP tool invocation
    HTTP = "http"  # FastAPI / REST POST
    CLI = "cli"  # interactive REPL / one-shot CLI
    SCHEDULED = "scheduled"  # cron / interval (paper §2.3)
    ESCALATION = "escalation"  # sub-agent escalating to parent / human (paper §2.4)
    WEBHOOK = "webhook"  # external feed (paper §2.1)
    FILESYSTEM = "filesystem"  # filesystem watch
    DELEGATION = "delegation"  # parent agent delegating to a sub-agent


class Trigger(KaosEvent, register=False):
    """Frozen typed event that opens an agent turn.

    Mirrors paper §Q1 / plan §5: ``kind`` + ``source_id`` + ``payload``
    + ``occurred_at`` + ``correlation_id`` + ``metadata``. Subclasses
    :class:`kaos_agents.base.event.KaosEvent` so triggers flow through
    the existing event-stream / serializer machinery.

    Phase 2.A intentionally **does not** register ``Trigger`` in
    :data:`kaos_agents.registry.event_registry.default_event_registry`
    — that wiring is a Phase 2.B/2.C concern. The ``register=False``
    keyword on the class declaration suppresses the auto-registration
    hook in :meth:`KaosEvent.__init_subclass__`.

    Use the source-tagged factory classmethods (:meth:`mcp`,
    :meth:`http`, :meth:`cli`, :meth:`scheduled`, :meth:`escalation`,
    :meth:`webhook`, :meth:`filesystem`, :meth:`delegation`) instead of
    constructing directly — they fill in the right ``kind`` and
    ``payload`` shape per source.
    """

    # ``frozen=True``: triggers are immutable post-construction.
    # ``extra="forbid"``: catch construction typos at the Python boundary
    #   (Trigger is a pre-turn input, not a wire-receive event — so the
    #   forward-compat ``extra="ignore"`` of the KaosEvent base does not
    #   apply here; mistyped keys should raise loudly).
    model_config = ConfigDict(frozen=True, extra="forbid", populate_by_name=True)

    kind: TriggerKind
    source_id: str = ""  # session_id / request_id / cron_name / parent_turn_id
    payload: Mapping[str, Any] = Field(default_factory=dict)
    occurred_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    correlation_id: str | None = None  # for trace correlation
    metadata: Mapping[str, str] = Field(default_factory=dict)

    # ---- source-tagged factories --------------------------------------

    @classmethod
    def mcp(
        cls,
        message: str,
        *,
        session_id: str | None = None,
        tool_name: str | None = None,
        correlation_id: str | None = None,
        metadata: Mapping[str, str] | None = None,
    ) -> Trigger:
        """Build a ``MCP``-kind trigger from an inbound MCP tool call.

        ``session_id`` becomes ``source_id``; ``tool_name`` is the name
        of the MCP tool that received the request (e.g.
        ``"kaos-agent-chat"``).
        """
        return cls._build(
            TriggerKind.MCP,
            source_id=session_id or "",
            payload={"message": message, "tool_name": tool_name},
            correlation_id=correlation_id,
            metadata=metadata,
        )

    @classmethod
    def http(
        cls,
        message: str,
        *,
        request_id: str | None = None,
        headers: Mapping[str, str] | None = None,
        correlation_id: str | None = None,
        metadata: Mapping[str, str] | None = None,
    ) -> Trigger:
        """Build an ``HTTP``-kind trigger from a FastAPI / REST POST.

        ``request_id`` becomes ``source_id`` (and is also echoed in
        ``payload`` so downstream consumers don't have to reach across
        layers). ``headers`` is copied to a plain dict — the original
        request's ``Headers`` mapping may not survive serialization.
        """
        return cls._build(
            TriggerKind.HTTP,
            source_id=request_id or "",
            payload={
                "message": message,
                "request_id": request_id,
                "headers": dict(headers or {}),
            },
            correlation_id=correlation_id,
            metadata=metadata,
        )

    @classmethod
    def cli(
        cls,
        message: str,
        *,
        session_id: str | None = None,
        tty: bool = False,
        correlation_id: str | None = None,
        metadata: Mapping[str, str] | None = None,
    ) -> Trigger:
        """Build a ``CLI``-kind trigger from an interactive REPL or
        one-shot CLI invocation.

        ``tty`` distinguishes interactive (``True``) from non-interactive
        / piped (``False``) sessions — relevant for ``--max-cost`` exit
        code disambiguation in ``kaos-agent chat``.
        """
        return cls._build(
            TriggerKind.CLI,
            source_id=session_id or "",
            payload={"message": message, "tty": tty},
            correlation_id=correlation_id,
            metadata=metadata,
        )

    @classmethod
    def scheduled(
        cls,
        *,
        job_name: str,
        fired_at: datetime | None = None,
        schedule: str | None = None,
        correlation_id: str | None = None,
        metadata: Mapping[str, str] | None = None,
    ) -> Trigger:
        """Build a ``SCHEDULED``-kind trigger from a cron / interval.

        ``fired_at`` defaults to ``datetime.now(UTC)`` and is serialized
        to ISO-8601 in ``payload`` (datetimes don't survive a generic
        ``Mapping[str, Any]`` round-trip cleanly; the canonical
        ``occurred_at`` field at the top level keeps the typed value).
        """
        fired = fired_at or datetime.now(UTC)
        return cls._build(
            TriggerKind.SCHEDULED,
            source_id=job_name,
            payload={
                "job_name": job_name,
                "fired_at": fired.isoformat(),
                "schedule": schedule,
            },
            correlation_id=correlation_id,
            metadata=metadata,
        )

    @classmethod
    def escalation(
        cls,
        *,
        reason: str,
        kind: str = "domain_specific",
        parent_turn_id: str | None = None,
        details: Mapping[str, Any] | None = None,
        correlation_id: str | None = None,
        metadata: Mapping[str, str] | None = None,
    ) -> Trigger:
        """Build an ``ESCALATION``-kind trigger from a sub-agent
        escalating to its parent (or human).

        ``kind`` is the escalation taxonomy member (e.g. ``"clarification_needed"``,
        ``"budget_exceeded"``, ``"domain_specific"``). It is namespaced
        under ``payload["kind"]`` to avoid colliding with the top-level
        :class:`TriggerKind` discriminator.
        """
        return cls._build(
            TriggerKind.ESCALATION,
            source_id=parent_turn_id or "",
            payload={
                "reason": reason,
                "kind": kind,
                "parent_turn_id": parent_turn_id,
                "details": dict(details or {}),
            },
            correlation_id=correlation_id,
            metadata=metadata,
        )

    @classmethod
    def webhook(
        cls,
        *,
        event_type: str,
        body: Any = None,
        signature: str | None = None,
        source_id: str | None = None,
        correlation_id: str | None = None,
        metadata: Mapping[str, str] | None = None,
    ) -> Trigger:
        """Build a ``WEBHOOK``-kind trigger from an external feed.

        ``source_id`` defaults to ``event_type`` when not supplied, so
        consumers can group by event class without the caller having to
        synthesize a redundant id. ``signature`` carries the verification
        material the webhook source provided (HMAC, JWS, etc.).
        """
        return cls._build(
            TriggerKind.WEBHOOK,
            source_id=source_id or event_type,
            payload={
                "event_type": event_type,
                "body": body,
                "signature": signature,
            },
            correlation_id=correlation_id,
            metadata=metadata,
        )

    @classmethod
    def filesystem(
        cls,
        *,
        path: str,
        event: str = "modified",
        stat: Mapping[str, Any] | None = None,
        correlation_id: str | None = None,
        metadata: Mapping[str, str] | None = None,
    ) -> Trigger:
        """Build a ``FILESYSTEM``-kind trigger from a path watcher.

        ``event`` is the watch event class (``"created"`` /
        ``"modified"`` / ``"deleted"`` / ``"moved"``). ``stat`` is a
        ``Mapping[str, Any]`` of stat fields the watcher captured — the
        loose typing accommodates the differences between ``watchdog``,
        ``watchfiles``, and platform-native APIs.
        """
        return cls._build(
            TriggerKind.FILESYSTEM,
            source_id=path,
            payload={"path": path, "event": event, "stat": dict(stat or {})},
            correlation_id=correlation_id,
            metadata=metadata,
        )

    @classmethod
    def delegation(
        cls,
        *,
        goal: str,
        parent_turn_id: str,
        sub_agent_hash: str | None = None,
        correlation_id: str | None = None,
        metadata: Mapping[str, str] | None = None,
    ) -> Trigger:
        """Build a ``DELEGATION``-kind trigger when a parent agent
        delegates to a sub-agent.

        ``parent_turn_id`` is required (delegation has no meaning
        without a parent context); ``sub_agent_hash`` identifies which
        sub-agent received the work — typically a content hash of the
        ``Agent`` config so identical sub-agents share trace history.
        """
        return cls._build(
            TriggerKind.DELEGATION,
            source_id=parent_turn_id,
            payload={
                "goal": goal,
                "parent_turn_id": parent_turn_id,
                "sub_agent_hash": sub_agent_hash,
            },
            correlation_id=correlation_id,
            metadata=metadata,
        )

    # ---- private helper -------------------------------------------------

    @classmethod
    def _build(
        cls,
        kind: TriggerKind,
        *,
        source_id: str,
        payload: Mapping[str, Any],
        correlation_id: str | None,
        metadata: Mapping[str, str] | None,
    ) -> Trigger:
        """Build a :class:`Trigger` with pre-turn defaults filled in.

        Triggers fire **before** a turn binds a session_id / run_id, so
        the inherited :class:`KaosEvent` base fields use empty defaults
        here. The :class:`kaos_agents.events.emitter.EventEmitter` will
        re-broadcast a ``Span(TURN, START)`` keyed by the loop's
        ``run_id`` once the turn opens.
        """
        return cls(
            timestamp=time.monotonic(),
            sequence=0,
            session_id="",
            run_id="",
            agent_id=None,
            kind=kind,
            source_id=source_id,
            payload=dict(payload),
            correlation_id=correlation_id,
            metadata=dict(metadata or {}),
        )


# ---- TriggerSource ABC (Phase 2 contract; concrete sources Phase 4+) ----


class TriggerSource(ABC):
    """An async iterable producing :class:`Trigger` events.

    Pull-based sources (cron, filesystem watch, message-queue
    consumers) implement this to plug into a Runner. Push-based
    sources (MCP / HTTP / CLI) construct a :class:`Trigger` directly
    in their request handlers and don't need a ``TriggerSource``.

    Phase 2.A ships the contract; concrete implementations land in
    Phase 4+ when the surrounding runner / loop machinery exists.
    """

    @abstractmethod
    def __aiter__(self) -> AsyncIterator[Trigger]:
        """Return an async iterator of :class:`Trigger` events.

        Implementations typically write ``async def __aiter__(self)``
        as an async generator — that satisfies both the abstract
        method and the protocol's ``AsyncIterator`` return type.
        """

    @abstractmethod
    async def close(self) -> None:
        """Stop emitting triggers and release any resources."""
