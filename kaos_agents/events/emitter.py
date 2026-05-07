"""Event emission helpers.

:class:`EventEmitter` is the boilerplate-eliminator the agent loop
uses to construct events with ``timestamp`` / ``sequence`` /
``session_id`` / ``run_id`` auto-filled. It also offers
:meth:`span_start` / :meth:`span_complete` / :meth:`span_error`
shortcuts for the common :class:`Span` cases.

:func:`emit_usage_observed` is a convenience for the
:class:`UsageObserved` field mapping that pattern dispatchers (chat,
plan, research) all share.
"""

from __future__ import annotations

import time
import uuid
from typing import TYPE_CHECKING, Any

from kaos_agents.base.event import KaosEvent
from kaos_agents.events.lifecycle import UsageObserved
from kaos_agents.events.spans import Span, SpanPhase, SpanSubject

if TYPE_CHECKING:
    from kaos_agents.types.usage import InvocationUsage


def _new_span_id() -> str:
    """Fresh 12-hex-char span ID. Independent of session/run/turn IDs
    so consumers can tie spans to those via ``parent_span_id`` instead
    of overloading the IDs."""
    return uuid.uuid4().hex[:12]


class EventEmitter:
    """Helper that auto-fills timestamp, sequence, session_id, run_id.

    Used by the agent loop to emit events without repeating boilerplate.

    Usage::

        emitter = EventEmitter(session_id="abc", run_id="run_01")
        yield emitter.span_start(SpanSubject.TURN, name="turn.1",
                                 attributes={"turn_number": 1})
        yield emitter.emit(IntentClassified, intent="tool_use", confidence=0.9)
        yield emitter.span_complete(SpanSubject.TURN, span_id=turn_span_id,
                                    duration_ms=1234.5)
    """

    __slots__ = ("_agent_id", "_run_id", "_sequence", "_session_id")

    def __init__(
        self,
        session_id: str,
        run_id: str,
        agent_id: str | None = None,
    ) -> None:
        self._session_id = session_id
        self._run_id = run_id
        self._agent_id = agent_id
        self._sequence = 0

    def emit(self, cls: type[KaosEvent], **kwargs: Any) -> KaosEvent:
        """Create an event instance with auto-filled base fields.

        Args:
            cls: The event class to instantiate.
            **kwargs: Event-specific fields.

        Returns:
            A fully-initialized event instance.
        """
        seq = self._sequence
        self._sequence += 1
        return cls(
            timestamp=time.monotonic(),
            sequence=seq,
            session_id=self._session_id,
            run_id=self._run_id,
            agent_id=kwargs.pop("agent_id", self._agent_id),
            **kwargs,
        )

    # --- Span shortcuts -------------------------------------------------

    def span(
        self,
        subject: SpanSubject,
        phase: SpanPhase,
        *,
        span_id: str | None = None,
        parent_span_id: str | None = None,
        name: str = "",
        duration_ms: float | None = None,
        error_type: str | None = None,
        error_message: str | None = None,
        attributes: dict[str, Any] | None = None,
    ) -> Span:
        """Generic Span emission. Most callers use the ``span_*`` shortcuts."""
        event = self.emit(
            Span,
            subject=subject,
            phase=phase,
            span_id=span_id or _new_span_id(),
            parent_span_id=parent_span_id,
            name=name,
            duration_ms=duration_ms,
            error_type=error_type,
            error_message=error_message,
            attributes=attributes or {},
        )
        # mypy / ty narrow: emit returns KaosEvent; we know it's a Span.
        assert isinstance(event, Span)
        return event

    def span_start(
        self,
        subject: SpanSubject,
        *,
        span_id: str | None = None,
        parent_span_id: str | None = None,
        name: str = "",
        attributes: dict[str, Any] | None = None,
    ) -> Span:
        """Emit a ``Span(subject, START)``. Returns the Span so the caller
        can capture ``span.span_id`` for later ``span_complete``."""
        return self.span(
            subject,
            SpanPhase.START,
            span_id=span_id,
            parent_span_id=parent_span_id,
            name=name,
            attributes=attributes,
        )

    def span_complete(
        self,
        subject: SpanSubject,
        *,
        span_id: str,
        parent_span_id: str | None = None,
        name: str = "",
        duration_ms: float | None = None,
        attributes: dict[str, Any] | None = None,
    ) -> Span:
        """Emit a ``Span(subject, COMPLETE)`` matched by ``span_id`` to a
        prior :meth:`span_start`."""
        return self.span(
            subject,
            SpanPhase.COMPLETE,
            span_id=span_id,
            parent_span_id=parent_span_id,
            name=name,
            duration_ms=duration_ms,
            attributes=attributes,
        )

    def span_error(
        self,
        subject: SpanSubject,
        *,
        span_id: str,
        error_type: str,
        error_message: str,
        parent_span_id: str | None = None,
        name: str = "",
        duration_ms: float | None = None,
        attributes: dict[str, Any] | None = None,
    ) -> Span:
        """Emit a ``Span(subject, ERROR)`` for a failed unit of work."""
        return self.span(
            subject,
            SpanPhase.ERROR,
            span_id=span_id,
            parent_span_id=parent_span_id,
            name=name,
            duration_ms=duration_ms,
            error_type=error_type,
            error_message=error_message,
            attributes=attributes,
        )

    @property
    def sequence(self) -> int:
        """Current sequence counter (next event will have this value)."""
        return self._sequence


def emit_usage_observed(
    emitter: EventEmitter, usage: InvocationUsage, *, source: str = ""
) -> KaosEvent:
    """Build a ``UsageObserved`` event from an ``InvocationUsage``.

    Returns ``KaosEvent`` (not ``UsageObserved``) because ``emitter.emit``
    is declared to return the common base type. Downstream consumers
    narrow with ``isinstance(event, UsageObserved)`` as usual.

    Centralized so pattern-level dispatchers (chat, plan, research)
    don't each re-learn the field mapping.
    """
    return emitter.emit(
        UsageObserved,
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        total_tokens=usage.total_tokens,
        cost_usd=usage.cost_usd,
        source=source,
    )


__all__ = ["EventEmitter", "emit_usage_observed"]
