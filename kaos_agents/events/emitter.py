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

import contextlib
import contextvars
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

from kaos_agents.base.event import KaosEvent
from kaos_agents.events.collector import active_collector, push_event
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

    __slots__ = ("_agent_id", "_run_id", "_sequence", "_session_id", "_span_starts")

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
        # Per-emitter map of span_id → monotonic start time. Populated
        # on span_start, drained on span_complete / span_error so we
        # can auto-compute duration_ms when callers don't provide one.
        # Bounded implicitly: the agent loop opens/closes spans in
        # roughly LIFO order and each turn empties the dict. We never
        # reach pathological size in normal use; if a START never gets
        # a matching COMPLETE the entry leaks until the emitter is
        # garbage collected — acceptable since EventEmitter is per-run.
        self._span_starts: dict[str, float] = {}

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
        event = cls(
            timestamp=time.monotonic(),
            sequence=seq,
            session_id=self._session_id,
            run_id=self._run_id,
            agent_id=kwargs.pop("agent_id", self._agent_id),
            **kwargs,
        )
        # Push to the active collector if one is in scope. No-op
        # otherwise, preserving backward compat for callers that emit
        # without opening a collect_events() block.
        push_event(event)
        return event

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
        """Generic Span emission. Most callers use the ``span_*`` shortcuts.

        When ``parent_span_id`` is omitted on a START phase and an
        :class:`EventCollector` is active for the current task, the
        parent is synthesized from the collector's span stack — this
        threads OTel parenting through nested ``span_start`` calls
        without making every caller pass the parent explicitly.

        COMPLETE / ERROR / CANCELLED / PROGRESS phases never auto-
        synthesize: those phases match a prior START's ``span_id``, and
        the caller already has the START's ``parent_span_id`` in scope
        if it wants to propagate it. Auto-synthesis here would read
        the collector stack *after* the START was popped (or not), and
        produce surprising parents.
        """
        # Synthesize parent from the active collector when the caller
        # didn't supply one. Only on START — see docstring above.
        if parent_span_id is None and phase is SpanPhase.START:
            coll = active_collector()
            if coll is not None:
                parent_span_id = coll.current_parent_span_id()

        resolved_span_id = span_id or _new_span_id()

        # Auto-track span durations. On START we record the monotonic
        # start time; on COMPLETE / ERROR we look it up and compute
        # duration_ms when the caller didn't pass one (most call sites
        # historically passed `duration_ms=0.0` since per-span timing
        # was not measured anywhere — explain records showed 0.0 for
        # every tool call). Caller-supplied durations win so existing
        # measured paths (e.g. TURN's turn_duration_ms in BaseAgent)
        # keep their values.
        if phase is SpanPhase.START:
            self._span_starts[resolved_span_id] = time.monotonic()
        elif phase in (SpanPhase.COMPLETE, SpanPhase.ERROR, SpanPhase.CANCELLED):
            t0 = self._span_starts.pop(resolved_span_id, None)
            if t0 is not None and (duration_ms is None or duration_ms == 0.0):
                duration_ms = (time.monotonic() - t0) * 1000.0

        event = self.emit(
            Span,
            subject=subject,
            phase=phase,
            span_id=resolved_span_id,
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


def emit_thinking_from_invocation(
    emitter: EventEmitter,
    invocation: Any,
) -> KaosEvent | None:
    """Emit a ``ThinkingDelta`` from an Invocation's native-thinking content.

    kaos-llm-client surfaces model reasoning blocks (Anthropic
    extended thinking, OpenAI ``o1`` reasoning summaries, Google
    ``thinkingConfig``) as ``Completion.thinking``. kaos-llm-core's
    ChainOfThought + Call paths copy the concatenated thinking text
    onto ``Invocation.extras["native_thinking"]`` so downstream
    consumers can read it without unpacking the raw provider response.

    Without this bridge, model reasoning was invisible in kaos-agents'
    JSONL audit log even when the underlying provider exposed it.
    Pattern dispatchers (chat / research / plan-execute) call this
    helper after ``Call.invoke`` to surface thinking before
    TextDelta.

    Args:
        emitter: Active EventEmitter for the current turn.
        invocation: A kaos-llm-core ``Invocation`` whose ``extras``
            may carry ``native_thinking``.

    Returns:
        The emitted ``ThinkingDelta`` event, or ``None`` when no
        thinking content was present (most calls — thinking is
        opt-in per provider config).
    """
    if invocation is None:
        return None
    extras = getattr(invocation, "extras", None) or {}
    thinking = str(extras.get("native_thinking") or "").strip()
    if not thinking:
        return None
    # Local import to avoid a cyclic import — this module is imported
    # by events/__init__.py which also defines ThinkingDelta.
    from kaos_agents.events.stream import ThinkingDelta

    return emitter.emit(ThinkingDelta, content=thinking)


# ---------------------------------------------------------------------------
# Active EventEmitter ContextVar
#
# Mirrors the ``_active_collector_var`` / :func:`collect_events` /
# :func:`active_collector` triple in :mod:`kaos_agents.events.collector`.
# The collector solves "where do emitted events accumulate"; this primitive
# solves the symmetric problem of "how do helpers deep in the call stack
# *produce* an event with full base-field metadata when no emitter was
# threaded through their signature".
#
# Theme A motivation (audit 2026-05-25):
#   ``kaos_agents/capabilities/retrieve.py:_invoke_tool``,
#   ``kaos_agents/actions/tool_bridge.py`` (asyncio.timeout handler),
#   ``kaos_agents/planning/act.py::_act_tool`` (asyncio.timeout handler)
# all swallowed tool-execution failures silently because no
# :class:`EventEmitter` was in scope. With this primitive each of those
# sites can call ``active_emitter().emit(...)`` (gracefully degrading to
# a no-op when no emitter is in scope, preserving the back-compat
# contract that callers without a ``use_emitter`` block continue to
# work unchanged).
# ---------------------------------------------------------------------------

_active_emitter_var: contextvars.ContextVar[EventEmitter | None] = contextvars.ContextVar(
    "kaos_agents_event_emitter", default=None
)


@contextmanager
def use_emitter(emitter: EventEmitter) -> Iterator[EventEmitter]:
    """Make ``emitter`` the active EventEmitter for the duration of the with-block.

    Helpers deep in the call stack (e.g. tool execution wrappers in
    :mod:`kaos_agents.capabilities.retrieve`,
    :mod:`kaos_agents.actions.tool_bridge`,
    :mod:`kaos_agents.planning.act`) can call
    ``active_emitter().emit(SomeEvent, ...)`` without needing the emitter
    threaded through every call signature.

    Mirrors the ``_active_collector_var`` / :func:`collect_events`
    pattern from :mod:`kaos_agents.events.collector`. Nested scopes
    restore the prior emitter on exit (via the ContextVar token).

    Usage::

        emitter = EventEmitter(session_id=sid, run_id=rid)
        with collect_events() as coll, use_emitter(emitter):
            ...  # any helper can call active_emitter().emit(...)
        # collector saw every event the helpers emitted.
    """
    token = _active_emitter_var.set(emitter)
    try:
        yield emitter
    finally:
        # ContextVar.reset() raises ValueError when the token was set
        # in a different Context (e.g. when use_emitter wraps an async
        # generator that yields control to a consumer in a different
        # task — same reason collect_events suppresses the same error
        # in collector.py). The contextvar auto-resets when its
        # owning context exits, so swallowing this is safe.
        with contextlib.suppress(ValueError):
            _active_emitter_var.reset(token)


def active_emitter() -> EventEmitter | None:
    """Return the active EventEmitter, or ``None`` if none is in scope.

    Callers MUST check the return for ``None`` — most call paths today
    don't open a :func:`use_emitter` scope, so this returns ``None``
    outside agent runs. Treat that as "skip the emit" rather than an
    error: the back-compat contract is that any caller not opening
    ``use_emitter`` continues to work unchanged.
    """
    return _active_emitter_var.get()


__all__ = [
    "EventEmitter",
    "active_emitter",
    "emit_thinking_from_invocation",
    "emit_usage_observed",
    "use_emitter",
]
