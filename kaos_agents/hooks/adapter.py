"""Adapter from KaosHook → kaos-llm-core CallHooks / ProgramHooks.

The audit found that pattern dispatchers construct kaos-llm-core
Programs (ReAct, RAG, Refine, BestOfN) without forwarding the runner's
KaosHook tuple. As a result, on_call_start / on_iteration /
on_validation_retry events from the inner Programs land in
kaos-llm-core's logger and never reach the agent's hook stream — so a
single ``OTelHook`` can observe one layer or the other, but not both.

This adapter wraps a tuple of :class:`KaosHook` into a
:class:`~kaos_llm_core.programs.hooks.CallHooks` (per-Call boundary)
and a :class:`~kaos_llm_core.programs.program_hooks.ProgramHooks`
(per-Program boundary). When ReAct fires ``on_call_start``, the
adapter dispatches a synthesized :class:`~kaos_agents.events.Span`
event into every :class:`KaosHook` in the tuple via
:func:`kaos_agents.hooks.dispatch.dispatch_hook`. Phase 0.C ships the
adapter; AgentLoop wiring is Phase 2.

Hook signatures (kaos-llm-core Phase 10A)::

    on_call_start(call, inputs, *, context=None)
    on_call_end(call, inputs, invocation, *, context=None)
    on_call_error(call, inputs, exception, *, context=None)
    on_validation_retry(call, inputs, attempt, error, *, context=None)
    on_program_start(program, inputs, *, context=None)
    on_iteration(program, iteration, payload, *, context=None)
    on_program_end(program, inputs, invocation, *, context=None)
    on_program_error(program, inputs, exception, *, context=None)

Design choices
--------------

**Sync → async bridge.** ``fire_hook`` and ``fire_program_hook`` in
kaos-llm-core invoke their callbacks **synchronously** — they do not
await coroutines. Our :func:`dispatch_hook` is ``async``. We bridge by
making the adapter callbacks ``def`` (sync) and using
``asyncio.get_running_loop().create_task(dispatch_hook(...))`` to
fire-and-forget the dispatch on the running loop. ReAct / Refine / RAG
all run inside ``async`` Call/Program code, so a running loop is
guaranteed in practice. If no running loop is present (e.g. the
adapter is exercised from a sync test or a thread without a loop), we
fall back to ``asyncio.run`` — this is best-effort observability and
matches the kaos-llm-core "hooks must never affect control flow"
contract.

**SpanSubject choice.** ``SpanSubject.LLM_CALL`` already exists for
"one LLM invocation (transport-level)" — that is exactly what
``CallHooks`` boundaries are. ``SpanSubject.LLM_PROGRAM`` does **not**
exist; rather than add a new enum member in Phase 0.C (a wider
change), ``ProgramHooks`` boundaries map onto ``SpanSubject.STEP`` —
a Program iteration is a step-shaped unit of work in the agent's
hook taxonomy. Phase 2 may introduce a dedicated ``LLM_PROGRAM``
subject and remap.

**Empty-tuple short-circuit.** ``adapt_hooks(())`` returns empty
:class:`CallHooks` and :class:`ProgramHooks` instances (all four
callback slots ``None``). This avoids the dispatch overhead for users
who don't register any hooks and keeps the kaos-llm-core "hook is
``None`` → skip" fast path active.

**Exception transparency.** :func:`dispatch_hook` already swallows
exceptions from individual ``KaosHook.on_*`` callbacks. ``fire_hook``
in kaos-llm-core also swallows exceptions raised by the adapter
callback itself. So a raising user hook propagates no further than
``dispatch_hook`` — consistent with both observability contracts.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from typing import TYPE_CHECKING, Any

from kaos_core.logging import get_logger
from kaos_llm_core.programs.hooks import CallHooks
from kaos_llm_core.programs.program_hooks import ProgramHooks

from kaos_agents.events.spans import Span, SpanPhase, SpanSubject
from kaos_agents.hooks.base import HookAction
from kaos_agents.hooks.dispatch import dispatch_hook

if TYPE_CHECKING:
    from kaos_agents.hooks.base import KaosHook

logger = get_logger(__name__)


# Subject mapping. ``LLM_CALL`` exists in ``SpanSubject``; ``LLM_PROGRAM``
# does not. We deliberately do not introduce a new enum member in
# Phase 0.C — Phase 2 owns that wider change.
_CALL_SUBJECT = SpanSubject.LLM_CALL
_PROGRAM_SUBJECT = SpanSubject.STEP


# Module-level strong-reference set for fire-and-forget dispatch tasks.
# Each task removes itself in a done-callback. Guards against GC of
# scheduled-but-not-yet-running tasks (RUF006). ``dispatch_hook`` returns
# :class:`HookAction`, hence the parametrization.
_PENDING_TASKS: set[asyncio.Task[HookAction]] = set()


def _new_span_id() -> str:
    """12-hex-char span id — independent of the agent's run/turn ids."""
    return uuid.uuid4().hex[:12]


def _safe_attr_str(value: Any) -> str:
    """Best-effort string for a hook payload field, capped to 500 chars.

    Hooks fire across arbitrary user-supplied programs/inputs/exceptions —
    a payload that doesn't ``repr`` cleanly must not break the adapter.
    """
    try:
        text = repr(value)
    except Exception as exc:  # pragma: no cover - defensive
        text = f"<unrepresentable: {type(exc).__name__}>"
    if len(text) > 500:
        return text[:497] + "..."
    return text


def _context_ids(context: Any) -> tuple[str, str]:
    """Pull (session_id, run_id) off a KaosContext, with empty fallbacks.

    ``KaosContext`` is the kaos-core per-task context object that
    kaos-llm-core threads through hooks for session/trace correlation.
    The adapter is tolerant: a ``None`` context (or any object missing
    those attributes) falls back to empty strings.
    """
    if context is None:
        return "", ""
    session_id = str(getattr(context, "session_id", "") or "")
    run_id = str(getattr(context, "run_id", "") or "")
    return session_id, run_id


def _make_span(
    *,
    subject: SpanSubject,
    phase: SpanPhase,
    name: str,
    span_id: str,
    attributes: dict[str, Any],
    context: Any,
    duration_ms: float | None = None,
    error_type: str | None = None,
    error_message: str | None = None,
) -> Span:
    """Build a synthesized :class:`Span` for forwarding to KaosHooks.

    The adapter is constructed before any agent loop runs, so we don't
    have an :class:`EventEmitter` in scope. We fill ``timestamp`` from
    :func:`time.monotonic` and read ``session_id`` / ``run_id`` off the
    optional :class:`KaosContext` — that's what kaos-llm-core threads
    through ``fire_hook`` for exactly this correlation purpose.
    ``sequence`` is left at ``0``: ``dispatch_hook`` doesn't read it,
    and the synthesized events are best-effort observability rather
    than authoritative replay material.
    """
    session_id, run_id = _context_ids(context)
    return Span(
        timestamp=time.monotonic(),
        sequence=0,
        session_id=session_id,
        run_id=run_id,
        subject=subject,
        phase=phase,
        span_id=span_id,
        name=name,
        duration_ms=duration_ms,
        error_type=error_type,
        error_message=error_message,
        attributes=attributes,
    )


def _dispatch(kaos_hooks: tuple[KaosHook, ...], span: Span) -> None:
    """Schedule ``dispatch_hook(kaos_hooks, span)`` on the running loop.

    The adapter callbacks are sync (because ``fire_hook`` in kaos-llm-core
    invokes them synchronously); ``dispatch_hook`` is async. We bridge
    by creating a fire-and-forget task on the running loop. Falls back
    to ``asyncio.run`` if there is no running loop — that's a
    best-effort path for sync test contexts.

    Errors are swallowed: hooks must never affect program control flow.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        # No running loop. This is rare in practice (kaos-llm-core
        # programs run in async code) but valid for sync tests.
        try:
            asyncio.run(dispatch_hook(kaos_hooks, span))
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(
                "KaosHook adapter dispatch (sync fallback) raised %s: %s",
                type(exc).__name__,
                exc,
            )
        return
    try:
        # Fire-and-forget: hooks are observability, not control flow.
        # Hold a reference in _PENDING_TASKS to satisfy RUF006 and to
        # prevent the task from being garbage-collected while running;
        # the task removes itself from the set on completion.
        task = loop.create_task(dispatch_hook(kaos_hooks, span))
        _PENDING_TASKS.add(task)
        task.add_done_callback(_PENDING_TASKS.discard)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning(
            "KaosHook adapter could not schedule dispatch: %s",
            exc,
        )


# ----- Forwarders -------------------------------------------------------


def _make_call_forwarders(
    kaos_hooks: tuple[KaosHook, ...],
) -> tuple[Any, Any, Any, Any]:
    """Build the four sync callbacks for :class:`CallHooks`.

    Each callback synthesizes a :class:`Span` (subject=LLM_CALL) with
    a phase that matches the kaos-llm-core lifecycle hook and
    dispatches it through :func:`dispatch_hook`. The span_id is fresh
    per ``on_call_start`` and shared with the matching ``on_call_end``
    / ``on_call_error`` via the ``call`` identity (``id(call)``) so a
    consumer can correlate start/complete pairs without threading
    bespoke state through the adapter.
    """

    # Per-call mapping from id(call) to the START span_id, so COMPLETE/ERROR
    # share the same span identity for tree assembly. Held as long as the
    # CallHooks instance lives — typically one ReAct iteration.
    span_ids: dict[int, str] = {}

    def on_call_start(call: Any, inputs: Any, *, context: Any = None) -> None:
        span_id = _new_span_id()
        span_ids[id(call)] = span_id
        span = _make_span(
            subject=_CALL_SUBJECT,
            phase=SpanPhase.START,
            name=f"llm_call.{type(call).__name__}",
            span_id=span_id,
            attributes={
                "call_type": type(call).__name__,
                "inputs": _safe_attr_str(inputs),
            },
            context=context,
        )
        _dispatch(kaos_hooks, span)

    def on_call_end(call: Any, inputs: Any, invocation: Any, *, context: Any = None) -> None:
        span_id = span_ids.pop(id(call), _new_span_id())
        usage = getattr(invocation, "usage", None)
        attrs: dict[str, Any] = {
            "call_type": type(call).__name__,
            "inputs": _safe_attr_str(inputs),
        }
        if usage is not None:
            for field in ("input_tokens", "output_tokens", "total_tokens", "cost_usd"):
                value = getattr(usage, field, None)
                if value is not None:
                    attrs[field] = value
        span = _make_span(
            subject=_CALL_SUBJECT,
            phase=SpanPhase.COMPLETE,
            name=f"llm_call.{type(call).__name__}",
            span_id=span_id,
            attributes=attrs,
            context=context,
        )
        _dispatch(kaos_hooks, span)

    def on_call_error(call: Any, inputs: Any, exception: Any, *, context: Any = None) -> None:
        span_id = span_ids.pop(id(call), _new_span_id())
        span = _make_span(
            subject=_CALL_SUBJECT,
            phase=SpanPhase.ERROR,
            name=f"llm_call.{type(call).__name__}",
            span_id=span_id,
            attributes={
                "call_type": type(call).__name__,
                "inputs": _safe_attr_str(inputs),
            },
            context=context,
            error_type=type(exception).__name__,
            error_message=str(exception),
        )
        _dispatch(kaos_hooks, span)

    def on_validation_retry(
        call: Any,
        inputs: Any,
        attempt: int,
        error: Any,
        *,
        context: Any = None,
    ) -> None:
        # Retries do not own a phase boundary of their own; we emit
        # PROGRESS on the parent call's span (or a fresh span if the
        # caller didn't fire on_call_start first — defensive).
        span_id = span_ids.get(id(call)) or _new_span_id()
        span = _make_span(
            subject=_CALL_SUBJECT,
            phase=SpanPhase.PROGRESS,
            name=f"llm_call.{type(call).__name__}.retry",
            span_id=span_id,
            attributes={
                "call_type": type(call).__name__,
                "inputs": _safe_attr_str(inputs),
                "retry_attempt": attempt,
                "retry_error": _safe_attr_str(error),
            },
            context=context,
        )
        _dispatch(kaos_hooks, span)

    return on_call_start, on_call_end, on_call_error, on_validation_retry


def _make_program_forwarders(
    kaos_hooks: tuple[KaosHook, ...],
) -> tuple[Any, Any, Any, Any]:
    """Build the four sync callbacks for :class:`ProgramHooks`."""

    span_ids: dict[int, str] = {}

    def on_program_start(program: Any, inputs: Any, *, context: Any = None) -> None:
        span_id = _new_span_id()
        span_ids[id(program)] = span_id
        span = _make_span(
            subject=_PROGRAM_SUBJECT,
            phase=SpanPhase.START,
            name=f"llm_program.{type(program).__name__}",
            span_id=span_id,
            attributes={
                "program_type": type(program).__name__,
                "inputs": _safe_attr_str(inputs),
            },
            context=context,
        )
        _dispatch(kaos_hooks, span)

    def on_iteration(
        program: Any,
        iteration: int,
        payload: Any,
        *,
        context: Any = None,
    ) -> None:
        # Each iteration is a PROGRESS event on the parent program span
        # — distinct from the program's own START / COMPLETE.
        parent_span_id = span_ids.get(id(program)) or _new_span_id()
        span = _make_span(
            subject=_PROGRAM_SUBJECT,
            phase=SpanPhase.PROGRESS,
            name=f"llm_program.{type(program).__name__}.iter.{iteration}",
            span_id=parent_span_id,
            attributes={
                "program_type": type(program).__name__,
                "iteration": iteration,
                "payload": _safe_attr_str(payload),
            },
            context=context,
        )
        _dispatch(kaos_hooks, span)

    def on_program_end(
        program: Any,
        inputs: Any,
        invocation: Any,
        *,
        context: Any = None,
    ) -> None:
        span_id = span_ids.pop(id(program), _new_span_id())
        usage = getattr(invocation, "usage", None)
        attrs: dict[str, Any] = {
            "program_type": type(program).__name__,
            "inputs": _safe_attr_str(inputs),
        }
        if usage is not None:
            for field in ("input_tokens", "output_tokens", "total_tokens", "cost_usd"):
                value = getattr(usage, field, None)
                if value is not None:
                    attrs[field] = value
        span = _make_span(
            subject=_PROGRAM_SUBJECT,
            phase=SpanPhase.COMPLETE,
            name=f"llm_program.{type(program).__name__}",
            span_id=span_id,
            attributes=attrs,
            context=context,
        )
        _dispatch(kaos_hooks, span)

    def on_program_error(
        program: Any,
        inputs: Any,
        exception: Any,
        *,
        context: Any = None,
    ) -> None:
        span_id = span_ids.pop(id(program), _new_span_id())
        span = _make_span(
            subject=_PROGRAM_SUBJECT,
            phase=SpanPhase.ERROR,
            name=f"llm_program.{type(program).__name__}",
            span_id=span_id,
            attributes={
                "program_type": type(program).__name__,
                "inputs": _safe_attr_str(inputs),
            },
            context=context,
            error_type=type(exception).__name__,
            error_message=str(exception),
        )
        _dispatch(kaos_hooks, span)

    return on_program_start, on_iteration, on_program_end, on_program_error


# ----- Public surface ---------------------------------------------------


def _to_call_hooks(kaos_hooks: tuple[KaosHook, ...]) -> CallHooks:
    """Build a :class:`CallHooks` that forwards Call lifecycle events.

    Returns an empty :class:`CallHooks` (all four slots ``None``) when
    ``kaos_hooks`` is empty — this preserves the kaos-llm-core
    "hook is ``None`` → skip" fast path.
    """
    if not kaos_hooks:
        return CallHooks()
    on_call_start, on_call_end, on_call_error, on_validation_retry = _make_call_forwarders(
        kaos_hooks
    )
    return CallHooks(
        on_call_start=on_call_start,
        on_call_end=on_call_end,
        on_call_error=on_call_error,
        on_validation_retry=on_validation_retry,
    )


def _to_program_hooks(kaos_hooks: tuple[KaosHook, ...]) -> ProgramHooks:
    """Build a :class:`ProgramHooks` that forwards Program lifecycle events.

    Returns an empty :class:`ProgramHooks` when ``kaos_hooks`` is empty.
    """
    if not kaos_hooks:
        return ProgramHooks()
    (
        on_program_start,
        on_iteration,
        on_program_end,
        on_program_error,
    ) = _make_program_forwarders(kaos_hooks)
    return ProgramHooks(
        on_program_start=on_program_start,
        on_iteration=on_iteration,
        on_program_end=on_program_end,
        on_program_error=on_program_error,
    )


def adapt_hooks(
    kaos_hooks: tuple[KaosHook, ...],
) -> tuple[CallHooks, ProgramHooks]:
    """Build a ``(CallHooks, ProgramHooks)`` pair from a tuple of KaosHook.

    AgentLoop / Runner construct adapted hooks at startup; pattern
    dispatchers thread them into every ``ReAct(...)`` / ``RAG(...)`` /
    ``Refine(...)`` they construct so a single :class:`KaosHook` tree
    observes both the agent's outer Span stream and kaos-llm-core's
    inner per-Call / per-Program lifecycle events.

    Phase 0.C ships the adapter; AgentLoop wiring lands in Phase 2.
    """
    return _to_call_hooks(kaos_hooks), _to_program_hooks(kaos_hooks)


__all__ = ["adapt_hooks"]
