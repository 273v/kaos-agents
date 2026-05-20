"""CircuitBreaker — a :class:`KaosHook` that trips on consecutive errors.

State machine (per tool name):

::

    CLOSED -- N consecutive failures --> OPEN
    OPEN   -- reset_timeout_seconds  --> HALF_OPEN (next call probes)
    HALF_OPEN -- probe success       --> CLOSED
    HALF_OPEN -- probe failure       --> OPEN

While OPEN, calls are refused via :attr:`HookAction.SKIP`. While
HALF_OPEN, exactly one probe call is allowed; further calls are
refused until the probe completes.

Phase 1.C tracks failures via :meth:`KaosHook.on_tool_call_result` —
the existing kaos-agents hook surface uses ``on_tool_call_result``
(``Span(TOOL_CALL, COMPLETE)``), with success/failure read from
``event.attributes["is_error"]`` (see :class:`kaos_agents.events.Span`
docstring for the conventional payload shape).

What counts as a failure
------------------------

A call is treated as a failure when:

1. ``event.attributes["is_error"]`` is True (a tool-bridge failure,
   an HTTP non-2xx, or any exception bubbled up by the tool wrapper),
   OR
2. ``uninformative_counts_as_failure`` is True (the default) AND
   :func:`kaos_agents.planning.result_check.is_uninformative_result`
   returns True for ``event.attributes["result_summary"]``.

The uninformative-result branch closes the gap exposed by session
``01KS2DEBYT341F1F16B3BRQRV0``, where the agent ran 12 consecutive
``kaos-web-search`` calls that each returned ``is_error=False`` with
the body ``"No results found for: ..."`` — every call "succeeded" by
the error-only predicate, the breaker never tripped, and the loop
ran out of budget instead of refusing cleanly.

Scope limitation (loop termination)
-----------------------------------

The current :class:`~kaos_agents.runtime.runner.Runner` honors
``HookAction.SKIP`` by suppressing the suppressed event from the
outbound event stream, but it does NOT pre-empt tool execution in the
internal agent — by the time the Runner sees the
``Span(TOOL_CALL, START)`` the inner ChatAgent / ReAct has already
dispatched the tool. The breaker is therefore an OBSERVABILITY +
COUNT primitive at the moment, not a hard tool-call block. To actually
terminate the loop on a tripped breaker we still rely on the upstream
:class:`~kaos_agents.patterns.agentic_loop.run_agentic_turn`
terminators (max_iterations / cost / wall_clock / stuck_no_progress).
Closing this loop end-to-end (Runner blocking tool execution on SKIP,
or an explicit ``CircuitBreakerTripped`` event the AgenticLoop watches
for) is tracked as a follow-up.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from enum import StrEnum, unique
from typing import TYPE_CHECKING

from kaos_agents.hooks.base import HookAction, KaosHook

if TYPE_CHECKING:
    from kaos_agents.events import Span


@unique
class CircuitState(StrEnum):
    """The three states of a per-tool circuit breaker."""

    CLOSED = "closed"
    """Normal — calls allowed."""

    OPEN = "open"
    """Tripped — calls refused until ``reset_timeout_seconds`` passes."""

    HALF_OPEN = "half_open"
    """One test (probe) call allowed; result determines next transition."""


@dataclass
class _PerToolState:
    """Mutable per-tool state for the circuit breaker."""

    state: CircuitState = CircuitState.CLOSED
    consecutive_failures: int = 0
    opened_at: float = 0.0
    # When in HALF_OPEN, exactly one probe is in flight; further calls
    # are refused until the probe completes.
    probe_in_flight: bool = False
    # Timestamps and counters retained for diagnostics.
    last_attempt_at: float = field(default_factory=time.monotonic)


class CircuitBreaker(KaosHook):
    """Per-tool circuit breaker with closed / open / half-open states.

    Trips OPEN when consecutive errors exceed ``failure_threshold``.
    After ``reset_timeout_seconds``, the next call transitions the
    circuit to HALF_OPEN as a probe; a successful probe closes the
    circuit, a failed probe re-opens it.

    Both predicates and state transitions are exposed for the Actor's
    ``classify_plan`` path (which doesn't go through the hook
    dispatcher) — see :meth:`allow`, :meth:`record_success`,
    :meth:`record_failure`.
    """

    def __init__(
        self,
        *,
        failure_threshold: int = 5,
        reset_timeout_seconds: float = 30.0,
        uninformative_counts_as_failure: bool = True,
        extra_uninformative_patterns: tuple[re.Pattern[str], ...] = (),
    ) -> None:
        if failure_threshold < 1:
            raise ValueError(
                f"failure_threshold must be >= 1 (got {failure_threshold}). "
                "Use 1 for an immediate-trip breaker; 5 is a common default. "
                "If you need to disable circuit breaking entirely, omit the hook."
            )
        if reset_timeout_seconds < 0.0:
            raise ValueError(
                f"reset_timeout_seconds must be >= 0 (got {reset_timeout_seconds}). "
                "Use 0 for an immediate-recovery breaker; 30s is a common default."
            )
        self._failure_threshold = failure_threshold
        self._reset_timeout = reset_timeout_seconds
        self._uninformative_counts_as_failure = uninformative_counts_as_failure
        self._extra_uninformative_patterns = extra_uninformative_patterns
        self._states: dict[str, _PerToolState] = {}

    def _get(self, tool_name: str) -> _PerToolState:
        st = self._states.get(tool_name)
        if st is None:
            st = _PerToolState()
            self._states[tool_name] = st
        return st

    def state_for(self, tool_name: str) -> CircuitState:
        """Return the current circuit state, applying time-based transitions."""
        st = self._get(tool_name)
        if st.state == CircuitState.OPEN:
            elapsed = time.monotonic() - st.opened_at
            if elapsed >= self._reset_timeout:
                st.state = CircuitState.HALF_OPEN
                st.probe_in_flight = False
        return st.state

    def allow(self, tool_name: str) -> bool:
        """Synchronous predicate — True iff a call to ``tool_name`` may proceed.

        Mutates state on HALF_OPEN to mark the probe as in-flight.
        """
        state = self.state_for(tool_name)
        st = self._get(tool_name)
        st.last_attempt_at = time.monotonic()
        if state == CircuitState.CLOSED:
            return True
        if state == CircuitState.OPEN:
            return False
        # HALF_OPEN: allow exactly one probe.
        if st.probe_in_flight:
            return False
        st.probe_in_flight = True
        return True

    def record_success(self, tool_name: str) -> None:
        """Record a successful call; closes the circuit if HALF_OPEN."""
        st = self._get(tool_name)
        st.consecutive_failures = 0
        if st.state in (CircuitState.HALF_OPEN, CircuitState.OPEN):
            st.state = CircuitState.CLOSED
        st.probe_in_flight = False

    def record_failure(self, tool_name: str) -> None:
        """Record a failure; trips OPEN past the threshold or on HALF_OPEN probe."""
        st = self._get(tool_name)
        st.consecutive_failures += 1
        # A failure during a HALF_OPEN probe immediately re-opens.
        if st.state == CircuitState.HALF_OPEN:
            st.state = CircuitState.OPEN
            st.opened_at = time.monotonic()
            st.probe_in_flight = False
            return
        if st.consecutive_failures >= self._failure_threshold:
            st.state = CircuitState.OPEN
            st.opened_at = time.monotonic()
            st.probe_in_flight = False

    async def on_tool_call_start(self, event: Span) -> HookAction:
        """Block tool calls when the per-tool circuit is OPEN."""
        tool_name = ""
        attrs = getattr(event, "attributes", None)
        if isinstance(attrs, dict):
            tool_name = str(attrs.get("tool_name", "")) or ""
        if not tool_name:
            return HookAction.CONTINUE
        return HookAction.CONTINUE if self.allow(tool_name) else HookAction.SKIP

    async def on_tool_call_result(self, event: Span) -> None:
        """Update circuit state from the COMPLETE span's attributes.

        A call counts as a failure when ``is_error`` is True OR — if
        ``uninformative_counts_as_failure`` was set on construction
        (default True) — when
        :func:`~kaos_agents.planning.result_check.is_uninformative_result`
        returns True for ``result_summary``. See the module docstring
        for the rationale.
        """
        tool_name = ""
        is_error = False
        result_summary = ""
        attrs = getattr(event, "attributes", None)
        if isinstance(attrs, dict):
            tool_name = str(attrs.get("tool_name", "")) or ""
            is_error = bool(attrs.get("is_error", False))
            result_summary = str(attrs.get("result_summary", "") or "")
        if not tool_name:
            return
        is_failure = is_error
        if not is_failure and self._uninformative_counts_as_failure:
            # Lazy-import: ``kaos_agents.planning`` eagerly loads modules
            # that depend on the ``[llm]`` optional extra. Importing it
            # at module top-level would break the base-install contract
            # tested by ``tests/unit/test_base_install_importable.py``.
            # Deferring to first-call defers the load to runtime, when
            # the extras are guaranteed present.
            from kaos_agents.planning.result_check import is_uninformative_result

            is_failure = is_uninformative_result(
                result_summary,
                extra_patterns=self._extra_uninformative_patterns,
            )
        if is_failure:
            self.record_failure(tool_name)
        else:
            self.record_success(tool_name)


__all__ = ["CircuitBreaker", "CircuitState"]
