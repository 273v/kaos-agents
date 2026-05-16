"""AgenticLoop — plan → elevate → execute → check → replan orchestrator.

Per `kaos-modules/docs/internal/agentic-loop-auto-elevation-plan.md`
§3.3. The flagship pattern: composes the existing
:func:`~kaos_agents.planning.policy.plan_turn_tool_policy` planner +
:func:`~kaos_agents.planning.goal_check.check_goal` Critic with a
new auto-elevation step + a replan-on-needs-more-work loop.

**Design at a glance.** One AgenticLoop call corresponds to one user
turn (one POST /v1/chat/sessions/{id}/messages). Within that turn:

  1. Plan tool groups (:func:`plan_turn_tool_policy`)
  2. Auto-elevate green-auto ``dropped_groups`` (silent + audit event)
  3. Emit :class:`CapabilityRequested` for yellow-confirm dropped groups
     (the caller pauses + asks the user; out of scope for this fn —
     the loop emits the event and continues with what it has)
  4. Run the worker (ReAct or whatever ``react_callable`` is injected)
  5. Goal-check (:func:`check_goal` — three-way verdict)
  6. If ``needs_more_work``: thread ``next_action`` into the agent's
     thinking context and replan from step 1
  7. If ``satisfied`` / ``insufficient_evidence``: return
  8. If any limiter trips (iterations, cost, wall-clock,
     stuck-no-progress, user interrupt): return early

**Independent limiters** — three guards, plus stuck-detection +
cancellation, per Pydantic AI usage_limits / LangGraph cycle best
practice (web SOTA research):

  - ``max_loop_iterations`` (default 3) — hard cap on iterations
  - ``max_loop_cost_usd`` (default $0.25) — cumulative LLM cost cap
  - ``max_loop_wall_clock_seconds`` (default 60s) — defense in depth
    against a hung LLM call the cost cap wouldn't catch
  - state-mutation stuck detection — if an iteration ends with no
    new tool calls AND no new text vs the previous iteration, the
    loop is stuck and terminates with reason ``"stuck_no_progress"``
  - asyncio cancellation — the chat router can abort by cancelling
    the task; the loop cleans up + emits ``LoopTerminated(reason=
    "user_interrupt")``

**ReAct injection.** The worker is passed in as ``react_callable``:

    react_callable: Callable[[str, SessionToolSet, ...], Awaitable[ReactResult]]

This decouples kaos-agents (the orchestrator) from any specific
ReAct implementation. The single-user-chat backend will wire its
existing ``stream_chat`` (an httpx proxy to a remote kaos-agents
service) as ``react_callable``. Tests stub it with a deterministic
fake.

**No new SessionPattern subclass.** The AgenticLoop is a pure async
generator function — not a :class:`KaosPattern` subclass — because
the existing patterns (Chat, PlanExecute, Research) compose ReAct
internally; the AgenticLoop sits one level ABOVE them, orchestrating
plan-execute-check-replan around an injected worker. Keeping it a
function (rather than a class) means tests don't have to mock a
runtime + memory + session shape.

The chat router calls ``async for event in run_agentic_turn(...)`` and
proxies events to its SSE stream. The yielded events are a mix of:

  - Local events (this module): :class:`ToolPolicyElevated`,
    :class:`CapabilityRequested`, :class:`GoalChecked`,
    :class:`LoopTerminated`.
  - Pass-through events from ``react_callable``: every event the
    worker yields (text deltas, tool calls, the worker's own
    UsageObserved, etc.) is forwarded verbatim.

The caller sees one continuous event stream per turn.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from kaos_agents.events.policy import (
    CapabilityRequested,
    GoalChecked,
    LoopTerminated,
    ToolPolicyElevated,
)
from kaos_agents.planning.goal_check import (
    GoalCheckInsufficientEvidence,
    GoalCheckNeedsMoreWork,
    GoalCheckSatisfied,
    check_goal,
)
from kaos_agents.planning.policy import (
    TurnToolPolicy,
    plan_turn_tool_policy,
)
from kaos_agents.types.session_policy import SessionPolicy

logger = logging.getLogger(__name__)


# ─── Worker contract ─────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class WorkerResult:
    """The shape :func:`run_agentic_turn` expects back from the injected worker.

    The worker (ReAct, ChatPattern, an httpx-proxied remote agent, etc.)
    returns this after consuming one turn's prompt + tool catalogue.
    Attribute names mirror :class:`AgentResponse` for easy adapter
    construction.
    """

    text: str
    tool_calls_made: list[dict]
    cost_usd: float
    latency_ms: float
    # Worker's raw events to forward into the SSE stream. The
    # orchestrator passes these through verbatim before emitting its
    # own AgenticLoop events.
    events: list[Any] = field(default_factory=list)


# Type alias for the injected worker callable.
WorkerCallable = Callable[..., Awaitable[WorkerResult]]


# ─── Loop state (internal) ───────────────────────────────────────────


@dataclass(slots=True)
class _LoopState:
    """Mutable per-turn state. Internal to :func:`run_agentic_turn`."""

    iteration: int = 0
    cumulative_cost_usd: float = 0.0
    elevation_count: int = 0
    t_start: float = 0.0
    last_text: str = ""
    last_tool_call_count: int = 0
    thinking_note: str = ""
    # Monotonic counter for KaosEvent.sequence — increments per event emission.
    sequence: int = 0

    def wall_clock_ms(self) -> float:
        return (time.monotonic() - self.t_start) * 1000

    def wall_clock_seconds(self) -> float:
        return time.monotonic() - self.t_start


# ─── Public entrypoint ───────────────────────────────────────────────


async def run_agentic_turn(
    *,
    user_message: str,
    policy: SessionPolicy,
    worker: WorkerCallable,
    available_groups: list[str],
    session_id: str = "",
    run_id: str = "",
    corpus_kinds: list[str] | None = None,
    session_intent: str | None = None,
    corpus_headlines: str = "",
    recent_turns: str = "",
    planner_model: str | None = None,
    goal_check_model: str | None = None,
) -> AsyncIterator[Any]:
    """Run one agent turn as an event-streaming loop.

    Yields a mix of:
      - :class:`ToolPolicyElevated` / :class:`CapabilityRequested` /
        :class:`GoalChecked` / :class:`LoopTerminated` (this module)
      - Pass-through events from ``worker`` (text deltas, tool spans, ...)

    Always ends with exactly one :class:`LoopTerminated` event.

    Args:
        user_message: The user's message starting this turn.
        policy: :class:`SessionPolicy` — the two-tier ceiling +
            elevation policy + loop budget for this session.
        worker: Async callable invoked once per iteration. Signature:
            ``worker(user_message, allowed_groups, thinking_note, ...)``;
            returns :class:`WorkerResult`. The orchestrator never
            constructs ReAct directly — keeps kaos-agents pure.
        available_groups: Every group registered in the runtime, for
            the planner + critic context.
        corpus_kinds: Magika-style content-type labels for uploaded
            files. Passed to the planner.
        session_intent: Persona chip name ("research"/"drafting"/
            "forensics"). Passed to the planner.
        corpus_headlines: One-line-per-file headline string. Passed
            to the planner.
        recent_turns: Compressed conversation history. Passed to the
            planner.
        planner_model / goal_check_model: Override the model for those
            calls. Useful for tests or per-tenant overrides.

    The loop terminates when one of:
      - GoalChecker returns ``satisfied`` or ``insufficient_evidence``
      - ``max_loop_iterations`` is hit
      - ``max_loop_cost_usd`` is exceeded
      - ``max_loop_wall_clock_seconds`` is exceeded
      - state-mutation stuck-detection fires
      - the surrounding task is cancelled
    """
    state = _LoopState(t_start=time.monotonic())
    raw_turn_groups: list[str] | None = None

    try:
        while state.iteration < policy.max_loop_iterations:
            state.iteration += 1
            iteration_started = time.monotonic()

            # ── 1. Plan ──────────────────────────────────────────────
            plan = await plan_turn_tool_policy(
                user_message=user_message,
                recent_turns=recent_turns,
                corpus_headlines=corpus_headlines,
                corpus_kinds=list(corpus_kinds or []),
                session_intent=session_intent,
                raw_turn_groups=raw_turn_groups,
                ceiling_groups=sorted(policy.allowed_groups),
                available_groups=available_groups,
                model=planner_model,
            )
            state.cumulative_cost_usd += plan.cost_usd
            raw_turn_groups = sorted(plan.kept_groups | plan.dropped_groups)

            if _budget_exceeded(state, policy):
                yield _terminate(state, policy, "cost_exceeded", session_id, run_id)
                return
            if _wall_clock_exceeded(state, policy):
                yield _terminate(state, policy, "wall_clock_exceeded", session_id, run_id)
                return

            # ── 2. Auto-elevate green-auto dropped_groups ────────────
            policy, elevated_event, capability_event = _consider_elevation(
                policy=policy,
                plan=plan,
                state=state,
                session_id=session_id,
                run_id=run_id,
            )
            if elevated_event is not None:
                yield elevated_event
            if capability_event is not None:
                yield capability_event

            # ── 3. Execute worker ────────────────────────────────────
            effective_groups = sorted(
                plan.kept_groups | (policy.allowed_groups & plan.dropped_groups)
            )
            worker_result = await worker(
                user_message=user_message,
                allowed_groups=effective_groups,
                thinking_note=state.thinking_note,
                iteration=state.iteration,
            )
            state.cumulative_cost_usd += worker_result.cost_usd

            # Forward worker events verbatim.
            for ev in worker_result.events:
                yield ev

            if _budget_exceeded(state, policy):
                yield _terminate(state, policy, "cost_exceeded", session_id, run_id)
                return
            if _wall_clock_exceeded(state, policy):
                yield _terminate(state, policy, "wall_clock_exceeded", session_id, run_id)
                return

            # ── 4. Goal check ────────────────────────────────────────
            # Elevation trail = groups added this iteration via auto-elevation
            # (i.e. groups in the effective set that weren't in `kept_groups`).
            elevation_trail = (
                sorted(set(effective_groups) - plan.kept_groups)
                if elevated_event is not None
                else []
            )
            outcome = await check_goal(
                user_message=user_message,
                agent_response=worker_result.text,
                tool_calls_made=worker_result.tool_calls_made,
                elevation_trail=elevation_trail,
                available_groups=available_groups,
                iteration=state.iteration,
                model=goal_check_model,
            )
            state.cumulative_cost_usd += outcome.cost_usd

            yield _build_goal_checked_event(outcome, state, session_id, run_id)

            # ── 5. Terminal verdicts ─────────────────────────────────
            if outcome.satisfied:
                yield _terminate(state, policy, "satisfied", session_id, run_id)
                return
            if outcome.insufficient_evidence:
                yield _terminate(state, policy, "insufficient_evidence", session_id, run_id)
                return

            # ── 6. Stuck detection ───────────────────────────────────
            new_text = worker_result.text
            new_tool_count = len(worker_result.tool_calls_made)
            if state.iteration > 1 and _is_stuck(
                last_text=state.last_text,
                new_text=new_text,
                last_tool_count=state.last_tool_call_count,
                new_tool_count=new_tool_count,
            ):
                yield _terminate(state, policy, "stuck_no_progress", session_id, run_id)
                return
            state.last_text = new_text
            state.last_tool_call_count = new_tool_count

            # ── 7. Plan next iteration via needs_more_work ──────────
            assert isinstance(outcome.result, GoalCheckNeedsMoreWork)
            state.thinking_note = (
                f"Critic noted iteration {state.iteration} is incomplete: "
                f"{outcome.result.rationale} "
                f"Next action: {outcome.result.next_action}"
            )
            # Track elapsed time for the wall-clock guard.
            _ = iteration_started

        # Fell out of the while loop without hitting a terminal verdict.
        yield _terminate(state, policy, "max_iterations", session_id, run_id)

    except asyncio.CancelledError:
        # Loop interrupt — chat router cancelled the task. Emit the
        # terminal event so the SPA finalizes the message + cleans up;
        # then re-raise so the task actually cancels.
        yield _terminate(state, policy, "user_interrupt", session_id, run_id)
        raise


# ─── Helpers ─────────────────────────────────────────────────────────


def _evt_base(state: _LoopState, session_id: str, run_id: str) -> dict[str, Any]:
    """Required KaosEvent base fields. Sequence increments per emission."""
    state.sequence += 1
    return {
        "timestamp": time.monotonic(),
        "sequence": state.sequence,
        "session_id": session_id,
        "run_id": run_id,
    }


def _consider_elevation(
    *,
    policy: SessionPolicy,
    plan: TurnToolPolicy,
    state: _LoopState,
    session_id: str,
    run_id: str,
) -> tuple[SessionPolicy, ToolPolicyElevated | None, CapabilityRequested | None]:
    """Decide whether to auto-elevate, emit a capability-request, or both.

    Walks ``plan.dropped_groups`` and consults
    :meth:`SessionPolicy.tier_for` for each:

      - ``green-auto`` → elevate silently; add to ``allowed_groups``;
        emit :class:`ToolPolicyElevated`.
      - ``yellow-confirm`` → emit :class:`CapabilityRequested` so the
        SPA renders an inline approval card. The loop continues
        without the requested group this iteration; the user's
        decision lands on the next turn via the SettingsSheet.
      - ``red-blocked`` → silent no-op. The agent will surface this in
        its response and the Critic will mark it
        ``insufficient_evidence`` if it really mattered.

    Returns ``(new_policy, elevated_event, capability_event)``. Either
    event may be ``None`` when no groups in that tier were dropped.
    """
    if not plan.dropped_groups or not policy.auto_elevate:
        return policy, None, None

    dropped = plan.dropped_groups
    previous_allowed = sorted(policy.allowed_groups)

    auto_elevate_targets = {g for g in dropped if policy.can_auto_elevate(g)}
    confirm_targets = {g for g in dropped if policy.needs_confirmation(g)}

    elevated_event: ToolPolicyElevated | None = None
    capability_event: CapabilityRequested | None = None
    new_policy = policy

    if auto_elevate_targets:
        new_policy = policy.with_added_groups(frozenset(auto_elevate_targets))
        state.elevation_count += len(auto_elevate_targets)
        elevated_event = ToolPolicyElevated(
            **_evt_base(state, session_id, run_id),
            elevated_groups=sorted(auto_elevate_targets),
            kept_groups=sorted(plan.kept_groups | frozenset(auto_elevate_targets)),
            previous_allowed=previous_allowed,
            rationale=plan.rationale,
            iteration=state.iteration,
        )

    if confirm_targets:
        capability_event = CapabilityRequested(
            **_evt_base(state, session_id, run_id),
            requested_groups=sorted(confirm_targets),
            justification=plan.rationale,
            iteration=state.iteration,
            previous_allowed=previous_allowed,
        )

    return new_policy, elevated_event, capability_event


def _build_goal_checked_event(
    outcome: Any, state: _LoopState, session_id: str, run_id: str
) -> GoalChecked:
    """Pack a :class:`GoalCheckOutcome` into a :class:`GoalChecked` event."""
    next_action = ""
    missing = ""
    confidence = 0.0

    if isinstance(outcome.result, GoalCheckSatisfied):
        confidence = outcome.result.confidence
    elif isinstance(outcome.result, GoalCheckNeedsMoreWork):
        next_action = outcome.result.next_action
        confidence = outcome.result.confidence
    elif isinstance(outcome.result, GoalCheckInsufficientEvidence):
        missing = outcome.result.missing

    return GoalChecked(
        **_evt_base(state, session_id, run_id),
        kind=outcome.kind,
        rationale=outcome.result.rationale,
        next_action=next_action,
        missing=missing,
        confidence=confidence,
        iteration=outcome.iteration,
        cost_usd=outcome.cost_usd,
        latency_ms=outcome.latency_ms,
    )


def _is_stuck(
    *,
    last_text: str,
    new_text: str,
    last_tool_count: int,
    new_tool_count: int,
) -> bool:
    """State-mutation stuck-detection.

    The iteration is "stuck" when BOTH:
      - The new response text is byte-identical OR a prefix/suffix of
        the last response (no new prose),
      - AND no new tool calls happened (``new_tool_count <= last_tool_count``).

    Per LangGraph cycle-optimization best practice — don't burn budget
    on iterations that aren't moving forward.
    """
    if new_tool_count > last_tool_count:
        return False
    if not last_text or not new_text:
        return False
    if last_text == new_text:
        return True
    # Substring relationship is a weaker but still useful signal.
    return last_text in new_text or new_text in last_text


def _budget_exceeded(state: _LoopState, policy: SessionPolicy) -> bool:
    return state.cumulative_cost_usd >= policy.max_loop_cost_usd


def _wall_clock_exceeded(state: _LoopState, policy: SessionPolicy) -> bool:
    return state.wall_clock_seconds() >= policy.max_loop_wall_clock_seconds


def _terminate(
    state: _LoopState,
    policy: SessionPolicy,
    reason: str,
    session_id: str = "",
    run_id: str = "",
) -> LoopTerminated:
    """Build the terminal :class:`LoopTerminated` event."""
    _ = policy  # carried for future per-policy fields on the event
    return LoopTerminated(
        **_evt_base(state, session_id, run_id),
        reason=reason,
        iterations_used=state.iteration,
        elevations_used=state.elevation_count,
        cost_usd=state.cumulative_cost_usd,
        wall_clock_ms=state.wall_clock_ms(),
    )


__all__ = [
    "WorkerCallable",
    "WorkerResult",
    "run_agentic_turn",
]
