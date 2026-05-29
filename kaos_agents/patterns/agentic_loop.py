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
import json
import re
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from kaos_core.logging import get_logger

from kaos_agents.events.lifecycle import TurnSummary
from kaos_agents.events.policy import (
    CapabilityRequested,
    CircuitBreakerTripped,
    ConsistencyChecked,
    GoalChecked,
    LoopTerminated,
    ToolPolicyElevated,
)
from kaos_agents.events.spans import Span, SpanPhase, SpanSubject
from kaos_agents.events.stream import TextDelta
from kaos_agents.planning.goal_check import (
    GoalCheckInsufficientEvidence,
    GoalCheckNeedsMoreWork,
    GoalCheckSatisfied,
    check_goal,
)
from kaos_agents.planning.m2_consistency import (
    judge_reasoning_action_consistency,
)
from kaos_agents.planning.m3_grounding import (
    judge_grounding_fabrication,
)
from kaos_agents.planning.m4_completeness import (
    judge_completeness,
)
from kaos_agents.planning.policy import (
    TurnToolPolicy,
    plan_turn_tool_policy,
)
from kaos_agents.types.session_policy import SessionPolicy

logger = get_logger(__name__)


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
    # 0.1.2 (R0.1): mirror of the most recent worker_result.text,
    # captured immediately after the worker call. Distinct from
    # ``last_text`` (which holds the PRIOR iteration's text for
    # stuck-detection comparison). Budget caps firing mid-iteration
    # consult this to preserve the worker's draft.
    current_iter_text: str = ""
    # The most recent GoalCheck rationale + next_action — captured so
    # the max_iterations refusal-override can render an honest "I
    # tried and failed" message instead of letting the last
    # hallucinated worker text through. See task #505.
    last_critic_rationale: str = ""
    last_critic_next_action: str = ""
    # 0.1.2 (R0.1 / R1.2): tracks whether the most recent terminal
    # verdict was an actual critic rejection (worker text was the
    # rejected hallucination) or something else (budget cap fired
    # before the critic could verdict, OR critic satisfied → loop
    # would terminate normally).
    #
    # Values:
    #   ""               — no critic verdict yet this iteration
    #                      (worker may have substantive text;
    #                      preserve it on budget-cap exit)
    #   "needs_more_work" — GoalCheck rejected (legacy behavior;
    #                      clobber worker text on max-iter exit)
    #   "override"       — M2 or M3 overrode satisfied (the
    #                      worker draft was flagged by a critic;
    #                      clobber)
    #
    # The refusal builder uses this to decide whether to preserve
    # ``last_text`` (worker draft was substantive + uncriticized)
    # or substitute the boilerplate template (worker draft was
    # criticized). Three audits independently flagged the failure
    # of the previous unconditional clobber: workers had drafted
    # nuanced 1k-5k char answers and the loop discarded them for
    # budget reasons (cost / wall-clock / stuck), shipping a
    # 400-char generic "I stopped..." template to the persisted
    # record. See ``docs/audits/2026-05-21-worker-honesty.md``,
    # ``docs/audits/2026-05-21-loop-state-machine.md``,
    # ``docs/audits/2026-05-21-persistence-audit.md``.
    last_terminal_verdict: str = ""
    # Monotonic counter for KaosEvent.sequence — increments per event emission.
    sequence: int = 0
    # Per-tool consecutive failure counter for the loop-level circuit
    # breaker (#506-followup). Each observed
    # ``Span(TOOL_CALL, COMPLETE)`` increments or resets the counter
    # for its tool. When ANY tool crosses
    # ``circuit_breaker_threshold``, the loop emits
    # :class:`CircuitBreakerTripped` and terminates.
    consecutive_tool_failures: dict[str, int] = field(default_factory=dict)
    tripped_tool: str = ""
    tripped_failures: int = 0

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
    m2_consistency_model: str | None = None,
    m3_grounding_model: str | None = None,
    m4_completeness_model: str | None = None,
    circuit_breaker_threshold: int = 5,
    max_tool_calls_per_iteration: int = 30,
    citation_verification_enabled: bool = False,
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
        m2_consistency_model: When set, after a ``satisfied`` GoalCheck
            verdict the loop runs the M2 reasoning-action consistency
            critic on the worker's response. A
            ``contradicts_reasoning`` or ``contradicts_tool_results``
            verdict overrides the satisfied terminator and forces one
            more iteration with an M2-derived ``thinking_note``
            directive. Off by default (None) to preserve the existing
            cost profile and contract for callers not opted-in.
            Mirrors the force-elevate-from-critic mechanism designed
            in ``2026-05-19-agentic-loop-honesty.md`` §3.2 — the
            critic gets the last word on whether the iteration is
            actually done.
        m3_grounding_model: When set, after a ``satisfied`` GoalCheck
            verdict the loop also runs the M3 document-grounding
            fabrication critic (task #445 P0-7). A
            ``fabricated_with_admission`` or
            ``fabricated_without_admission`` verdict overrides the
            satisfied terminator and forces one more iteration with
            an M3-derived directive. Composes cleanly with
            ``m2_consistency_model`` — both critics run; either's
            override fires the replan. Off by default (None).
        m4_completeness_model: When set, after a ``satisfied`` GoalCheck
            verdict the loop ALSO runs the M4 silent-incompleteness
            critic (P0.4 / S12 confident-partial-answer). A ``partial``
            verdict overrides the satisfied terminator and forces one
            more iteration with an M4-derived directive ("you returned
            fewer items than requested without acknowledging
            exhaustion"). M4 runs AFTER M3 — order matters: a partial
            answer that's ALSO ungrounded should be flagged on
            grounding first; M4 catches the cases where the answer IS
            grounded but quantitatively incomplete. Off by default
            (None) to preserve cost profile.

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
            # 0.1.2 (R0.1): reset the terminal verdict at iteration
            # start so a budget cap firing mid-iteration sees "" (no
            # critic verdict yet) rather than the previous iteration's
            # value. The verdict is set later in the iteration when
            # GoalCheck or M2/M3 actually runs.
            state.last_terminal_verdict = ""

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
                async for ev in _emit_failure_refusal(
                    reason="cost_exceeded",
                    state=state,
                    session_id=session_id,
                    run_id=run_id,
                ):
                    yield ev
                yield _terminate(state, policy, "cost_exceeded", session_id, run_id)
                return
            if _wall_clock_exceeded(state, policy):
                async for ev in _emit_failure_refusal(
                    reason="wall_clock_exceeded",
                    state=state,
                    session_id=session_id,
                    run_id=run_id,
                ):
                    yield ev
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
            # 0.1.2 (R0.1): capture the worker's draft text in a
            # separate field — DO NOT overwrite ``state.last_text``
            # here because stuck-detection (below) compares prior vs
            # current iteration text via that field. The dedicated
            # ``current_iter_text`` field lets budget caps firing
            # mid-iteration (cost / wall-clock / circuit-breaker)
            # preserve the worker's draft via
            # ``_should_preserve_worker_draft``.
            state.current_iter_text = worker_result.text

            # Plan §Issue 9 / B1.7 — per-call cost cap. Disabled by
            # default (max_per_tool_cost_usd == 0.0); operators tighten
            # to e.g. 0.05 to catch a misconfigured-model runaway that
            # the cumulative loop cap wouldn't trip on a single shot.
            if _per_tool_budget_exceeded(worker_result.cost_usd, policy):
                async for ev in _emit_failure_refusal(
                    reason="cost_exceeded",
                    state=state,
                    session_id=session_id,
                    run_id=run_id,
                ):
                    yield ev
                yield _terminate(state, policy, "cost_exceeded", session_id, run_id)
                return

            # B0.8: post-worker citation verification gate. Opt-in via
            # ``citation_verification_enabled`` so existing callers
            # don't acquire a CourtListener dependency or per-turn
            # network cost. When enabled, we extract every case
            # citation in the worker's draft, resolve each against
            # CourtListener (gated separately by
            # ``KAOS_AGENT_CITATION_VERIFY_ENABLED``), emit a
            # :class:`CitationVerified` event per result, and fold any
            # ``mismatch`` / ``not_found`` diagnostic into
            # ``thinking_note`` so the next iteration sees the
            # specific failure. ``unreachable`` / ``skipped`` are
            # recorded but do NOT force a replan (the cite might be
            # correct; we just can't confirm).
            if citation_verification_enabled and worker_result.text.strip():
                from kaos_agents.citations import verify_citations_in_text
                from kaos_agents.events.research import CitationVerified

                verification_results = await verify_citations_in_text(worker_result.text)
                _replan_diagnostics: list[str] = []
                for v_result in verification_results:
                    yield CitationVerified(
                        **_evt_base(state, session_id, run_id),
                        raw_cite=v_result.raw_cite,
                        status=v_result.status,
                        courtlistener_url=v_result.courtlistener_url or "",
                        observed_case_name=v_result.observed_case_name or "",
                        observed_year=v_result.observed_year or 0,
                        diagnostic=v_result.diagnostic or "",
                    )
                    if v_result.is_problem:
                        _replan_diagnostics.append(
                            f"cite {v_result.raw_cite!r}: {v_result.diagnostic}"
                        )
                if _replan_diagnostics:
                    state.thinking_note = (
                        "Citation verification failed for one or more cites — "
                        "re-check authority before answering: " + "; ".join(_replan_diagnostics)
                    )

            # Forward worker events verbatim. While we're walking them,
            # observe ``Span(TOOL_CALL, COMPLETE)`` events for the
            # loop-level circuit breaker (#506-followup): when a single
            # tool returns ``circuit_breaker_threshold`` consecutive
            # failures or uninformative results, emit
            # CircuitBreakerTripped + terminate the loop. This closes
            # the gap left open by the Runner-layer breaker (which
            # only suppresses event emission, not tool dispatch).
            for ev in worker_result.events:
                yield ev
                _observe_for_circuit_breaker(
                    ev,
                    state=state,
                    threshold=circuit_breaker_threshold,
                )

            # If a per-tool circuit breaker tripped this iteration,
            # emit the typed event + refusal + LoopTerminated and stop.
            if state.tripped_tool:
                yield CircuitBreakerTripped(
                    **_evt_base(state, session_id, run_id),
                    tool_name=state.tripped_tool,
                    consecutive_failures=state.tripped_failures,
                    failure_threshold=circuit_breaker_threshold,
                    reset_timeout_seconds=0.0,
                    uninformative_counted=True,
                )
                async for ev in _emit_failure_refusal(
                    reason="circuit_breaker_tripped",
                    state=state,
                    session_id=session_id,
                    run_id=run_id,
                ):
                    yield ev
                yield _terminate(state, policy, "circuit_breaker_tripped", session_id, run_id)
                return

            # 0.1.2 (R1.3 — reliability roadmap #563): per-iteration
            # tool-call cap. The audit anchor is Agent 1's Sonnet P5
            # case which ran 32 tool calls in iteration 1 and burned
            # $0.67 before ``cost_exceeded`` fired mid-synthesis. The
            # iteration-level cap returns control to the orchestrator
            # so M2/circuit-breaker/budget-cap layers can intervene
            # BEFORE the cost-storm completes. Default 10. Disable
            # with 0.
            if (
                max_tool_calls_per_iteration > 0
                and len(worker_result.tool_calls_made) >= max_tool_calls_per_iteration
            ):
                logger.warning(
                    "agentic_loop: per-iteration tool-call cap fired "
                    "iteration=%d tool_calls=%d cap=%d",
                    state.iteration,
                    len(worker_result.tool_calls_made),
                    max_tool_calls_per_iteration,
                )
                async for ev in _emit_failure_refusal(
                    reason="tool_call_cap_exceeded",
                    state=state,
                    session_id=session_id,
                    run_id=run_id,
                ):
                    yield ev
                yield _terminate(state, policy, "tool_call_cap_exceeded", session_id, run_id)
                return

            if _budget_exceeded(state, policy):
                async for ev in _emit_failure_refusal(
                    reason="cost_exceeded",
                    state=state,
                    session_id=session_id,
                    run_id=run_id,
                ):
                    yield ev
                yield _terminate(state, policy, "cost_exceeded", session_id, run_id)
                return
            if _wall_clock_exceeded(state, policy):
                async for ev in _emit_failure_refusal(
                    reason="wall_clock_exceeded",
                    state=state,
                    session_id=session_id,
                    run_id=run_id,
                ):
                    yield ev
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

            # ── 4b. Post-satisfied grounding critics (M2 / M3 / M4) ──
            # Opt-in (the caller passes a model) and only after a
            # satisfied GoalCheck. Each critic is a JudgeSignature rubric
            # that can override the satisfied verdict and force one more
            # iteration with a re-write directive. All three run through
            # the shared _process_critic path (see the _GroundingCritic
            # specs above); M2/M3 judge the worker text against the tool
            # results, M4 judges the response against the user's request.
            # Order is M2 → M3 → M4 (grounding before completeness): the
            # first critic to override supplies the refusal rationale if
            # a later iteration bails. See 2026-05-19-agentic-loop-honesty.md §3.2.
            override_notes: list[str] = []
            critic_rationale = ""
            if outcome.satisfied:
                tool_results_text = _format_tool_results_for_m2(worker_result.tool_calls_made)
                critic_calls: list[tuple[_GroundingCritic, Awaitable[Any]]] = []
                if m2_consistency_model is not None:
                    critic_calls.append(
                        (
                            _M2_CRITIC,
                            judge_reasoning_action_consistency(
                                response_text=worker_result.text,
                                model=m2_consistency_model,
                                tool_results_text=tool_results_text,
                            ),
                        )
                    )
                if m3_grounding_model is not None:
                    critic_calls.append(
                        (
                            _M3_CRITIC,
                            judge_grounding_fabrication(
                                response_text=worker_result.text,
                                model=m3_grounding_model,
                                tool_results_text=tool_results_text,
                            ),
                        )
                    )
                if m4_completeness_model is not None:
                    critic_calls.append(
                        (
                            _M4_CRITIC,
                            judge_completeness(
                                user_prompt=user_message,
                                response_text=worker_result.text,
                                model=m4_completeness_model,
                            ),
                        )
                    )
                for spec, coro in critic_calls:
                    note, rationale, event = _process_critic(
                        spec=spec,
                        verdict=await coro,
                        state=state,
                        session_id=session_id,
                        run_id=run_id,
                    )
                    yield event
                    if note:
                        override_notes.append(note)
                        critic_rationale = critic_rationale or rationale

            # When multiple critics fire, the worker sees every directive
            # in the next iteration's thinking_note.
            override_note = "\n\n".join(override_notes)

            # ── 5. Terminal verdicts ─────────────────────────────────
            if outcome.satisfied and not override_note:
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
                async for ev in _emit_failure_refusal(
                    reason="stuck_no_progress",
                    state=state,
                    session_id=session_id,
                    run_id=run_id,
                ):
                    yield ev
                yield _terminate(state, policy, "stuck_no_progress", session_id, run_id)
                return
            state.last_text = new_text
            state.last_tool_call_count = new_tool_count

            # ── 7. Plan next iteration via needs_more_work ──────────
            # Two paths into replan: the goal check returned
            # needs_more_work, OR a post-satisfied critic overrode the
            # satisfied verdict (override_note is non-empty). The
            # override path bypasses the GoalCheckNeedsMoreWork assertion
            # because the outcome object is still satisfied.
            if override_note:
                state.thinking_note = override_note
                # The first critic to override (M2 → M3 → M4) already
                # built critic_rationale; surface it so a downstream
                # max-iterations refusal names the critic that flagged
                # the response rather than a stale GoalCheck rationale
                # (task #505 / R1.2). The override_note carries the
                # actionable directive, so it doubles as next_action.
                state.last_critic_rationale = critic_rationale
                state.last_critic_next_action = override_note
                state.last_terminal_verdict = "override"  # R0.1
            else:
                assert isinstance(outcome.result, GoalCheckNeedsMoreWork)
                state.thinking_note = (
                    f"Critic noted iteration {state.iteration} is incomplete: "
                    f"{outcome.result.rationale} "
                    f"Next action: {outcome.result.next_action}"
                )
                # Stash for the max_iterations refusal-override path.
                state.last_critic_rationale = outcome.result.rationale
                state.last_critic_next_action = outcome.result.next_action
                state.last_terminal_verdict = "needs_more_work"  # R0.1

            # ── 7a. 0.1.1 (#549.B) — Remediation-hint threading ────
            # If any tool call this iteration returned is_error=True
            # AND its summary_excerpt contains a "Try kaos-{module}-
            # {tool}" remediation hint (the kaos-mcp tool-design
            # standard for error messages), thread the hint into the
            # next iteration's thinking_note. Pre-fix the worker
            # would retry the same broken tool 4x because nothing
            # surfaced the hint at the loop level. See task #549.B.
            remediation_hints = _extract_remediation_hints(worker_result.tool_calls_made)
            if remediation_hints:
                state.thinking_note = (
                    f"{state.thinking_note}\n\n"
                    "Tool-error remediation hints from iteration "
                    f"{state.iteration} (act on these instead of "
                    "retrying the same broken tools):\n"
                    + "\n".join(f"- {hint}" for hint in remediation_hints)
                )

            # ── 7b. 0.1.1 (P2-B) — Prior-tool-call summary threading ─
            # Append a short summary of what tools have been called
            # this turn so the next iteration doesn't re-issue near-
            # duplicate queries. This is the loop-level fix for the
            # over-specified-query-storm pattern documented in WU-K
            # v2 Case C1 (13 near-identical site:sec.gov searches
            # before the cost-cap fired). Note: the deeper fix is
            # planner / ranker work — see the plan's deferred
            # backlog. This summary is the cheap, immediate mitigant.
            prior_summary = _summarize_prior_tool_calls(worker_result.tool_calls_made)
            if prior_summary:
                state.thinking_note = (
                    f"{state.thinking_note}\n\n"
                    "Tool calls you already ran this turn (do NOT "
                    "re-issue near-duplicate queries — try a "
                    "different angle if you need more evidence):\n"
                    f"{prior_summary}"
                )
            # Track elapsed time for the wall-clock guard.
            _ = iteration_started

        # Fell out of the while loop without hitting a terminal verdict —
        # i.e. ``max_loop_iterations`` iterations of needs_more_work.
        # Per task #505: do NOT ship the last worker text (it's the
        # response the critic JUST rejected). Synthesize a clean refusal
        # informed by the last GoalCheck rationale + next_action, emit
        # it as a final TextDelta + TurnSummary so the SPA's
        # ``last_iter_text`` accumulator picks it up as the canonical
        # turn text, then emit the LoopTerminated as before.
        async for ev in _emit_failure_refusal(
            reason="max_iterations",
            state=state,
            session_id=session_id,
            run_id=run_id,
        ):
            yield ev
        yield _terminate(state, policy, "max_iterations", session_id, run_id)

    except asyncio.CancelledError:
        # Loop interrupt — chat router cancelled the task. Emit the
        # terminal event so the SPA finalizes the message + cleans up;
        # then re-raise so the task actually cancels.
        yield _terminate(state, policy, "user_interrupt", session_id, run_id)
        raise


# ─── Helpers ─────────────────────────────────────────────────────────


_REFUSAL_LEAD_BY_REASON: dict[str, str] = {
    "max_iterations": (
        "I tried {iterations} times and could not produce a grounded "
        "answer. Each attempt was flagged for the reason below:"
    ),
    "stuck_no_progress": (
        "I stopped after {iterations} attempt(s) — I was repeating "
        "myself without finding new evidence. Last issue:"
    ),
    "cost_exceeded": (
        "I stopped after {iterations} attempt(s) — this request hit "
        "its spending limit before I could finish. Last issue:"
    ),
    "wall_clock_exceeded": (
        "I stopped after {iterations} attempt(s) — this request hit "
        "its time limit before I could finish. Last issue:"
    ),
    "circuit_breaker_tripped": (
        "I stopped after {iterations} attempt(s) — a tool I needed kept failing. Last issue:"
    ),
    "tool_call_cap_exceeded": (
        "I stopped after {iterations} attempt(s) — I ran out of tool "
        "calls before I could finish. Last issue:"
    ),
}


# 0.1.2 (R0.1): the minimum worker-draft length we consider "substantive"
# for the preserve-on-budget-exit path. Below this threshold (in chars
# stripped of whitespace) we still substitute the boilerplate template —
# the user gets a more useful refusal than 3 chars of "ok".
_MIN_WORKER_DRAFT_CHARS = 40


def _build_budget_footer(
    *,
    reason: str,
    iterations: int,
) -> str:
    """Render the short footer appended to a preserved worker draft.

    0.1.2 (R0.1). When the worker had drafted substantive text and the
    loop terminated for a budget reason BEFORE any critic could reject
    the draft, we preserve the worker text and tack a short, honest
    footer on the end explaining what budget ran out. The reader sees
    the work-in-progress + the caveat, not 400 chars of generic
    boilerplate that throws away the work.
    """
    reason_phrase = _BUDGET_REASON_PHRASE.get(reason, "I ran out of attempts")
    return (
        "\n\n---\n"
        f"_Note: I stopped after {iterations} iteration(s) because "
        f"{reason_phrase}. The answer above represents my work-in-"
        "progress at that point; please verify any specific claims "
        "before relying on them, and consider re-prompting with a "
        "higher budget if you need a more complete response._"
    )


# 0.1.2 (R0.1): per-reason phrasing for the budget-cap footer. Distinct
# from ``_REFUSAL_LEAD_BY_REASON`` because the footer is rendered AFTER
# the worker draft (not as a replacement) and uses different grammar.
_BUDGET_REASON_PHRASE: dict[str, str] = {
    "max_iterations": "I ran out of attempts",
    "stuck_no_progress": "my last few attempts were not making progress",
    "cost_exceeded": "this request hit its spending limit",
    "wall_clock_exceeded": "this request hit its time limit",
    "circuit_breaker_tripped": "a tool I needed kept failing",
    "tool_call_cap_exceeded": "I ran out of tool calls",
}


def _build_failure_refusal(
    *,
    reason: str,
    iterations: int,
    critic_rationale: str,
    critic_next_action: str,
) -> str:
    """Render an honest refusal for a non-satisfied termination path.

    Per task #505: when the loop terminates with the last GoalCheck
    verdict still ``needs_more_work``, we MUST NOT ship the worker's
    last attempt — that text is exactly what the critic just rejected
    (typically a confident-wrong hallucination). Instead, surface an
    honest "I tried and failed" message derived from the last critic
    rationale + next_action so the user sees the failure mode, not
    the fabrication.

    This path is used when the worker draft was actually CRITICIZED
    (last_terminal_verdict in ("needs_more_work", "override")). For
    the case where the budget cap fired BEFORE any critic verdict,
    see ``_build_budget_footer`` + the preserve-worker-text branch
    in ``_emit_failure_refusal``.

    Args:
        reason: One of the ``_REFUSAL_LEAD_BY_REASON`` keys. Unknown
            reasons fall back to the ``max_iterations`` phrasing — the
            renderer never crashes on a new reason string.
        iterations: How many iterations ran (1..N).
        critic_rationale: Last GoalCheck rationale (may be empty).
        critic_next_action: Last GoalCheck next_action (may be empty).
    """
    lead = _REFUSAL_LEAD_BY_REASON.get(reason, _REFUSAL_LEAD_BY_REASON["max_iterations"])
    lines = [lead.format(iterations=iterations)]
    if critic_rationale:
        lines.append("")
        lines.append(f"- Critic's diagnosis: {critic_rationale}")
    if critic_next_action:
        lines.append(f"- What was still needed: {critic_next_action}")
    if not critic_rationale and not critic_next_action:
        lines.append("")
        lines.append(
            "- The tool calls I made did not produce evidence sufficient "
            "to ground a reliable answer."
        )
    lines.append("")
    lines.append(
        "Rather than ship a confident-but-unverified answer, I'm "
        "stopping here so you can re-prompt with a different angle "
        "(e.g. supply a specific URL, attach a document, or relax the "
        "scope) or retry with a higher iteration / cost budget."
    )
    return "\n".join(lines)


# Backwards-compat alias for the older name used while #505 was being
# scoped. Will be removed in the next refactor pass.
_build_max_iterations_refusal = _build_failure_refusal


def _draft_for_preserve(state: _LoopState) -> str:
    """The worker draft to consider for preserve-on-budget-exit.

    0.1.2 (R0.1). Prefer the current iteration's draft (captured
    immediately after the worker call into ``current_iter_text``);
    fall back to ``last_text`` when ``current_iter_text`` is empty
    (rare: only happens on max-iter exit after the post-critic
    promote step has moved the text).
    """
    return state.current_iter_text or state.last_text


def _should_preserve_worker_draft(state: _LoopState) -> bool:
    """Decide whether a non-satisfied exit should preserve the worker draft.

    0.1.2 (R0.1). Three audits found the same bug: the loop discarded
    substantive worker drafts on budget-cap exits, replacing them with
    a generic 400-char boilerplate. Cases observed:

    - Sonnet 4.6 SCOTUS table: 4827-char draft with 14 citations →
      cost-cap fired during synthesis → persisted 426-char template.
    - gpt-5.4-mini SEC RIA: 1102-char honest "couldn't verify, here
      are 2 candidates" answer → max-iter fired → persisted 432-char
      template.
    - Multiple Memory ≠ UI sessions where the streamed bubble showed
      a real answer and the persisted record showed boilerplate.

    Preserve worker draft when ALL of the following hold:
      1. ``state.last_text`` is at least ``_MIN_WORKER_DRAFT_CHARS``
         non-whitespace characters (rules out empty or 3-char-yes
         answers).
      2. ``state.last_terminal_verdict`` is "" (no critic verdict yet
         this iteration — budget cap fired before GoalCheck/M2/M3
         could weigh in) OR "satisfied" (critic accepted the draft;
         should never reach refusal, but be defensive).

    Clobber the draft (existing template behavior) when:
      - "needs_more_work" — GoalCheck rejected the draft as
        confident-but-wrong. The text was exactly the hallucination
        the critic caught; do NOT ship.
      - "override" — M2 or M3 overrode satisfied. The draft was
        flagged by the consistency / grounding critic. The text was
        rejected by a second critic; do NOT ship.
    """
    return len(
        _draft_for_preserve(state).strip()
    ) >= _MIN_WORKER_DRAFT_CHARS and state.last_terminal_verdict in ("", "satisfied")


async def _emit_failure_refusal(
    *,
    reason: str,
    state: _LoopState,
    session_id: str,
    run_id: str,
) -> AsyncIterator[Any]:
    """Emit the TextDelta + TurnSummary pair carrying a clean refusal.

    Used by every non-satisfied terminator (max_iterations,
    stuck_no_progress, cost_exceeded, wall_clock_exceeded,
    circuit_breaker_tripped). Per 0.1.2 R0.1 we now distinguish two
    cases:

    1. **Worker draft preservation path** (``_should_preserve_worker_draft``
       returns True): the worker emitted substantive text AND no critic
       has rejected it. The budget cap fired before the critic could
       verdict. Ship the draft + a short footer explaining the cap.
       The user gets the answer they paid for; the loop notes that
       budget ran out so caveats apply.

    2. **Refusal template path** (legacy): the worker text was either
       empty/trivial OR a critic explicitly rejected it. Render the
       honest "I tried and failed" template per task #505.

    Three audits independently flagged the unconditional-clobber
    behavior as the single biggest user-impact bug in the system
    (see ``docs/audits/2026-05-21-{worker-honesty,loop-state-machine,
    persistence-audit}.md`` for the convergent finding). Fixing this
    one path is the highest-leverage change in the 0.1.2 release.
    """
    if _should_preserve_worker_draft(state):
        # Preserve worker draft + budget footer. The user sees the
        # work-in-progress + a clear caveat — much better than
        # discarding the work for a generic template.
        worker_draft = _draft_for_preserve(state).rstrip()
        footer = _build_budget_footer(reason=reason, iterations=state.iteration)
        refusal_text = worker_draft + footer
        intent = "respond_with_caveat"
    else:
        refusal_text = _build_failure_refusal(
            reason=reason,
            iterations=state.iteration,
            critic_rationale=state.last_critic_rationale,
            critic_next_action=state.last_critic_next_action,
        )
        intent = "refuse"
    yield TextDelta(
        **_evt_base(state, session_id, run_id),
        content=refusal_text,
    )
    yield TurnSummary(
        **_evt_base(state, session_id, run_id),
        text=refusal_text,
        intent=intent,
        cost_usd=state.cumulative_cost_usd,
    )


def _tool_call_complete_attrs(event: Any) -> dict[str, Any] | None:
    """Extract the tool-call-complete attribute bag from either an event shape.

    0.1.2 (R1.1 — reliability roadmap #561). The AgenticLoop's circuit
    breaker observer was originally written against typed ``Span``
    instances, but the SPA's ``agentic_worker`` (see
    ``kaos-ui/examples/single-user-chat/backend/app/services/agentic_worker.py:158-171``)
    forwards raw SSE records to the orchestrator — ``WorkerResult.events``
    is ``list[dict[str, str]]`` of ``{"event": "<type>", "data": "<json>"}``
    pairs, not deserialized ``Span`` objects. Pre-R1.1 the
    ``isinstance(event, Span)`` guard at the head of
    ``_observe_for_circuit_breaker`` silently returned for every
    forwarded SSE record, so the loop-level breaker was dead code in the
    SPA. Cost-storm sessions (WU-K v3 C1 with 17 tool calls, Agent 4's
    C7 with 12 consecutive "No results found") ran all the way to
    cost/wall-clock cap because the breaker never fired.

    Returns the ``attributes`` dict for the event if and only if it is a
    ``TOOL_CALL`` span in the ``COMPLETE`` phase. Returns ``None`` for
    any other event (so the caller can return early). Handles three
    shapes:

    - **Typed Span** — checks ``event.subject`` /  ``event.phase`` /
      ``event.attributes``.
    - **SSE record dict** (``{"event": "span", "data": "<json>"}``) —
      JSON-decodes the data field and routes through the parsed-dict
      branch.
    - **Already-parsed payload dict** (``{"type": "span", "subject":
      "tool_call", "phase": "complete", "attributes": {...}}``) — the
      shape ``app/services/agentic_worker.py:_parse_payload`` produces.

    Any other shape returns ``None``. JSON-decoding failures are
    swallowed — a malformed event must never take down a turn.
    """
    if isinstance(event, Span):
        if event.subject != SpanSubject.TOOL_CALL or event.phase != SpanPhase.COMPLETE:
            return None
        attrs = event.attributes
        return attrs if isinstance(attrs, dict) else None
    if not isinstance(event, dict):
        return None
    # SSE record shape — {"event": "<type>", "data": "<json-string>"}
    if "event" in event and "data" in event:
        raw = event.get("data")
        if not isinstance(raw, str):
            return None
        try:
            payload: Any = json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            return None
        if not isinstance(payload, dict):
            return None
        return _tool_call_complete_attrs(payload)
    # Already-parsed payload shape — {"type": "span", "subject": "tool_call",
    # "phase": "complete", "attributes": {...}}. Match the wire
    # discriminator ``type`` (kaos_agents.events.serde) AND the legacy
    # ``subject``/``phase`` fields the SPA's ``_parse_payload`` produces.
    if event.get("type") not in (None, "span"):
        return None
    if event.get("subject") != "tool_call":
        return None
    if event.get("phase") != "complete":
        return None
    attrs = event.get("attributes")
    return attrs if isinstance(attrs, dict) else None


def _observe_for_circuit_breaker(
    event: Any,
    *,
    state: _LoopState,
    threshold: int,
) -> None:
    """Update the loop-level circuit breaker from a forwarded worker event.

    Walks every event the worker yields; only acts on
    ``Span(TOOL_CALL, COMPLETE)``. For each such span, looks at
    ``attributes["is_error"]`` and ``attributes["result_summary"]``:

    - error or uninformative result → increment the per-tool
      consecutive-failure counter.
    - informative result → reset the counter to 0.

    If the counter for any tool crosses ``threshold``, sets
    ``state.tripped_tool`` so the loop terminates after the current
    iteration. The check is generic — it does NOT enumerate tool
    names — and shares
    :func:`kaos_agents.planning.result_check.is_uninformative_result`
    with the Runner-layer ``CircuitBreaker`` so the two layers agree
    on the predicate.

    0.1.2 (R1.1): accepts both typed ``Span`` instances AND SSE record
    dicts (the shape the SPA's worker forwards). See
    :func:`_tool_call_complete_attrs` for the supported shapes.
    """
    if state.tripped_tool:
        return  # already tripped — don't keep counting
    if threshold < 1:
        return  # circuit breaker disabled
    attrs = _tool_call_complete_attrs(event)
    if attrs is None:
        return
    tool_name = str(attrs.get("tool_name", "") or "")
    if not tool_name:
        return
    is_error = bool(attrs.get("is_error", False))
    result_summary = str(attrs.get("result_summary", "") or "")
    is_failure = is_error
    if not is_failure:
        # Lazy-import: same rationale as kaos_agents.action.circuit —
        # planning.__init__ chains through optional [llm] extras.
        from kaos_agents.planning.result_check import is_uninformative_result

        is_failure = is_uninformative_result(result_summary)
    if is_failure:
        new_count = state.consecutive_tool_failures.get(tool_name, 0) + 1
        state.consecutive_tool_failures[tool_name] = new_count
        if new_count >= threshold:
            state.tripped_tool = tool_name
            state.tripped_failures = new_count
            logger.warning(
                "agentic_loop: circuit breaker tripped tool=%s failures=%d threshold=%d",
                tool_name,
                new_count,
                threshold,
            )
    else:
        state.consecutive_tool_failures[tool_name] = 0


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


# ── Post-satisfied grounding critics (M2 / M3 / M4) ──────────────────
#
# After a *satisfied* GoalCheck the loop optionally runs one or more
# grounding critics (opt-in: the caller passes a model). Each is a
# JudgeSignature rubric living in planning.{m2_consistency,m3_grounding,
# m4_completeness}; a failing verdict overrides the satisfied terminator
# and forces one more iteration with a re-write directive.
#
# All three are described declaratively by a `_GroundingCritic` spec and
# share one runner (`_process_critic`), so the loop body holds three
# small guarded calls instead of three ~55-line copies. Adding a critic
# is a new spec + one guarded call.

# Minimum verdict confidence for a critic to override a satisfied
# GoalCheck. The JudgeSignature rubrics themselves say "high confidence
# (>= 0.85) when the contradiction is explicit"; the loop honours that
# threshold so an over-eager critic can't flip a correctly-grounded
# answer into a replan on a weak heuristic (WU-K v2 Case E1, 2026-05-21).
_CRITIC_OVERRIDE_CONFIDENCE_FLOOR = 0.85

# Failing verdict label → re-write directive threaded into the next
# iteration's thinking_note. These strings are LLM-VISIBLE prompts: they
# name the actual problem and the fix, never the internal critic id or
# the rubric label (feedback_llm_visible_prompt_prose).
_M2_DIRECTIVES: dict[str, str] = {
    "contradicts_reasoning": (
        "Your previous response's opening headline/announcement/summary "
        "contradicted its own body. Re-write so the opening matches the "
        "body's actual computation and final conclusion. Do not leave a "
        "stale headline from a first-draft attempt."
    ),
    "contradicts_tool_results": (
        "Your previous response asserted facts that contradict what the "
        "tool calls actually returned. Re-write using only what the tools "
        "returned; do not fall back to training memory for the disputed "
        "claim."
    ),
}
_M3_DIRECTIVES: dict[str, str] = {
    "fabricated_with_admission": (
        "Your previous response ADMITTED it could not read / fetch / "
        "access the document AND THEN described the document's contents "
        "anyway. Re-write so EITHER (a) you actually invoke a fetch / "
        "search / read tool for the document and ground your claims in "
        "what it returns, OR (b) you refuse cleanly and do NOT describe "
        "contents you cannot verify. Never both — that's the worst case "
        "for the user."
    ),
    "fabricated_without_admission": (
        "Your previous response made substantive claims about a "
        "document/source that the tool calls did not actually return. "
        "Re-write using ONLY what the tools returned (or what the user "
        "supplied verbatim in the prompt). If the tools returned nothing "
        "useful, refuse honestly — do not silently fall back to training "
        "memory for specifics about named documents."
    ),
}
_M4_DIRECTIVES: dict[str, str] = {
    "partial": (
        "Your previous response returned fewer items than the user "
        "requested without acknowledging exhaustion. EITHER continue "
        "searching the corpus for the missing items, OR re-write to "
        "explicitly state which items were found and which were not "
        "(cite what you searched — 'I reviewed N documents and found M')."
    ),
}


@dataclass(frozen=True, slots=True)
class _GroundingCritic:
    """Declarative spec for one post-satisfied grounding critic.

    Fields:
        name: Internal id used in events, logs, and the refusal
            rationale ONLY — never threaded into an LLM prompt.
        directives: Failing verdict label → plain-English re-write
            instruction (LLM-visible; see ``_M2_DIRECTIVES`` et al.).
            A verdict whose label is not a key never overrides.
        confidence_floor: Minimum verdict confidence to override. M2 and
            M4 use ``_CRITIC_OVERRIDE_CONFIDENCE_FLOOR``; M3 uses ``0.0``
            (any trustworthy fabrication label overrides) because an
            admitted-or-silent fabrication is the highest-severity
            failure family. This asymmetry is intentional and pre-dates
            the unification — see docs/plans/2026-05-29-kaos-agents-
            simplification-diary.md before "fixing" it.
    """

    name: str
    directives: Mapping[str, str]
    confidence_floor: float


_M2_CRITIC = _GroundingCritic("M2", _M2_DIRECTIVES, _CRITIC_OVERRIDE_CONFIDENCE_FLOOR)
_M3_CRITIC = _GroundingCritic("M3", _M3_DIRECTIVES, 0.0)
_M4_CRITIC = _GroundingCritic("M4", _M4_DIRECTIVES, _CRITIC_OVERRIDE_CONFIDENCE_FLOOR)


def _critic_override(spec: _GroundingCritic, verdict: Any) -> str:
    """Return this critic's thinking_note directive, or '' if it doesn't override.

    A critic overrides the satisfied terminator iff the verdict is
    trustworthy (not ``fell_back``), clears the spec's confidence floor,
    and carries a failing label the spec has a directive for.
    """
    if verdict.fell_back or verdict.confidence < spec.confidence_floor:
        return ""
    return spec.directives.get(verdict.label, "")


def _process_critic(
    *,
    spec: _GroundingCritic,
    verdict: Any,
    state: _LoopState,
    session_id: str,
    run_id: str,
) -> tuple[str, str, ConsistencyChecked]:
    """Shared post-processing for one post-satisfied grounding critic.

    Folds the verdict's cost into ``state``, decides whether it overrides
    the satisfied verdict, builds the :class:`ConsistencyChecked`
    observability event, and logs once (INFO when trusted, WARNING when
    the critic fell back). Returns ``(override_note, rationale, event)``:

    * ``override_note`` — the directive to thread into the next
      iteration's thinking_note, or '' when the critic does not override.
    * ``rationale`` — human-facing diagnosis stashed for a downstream
      max-iterations refusal; '' unless the critic overrode.
    * ``event`` — the caller yields this to preserve stream order.
    """
    state.cumulative_cost_usd += verdict.cost_usd
    note = _critic_override(spec, verdict)
    rationale = ""
    if note:
        rationale = (
            f"{spec.name} ({verdict.label}): {verdict.reasoning}"
            if verdict.reasoning
            else f"{spec.name} flagged {verdict.label} at confidence {verdict.confidence:.2f}"
        )
    event = _build_consistency_checked_event(
        verdict=verdict,
        iteration=state.iteration,
        overrode_satisfied=bool(note),
        state=state,
        session_id=session_id,
        run_id=run_id,
    )
    if verdict.fell_back:
        logger.warning(
            "%s critic fell back (treated as pass) session=%s iteration=%d cost=$%.4f reason=%r",
            spec.name,
            session_id,
            state.iteration,
            verdict.cost_usd,
            verdict.reasoning,
        )
    else:
        logger.info(
            "%s critic verdict=%s confidence=%.2f overrode_satisfied=%s "
            "session=%s iteration=%d cost=$%.4f latency_ms=%.0f",
            spec.name,
            verdict.label,
            verdict.confidence,
            bool(note),
            session_id,
            state.iteration,
            verdict.cost_usd,
            verdict.latency_ms,
        )
    return note, rationale, event


def _build_consistency_checked_event(
    *,
    verdict: Any,
    iteration: int,
    overrode_satisfied: bool,
    state: _LoopState,
    session_id: str,
    run_id: str,
) -> ConsistencyChecked:
    """Pack a :class:`JudgeVerdict` from M2 into a :class:`ConsistencyChecked` event.

    Kept-import-light: ``verdict`` is typed ``Any`` to avoid forcing
    the kaos-llm-core import path on consumers of the events module.
    """
    return ConsistencyChecked(
        **_evt_base(state, session_id, run_id),
        label=getattr(verdict, "label", "") or "",
        confidence=float(getattr(verdict, "confidence", 0.0) or 0.0),
        reasoning=str(getattr(verdict, "reasoning", "") or ""),
        iteration=iteration,
        cost_usd=float(getattr(verdict, "cost_usd", 0.0) or 0.0),
        latency_ms=float(getattr(verdict, "latency_ms", 0.0) or 0.0),
        fell_back=bool(getattr(verdict, "fell_back", False)),
        overrode_satisfied=overrode_satisfied,
    )


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


def _char_3grams(text: str) -> set[str]:
    """Character-level 3-gram set used for semantic similarity.

    0.1.2 (R1.4 — reliability roadmap #564). Lowercased and stripped of
    runs of whitespace so cosmetic reformatting between iterations
    (e.g., the model adding bullet markers but otherwise saying the
    same thing) does not defeat the comparison.
    """
    if not text:
        return set()
    normalized = re.sub(r"\s+", " ", text.strip().lower())
    if len(normalized) < 3:
        return {normalized} if normalized else set()
    return {normalized[i : i + 3] for i in range(len(normalized) - 2)}


def _jaccard_similarity(a: set[str], b: set[str]) -> float:
    """Standard Jaccard similarity over n-gram sets. 0.0 when both empty."""
    if not a and not b:
        return 0.0
    intersection = len(a & b)
    union = len(a | b)
    if union == 0:
        return 0.0
    return intersection / union


# 0.1.2 (R1.4): two semantically-identical refusals with cosmetic wording
# differences (Agent 4's C5 case) score ~0.85 Jaccard 3-gram similarity.
# Pre-R1.4 the prefix/substring check missed these and the loop kept
# burning budget on near-duplicate iterations. Threshold tuned to fire
# on near-duplicates without flagging legitimate refinements (which
# typically score < 0.7 on the 3-gram measure).
_SEMANTIC_STUCK_JACCARD_THRESHOLD = 0.85


def _is_stuck(
    *,
    last_text: str,
    new_text: str,
    last_tool_count: int,
    new_tool_count: int,
) -> bool:
    """State-mutation stuck-detection.

    The iteration is "stuck" when BOTH:
      - The new response text is byte-identical, a prefix/suffix of
        the last response, OR semantically near-identical (Jaccard
        3-gram similarity ≥ ``_SEMANTIC_STUCK_JACCARD_THRESHOLD``) —
        i.e., no genuinely new prose,
      - AND no new tool calls happened
        (``new_tool_count <= last_tool_count``).

    0.1.2 (R1.4): adds the Jaccard 3-gram semantic check. Pre-R1.4 only
    byte-identity + substring relationship were considered, which let
    two semantically-identical refusals (e.g., "I cannot find this" vs
    "I'm unable to find that information") bypass detection. The new
    semantic check fires on cosmetic-only rewording while still
    tolerating substantive refinements.

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
    if last_text in new_text or new_text in last_text:
        return True
    # 0.1.2 (R1.4): Jaccard 3-gram semantic check. Catches the
    # cosmetic-rewording pattern the substring check misses.
    similarity = _jaccard_similarity(_char_3grams(last_text), _char_3grams(new_text))
    return similarity >= _SEMANTIC_STUCK_JACCARD_THRESHOLD


def _budget_exceeded(state: _LoopState, policy: SessionPolicy) -> bool:
    return state.cumulative_cost_usd >= policy.max_loop_cost_usd


def _per_tool_budget_exceeded(cost_delta_usd: float, policy: SessionPolicy) -> bool:
    """Plan §Issue 9 / B1.7 — single-call cost cap check.

    Returns True when the operator has tightened ``max_per_tool_cost_usd``
    to a non-zero value AND the most recent cost delta (one LLM call
    inside the loop iteration — planner, worker, critic, or judge)
    billed above that ceiling. The loop terminates with
    ``LoopTerminated(reason="cost_exceeded")`` exactly as the
    cumulative-cap path does; downstream consumers don't need to
    distinguish the two.

    The 0.0 default means "disabled" — historic behavior preserved
    for every caller that hasn't opted in.
    """
    if policy.max_per_tool_cost_usd <= 0.0:
        return False
    return cost_delta_usd > policy.max_per_tool_cost_usd


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


def _format_tool_results_for_m2(
    tool_calls_made: list[dict],
    *,
    per_call_char_budget: int | None = None,
) -> str:
    """Render worker tool-call results as ``context`` for the M2 critic.

    The M2 rubric distinguishes ``contradicts_tool_results`` from
    ``contradicts_reasoning`` by comparing the response against this
    context. Empty context means "no tools invoked" — the critic will
    not return ``contradicts_tool_results`` in that case (per its
    decision rule 1).

    Output is one line per tool call:
        ``<tool_name>: <result_summary>``

    Args:
        tool_calls_made: WorkerResult.tool_calls_made list.
        per_call_char_budget: When set, truncates each call's
            result_summary at this many characters (appending ``…``).
            **Default ``None`` — no truncation.** The judge sees the
            FULL summary so it can spot fabrication anywhere in the
            result, not just in the first N characters. Pass an
            explicit budget only when the operator has measured a
            specific provider context-window pressure; never as a
            default-on safety net (silent truncation makes the judge
            dishonest about what it actually evaluated).
    """
    if not tool_calls_made:
        return ""
    lines: list[str] = []
    for call in tool_calls_made:
        name = str(call.get("tool_name") or call.get("name") or "tool")
        summary = call.get("result_summary") or call.get("result") or call.get("output") or ""
        summary_str = str(summary).strip().replace("\n", " ")
        if per_call_char_budget is not None and len(summary_str) > per_call_char_budget:
            summary_str = summary_str[:per_call_char_budget] + "…"
        lines.append(f"{name}: {summary_str}")
    return "\n".join(lines)


# 0.1.1: matches kaos-* tool references inside remediation hints.
# The kaos-mcp tool-design contract says errors must name an
# alternative tool, typically as "Try kaos-X" or in a comma-separated
# list "Try kaos-X, kaos-Y for Z". This regex catches every "kaos-..."
# token appearing in such a hint. The "Try" anchor is checked
# separately to avoid false positives on bare tool names in unrelated
# error text.
_TRY_HINT_ANCHOR_RE = re.compile(r"\bTry\b", re.IGNORECASE)
_KAOS_TOOL_RE = re.compile(r"\bkaos-[a-z0-9]+(?:-[a-z0-9]+)+\b")


def _extract_remediation_hints(tool_calls_made: list[dict]) -> list[str]:
    """Pull "Try kaos-{module}-{tool}" hints from this iteration's errored calls.

    0.1.1 (#549.B). Worker errors carry remediation hints per the
    kaos-mcp tool-design contract: every error must include
    *(1) what went wrong, (2) how to fix it, (3) alternative tool*.
    When the agent's worker hits an error, the AgenticLoop threads
    these hints into the next iteration's ``thinking_note`` so the
    worker doesn't retry the same broken tool blindly.

    Returns a list of unique ``"Try {tool}"`` hint sentences (max 3
    hints). Multi-tool "Try X, Y, Z" remediations expand into
    individual entries so each alternative is visible to the next
    iteration. Empty list when there were no errored calls or none
    carried a Try-X remediation.
    """
    if not tool_calls_made:
        return []
    hints: list[str] = []
    seen: set[str] = set()
    for call in tool_calls_made:
        if not bool(call.get("is_error") or False):
            continue
        text = str(call.get("summary_excerpt") or call.get("result_summary") or "")
        if not text:
            continue
        # The "Try" anchor must be in the same string as the tool
        # name — i.e. inside a remediation sentence — to avoid
        # promoting an unrelated kaos-X mention into a "hint".
        if not _TRY_HINT_ANCHOR_RE.search(text):
            continue
        for match in _KAOS_TOOL_RE.finditer(text):
            tool_name = match.group(0)
            hint = f"Try {tool_name}"
            if hint not in seen:
                seen.add(hint)
                hints.append(hint)
                if len(hints) >= 3:
                    return hints
    return hints


def _summarize_prior_tool_calls(tool_calls_made: list[dict]) -> str:
    """Render a short bullet summary of this iteration's tool calls.

    0.1.1 (P2-B). Threaded into the next iteration's thinking_note so
    the worker sees what was already tried and avoids re-issuing
    near-duplicate queries (the over-specified-search-storm pattern
    documented in WU-K v2 Case C1).

    Output shape (one bullet per call, max 10):

        - tool_name(is_error=False) — first 120 chars of result
        - tool_name(is_error=True) — first 120 chars of result

    Empty string when no calls were made.
    """
    if not tool_calls_made:
        return ""
    lines: list[str] = []
    for call in tool_calls_made[:10]:
        name = str(call.get("name") or call.get("tool_name") or "tool")
        is_error = bool(call.get("is_error") or False)
        summary = call.get("summary_excerpt") or call.get("result_summary") or ""
        summary_str = str(summary).strip().replace("\n", " ")
        if len(summary_str) > 120:
            summary_str = summary_str[:120] + "…"
        lines.append(f"- {name}(is_error={is_error}) — {summary_str}")
    return "\n".join(lines)


__all__ = [
    "WorkerCallable",
    "WorkerResult",
    "run_agentic_turn",
]
