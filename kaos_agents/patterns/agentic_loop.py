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
import time
from collections.abc import AsyncIterator, Awaitable, Callable
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
    # The most recent GoalCheck rationale + next_action — captured so
    # the max_iterations refusal-override can render an honest "I
    # tried and failed" message instead of letting the last
    # hallucinated worker text through. See task #505.
    last_critic_rationale: str = ""
    last_critic_next_action: str = ""
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
    circuit_breaker_threshold: int = 5,
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

            # ── 4b. M2 reasoning-action consistency override ─────────
            # When the caller opted-in to M2 (m2_consistency_model is
            # set) AND the goal check returned satisfied, run the M2
            # critic on the worker's response. A contradicts_* verdict
            # overrides the terminator and forces a re-write iteration.
            # See 2026-05-19-agentic-loop-honesty.md §3.2 + the M2
            # rubric in kaos_agents.planning.m2_consistency.
            m2_override_note: str = ""
            if m2_consistency_model is not None and outcome.satisfied:
                m2_verdict = await judge_reasoning_action_consistency(
                    response_text=worker_result.text,
                    model=m2_consistency_model,
                    tool_results_text=_format_tool_results_for_m2(worker_result.tool_calls_made),
                )
                state.cumulative_cost_usd += m2_verdict.cost_usd
                # Confidence floor — only an explicit contradiction (per
                # the rubric's own ">= 0.85" guidance) overrides a
                # satisfied verdict. Lower-confidence flags are real
                # observability signals (emitted via the event +
                # structured log below) but must not flip a passing
                # turn into a replan. Without this gate, an over-eager
                # critic LLM can refuse correctly-grounded answers on
                # weak heuristics — WU-K v2 Case E1 (2026-05-21):
                # response cited "Meridian Financial, LLC" verbatim
                # present in tool results, M2 returned
                # contradicts_tool_results @ 0.92 anyway. The rubric
                # has been tightened with explicit "RAG pick-one" and
                # "honest can't-verify" carve-outs (see
                # planning/m2_consistency.py); this floor is the
                # defensive backstop when the model still flags those
                # patterns despite the carve-outs.
                M2_OVERRIDE_CONFIDENCE_FLOOR: float = 0.85
                if (
                    not m2_verdict.fell_back
                    and m2_verdict.confidence >= M2_OVERRIDE_CONFIDENCE_FLOOR
                ):
                    if m2_verdict.label == "contradicts_reasoning":
                        m2_override_note = (
                            "M2 consistency critic flagged contradicts_reasoning: "
                            "your prior response's headline contradicted its own "
                            "body. Re-write the response so the opening "
                            "branch/announcement/summary matches the body's "
                            "actual computation and final conclusion. Do not "
                            "leave a stale headline from a first-draft attempt."
                        )
                    elif m2_verdict.label == "contradicts_tool_results":
                        m2_override_note = (
                            "M2 consistency critic flagged contradicts_tool_results: "
                            "your prior response asserted facts that contradict "
                            "what the tool calls actually returned. Re-write the "
                            "response using only what the tools returned; do not "
                            "fall back to training memory for the disputed claim."
                        )
                # Emit a typed observability event so operators, SPA
                # run-inspectors, and OTel exporters all see the
                # verdict (regardless of whether it triggered an
                # override). Mirrors GoalChecked + ToolPolicyElevated.
                yield _build_consistency_checked_event(
                    verdict=m2_verdict,
                    iteration=state.iteration,
                    overrode_satisfied=bool(m2_override_note),
                    state=state,
                    session_id=session_id,
                    run_id=run_id,
                )
                # Structured log so default log filters surface M2
                # firings without grepping memory.json. INFO when the
                # verdict is trustworthy, WARNING on fell_back.
                if m2_verdict.fell_back:
                    logger.warning(
                        "M2 consistency critic fell back (treated as consistent) "
                        "session=%s iteration=%d cost=$%.4f reason=%r",
                        session_id,
                        state.iteration,
                        m2_verdict.cost_usd,
                        m2_verdict.reasoning,
                    )
                else:
                    logger.info(
                        "M2 consistency critic verdict=%s confidence=%.2f "
                        "overrode_satisfied=%s session=%s iteration=%d "
                        "cost=$%.4f latency_ms=%.0f",
                        m2_verdict.label,
                        m2_verdict.confidence,
                        bool(m2_override_note),
                        session_id,
                        state.iteration,
                        m2_verdict.cost_usd,
                        m2_verdict.latency_ms,
                    )
                # When M2 fires, do NOT terminate — fall through to the
                # replan section below with the M2 note in thinking_note.

            # ── 4c. M3 document-grounding fabrication override ───────
            # Composes with M2 — both critics run when satisfied. Either
            # one's override fires the replan. M3 targets the "agent
            # admits it didn't read X then describes X" pattern
            # (task #445 P0-7 / R1-REAL recurrence).
            m3_override_note: str = ""
            if m3_grounding_model is not None and outcome.satisfied:
                m3_verdict = await judge_grounding_fabrication(
                    response_text=worker_result.text,
                    model=m3_grounding_model,
                    tool_results_text=_format_tool_results_for_m2(worker_result.tool_calls_made),
                )
                state.cumulative_cost_usd += m3_verdict.cost_usd
                if not m3_verdict.fell_back:
                    if m3_verdict.label == "fabricated_with_admission":
                        m3_override_note = (
                            "M3 grounding critic flagged "
                            "fabricated_with_admission: your prior response "
                            "ADMITTED it could not read / fetch / access the "
                            "document AND THEN described the document's "
                            "contents anyway. Re-write so EITHER (a) you "
                            "actually invoke a fetch / search / read tool "
                            "for the document and ground your claims in what "
                            "the tool returns, OR (b) you refuse cleanly and "
                            "do NOT describe contents you cannot verify. "
                            "Never both — that's the worst case for the user."
                        )
                    elif m3_verdict.label == "fabricated_without_admission":
                        m3_override_note = (
                            "M3 grounding critic flagged "
                            "fabricated_without_admission: your prior response "
                            "made substantive claims about a document/source "
                            "that the tool calls did not actually return. "
                            "Re-write the response using ONLY what the tools "
                            "returned (or what the user supplied verbatim in "
                            "the prompt). If the tools returned nothing "
                            "useful, refuse honestly — do not silently fall "
                            "back to training memory for specifics about "
                            "named documents."
                        )
                yield _build_consistency_checked_event(
                    verdict=m3_verdict,
                    iteration=state.iteration,
                    overrode_satisfied=bool(m3_override_note),
                    state=state,
                    session_id=session_id,
                    run_id=run_id,
                )
                if m3_verdict.fell_back:
                    logger.warning(
                        "M3 grounding critic fell back (treated as grounded) "
                        "session=%s iteration=%d cost=$%.4f reason=%r",
                        session_id,
                        state.iteration,
                        m3_verdict.cost_usd,
                        m3_verdict.reasoning,
                    )
                else:
                    logger.info(
                        "M3 grounding critic verdict=%s confidence=%.2f "
                        "overrode_satisfied=%s session=%s iteration=%d "
                        "cost=$%.4f latency_ms=%.0f",
                        m3_verdict.label,
                        m3_verdict.confidence,
                        bool(m3_override_note),
                        session_id,
                        state.iteration,
                        m3_verdict.cost_usd,
                        m3_verdict.latency_ms,
                    )

            # Compose M2 + M3 override notes. When both fire, the user
            # sees both directives in the next iteration's thinking_note.
            override_note = "\n\n".join(n for n in (m2_override_note, m3_override_note) if n)

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
            # Three paths into replan: the goal check returned
            # needs_more_work, OR the M2 override fired on a satisfied
            # verdict, OR the M3 override fired. M2/M3 paths bypass
            # the GoalCheckNeedsMoreWork assertion because the outcome
            # object is still satisfied.
            if override_note:
                state.thinking_note = override_note
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
        "I was unable to answer this within the {iterations}-iteration "
        "budget. The critic kept flagging my attempts as ungrounded or "
        "incomplete:"
    ),
    "stuck_no_progress": (
        "I stopped after {iterations} iteration(s) because my responses "
        "were no longer making progress (identical or near-identical "
        "text with no new tool calls). The critic's last diagnosis:"
    ),
    "cost_exceeded": (
        "I stopped after {iterations} iteration(s) because the loop's "
        "cost budget was exhausted. The critic's last diagnosis:"
    ),
    "wall_clock_exceeded": (
        "I stopped after {iterations} iteration(s) because the loop's "
        "wall-clock budget was exhausted. The critic's last diagnosis:"
    ),
    "circuit_breaker_tripped": (
        "I stopped after {iterations} iteration(s) because a single "
        "tool kept failing or returning no usable result (circuit "
        "breaker tripped). The critic's last diagnosis:"
    ),
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


async def _emit_failure_refusal(
    *,
    reason: str,
    state: _LoopState,
    session_id: str,
    run_id: str,
) -> AsyncIterator[Any]:
    """Emit the TextDelta + TurnSummary pair carrying a clean refusal.

    Used by every non-satisfied terminator (max_iterations,
    stuck_no_progress, cost_exceeded, wall_clock_exceeded) so the
    user sees an honest "I tried and failed" message instead of the
    last worker attempt the critic just rejected. See task #505.
    """
    refusal_text = _build_failure_refusal(
        reason=reason,
        iterations=state.iteration,
        critic_rationale=state.last_critic_rationale,
        critic_next_action=state.last_critic_next_action,
    )
    yield TextDelta(
        **_evt_base(state, session_id, run_id),
        content=refusal_text,
    )
    yield TurnSummary(
        **_evt_base(state, session_id, run_id),
        text=refusal_text,
        intent="refuse",
        cost_usd=state.cumulative_cost_usd,
    )


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
    """
    if state.tripped_tool:
        return  # already tripped — don't keep counting
    if threshold < 1:
        return  # circuit breaker disabled
    if not isinstance(event, Span):
        return
    if event.subject != SpanSubject.TOOL_CALL or event.phase != SpanPhase.COMPLETE:
        return
    attrs = event.attributes
    if not isinstance(attrs, dict):
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


__all__ = [
    "WorkerCallable",
    "WorkerResult",
    "run_agentic_turn",
]
