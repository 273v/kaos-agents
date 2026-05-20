"""AgenticLoop policy + loop-control events.

Per `kaos-modules/docs/internal/agentic-loop-auto-elevation-plan.md`
§4 Layer 2 + §7. The four new event types the AgenticLoop yields:

- :class:`ToolPolicyElevated` — auto-elevation just happened
  silently (green-auto tier). The SPA renders an inline badge.
- :class:`CapabilityRequested` — the planner wants a yellow-confirm
  group that needs user approval. The chat router pauses and emits
  this event; the SPA renders an inline approval card. The agent
  receives the user's decision on the next turn.
- :class:`GoalChecked` — the Critic ran. Carries the three-way
  verdict. Drives the SPA's GoalCheckBadge color.
- :class:`LoopTerminated` — the AgenticLoop stopped iterating. Carries
  the reason (one of: ``"satisfied"``, ``"insufficient_evidence"``,
  ``"max_iterations"``, ``"cost_exceeded"``, ``"wall_clock_exceeded"``,
  ``"stuck_no_progress"``, ``"user_interrupt"``).

All four subclass :class:`LifecycleEvent` so existing event-routing
code (hooks, SSE wire serializer, registry) handles them
automatically. Auto-registration into ``default_event_registry``
happens on subclass creation via the :class:`KaosEvent` metaclass
hook.
"""

from __future__ import annotations

from pydantic import Field

from kaos_agents.events._intermediates import LifecycleEvent


class ToolPolicyElevated(LifecycleEvent):
    """Auto-elevation just happened (green-auto tier).

    The chat router consulted ``SessionPolicy.tier_for(group)`` for each
    group in ``TurnToolPolicy.dropped_groups``, found at least one
    green-auto group, and added it to ``allowed_groups`` without
    pausing the turn. This event is the audit trail.

    Attributes:
        elevated_groups: Groups added to ``allowed_groups`` this turn.
        kept_groups: Final ``kept_groups`` after elevation.
        previous_allowed: ``allowed_groups`` BEFORE the elevation.
        rationale: One-sentence explanation (lifted from the planner).
        iteration: Loop iteration number (1-indexed).
    """

    elevated_groups: list[str] = Field(default_factory=list)
    kept_groups: list[str] = Field(default_factory=list)
    previous_allowed: list[str] = Field(default_factory=list)
    rationale: str = ""
    iteration: int = 1


class CapabilityRequested(LifecycleEvent):
    """The planner wants a yellow-confirm group that needs user approval.

    The chat router pauses the loop, emits this event, and yields the
    in-progress turn. The SPA renders an inline approval card with
    [Enable for this turn] [Enable for session] [Deny + continue]
    [Deny + stop]. On approval, the chat router resumes from the same
    step with the elevated ceiling; on denial, the loop continues
    without the requested capability (the agent's next iteration may
    still get a ``satisfied`` from the Critic if the question can be
    answered without it).

    Attributes:
        requested_groups: Groups the planner wants in this tier.
        justification: One-sentence explanation from the planner
            (e.g., "needs Playwright to scrape this JavaScript-rendered
            page"). Surfaced verbatim in the approval card.
        iteration: Loop iteration number.
        previous_allowed: Current ``allowed_groups`` so the SPA can
            show the delta.
    """

    requested_groups: list[str] = Field(default_factory=list)
    justification: str = ""
    iteration: int = 1
    previous_allowed: list[str] = Field(default_factory=list)


class GoalChecked(LifecycleEvent):
    """The Critic ran after this iteration's ReAct.

    Carries the three-way verdict, which drives both the AgenticLoop's
    next step (return / replan / refuse) and the SPA's
    ``GoalCheckBadge`` rendering (green / amber / gray).

    Attributes:
        kind: One of ``"satisfied"`` / ``"needs_more_work"`` /
            ``"insufficient_evidence"``.
        rationale: One-sentence explanation the user will see.
        next_action: When ``kind == "needs_more_work"``, the imperative
            one-liner the next iteration's agent thinks about (NOT a
            fake user message — preserves transcript hygiene).
            Empty otherwise.
        missing: When ``kind == "insufficient_evidence"``, what the
            corpus lacks. Empty otherwise.
        confidence: Critic's self-rated [0.0, 1.0].
        iteration: Loop iteration number.
        cost_usd: Critic's LLM cost this call.
        latency_ms: Critic's wall-clock time this call.
    """

    kind: str = ""
    rationale: str = ""
    next_action: str = ""
    missing: str = ""
    confidence: float = 0.0
    iteration: int = 1
    cost_usd: float = 0.0
    latency_ms: float = 0.0


class ConsistencyChecked(LifecycleEvent):
    """The M2 reasoning-action consistency critic ran after the worker.

    Emitted by ``run_agentic_turn`` when ``m2_consistency_model`` is
    set and the worker has produced a response. Carries the M2
    verdict alongside whether the critic actually overrode the
    GoalCheck terminator (i.e. forced a re-iteration).

    Mirrors :class:`GoalChecked` shape — same audit-trail discipline,
    different rubric. SPA run-inspectors render this as an M2 chip
    alongside the GoalCheck chip. OTel exporters get it as a
    Span(SUBJECT.JUDGE, phase=COMPLETE) sibling.

    Attributes:
        label: One of ``"consistent"`` / ``"contradicts_reasoning"``
            / ``"contradicts_tool_results"`` / ``""`` (when
            ``fell_back=True``). Lowercase, verbatim from the rubric.
        confidence: M2's self-rated [0.0, 1.0]. ``0.0`` when
            ``fell_back=True``.
        reasoning: One-paragraph justification from the critic.
            Empty when ``fell_back=True``.
        iteration: Loop iteration number that produced the response
            this verdict applies to.
        cost_usd: M2's LLM cost this call.
        latency_ms: M2's wall-clock time this call.
        fell_back: True when the critic invocation errored / emitted
            a disallowed label. Loop treats fell_back as ``consistent``
            to avoid loop-on-broken-critic, but the event preserves
            the signal so operators can see it.
        overrode_satisfied: True when the GoalCheck verdict for the
            same iteration was ``satisfied`` AND the M2 label was a
            ``contradicts_*`` — i.e. M2 forced another iteration.
            The single most useful column for "did M2 do anything?"
    """

    label: str = ""
    confidence: float = 0.0
    reasoning: str = ""
    iteration: int = 1
    cost_usd: float = 0.0
    latency_ms: float = 0.0
    fell_back: bool = False
    overrode_satisfied: bool = False


class CircuitBreakerTripped(LifecycleEvent):
    """A per-tool :class:`~kaos_agents.action.circuit.CircuitBreaker` opened.

    Emitted when a CircuitBreaker transitions from CLOSED to OPEN —
    either via :meth:`record_failure` crossing the configured threshold
    or via a HALF_OPEN probe failing. Carries the per-tool diagnostic
    metadata downstream consumers (AgenticLoop terminators, the SPA
    chat surface, audit logs) need to render an honest refusal.

    Attributes:
        tool_name: The tool whose breaker opened.
        consecutive_failures: How many consecutive failures (or
            uninformative results when
            ``uninformative_counts_as_failure`` is set) led to the
            trip. At least ``failure_threshold``.
        failure_threshold: The threshold the breaker was configured
            with (so consumers can render "5/5" without inferring).
        reset_timeout_seconds: How long the breaker will stay OPEN
            before allowing a HALF_OPEN probe.
        uninformative_counted: True iff
            ``uninformative_counts_as_failure`` was on when the trip
            fired (so consumers can phrase the refusal correctly —
            "tool kept failing" vs "tool kept returning empty").
    """

    tool_name: str = ""
    consecutive_failures: int = 0
    failure_threshold: int = 0
    reset_timeout_seconds: float = 0.0
    uninformative_counted: bool = True


class LoopTerminated(LifecycleEvent):
    """The AgenticLoop exited.

    Always the last event yielded by an AgenticLoop turn. The SPA
    finalizes the streaming message + renders any terminal banner
    based on ``reason``.

    Attributes:
        reason: One of:
          - ``"satisfied"`` — Critic returned Satisfied; happy path.
          - ``"insufficient_evidence"`` — Critic confirmed the corpus
            cannot answer; refusal-with-explanation.
          - ``"max_iterations"`` — Hit ``max_loop_iterations`` cap.
          - ``"cost_exceeded"`` — Hit ``max_loop_cost_usd`` cap.
          - ``"wall_clock_exceeded"`` — Hit
            ``max_loop_wall_clock_seconds`` cap.
          - ``"stuck_no_progress"`` — State-mutation check fired
            (an iteration completed without new tool calls or new text).
          - ``"circuit_breaker_tripped"`` — A single tool returned
            N consecutive failures or uninformative results (#506
            follow-up). The :class:`CircuitBreakerTripped` event
            emitted just before this carries the per-tool diagnostic.
          - ``"user_interrupt"`` — FastAPI request was aborted.
        iterations_used: How many iterations the loop ran (1..N).
        elevations_used: Total groups auto-elevated across iterations.
        cost_usd: Cumulative LLM cost across the entire loop (planner +
            critic + worker + any re-planner calls).
        wall_clock_ms: Total wall-clock time of the loop.
    """

    reason: str = ""
    iterations_used: int = 0
    elevations_used: int = 0
    cost_usd: float = 0.0
    wall_clock_ms: float = 0.0


__all__ = [
    "CapabilityRequested",
    "CircuitBreakerTripped",
    "ConsistencyChecked",
    "GoalChecked",
    "LoopTerminated",
    "ToolPolicyElevated",
]
