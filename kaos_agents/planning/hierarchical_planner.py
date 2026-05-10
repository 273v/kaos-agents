"""HierarchicalPlanner — paper §7.1 hierarchical decomposition.

Law-firm-style decomposition: the matter (parent) breaks into
workstreams, each delegated to an associate (sub-agent). Each
sub-agent runs as a fresh AgentLoop turn; results aggregate back
to the parent.

Sub-agent definitions are AgentEnvelopes (content-addressed,
serializable, replayable). Phase 3.C ships static decomposition —
the parent specifies sub-agents up front via constructor — and a
heuristic decomposer based on intent.constraints. Phase 3+ wires
an LLM-driven decomposer Call.

Resolved Decision #4 (stream_mode): sub-agent events flow upward
through three modes:
  - "value-only" (default): collapse stream-deltas, propagate value
    events (UsageObserved, CitationFound, EvidenceInsufficient,
    Span(SUBAGENT/STEP/TURN/...), TurnSummary, MemoryEvent).
  - "full": every sub-event flows up, including TextDelta /
    ThinkingDelta / ToolCallArgsDelta. Most expensive.
  - "summary-only": only Span(SUBAGENT, COMPLETE) + the final
    TurnSummary flow up. Cheapest.

Resolved Decision #3 ensures the AgentLoop selects this planner
based on intent.pattern; that wiring lives in Phase 3.D.

Depth tracking
--------------

Hierarchical delegation uses its own ``contextvars.ContextVar``
(``_hierarchical_depth``) to detect runaway nesting. The runtime
delegation system in :mod:`kaos_agents.runtime.delegation` keeps a
parallel counter; we don't share it because the hierarchical-planner
recursion semantics differ subtly (one parent → N sub-agents in a
single execute, vs. one parent → one sub-agent per call). When
``execute()`` exceeds ``max_delegation_depth``, a :class:`RuntimeError`
is raised with the agent-friendly format (what + how-to-fix +
alternative).

agent_loop_factory
------------------

The factory is the seam where tests inject stubs and the production
runtime injects a real :class:`AgentLoop`. The default builder reads
the envelope, hydrates an :class:`Agent`, and constructs an
``AgentLoop`` with an :class:`IntentExtractor` configured to the
envelope's model. Tests pass a custom factory (or override
``_default_agent_loop_factory`` indirectly through it) so the
sub-agent is a stub :class:`SimpleNamespace` that pushes pre-canned
events.
"""

from __future__ import annotations

import contextvars
import json
from typing import Any

from pydantic import Field

from kaos_agents.core.envelope import AgentEnvelope
from kaos_agents.events.collector import collect_events, push_event
from kaos_agents.events.lifecycle import (
    IntentClassified,
    TurnSummary,
    UsageObserved,
)
from kaos_agents.events.memory import MemoryEvent
from kaos_agents.events.research import (
    CitationFound,
    EvidenceInsufficient,
    GroundingRefusalTriggered,
)
from kaos_agents.events.spans import Span, SpanPhase, SpanSubject
from kaos_agents.events.stream import TextDelta, ThinkingDelta, ToolCallArgsDelta
from kaos_agents.intent.types import IntentResult
from kaos_agents.planning.planner import Plan, PlanResult
from kaos_agents.triggers.base import Trigger
from kaos_agents.types.usage import ZERO_USAGE, InvocationUsage

# Event types we forward in "value-only" mode (Resolved Decision #4).
_VALUE_EVENT_TYPES: tuple[type, ...] = (
    UsageObserved,
    CitationFound,
    EvidenceInsufficient,
    GroundingRefusalTriggered,
    IntentClassified,
    TurnSummary,
    MemoryEvent,
    Span,
)
# Event types we suppress in "value-only" mode.
_STREAM_DELTA_TYPES: tuple[type, ...] = (TextDelta, ThinkingDelta, ToolCallArgsDelta)


# Module-level depth tracking for hierarchical delegation. Mirrors
# ``kaos_agents.runtime.delegation._delegation_depth`` but in its own
# namespace so the two systems can coexist without one stepping on the
# other's counter. ``execute()`` increments at entry, decrements at
# exit; if the new value would exceed ``max_delegation_depth``, the
# planner raises before constructing any sub-agent.
_hierarchical_depth: contextvars.ContextVar[int] = contextvars.ContextVar(
    "kaos_agents_hierarchical_depth",
    default=0,
)


def current_hierarchical_depth() -> int:
    """Return the current hierarchical-delegation depth (read-only).

    0 = no hierarchical planner is active. 1 = inside one
    HierarchicalPlanner.execute(). 2+ = nested hierarchical planners.
    """
    return _hierarchical_depth.get()


class SubAgentSpec(Plan):
    """One sub-agent declaration in a hierarchical plan.

    Carries the sub-agent's envelope (as wire-safe JSON so the spec
    can be serialised across processes) and the sub-goal text the
    parent is delegating. Subclasses :class:`Plan` so the same
    serialisation discipline applies — frozen, ``extra="allow"``, JSON
    round-trip via ``model_dump_json`` / ``model_validate_json``.
    """

    sub_agent_envelope_json: str  # AgentEnvelope.model_dump_json() — wire-safe
    sub_goal: str
    sub_agent_name: str = ""  # human-readable; defaults to envelope.name
    inputs: dict[str, Any] = Field(default_factory=dict)
    pattern: str = "hierarchical_subagent"


class HierarchicalPlan(Plan):
    """A hierarchical plan: a list of sub-agent specs to delegate.

    The aggregation strategy controls how sub-results combine into the
    parent's :class:`PlanResult.text`:

    * ``"concat"`` (default) — newline-joined sub-outputs.
    * ``"first"`` — the first non-empty sub-output.
    * ``"json"`` — a JSON string mapping sub-agent name → output.

    ``parent_goal`` is the goal text the planner started from — kept
    for replay / telemetry context.
    """

    pattern: str = "hierarchical"
    parent_goal: str = ""
    sub_agents: tuple[SubAgentSpec, ...] = ()
    aggregation_strategy: str = "concat"  # "concat" | "first" | "json"


class HierarchicalPlanner:
    """Planner that decomposes via delegation.

    Args:
        sub_agent_envelopes: Tuple of :class:`AgentEnvelope` to delegate
            to. When provided, :meth:`plan` uses these directly. When
            ``None``, :meth:`plan` falls back to a heuristic decomposer
            that produces a single-sub-agent plan from the intent
            (Phase 3.C minimum).
        stream_mode: ``"value-only"`` (default), ``"full"``, or
            ``"summary-only"``. Controls which sub-agent events flow
            up to the parent collector — see Resolved Decision #4.
        max_concurrent_subagents: Cap on parallel sub-agent runs
            (default 3). Phase 3.C runs sub-agents sequentially; the
            cap is recorded but not enforced concurrency-wise yet.
        decomposer_call: Optional kaos-llm-core ``Call`` (or any
            duck-typed object with an awaitable ``invoke`` method)
            returning a :class:`HierarchicalPlan`. Phase 3.C default
            is ``None``; Phase 3+ wires an LLM-driven decomposer.
        agent_loop_factory: Optional callable producing an
            ``AgentLoop`` for a given :class:`AgentEnvelope`. Defaults
            to :func:`_default_agent_loop_factory`. Tests inject a
            stub here.
        max_delegation_depth: Hard limit on recursion (default 3).
            Tracked via :data:`_hierarchical_depth` ContextVar.
        aggregation_strategy: Default aggregation strategy applied
            when the heuristic / decomposer plan does not specify one
            (default ``"concat"``).
    """

    def __init__(
        self,
        *,
        sub_agent_envelopes: tuple[AgentEnvelope, ...] | None = None,
        stream_mode: str = "value-only",
        max_concurrent_subagents: int = 3,
        decomposer_call: Any | None = None,
        agent_loop_factory: Any | None = None,
        max_delegation_depth: int = 3,
        aggregation_strategy: str = "concat",
    ) -> None:
        if stream_mode not in ("value-only", "full", "summary-only"):
            raise ValueError(
                f"stream_mode must be one of "
                f"('value-only', 'full', 'summary-only'); got {stream_mode!r}"
            )
        if aggregation_strategy not in ("concat", "first", "json"):
            raise ValueError(
                f"aggregation_strategy must be one of "
                f"('concat', 'first', 'json'); got {aggregation_strategy!r}"
            )
        self._sub_agents = sub_agent_envelopes
        self._stream_mode = stream_mode
        self._max_concurrent = max_concurrent_subagents
        self._decomposer_call = decomposer_call
        self._agent_loop_factory = agent_loop_factory or _default_agent_loop_factory
        self._max_depth = max_delegation_depth
        self._aggregation = aggregation_strategy

    @property
    def stream_mode(self) -> str:
        """Active stream-merge mode (Resolved Decision #4)."""
        return self._stream_mode

    @property
    def aggregation_strategy(self) -> str:
        """Active aggregation strategy."""
        return self._aggregation

    @property
    def max_concurrent_subagents(self) -> int:
        """Cap on parallel sub-agent runs."""
        return self._max_concurrent

    @property
    def max_delegation_depth(self) -> int:
        """Hard limit on recursive hierarchical delegation."""
        return self._max_depth

    # ------------------------------------------------------------------
    # Planner Protocol
    # ------------------------------------------------------------------

    async def plan(
        self,
        intent: IntentResult,
        memory: Any | None = None,
    ) -> HierarchicalPlan:
        """Produce a :class:`HierarchicalPlan` from intent.

        Phase 3.C order:
          1. If ``decomposer_call`` is provided: invoke it.
          2. Else if ``sub_agent_envelopes`` is provided: build plan
             from those.
          3. Else: heuristic 1-sub-agent plan that wraps the goal.

        ``memory`` is accepted for protocol compliance but ignored in
        Phase 3.C — Phase 4 wires memory-aware decomposition (e.g.
        prior workstream summaries influencing sub-agent selection).
        """
        del memory  # Phase 4 wires memory-aware decomposition.

        if self._decomposer_call is not None:
            invocation = await self._decomposer_call.invoke(
                goal=intent.goal.statement,
                constraints=tuple(c.value for c in intent.constraints),
            )
            return invocation.output

        if self._sub_agents is not None:
            specs = tuple(
                SubAgentSpec(
                    pattern="hierarchical_subagent",
                    sub_agent_envelope_json=env.model_dump_json(),
                    sub_goal=intent.goal.statement,
                    sub_agent_name=env.name or f"subagent_{i}",
                )
                for i, env in enumerate(self._sub_agents)
            )
            return HierarchicalPlan(
                pattern="hierarchical",
                parent_goal=intent.goal.statement,
                sub_agents=specs,
                aggregation_strategy=self._aggregation,
                metadata={
                    "source": "explicit_envelopes",
                    "intent_type": intent.goal.intent_type.value,
                    "sub_agent_count": len(specs),
                },
            )

        # Heuristic fallback: a single research sub-agent.
        env = self._heuristic_subagent_envelope(intent)
        spec = SubAgentSpec(
            pattern="hierarchical_subagent",
            sub_agent_envelope_json=env.model_dump_json(),
            sub_goal=intent.goal.statement,
            sub_agent_name=env.name or "subagent_0",
        )
        return HierarchicalPlan(
            pattern="hierarchical",
            parent_goal=intent.goal.statement,
            sub_agents=(spec,),
            aggregation_strategy=self._aggregation,
            metadata={
                "source": "heuristic",
                "intent_type": intent.goal.intent_type.value,
                "sub_agent_count": 1,
            },
        )

    async def execute(
        self,
        plan: Plan,
        *,
        perceiver: Any | None = None,
        actor: Any | None = None,
    ) -> PlanResult:
        """Run each sub-agent, propagate events per stream_mode, aggregate.

        ``perceiver`` / ``actor`` are accepted for protocol compliance
        but not consulted in Phase 3.C — sub-agents bring their own
        perception/action surface via their envelope. Phase 3+ may
        forward these as default subsystems for sub-agents that don't
        declare their own.
        """
        del perceiver, actor  # Phase 3+ may forward these.

        h_plan = self._coerce_plan(plan)

        # Depth check — increment first, decrement in finally.
        current = _hierarchical_depth.get()
        if current >= self._max_depth:
            raise RuntimeError(
                f"Hierarchical delegation exceeded max_depth={self._max_depth}. "
                f"Reduce nesting in the agent graph, or raise max_delegation_depth "
                f"on the HierarchicalPlanner. Alternative: flatten via "
                f"PlanExecutePlanner (sequential workstreams without recursion)."
            )
        token = _hierarchical_depth.set(current + 1)
        try:
            return await self._run_sub_agents(h_plan)
        finally:
            _hierarchical_depth.reset(token)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    async def _run_sub_agents(self, h_plan: HierarchicalPlan) -> PlanResult:
        """Execute every :class:`SubAgentSpec` in ``h_plan`` sequentially.

        Phase 3.C is sequential — :attr:`max_concurrent_subagents` is
        the hard cap (default 3), but the actual concurrency wiring
        lands in Phase 3+ when an asyncio TaskGroup harness is
        available. The aggregation contract (concat / first / json)
        is identical regardless of ordering.
        """
        # Per-sub-agent accumulators.
        sub_outputs: list[str] = []
        sub_names: list[str] = []
        total_usage = ZERO_USAGE
        all_tool_executions: list[Any] = []
        sub_agent_records: list[dict[str, Any]] = []

        # Empty plan: short-circuit to an empty result.
        if not h_plan.sub_agents:
            return PlanResult(
                text="",
                output="",
                tool_executions=(),
                usage=ZERO_USAGE,
                metadata={
                    "sub_agents_run": 0,
                    "sub_agent_records": (),
                    "aggregation_strategy": h_plan.aggregation_strategy,
                    "stream_mode": self._stream_mode,
                },
            )

        # Determine aggregation: prefer the plan's value, fall back to
        # the planner default.
        aggregation = (
            h_plan.aggregation_strategy
            if h_plan.aggregation_strategy in ("concat", "first", "json")
            else self._aggregation
        )

        for spec in h_plan.sub_agents:
            # Hydrate envelope from JSON; tolerant of dict input via
            # AgentEnvelope.model_validate_json.
            env = AgentEnvelope.model_validate_json(spec.sub_agent_envelope_json)
            sub_loop = self._agent_loop_factory(env)
            sub_name = spec.sub_agent_name or env.name or env.agent_hash()

            # Build the delegation trigger. The parent_turn_id is best-
            # effort: read from the active TurnInvocation if any.
            parent_turn_id = _current_turn_id()
            trigger = Trigger.delegation(
                goal=spec.sub_goal,
                parent_turn_id=parent_turn_id or "",
                sub_agent_hash=env.agent_hash(),
            )

            # Open a child collect_events scope so sub-agent events
            # don't pollute the parent collector. We re-publish the
            # filtered events (per stream_mode) into the parent
            # collector after the inner block exits.
            with collect_events() as child_collector:
                sub_invocation = await sub_loop.invoke(trigger=trigger)

            # Forward filtered events to the parent collector. The
            # parent collector is now the active one again because
            # collect_events used a context-manager-scoped ContextVar.
            for event in child_collector.events:
                if self._should_forward(event):
                    push_event(event)

            # Project the sub-invocation into our accumulators.
            sub_output = self._extract_output(sub_invocation)
            sub_usage = self._extract_usage(sub_invocation)
            sub_tools = self._extract_tool_executions(sub_invocation)

            sub_outputs.append(sub_output)
            sub_names.append(sub_name)
            total_usage = total_usage + sub_usage
            all_tool_executions.extend(sub_tools)
            sub_agent_records.append(
                {
                    "sub_agent_name": sub_name,
                    "sub_agent_hash": env.agent_hash(),
                    "output_chars": len(sub_output),
                    "usage": {
                        "input_tokens": sub_usage.input_tokens,
                        "output_tokens": sub_usage.output_tokens,
                        "total_tokens": sub_usage.total_tokens,
                        "cost_usd": sub_usage.cost_usd,
                    },
                }
            )

        # Aggregate sub-outputs into the parent text.
        aggregated = self._aggregate(sub_outputs, sub_names, aggregation)

        return PlanResult(
            text=aggregated,
            output=aggregated,
            tool_executions=tuple(all_tool_executions),
            usage=total_usage,
            metadata={
                "sub_agents_run": len(h_plan.sub_agents),
                "sub_agent_records": tuple(sub_agent_records),
                "aggregation_strategy": aggregation,
                "stream_mode": self._stream_mode,
            },
        )

    def _should_forward(self, event: Any) -> bool:
        """Resolved Decision #4: filter sub-agent events per stream_mode.

        ``"full"``: every event is forwarded. ``"summary-only"``: only
        :class:`TurnSummary` and ``Span(SUBAGENT, COMPLETE)`` are
        forwarded. ``"value-only"`` (default): stream-deltas are
        suppressed; value events (UsageObserved / CitationFound /
        IntentClassified / Span / TurnSummary / MemoryEvent / etc.)
        are forwarded.
        """
        if self._stream_mode == "full":
            return True
        if self._stream_mode == "summary-only":
            if isinstance(event, TurnSummary):
                return True
            return (
                isinstance(event, Span)
                and event.subject == SpanSubject.SUBAGENT
                and event.phase == SpanPhase.COMPLETE
            )
        # value-only: collapse stream-deltas, propagate value events
        if isinstance(event, _STREAM_DELTA_TYPES):
            return False
        return isinstance(event, _VALUE_EVENT_TYPES)

    @staticmethod
    def _coerce_plan(plan: Plan) -> HierarchicalPlan:
        """Coerce a base :class:`Plan` into a typed :class:`HierarchicalPlan`.

        Defensive — keeps the Protocol open to plans constructed
        through the base :class:`Plan` constructor (e.g. by tests or
        by serialisation round-trips that lose the subclass type).
        """
        if isinstance(plan, HierarchicalPlan):
            return plan
        return HierarchicalPlan.model_validate(plan.model_dump())

    @staticmethod
    def _heuristic_subagent_envelope(intent: IntentResult) -> AgentEnvelope:
        """Build a default research-style sub-agent envelope.

        Phase 3.C minimum: a CHAT-pattern sub-agent with a generic
        research instruction. Phase 3+ will swap to LLM-driven
        decomposition that produces multiple specialised envelopes
        per workstream.
        """
        from kaos_agents.config import AgentPattern

        return AgentEnvelope(
            pattern=AgentPattern.CHAT,
            instructions=(
                f"You are a research sub-agent. Answer this sub-goal: {intent.goal.statement}"
            ),
            model="anthropic:claude-haiku-4-5",
            name="research_subagent",
        )

    @staticmethod
    def _aggregate(outputs: list[str], names: list[str], strategy: str) -> str:
        """Combine sub-agent outputs per the chosen strategy."""
        if strategy == "first":
            for out in outputs:
                if out:
                    return out
            return ""
        if strategy == "json":
            payload = dict(zip(names, outputs, strict=False))
            return json.dumps(payload, separators=(",", ":"), sort_keys=True)
        # Default "concat"
        return "\n".join(out for out in outputs if out)

    @staticmethod
    def _extract_output(invocation: Any) -> str:
        """Pull the response text out of a TurnInvocation-like object."""
        out = getattr(invocation, "output", None)
        if out is None:
            return ""
        return str(out)

    @staticmethod
    def _extract_usage(invocation: Any) -> InvocationUsage:
        """Pull the :class:`InvocationUsage` off a TurnInvocation-like
        object. Falls back to :data:`ZERO_USAGE` when absent.
        """
        usage = getattr(invocation, "usage", None)
        if usage is None:
            return ZERO_USAGE
        if isinstance(usage, InvocationUsage):
            return usage
        # Duck-typed: build from .input_tokens / .output_tokens / ...
        return InvocationUsage.from_llm_usage(usage)

    @staticmethod
    def _extract_tool_executions(invocation: Any) -> tuple[Any, ...]:
        """Pull the ``tool_executions`` tuple off a TurnInvocation-like
        object. Falls back to ``()`` when absent.
        """
        tool_execs = getattr(invocation, "tool_executions", None)
        if tool_execs is None:
            return ()
        return tuple(tool_execs)


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _default_agent_loop_factory(env: AgentEnvelope) -> Any:
    """Default factory: build a Phase-2 :class:`AgentLoop` from an envelope.

    Imports are local so tests that inject their own factory don't
    need the production agent / intent extractor stack on the import
    path. Returns ``Any`` (rather than ``AgentLoop``) to keep the
    factory signature tolerant — tests typically return a
    :class:`SimpleNamespace` stub.
    """
    from kaos_agents.config import Agent
    from kaos_agents.intent import IntentExtractor
    from kaos_agents.loop import AgentLoop

    agent = Agent.from_envelope(env)
    return AgentLoop(
        intent_extractor=IntentExtractor(model=agent.model),
        agent_envelope_hash=env.agent_hash(),
    )


def _current_turn_id() -> str | None:
    """Best-effort read of the active TurnInvocation id, or ``None``.

    Used to populate ``Trigger.delegation(parent_turn_id=...)`` so the
    sub-agent's events carry the parent's correlation id. Returns
    ``None`` when no turn is active (e.g. tests calling ``execute``
    directly without :class:`AgentLoop`).
    """
    from kaos_agents.core.invocation import current_turn

    turn = current_turn()
    return turn.id if turn is not None else None


__all__ = [
    "HierarchicalPlan",
    "HierarchicalPlanner",
    "SubAgentSpec",
    "current_hierarchical_depth",
]
