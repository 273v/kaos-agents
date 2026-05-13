"""PlanExecutePlanner — paper §7.1 Plan-Execute pattern.

Produces a typed plan graph upfront from the intent, then executes
it via kaos-llm-core's :class:`~kaos_llm_core.programs.loop_runner.LoopRunner`.
The plan is static (well-defined task), unlike ReAct where reasoning
interleaves with action.

Variants:
  - ``"vanilla"`` (Phase 3.B): linear plan, sequential execution.
  - ``"rewoo"`` (Phase 3+ stub): separate reasoning from observation;
    parallel tool calling (Xu et al. 2023 ReWOO).
  - ``"compiler"`` (Phase 3+ stub): LLMCompiler-style execution graph
    optimization (Kim et al. 2024).

Phase 3.B ships ``"vanilla"`` with a heuristic plan generator that
decomposes the goal into 1-3 steps. An LLM-driven planner Call is
deferred to Phase 3+ when an optimizer harness is available to tune
it. ``"rewoo"`` and ``"compiler"`` are accepted at construction but
silently downgraded to ``"vanilla"`` (caller can introspect via the
:attr:`PlanExecutePlanner.strategy` property).

LoopRunner usage notes
----------------------

The Phase 11 plan mandates :class:`LoopRunner` as the iteration kernel.
For ``"vanilla"`` (linear, no early-stop logic beyond
``per-step skip``), the LoopRunner generic-state shape is straightforward:

* ``State`` = a mutable accumulator dataclass holding ``step_outputs``
  and ``executed`` counters.
* ``Record`` = a per-step :class:`PlanStepRecord` summary.
* ``Result`` = the final :class:`PlanResult`.

The runner fires :meth:`PlanExecutePlanner._execute_step` once per
plan step, capped at ``min(max_iterations, len(plan.steps))``. Failures
inside a step do NOT propagate by default — they are caught and
recorded as skipped steps so the plan continues, mirroring Phase 2.B's
degraded-skeleton tolerance. ReWOO/LLMCompiler variants will introduce
parallel-fan-out or DAG-aware execution; both subclass the same
:class:`PlanExecutePlan` schema.

When ``perceiver`` / ``actor`` is ``None`` for a step that needs it,
the step is skipped with a metadata note and execution proceeds.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from kaos_llm_core.programs.loop_runner import (
    LoopConfig,
    LoopRunner,
    StepOutcome,
)
from pydantic import Field

from kaos_agents.action.types import ActionPlan
from kaos_agents.intent.types import IntentResult
from kaos_agents.perception.types import PerceptionQuery, PerceptionQueryKind
from kaos_agents.planning.planner import Plan, PlanResult
from kaos_agents.types.intents import IntentType
from kaos_agents.types.usage import ZERO_USAGE


class PlanStep(Plan):
    """A single step in a :class:`PlanExecutePlan`.

    Subclassed from :class:`Plan` so it serializes consistently.
    ``kind`` discriminates how to execute this step:

    * ``"perceive"`` — route to ``perceiver`` (read-only fact-finding).
    * ``"act"`` — route to ``actor`` (mutation).
    * ``"respond"`` — produce a text result (no tool call).

    ``depends_on`` is recorded but ignored by the Phase 3.B "vanilla"
    executor; it exists for the Phase 3+ ReWOO / LLMCompiler variants.
    """

    step_id: str
    kind: str  # "perceive" | "act" | "respond"
    description: str
    inputs: dict[str, Any] = Field(default_factory=dict)
    depends_on: tuple[str, ...] = ()  # other step_ids; sequential default
    pattern: str = "plan_execute_step"


class PlanExecutePlan(Plan):
    """A multi-step plan ready for :class:`LoopRunner` execution.

    Steps in :attr:`steps` execute in declared order under the Phase
    3.B ``"vanilla"`` strategy. ``depends_on`` is recorded for future
    ReWOO / LLMCompiler variants but ignored by the vanilla executor.

    The ``goal_statement`` mirrors :attr:`IntentResult.goal.statement`
    so :meth:`PlanExecutePlanner.execute` can fall back to it when a
    "respond" step has no accumulated context to summarise.
    """

    pattern: str = "plan_execute"
    strategy: str = "vanilla"  # "vanilla" | "rewoo" | "compiler"
    goal_statement: str = ""
    steps: tuple[PlanStep, ...] = ()


@dataclass(slots=True)
class _PlanStepRecord:
    """Per-step record accumulated by :class:`LoopRunner`.

    Internal value type — never crosses the Planner boundary. Surfaces
    onto :attr:`PlanResult.metadata` after :meth:`_build_plan_result`
    aggregates the records.
    """

    step_id: str
    kind: str
    output: str = ""
    skipped: bool = False
    skip_reason: str | None = None


@dataclass(slots=True)
class _ExecutionState:
    """Mutable per-run state threaded through :class:`LoopRunner`."""

    plan: PlanExecutePlan
    perceiver: Any | None
    actor: Any | None
    step_outputs: list[str] = field(default_factory=list)
    """Concatenated string outputs from successful steps (in order)."""
    executed_count: int = 0
    skipped_count: int = 0


class PlanExecutePlanner:
    """Planner that produces a static plan graph and executes it.

    Args:
        strategy: ``"vanilla"`` (default), ``"rewoo"``, or
            ``"compiler"``. Phase 3.B only honors ``"vanilla"``; the
            others are accepted but downgraded to ``"vanilla"``.
            Caller can introspect the active strategy via the
            :attr:`strategy` property.
        max_steps: Cap on plan length (default 8). Forwarded to
            :class:`LoopConfig.max_iterations` at execute time.
        step_timeout_seconds: Per-step wall-clock cap (default 60.0).
            Phase 3.B records the field but does not enforce it; Phase
            3+ wires it into :class:`asyncio.wait_for` per step.
        planner_call: Optional kaos-llm-core Call (or any object with
            an awaitable ``invoke`` method) that produces a
            :class:`PlanExecutePlan` from the intent. Phase 3.B default
            is ``None`` (heuristic plan generator).
    """

    def __init__(
        self,
        *,
        strategy: str = "vanilla",
        max_steps: int = 8,
        step_timeout_seconds: float = 60.0,
        planner_call: Any | None = None,
    ) -> None:
        # Phase 3.B downgrades non-vanilla strategies to vanilla rather
        # than raising — caller can introspect via .strategy. ReWOO /
        # LLMCompiler land in a follow-up phase.
        self._strategy = strategy if strategy == "vanilla" else "vanilla"
        self._requested_strategy = strategy
        self._max_steps = max_steps
        self._step_timeout = step_timeout_seconds
        self._planner_call = planner_call

    @property
    def strategy(self) -> str:
        """Active execution strategy. Always ``"vanilla"`` in Phase 3.B."""
        return self._strategy

    @property
    def requested_strategy(self) -> str:
        """The strategy the caller asked for at construction.

        Distinct from :attr:`strategy` because non-vanilla requests are
        silently downgraded — telemetry / logging consumers want to
        observe the original request even after the downgrade.
        """
        return self._requested_strategy

    async def plan(
        self,
        intent: IntentResult,
        memory: Any | None = None,
    ) -> PlanExecutePlan:
        """Produce a typed :class:`PlanExecutePlan` from the intent.

        Phase 3.B uses a deterministic heuristic; when
        :attr:`_planner_call` is provided it is invoked instead and
        its ``invocation.output`` is trusted to be a
        :class:`PlanExecutePlan`.

        ``memory`` is accepted for protocol compliance but ignored in
        Phase 3.B — Phase 4 wires memory-aware plan generation
        (e.g. corpus-summary-conditioned step counts).
        """
        del memory  # Phase 4 wires memory-aware planning.

        if self._planner_call is not None:
            invocation = await self._planner_call.invoke(
                goal=intent.goal.statement,
                constraints=tuple(c.value for c in intent.constraints),
            )
            # Trust the call's output to be a PlanExecutePlan.
            return invocation.output

        steps = self._heuristic_steps(intent)
        return PlanExecutePlan(
            strategy=self._strategy,
            goal_statement=intent.goal.statement,
            steps=tuple(steps),
            metadata={
                "source": "heuristic",
                "intent_type": intent.goal.intent_type.value,
            },
        )

    async def execute(
        self,
        plan: Plan,
        *,
        perceiver: Any | None = None,
        actor: Any | None = None,
    ) -> PlanResult:
        """Execute the plan via :class:`LoopRunner`.

        Each step dispatches based on ``step.kind``:

        * ``"perceive"`` → ``perceiver.forward(query)`` (Phase 1.B).
        * ``"act"`` → ``actor.forward(plan=ActionPlan(...))`` (Phase 1.C).
        * ``"respond"`` → produce a text result from accumulated context.

        When a step needs a perceiver / actor that wasn't passed, the
        step is skipped with a metadata note rather than failing the
        whole plan (Phase 2.B's degraded-skeleton tolerance).
        """
        # Coerce a loose Plan into PlanExecutePlan-shaped access. Pydantic
        # ``extra="allow"`` lets a base Plan carry the same fields, but
        # the typed accessors are cleaner when we have the real subclass.
        pe_plan = self._coerce_plan(plan)

        # Empty-plan short-circuit: skip the LoopRunner entirely so
        # ``LoopConfig.max_iterations`` never has to be 0 (which is
        # rejected by ``LoopConfig.__post_init__``).
        if not pe_plan.steps:
            return PlanResult(
                text="",
                output="",
                tool_executions=(),
                usage=ZERO_USAGE,
                metadata={
                    "steps_executed": 0,
                    "steps_skipped": 0,
                    "strategy": self._strategy,
                    "plan_step_count": 0,
                    "step_records": (),
                },
            )

        max_iter = min(self._max_steps, len(pe_plan.steps))

        def make_state() -> _ExecutionState:
            return _ExecutionState(plan=pe_plan, perceiver=perceiver, actor=actor)

        async def step(i: int, state: _ExecutionState) -> StepOutcome[_PlanStepRecord]:
            step_def = state.plan.steps[i]
            record = await self._execute_step(step_def, state)
            # Stop after the last step we plan to run; LoopRunner would
            # also stop at max_iterations but being explicit lets ReWOO
            # / Compiler subclasses break early on a terminal record.
            stop = "completed" if (i + 1) >= max_iter else None
            return StepOutcome(record=record, stop_reason=stop)

        def build_result(state: _ExecutionState, records: list[_PlanStepRecord]) -> PlanResult:
            return self._build_plan_result(state, records)

        runner: LoopRunner[_ExecutionState, _PlanStepRecord, PlanResult] = LoopRunner(
            config=LoopConfig(max_iterations=max_iter, propagate_errors=False),
            make_state=make_state,
            step=step,
            build_result=build_result,
        )
        loop_result = await runner.run()
        return loop_result.result

    # ---- internals -------------------------------------------------

    def _heuristic_steps(self, intent: IntentResult) -> list[PlanStep]:
        """Decompose ``intent`` into 1-3 :class:`PlanStep`s.

        Rules (deterministic, no LLM call):

        * :class:`IntentType.RESPOND` → 1 ``"respond"`` step.
        * :class:`IntentType.RESEARCH` → 1 ``"perceive"`` + 1
          ``"respond"`` step.
        * :class:`IntentType.TOOL_USE` → 1 ``"act"`` + 1 ``"respond"`` step.
        * :class:`IntentType.PLAN` → 1 ``"perceive"`` + 1 ``"act"`` +
          1 ``"respond"`` step.
        * :class:`IntentType.CLARIFY` → 1 ``"respond"`` step (the
          clarification goes back to the user).
        * Any unknown / future enum → 1 ``"respond"`` step (safe
          fallback).

        Step ids are deterministic (``s1``, ``s2``, ``s3``) so plan
        replay / dedup works without UUID churn.
        """
        intent_type = intent.goal.intent_type
        statement = intent.goal.statement or "Respond to the user's request."

        if intent_type == IntentType.RESPOND:
            return [
                PlanStep(
                    pattern="plan_execute_step",
                    step_id="s1",
                    kind="respond",
                    description=f"Respond to: {statement}",
                ),
            ]
        if intent_type == IntentType.RESEARCH:
            return [
                PlanStep(
                    pattern="plan_execute_step",
                    step_id="s1",
                    kind="perceive",
                    description=f"Find facts relevant to: {statement}",
                    inputs={"query_text": statement},
                ),
                PlanStep(
                    pattern="plan_execute_step",
                    step_id="s2",
                    kind="respond",
                    description=f"Synthesize an answer to: {statement}",
                    depends_on=("s1",),
                ),
            ]
        if intent_type == IntentType.TOOL_USE:
            return [
                PlanStep(
                    pattern="plan_execute_step",
                    step_id="s1",
                    kind="act",
                    description=f"Invoke a tool to satisfy: {statement}",
                    inputs={"goal": statement},
                ),
                PlanStep(
                    pattern="plan_execute_step",
                    step_id="s2",
                    kind="respond",
                    description=f"Report tool result for: {statement}",
                    depends_on=("s1",),
                ),
            ]
        if intent_type == IntentType.PLAN:
            return [
                PlanStep(
                    pattern="plan_execute_step",
                    step_id="s1",
                    kind="perceive",
                    description=f"Gather context for: {statement}",
                    inputs={"query_text": statement},
                ),
                PlanStep(
                    pattern="plan_execute_step",
                    step_id="s2",
                    kind="act",
                    description=f"Take action to advance: {statement}",
                    inputs={"goal": statement},
                    depends_on=("s1",),
                ),
                PlanStep(
                    pattern="plan_execute_step",
                    step_id="s3",
                    kind="respond",
                    description=f"Summarise outcome of: {statement}",
                    depends_on=("s2",),
                ),
            ]
        # CLARIFY and any future enum: defer to a single respond step.
        return [
            PlanStep(
                pattern="plan_execute_step",
                step_id="s1",
                kind="respond",
                description=f"Respond to: {statement}",
            ),
        ]

    @staticmethod
    def _coerce_plan(plan: Plan) -> PlanExecutePlan:
        """Coerce a base :class:`Plan` (carrying steps via
        ``extra='allow'``) into a typed :class:`PlanExecutePlan`.

        Defensive — keeps the Protocol open to plans constructed
        through the base :class:`Plan` constructor (e.g. by tests or
        by serialisation round-trips that lose the subclass type).
        """
        if isinstance(plan, PlanExecutePlan):
            return plan
        # Re-validate as a PlanExecutePlan via pydantic. ``extra="allow"``
        # on Plan means unknown fields are stored on the instance; we
        # round-trip through model_dump so they re-bind to the subclass.
        return PlanExecutePlan.model_validate(plan.model_dump())

    async def _execute_step(self, step_def: PlanStep, state: _ExecutionState) -> _PlanStepRecord:
        """Dispatch a single :class:`PlanStep` based on ``step.kind``.

        Records skip metadata when the matching dispatcher
        (perceiver / actor) is ``None``. Returns a populated
        :class:`_PlanStepRecord` either way.
        """
        kind = step_def.kind
        if kind == "perceive":
            if state.perceiver is None:
                state.skipped_count += 1
                return _PlanStepRecord(
                    step_id=step_def.step_id,
                    kind=kind,
                    skipped=True,
                    skip_reason="no perceiver",
                )
            output = await self._dispatch_perceive(step_def, state.perceiver)
            state.executed_count += 1
            state.step_outputs.append(output)
            return _PlanStepRecord(
                step_id=step_def.step_id,
                kind=kind,
                output=output,
            )

        if kind == "act":
            if state.actor is None:
                state.skipped_count += 1
                return _PlanStepRecord(
                    step_id=step_def.step_id,
                    kind=kind,
                    skipped=True,
                    skip_reason="no actor",
                )
            output = await self._dispatch_act(step_def, state.actor)
            state.executed_count += 1
            state.step_outputs.append(output)
            return _PlanStepRecord(
                step_id=step_def.step_id,
                kind=kind,
                output=output,
            )

        if kind == "respond":
            output = self._dispatch_respond(step_def, state)
            state.executed_count += 1
            state.step_outputs.append(output)
            return _PlanStepRecord(
                step_id=step_def.step_id,
                kind=kind,
                output=output,
            )

        # Unknown kind — treat as skipped rather than crashing the
        # whole plan. Phase 3+ may upgrade this to a hard failure once
        # the kind enum is locked.
        state.skipped_count += 1
        return _PlanStepRecord(
            step_id=step_def.step_id,
            kind=kind,
            skipped=True,
            skip_reason=f"unknown kind: {kind!r}",
        )

    async def _dispatch_perceive(self, step_def: PlanStep, perceiver: Any) -> str:
        """Build a :class:`PerceptionQuery` and call ``perceiver.forward``.

        The query text falls back to the step description when
        ``inputs["query_text"]`` is missing. The result's items are
        joined into a single string output for the respond step's
        consumption; a refusal is rendered as ``"[no evidence]"``.
        """
        query_text = str(step_def.inputs.get("query_text") or step_def.description)
        query = PerceptionQuery(
            query_text=query_text,
            kind=PerceptionQueryKind.GENERAL_RECALL,
        )
        result = await perceiver.forward(query)
        if getattr(result, "refusal", None) is not None:
            return "[no evidence]"
        items = getattr(result, "items", ())
        if not items:
            return "[no items]"
        # Join item contents — keep it simple; Phase 4 will introduce
        # citation-aware rendering.
        return " ".join(str(getattr(it, "content", "")) for it in items)

    async def _dispatch_act(self, step_def: PlanStep, actor: Any) -> str:
        """Build an :class:`ActionPlan` and call ``actor.forward``.

        Phase 3.B treats the actor as a thin pass-through and lets
        the actor's own gating / refusal logic surface to the result
        text. The action plan's ``tool_name`` is taken from
        ``step.inputs["tool_name"]`` if present, otherwise the
        step description (fallback for the heuristic generator).
        """
        tool_name = str(step_def.inputs.get("tool_name") or step_def.description[:80])
        args = step_def.inputs.get("args") or {}
        if not isinstance(args, dict):
            args = {}
        action_plan = ActionPlan(
            tool_name=tool_name,
            args=args,
            rationale=step_def.description,
        )
        outcome = await actor.forward(plan=action_plan)
        # Render whatever came back as a string for the respond step.
        # ActionResult / ActionRefusal both expose a meaningful str().
        output = getattr(outcome, "output", None)
        if output is not None:
            return str(output)
        reason = getattr(outcome, "reason", None)
        if reason is not None:
            return f"[{reason}]"
        return str(outcome)

    @staticmethod
    def _dispatch_respond(step_def: PlanStep, state: _ExecutionState) -> str:
        """Compose a text response from the accumulated step outputs.

        When prior steps produced output, concatenate them; otherwise
        fall back to the plan's goal statement so an empty plan still
        yields *something* in :attr:`PlanResult.text`.
        """
        prior = "\n".join(o for o in state.step_outputs if o)
        if prior:
            return prior
        if state.plan.goal_statement:
            return state.plan.goal_statement
        return step_def.description

    def _build_plan_result(
        self, state: _ExecutionState, records: list[_PlanStepRecord]
    ) -> PlanResult:
        """Aggregate :class:`_PlanStepRecord`s into the final
        :class:`PlanResult`.

        ``text`` is the last respond-step's output (or the
        concatenated outputs when no respond step ran). Both ``text``
        and ``output`` carry the same value so the AgentLoop's
        ``getattr(..., "text") or getattr(..., "output")`` chain works
        either way.
        """
        # Pick the terminal text: the last ``respond`` step output if
        # any, else the joined non-empty outputs, else "".
        respond_records = [r for r in records if r.kind == "respond" and not r.skipped]
        if respond_records:
            text = respond_records[-1].output
        else:
            text = "\n".join(r.output for r in records if r.output and not r.skipped)

        step_metadata = tuple(
            {
                "step_id": r.step_id,
                "kind": r.kind,
                "skipped": r.skipped,
                "skip_reason": r.skip_reason,
            }
            for r in records
        )

        return PlanResult(
            text=text,
            output=text,
            tool_executions=(),  # Phase 4 wires tool-execution capture
            usage=ZERO_USAGE,  # Phase 3.B is heuristic; no LLM cost yet
            metadata={
                "steps_executed": state.executed_count,
                "steps_skipped": state.skipped_count,
                "strategy": self._strategy,
                "plan_step_count": len(state.plan.steps),
                "step_records": step_metadata,
            },
        )


__all__ = [
    "PlanExecutePlan",
    "PlanExecutePlanner",
    "PlanStep",
]
