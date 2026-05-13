"""AgentLoop — the canonical outer loop (Phase 2.B skeleton).

Replaces the :class:`kaos_agents.runtime.agent.BaseAgent` ``run`` monolith
with a :class:`~kaos_llm_core.programs.base.Program` that:

  1. Hydrates memory + classifies intent (:meth:`prepare_turn`).
  2. Dispatches to a Planner based on ``intent.pattern`` (Resolved
     decision §13 #3 — classifier picks at runtime).
  3. Captures events via the active
     :class:`~kaos_agents.events.collector.EventCollector` (Phase 0.B).
  4. Streams via a Task-backed asyncio Queue (Resolved decision §13 #8 —
     symmetric ``forward`` / ``stream`` API).
  5. Returns a :class:`~kaos_agents.core.invocation.TurnInvocation` as
     the canonical record (Phase 0.A).

Phase 2 ships AgentLoop alongside the existing ``BaseAgent`` /
``Runner`` machinery; Phase 6 cuts over.

Subsystem dependencies that **do not yet exist** in code:

* ``Planner`` ABC (Phase 3) — typed :class:`~typing.Any` here. The loop
  falls back to a "skeleton" path when ``planner is None`` (no LLM
  call, ``output=""``, ``extras["phase"]="skeleton"``).
* ``TerminationJudge`` (Phase 4) — typed :class:`~typing.Any`. When
  ``None`` the loop assumes the planner's result is terminal.
* ``EscalationPolicy`` (Phase 4) — typed :class:`~typing.Any`. When
  ``None`` the loop emits a ``Span(STEP, ERROR)`` placeholder and
  finalizes early on ``intent.requires_clarification``.
* ``DelegationRouter`` (Phase 4) — typed :class:`~typing.Any`. Phase 2.B
  does not delegate.
* ``GovernanceRecorder`` (Phase 5) — typed :class:`~typing.Any`. The
  loop still emits ``Span(TURN, START)`` / ``Span(TURN, COMPLETE)``
  without a recorder; the recorder integration is additive.
* ``KnowledgeBase`` (Phase 4) — typed :class:`~typing.Any`. Phase 2.B
  skips memory promotion.

When those subsystems land, AgentLoop's typed parameter signatures
will tighten — the constructor and :meth:`forward` are designed so
adding the real types is non-breaking for callers.

Streaming pattern (Pattern A — queue-tapping EventCollector):
    :meth:`stream` constructs a :class:`_QueueEventCollector` (subclass
    of :class:`~kaos_agents.events.collector.EventCollector` whose
    ``append`` also pushes into an :class:`asyncio.Queue`) and stashes
    it on a per-instance attribute. :meth:`forward` checks for the
    stashed collector and, when present, opens a
    :func:`~kaos_agents.events.collector.collect_events` context that
    swaps the active collector to the queue-tapping subclass — events
    flow into the queue as they happen, not just at the end. Pattern B
    (post-hoc patching of the active collector) was rejected because
    it has unclear ordering semantics across nested ``collect_events``
    scopes.
"""

from __future__ import annotations

import asyncio
import contextlib
import contextvars
from collections.abc import AsyncIterator
from typing import Any
from uuid import uuid4

from kaos_llm_core.programs.base import Program

from kaos_agents.base.event import KaosEvent
from kaos_agents.core.invocation import TurnInvocation, _active_turn_var
from kaos_agents.core.plan import TurnPlan
from kaos_agents.events.collector import (
    EventCollector,
    _active_collector_var,
)
from kaos_agents.events.emitter import EventEmitter
from kaos_agents.events.lifecycle import IntentClassified, TurnSummary, UsageObserved
from kaos_agents.events.spans import SpanSubject
from kaos_agents.hooks.base import KaosHook
from kaos_agents.intent import IntentExtractor, IntentResult
from kaos_agents.memory.session import SessionMemory
from kaos_agents.triggers.base import Trigger
from kaos_agents.types.usage import ZERO_USAGE, InvocationUsage

# ---------------------------------------------------------------------------
# Queue-tapping EventCollector (streaming Pattern A)
# ---------------------------------------------------------------------------


class _QueueEventCollector(EventCollector):
    """An :class:`EventCollector` that re-publishes every appended event
    into an :class:`asyncio.Queue` so streaming consumers see events
    live, not at the end of the turn.

    Used by :meth:`AgentLoop.stream`. The queue is a free attribute
    rather than a slotted field so we don't have to touch the slots of
    the parent dataclass.
    """

    # NB: parent is a slotted dataclass; we override append() without
    # adding new slots. The queue is stored on the instance via
    # __dict__ — Python permits this for subclasses unless we explicitly
    # slot it. We add ``__slots__ = ("_queue",)`` so the storage is
    # explicit and lookup stays cheap.
    __slots__ = ("_queue",)

    def __init__(self, queue: asyncio.Queue[KaosEvent | None]) -> None:
        super().__init__()
        self._queue = queue

    def append(self, event: KaosEvent) -> None:
        super().append(event)
        # ``put_nowait`` cannot block (queue is unbounded by default in
        # AgentLoop.stream), so this stays sync-safe inside emit().
        self._queue.put_nowait(event)


# ---------------------------------------------------------------------------
# AgentLoop
# ---------------------------------------------------------------------------


# Per-task active queue — set by stream() right before forward() is
# entered, read by forward() to decide which EventCollector class to
# open. ContextVar isolation matches _active_turn_var so concurrent
# streams in sibling tasks don't bleed events into each other.
_active_event_queue_var: contextvars.ContextVar[asyncio.Queue[KaosEvent | None] | None] = (
    contextvars.ContextVar("kaos_agents_loop_event_queue", default=None)
)


class AgentLoop(Program):
    """The canonical agent turn loop.

    Construct with whichever subsystems are available. Phase 2 supports
    only :class:`IntentExtractor` (a default is provided when ``None``
    is passed) plus optional ``perceiver`` / ``actor`` / ``memory`` /
    ``permission_policy`` / ``planner``-stub. Other params are accepted
    as typed ``Any`` forward references so the constructor surface is
    stable across Phase 3+ subsystem rollouts.

    Args:
        intent_extractor: The :class:`IntentExtractor` to use. When
            ``None`` (default), a fresh ``IntentExtractor()`` is built
            with the package's default model.
        perceiver: Phase 1 :class:`~kaos_agents.perception.Perceiver`
            (typed ``Any`` here so Phase 2 doesn't import perception).
        actor: Phase 1 :class:`~kaos_agents.action.Actor` (typed
            ``Any``).
        planner: Phase 3 ``Planner`` (typed ``Any``). When ``None`` the
            loop runs a skeleton path; when present it is dispatched
            via ``planner.plan(intent, memory)`` →
            ``planner.execute(plan_obj, perceiver=..., actor=...)``.
        memory: :class:`~kaos_agents.memory.session.SessionMemory` for
            this run. When ``None`` the loop tolerates and skips memory
            interactions.
        termination_judge: Phase 4 ``TerminationJudge`` (typed ``Any``).
        escalation_policy: Phase 4 ``EscalationPolicy`` (typed ``Any``).
        delegation_router: Phase 4 ``DelegationRouter`` (typed ``Any``).
        governance: Phase 5 ``GovernanceRecorder`` (typed ``Any``).
        permission_policy: Phase 4 ``PermissionPolicy`` (typed ``Any``).
        hooks: Tuple of :class:`KaosHook` instances; passed through as
            metadata for now (full hook dispatch lands in Phase 4).
        agent_envelope_hash: Content hash of the underlying
            :class:`~kaos_agents.core.envelope.AgentEnvelope`. Stamped
            on every :class:`TurnInvocation` for trace correlation.
        run_id_factory: Callable returning a fresh ``run_id`` string per
            turn. Defaults to a uuid4-hex factory.
    """

    def __init__(
        self,
        *,
        intent_extractor: IntentExtractor | None = None,
        perceiver: Any | None = None,
        actor: Any | None = None,
        planner: Any | None = None,
        memory: SessionMemory | None = None,
        termination_judge: Any | None = None,
        escalation_policy: Any | None = None,
        knowledge_base: Any | None = None,
        promotion_policy: Any | None = None,
        delegation_router: Any | None = None,
        governance: Any | None = None,
        permission_policy: Any | None = None,
        hooks: tuple[KaosHook, ...] = (),
        agent_envelope_hash: str = "",
        run_id_factory: Any | None = None,
        auto_select_planner: bool = True,
        default_planner_model: str = "anthropic:claude-haiku-4-5",
        tools: tuple[Any, ...] = (),
    ) -> None:
        super().__init__()
        # Use ``__dict__`` directly for non-Program children so
        # Program.__setattr__'s child-registry side-effect doesn't pull
        # them into the optimizer graph. The IntentExtractor is the one
        # exception — it's a Program subclass and SHOULD be auto-
        # registered as a child for trace collection.
        self._intent_extractor = intent_extractor or IntentExtractor()
        self._perceiver = perceiver
        self._actor = actor
        self._planner = planner
        self._memory = memory
        self._termination_judge = termination_judge
        self._escalation_policy = escalation_policy
        self._knowledge_base = knowledge_base
        self._promotion_policy = promotion_policy
        self._delegation_router = delegation_router
        self._governance = governance
        self._permission_policy = permission_policy
        self._hooks = hooks
        self._agent_envelope_hash = agent_envelope_hash
        self._run_id_factory = run_id_factory or _default_run_id
        # Phase 3.D fix: bridged kaos-llm-core Tool instances threaded
        # to auto-selected planners. Without this, an auto-selected
        # ReActPlanner is constructed with no tools and kaos-llm-core
        # warns "ReAct(tools=[]) is a degenerate construction" — the
        # turn yields zero tool calls and an empty answer (bug #3 in
        # the workflow telemetry audit). Runner._build_agent_loop is
        # responsible for calling bridge_runtime_tools and passing the
        # result through here.
        self._tools = tools
        # Phase 3.D: classifier-driven planner selection (Resolved #3).
        # When ``auto_select_planner=True`` (default) and no explicit
        # ``planner=`` was passed, AgentLoop selects a Planner based on
        # ``intent.pattern`` at dispatch time:
        #   AgentPattern.CHAT     → ReActPlanner
        #   AgentPattern.PLAN     → PlanExecutePlanner
        #   AgentPattern.RESEARCH → HierarchicalPlanner
        # Set ``auto_select_planner=False`` to preserve the Phase 2.B
        # skeleton path (used by tests that exercise the no-planner
        # invariants). Explicit ``planner=`` always wins.
        self._auto_select_planner = auto_select_planner
        self._default_planner_model = default_planner_model

    # ------------------------------------------------------------------
    # Public composition surface
    # ------------------------------------------------------------------

    async def prepare_turn(self, trigger: Trigger) -> TurnPlan:
        """Resolve the per-turn bundle (intent + memory + emitter).

        External composers (delegation, MCP wrappers, FastAPI route,
        evaluation harness) consume :class:`TurnPlan` instead of
        reaching into private state. Mirrors
        :meth:`kaos_llm_core.programs.call.Call.prepare_call` at the
        agent layer.

        Args:
            trigger: The :class:`Trigger` that opened the turn.

        Returns:
            A frozen :class:`TurnPlan` with intent classified, emitter
            constructed, and memory passed through. Subsystems that
            were not configured on construction propagate as ``None``
            on the plan — downstream steps tolerate this.
        """
        # 1. Resolve session_id from trigger.source_id (when MCP/HTTP/CLI
        #    populate it), else mint a fresh id.
        session_id = trigger.source_id or _new_session_id()
        run_id = self._run_id_factory()

        # 2. Phase 2 hydration is pass-through: accept the SessionMemory
        #    the caller configured. Phase 3 will own the hydration
        #    policy via SessionStore.
        memory = self._memory

        # 3. Build the emitter for this turn.
        emitter = EventEmitter(session_id=session_id, run_id=run_id)

        # 4. Extract the natural-language message from the trigger
        #    payload (kind-discriminated; MCP/HTTP/CLI use ``message``,
        #    DELEGATION uses ``goal``, ESCALATION uses ``reason``).
        message = self._message_from_trigger(trigger)

        # 5. Classify intent. Use the kaos-llm-core Call surface via
        #    ``IntentExtractor.invoke`` so callers get the full
        #    Invocation (trace, usage) for cost attribution.
        intent_invocation = await self._intent_extractor.invoke(
            message=message,
            recent_messages=self._recent_messages_summary(memory),
            domain_examples="",
        )
        intent: IntentResult = intent_invocation.output

        # 6. turn_number from memory if available, else 1.
        turn_number = (memory.turn_count + 1) if memory is not None else 1

        # DEFECT-5 fix: capture the intent_invocation's usage on the
        # TurnPlan so _run_8_step_turn can emit a UsageObserved inside
        # the collector scope. Without this, the IntentExtractor's
        # LLM call cost was lost on every turn (prepare_turn runs
        # before the collect_events context opens).
        intent_usage = getattr(intent_invocation, "usage", None)

        return TurnPlan(
            session_id=session_id,
            run_id=run_id,
            turn_number=turn_number,
            trigger=trigger,
            emitter=emitter,
            intent=intent,
            memory=memory,
            perceiver=self._perceiver,
            actor=self._actor,
            planner=self._planner,
            termination_judge=self._termination_judge,
            escalation_policy=self._escalation_policy,
            permission_policy=self._permission_policy,
            parent_span_id=None,
            intent_usage=intent_usage,
        )

    # ------------------------------------------------------------------
    # Runtime contract
    # ------------------------------------------------------------------

    async def forward(self, **kwargs: Any) -> TurnInvocation:
        """Run one turn end-to-end and return the canonical
        :class:`TurnInvocation`.

        The :class:`Program.forward` signature is ``(self, **kwargs)``
        per kaos-llm-core convention; we extract ``trigger=`` from
        ``kwargs`` and route all per-turn state through the
        :class:`TurnPlan` + the active TurnInvocation.

        On exception, the partial :class:`TurnInvocation` is finalized
        with the error stamp and the exception is re-raised with
        ``exc.turn_invocation`` set so callers in ``except`` blocks can
        recover the partial trace (mirrors :meth:`Program.invoke`'s
        ``exc.invocation`` convention from kaos-llm-core).
        """
        trigger = self._extract_trigger_kwarg(kwargs)
        plan = await self.prepare_turn(trigger)

        # Construct the TurnInvocation up front; mutate as the turn
        # progresses; finalize at the end.
        invocation = TurnInvocation(
            session_id=plan.session_id,
            run_id=plan.run_id,
            turn_number=plan.turn_number,
            trigger=trigger,
            intent=plan.intent,
            agent_envelope_hash=self._agent_envelope_hash,
        )
        token = _active_turn_var.set(invocation)
        try:
            collector = self._open_collector()
            ctoken = _active_collector_var.set(collector)
            try:
                await self._run_8_step_turn(plan, invocation, collector)
                invocation.events = tuple(collector.events)
                invocation.finalize(output=invocation.output)
                return invocation
            except BaseException as exc:
                # Capture whatever events we accumulated before the
                # failure so the partial invocation carries them.
                invocation.events = tuple(collector.events)
                invocation.error = exc
                # Tag the partial bundle onto the exception per the
                # rewrite plan §6 contract; some exceptions disallow
                # arbitrary attribute assignment, so guard.
                with contextlib.suppress(Exception):
                    exc.turn_invocation = invocation  # ty: ignore[unresolved-attribute]
                invocation.finalize(error=exc)
                raise
            finally:
                _active_collector_var.reset(ctoken)
        finally:
            _active_turn_var.reset(token)

    async def invoke(self, **kwargs: Any) -> Any:
        """Blocking invocation that returns a :class:`TurnInvocation`.

        :class:`Program.invoke` returns a kaos-llm-core
        :class:`~kaos_llm_core.programs._invocation.Invocation` wrapping
        the bare output. The agent-loop runtime contract is the
        :class:`TurnInvocation` itself, so we override and bridge:
        every call site that ``await``s ``loop.invoke(trigger=t)`` gets
        the canonical TurnInvocation, not the kaos-llm-core wrapper.

        Equivalent to ``await self.forward(trigger=t)``.
        """
        return await self.forward(**kwargs)

    # ------------------------------------------------------------------
    # Internal: the 8-step turn body
    # ------------------------------------------------------------------

    async def _run_8_step_turn(
        self,
        plan: TurnPlan,
        invocation: TurnInvocation,
        collector: EventCollector,
    ) -> None:
        """Execute the 8-step turn body.

        Steps that need subsystems we don't have yet (planner /
        termination / escalation / governance / KB) skip gracefully
        when those are ``None``. The Phase 2.B skeleton path produces
        ``output=""`` and stamps ``extras["phase"]="skeleton"`` so
        tests can distinguish it from a real planner result.
        """
        emitter = plan.emitter

        # Step 1 — turn-start span (governance) + IntentClassified.
        turn_span = emitter.span_start(
            SpanSubject.TURN,
            name=f"turn.{plan.turn_number}",
            attributes={
                "session_id": plan.session_id,
                "run_id": plan.run_id,
                "turn_number": plan.turn_number,
                "trigger_kind": plan.trigger.kind.value,
            },
        )
        emitter.emit(
            IntentClassified,
            intent=plan.intent.pattern.value,
            confidence=plan.intent.confidence,
        )
        # DEFECT-5 fix (May 2026): the IntentExtractor's LLM call
        # happens inside prepare_turn — BEFORE the collect_events
        # scope opens — so its usage is invisible to the active
        # collector unless we re-emit here. Bridges the same way
        # _emit_planner_usage handles ReActPlanner/PlanExecute usage.
        self._emit_planner_usage(emitter, plan.intent_usage, source_label="intent")

        # Step 2 — early-escalate on requires_clarification.
        # Phase 4.D: when EscalationPolicy is configured, route through
        # it and emit a typed EscalationRequired event. When the policy
        # is None, fall back to the Phase 2.B span-only sentinel so
        # tests that don't wire an escalation policy still see a
        # well-formed span tree.
        if plan.intent.requires_clarification:
            clarification_msg = self._clarification_message(plan.intent)
            step_span = emitter.span_start(
                SpanSubject.STEP,
                name="step.clarification_required",
                attributes={"reason": clarification_msg},
            )
            self._emit_escalation_for_clarification(
                plan=plan,
                invocation=invocation,
                emitter=emitter,
                clarification_msg=clarification_msg,
            )
            emitter.span_error(
                SpanSubject.STEP,
                span_id=step_span.span_id,
                error_type="ClarificationRequired",
                error_message=clarification_msg,
            )
            invocation.extras["phase"] = "skeleton"
            invocation.extras["clarification_required"] = True
            emitter.span_complete(
                SpanSubject.TURN,
                span_id=turn_span.span_id,
                duration_ms=0.0,
            )
            return

        # Step 3 — Plan + execute via Planner.
        # Phase 3.D: When no explicit planner is configured AND
        # auto_select_planner is True, classify a Planner from
        # intent.pattern (Resolved Decision #3). When
        # auto_select_planner is False, fall back to the Phase 2.B
        # skeleton path.
        active_planner = self._planner
        if active_planner is None and self._auto_select_planner:
            active_planner = self._select_planner_for_intent(plan.intent)
            if active_planner is not None:
                invocation.extras["selected_planner"] = type(active_planner).__name__

        plan_result_text = ""
        exec_result: Any = None
        if active_planner is not None:
            # Planner protocol (kaos_agents.planning.planner.Planner):
            #   planner.plan(intent, memory) → Plan
            #   planner.execute(plan_obj, perceiver=..., actor=...) → PlanResult
            plan_obj = await _maybe_await(active_planner.plan(plan.intent, plan.memory))
            invocation.plan = plan_obj
            exec_result = await _maybe_await(
                active_planner.execute(
                    plan_obj,
                    perceiver=plan.perceiver,
                    actor=plan.actor,
                )
            )
            plan_result_text = (
                getattr(exec_result, "text", None)
                or getattr(exec_result, "output", "")
                or str(exec_result)
            )
            # DEFECT-2 fix: planners that wrap kaos-llm-core programs
            # (ReAct/RAG/Refine) capture usage in the inner Invocation;
            # the AgentLoop's collector never sees it because the inner
            # call doesn't emit UsageObserved into the active scope.
            # Bridge: when the planner's PlanResult carries a non-zero
            # usage, emit a UsageObserved event so the Step-7 roll-up
            # picks it up.
            self._emit_planner_usage(emitter, exec_result)
            # Emit TOOL_CALL spans by walking the planner's trajectory.
            # Without this v2 had no tool-call telemetry — the v1
            # chat / research patterns walk `result.trajectory` for
            # exactly this reason. The collector's span stack threads
            # parent_span_id from the active TURN automatically.
            self._emit_tool_call_spans_from_trajectory(emitter, exec_result)
            # Emit CitationFound events for every detected citation in
            # the planner's text output. Bug #5 of the workflow audit:
            # explain records reported `citations: 0` even when answers
            # contained URLs / FR document numbers / statute cites,
            # because kaos-citations was never auto-applied to the
            # assistant's output. The helper is silent when
            # kaos-citations isn't installed (it's an optional dep).
            from kaos_agents.grounding import emit_citations_for_text

            emit_citations_for_text(emitter, plan_result_text)
        else:
            # Skeleton path: no planner, no auto-select (or unrecognised
            # intent.pattern). Used by tests; Phase 2.B legacy semantics.
            invocation.extras["phase"] = "skeleton"
            plan_result_text = ""

        invocation.output = plan_result_text

        # Step 4 — TerminationJudge (Phase 4.D wiring).
        # Calls the judge with the planner's partial output + the
        # current event collector (so the judge sees RunError /
        # EvidenceInsufficient if they fired). When the judge says
        # should_escalate, emit an EscalationRequired event. When it
        # says DEGRADED, swap in the partial_result.
        if self._termination_judge is not None and exec_result is not None:
            await self._run_termination_judge(
                plan=plan,
                invocation=invocation,
                emitter=emitter,
                collector=collector,
                exec_result=exec_result,
            )

        # Step 5 — Memory persist + Step 6 KB promotion (Phase 4.D).
        # Best-effort: if the planner produced findings on the
        # invocation extras (under the "findings" key) AND a
        # knowledge_base + promotion_policy + intent.matter_client
        # are all configured, the policy decides whether to promote.
        self._maybe_promote_findings(plan=plan, invocation=invocation)

        # Step 7 — finalize: usage roll-up + TurnSummary.
        usage = self._sum_usage_from_collector(collector)
        invocation.usage = usage
        invocation.cost_usd = usage.cost_usd

        emitter.emit(
            TurnSummary,
            text=invocation.output,
            intent=plan.intent.pattern.value,
            tool_calls=(),
            tokens_used=usage.total_tokens,
            cost_usd=usage.cost_usd,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
        )

        # Step 8 — close the TURN span. Phase 5 governance will fill in
        # real timing from the recorder; Phase 2 leaves duration_ms at
        # 0.0 because we have no monotonic-start delta on hand here
        # (the emitter's timestamp is the right place to read it).
        emitter.span_complete(
            SpanSubject.TURN,
            span_id=turn_span.span_id,
            duration_ms=0.0,
        )

    # ------------------------------------------------------------------
    # Streaming surface
    # ------------------------------------------------------------------

    async def stream(self, trigger: Trigger) -> AsyncIterator[KaosEvent]:
        """Run :meth:`forward` in a Task; yield events live from a Queue.

        Implements Resolved Decision §13 #8 — every Program has
        :meth:`invoke` (blocking, returns the full record) and
        :meth:`stream` (yields events as they happen).

        Pattern A — queue-tapping :class:`EventCollector` subclass: a
        :class:`_QueueEventCollector` is constructed, stashed on the
        ``_active_event_queue_var`` ContextVar, and then ``forward()``
        opens a ``collect_events`` scope that uses it. Each
        :meth:`EventCollector.append` call inside the turn pushes the
        event to the queue *and* the events list, so consumers see the
        same ordering the final ``invocation.events`` tuple carries.
        """
        queue: asyncio.Queue[KaosEvent | None] = asyncio.Queue()

        async def _runner() -> TurnInvocation:
            qtoken = _active_event_queue_var.set(queue)
            try:
                return await self.forward(trigger=trigger)
            finally:
                _active_event_queue_var.reset(qtoken)
                # Sentinel: signals end-of-stream to the consumer.
                await queue.put(None)

        task: asyncio.Task[TurnInvocation] = asyncio.create_task(_runner())
        try:
            while True:
                event = await queue.get()
                if event is None:
                    break
                yield event
        finally:
            # Make sure the runner is awaited so its exception (if any)
            # is observed and not swallowed as an unhandled task. The
            # caller already saw the events; the exception only matters
            # for cleanup logging.
            with contextlib.suppress(Exception):
                await task

    def _open_collector(self) -> EventCollector:
        """Return the right collector class for the current context.

        When ``stream()`` is the entry point, ``_active_event_queue_var``
        is set and we return a queue-tapping subclass. Otherwise the
        plain :class:`EventCollector` (no queue) is returned.
        """
        queue = _active_event_queue_var.get()
        if queue is not None:
            return _QueueEventCollector(queue)
        return EventCollector()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_trigger_kwarg(kwargs: dict[str, Any]) -> Trigger:
        try:
            return kwargs["trigger"]
        except KeyError as e:
            raise TypeError("AgentLoop.forward requires `trigger=` keyword argument") from e

    @staticmethod
    def _message_from_trigger(trigger: Trigger) -> str:
        """Pull the natural-language message out of a trigger payload.

        MCP / HTTP / CLI carry ``payload["message"]``; DELEGATION carries
        ``goal``; ESCALATION carries ``reason``. Other kinds may not
        carry a message at all (FILESYSTEM, SCHEDULED), in which case
        the empty string is returned and the IntentExtractor still
        runs on the empty input — Phase 4 sources will swap in
        kind-specific narration.
        """
        payload = trigger.payload or {}
        for key in ("message", "goal", "reason"):
            if payload.get(key):
                return str(payload[key])
        return ""

    @staticmethod
    def _recent_messages_summary(memory: SessionMemory | None) -> str:
        """Phase 2 stub: skip recent-messages summary.

        Phase 3 will wire this through ``SessionMemory.search`` so the
        IntentExtractor sees the conversational context. For now an
        empty string keeps the extractor's signature happy.
        """
        return ""

    # ------------------------------------------------------------------
    # Phase 4.D wiring helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _emit_planner_usage(
        emitter: EventEmitter,
        source: Any,
        *,
        source_label: str = "planner",
    ) -> None:
        """Bridge LLM-call usage into the loop's collector.

        DEFECT-2 (May 2026) found that planners wrapping kaos-llm-core
        programs (ReAct/RAG/Refine) capture usage in the
        :class:`Invocation` they return; that usage stays inside the
        inner program and the agent loop's :class:`EventCollector`
        never sees it. The fix re-emits a :class:`UsageObserved` event
        from the planner result so the loop's roll-up picks it up.

        DEFECT-5 (May 2026) extended the same problem to the
        :class:`IntentExtractor` LLM call: it runs inside
        :meth:`prepare_turn` *before* the ``collect_events`` scope is
        open, so its usage was invisible to the collector. The
        :class:`TurnPlan.intent_usage` field carries the
        IntentExtractor's usage forward so this helper can re-emit it
        at Step 1 of the turn loop.

        Args:
            emitter: The active :class:`EventEmitter` for the turn.
            source: Either a kaos-llm-core ``Invocation`` /
                :class:`PlanResult` (anything with a ``.usage`` attribute)
                OR a usage object directly (an
                :class:`InvocationUsage` /
                :class:`kaos_llm_core...TokenUsage`-shaped value with
                ``input_tokens`` / ``output_tokens`` / ``total_tokens``
                / ``cost_usd``). The helper probes ``.usage`` first;
                if absent it treats ``source`` as the usage object
                itself.
            source_label: Stamped on the emitted ``UsageObserved`` so
                downstream consumers can attribute cost back to the
                originating layer (``"planner"``, ``"intent"``).

        No-op when ``source`` is ``None`` or carries no observable
        usage — preserves the skeleton path's behaviour.
        """
        if source is None:
            return
        # Probe .usage first (PlanResult / Invocation-like). Fall back
        # to treating source as the usage object directly.
        usage = getattr(source, "usage", None)
        if usage is None:
            usage = source
        try:
            input_tokens = int(getattr(usage, "input_tokens", 0) or 0)
            output_tokens = int(getattr(usage, "output_tokens", 0) or 0)
            total_tokens = int(getattr(usage, "total_tokens", 0) or 0)
            cost_usd = float(getattr(usage, "cost_usd", 0.0) or 0.0)
        except (TypeError, ValueError):
            return
        # Skip the emission entirely when usage is the zero-value tuple —
        # avoids spurious UsageObserved noise on the skeleton path.
        if input_tokens == 0 and output_tokens == 0 and total_tokens == 0 and cost_usd == 0.0:
            return
        emitter.emit(
            UsageObserved,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            cost_usd=cost_usd,
            source=source_label,
        )

    @staticmethod
    def _clarification_message(intent: IntentResult) -> str:
        """Pick the most-blocking clarifying question from intent.ambiguities."""
        if intent.ambiguities and intent.ambiguities[0].preferred_clarification:
            return intent.ambiguities[0].preferred_clarification
        return "ambiguous request"

    def _emit_escalation_for_clarification(
        self,
        *,
        plan: TurnPlan,
        invocation: TurnInvocation,
        emitter: EventEmitter,
        clarification_msg: str,
    ) -> None:
        """Phase 4.D: emit a typed EscalationRequired when the
        EscalationPolicy says clarification is warranted.

        Falls back to a no-op when no policy is configured (the
        outer span-only sentinel still fires).
        """
        if self._escalation_policy is None:
            return
        decision = self._escalation_policy.evaluate_intent(plan.intent)
        if not getattr(decision, "escalate", False):
            return
        from kaos_agents.events.escalation import EscalationRequired

        kind_value = decision.kind.value if decision.kind is not None else "clarification_needed"
        event = emitter.emit(
            EscalationRequired,
            kind=kind_value,
            reason=decision.reason or clarification_msg,
            details=dict(decision.details or {}),
            resume_token=invocation.id,
            escalation_id=f"esc_{invocation.id[:12]}",
        )
        # Track on the invocation per Phase 0.A canonical bundle.
        invocation.escalations = (*invocation.escalations, event)

    async def _run_termination_judge(
        self,
        *,
        plan: TurnPlan,
        invocation: TurnInvocation,
        emitter: EventEmitter,
        collector: EventCollector,
        exec_result: Any,
    ) -> None:
        """Phase 4.D: call TerminationJudge with the live turn state.

        The judge sees the partial_text (planner output), the running
        usage, the iteration count (Phase 4.D = 1; replan loops are
        Phase 4+), and the events collected so far so failure-axis
        signals (RunError, EvidenceInsufficient) are observed.
        """
        if self._termination_judge is None:
            return  # caller checks this; defensive for ty narrowing
        usage = self._sum_usage_from_collector(collector)
        judge: Any = self._termination_judge
        # DEFECT-3 fix (May 2026): kaos-llm-core ``Program.invoke``
        # returns an :class:`Invocation` wrapper around the bare
        # forward() output. ``Invocation`` does not have ``.kind`` /
        # ``.partial_result`` / ``.should_escalate`` — those live on
        # the wrapped ``Decision``. Read ``invocation.output`` to
        # unwrap. Test stubs that bypass the Invocation wrapping
        # (returning a Decision directly) are still supported via
        # the duck-typed _unwrap_decision helper below.
        raw = await _maybe_await(
            judge.invoke(
                intent=plan.intent,
                usage=usage,
                events=tuple(collector.events),
                iteration=1,
                partial_text=invocation.output,
                step_signature=str(getattr(exec_result, "text", "") or ""),
            )
        )
        decision = _unwrap_decision(raw)
        kind_value = getattr(decision, "kind", "")
        # Decision.kind is a StrEnum; coerce to its .value for the
        # extras stamp so consumers see the string discriminator.
        kind_str = kind_value.value if hasattr(kind_value, "value") else str(kind_value)
        invocation.extras["termination_decision_kind"] = kind_str

        # DEGRADED: swap in the partial_result (already a substring of
        # invocation.output by construction; keep the longer of the two
        # for safety).
        partial = getattr(decision, "partial_result", None)
        if partial and (not invocation.output or len(str(partial)) >= len(invocation.output)):
            invocation.output = str(partial)

        # ESCALATE: when the judge says should_escalate AND a policy
        # is configured, emit EscalationRequired.
        if getattr(decision, "should_escalate", False) and self._escalation_policy is not None:
            from kaos_agents.events.escalation import EscalationRequired

            esc_decision = self._escalation_policy.evaluate_decision(decision)
            if esc_decision.escalate:
                kind_value = (
                    esc_decision.kind.value if esc_decision.kind is not None else "domain_specific"
                )
                event = emitter.emit(
                    EscalationRequired,
                    kind=kind_value,
                    reason=esc_decision.reason or str(getattr(decision, "feedback", "")),
                    details=dict(esc_decision.details or {}),
                    resume_token=invocation.id,
                    escalation_id=f"esc_{invocation.id[:12]}",
                )
                invocation.escalations = (*invocation.escalations, event)

    def _maybe_promote_findings(
        self,
        *,
        plan: TurnPlan,
        invocation: TurnInvocation,
    ) -> None:
        """Phase 4.D: route findings into the institutional KB.

        Findings live on ``invocation.extras["findings"]`` (a tuple of
        Cited[T]-shaped objects). When a KnowledgeBase + PromotionPolicy
        + intent.matter_client are all configured, each finding is
        evaluated by the policy and (if it qualifies) added to the KB.

        Phase 4.D no-ops when any of (kb, policy, matter_client) is
        missing. Phase 4+ may extend with summarisation / dedup.
        """
        if self._knowledge_base is None or self._promotion_policy is None or plan.intent is None:
            return
        matter_client = getattr(plan.intent.goal, "matter_client", None)
        if matter_client is None:
            return
        findings = invocation.extras.get("findings") or ()
        if not findings:
            return
        promoted_count = 0
        for finding in findings:
            outcome = self._promotion_policy.consider(
                finding,
                matter_client=matter_client,
                knowledge_base=self._knowledge_base,
            )
            if outcome.promoted:
                promoted_count += 1
        if promoted_count > 0:
            invocation.extras["promoted_findings"] = promoted_count

    @staticmethod
    def _emit_tool_call_spans_from_trajectory(
        emitter: EventEmitter,
        exec_result: Any,
    ) -> None:
        """Walk a planner's tool_executions and emit TOOL_CALL spans.

        Source-of-truth is `PlanResult.tool_executions` — the typed
        record planners populate from their inner ReAct trajectory
        (see `ReActPlanner._extract_tool_executions`). v1 patterns
        (chat / research) walk this themselves; v2's AgentLoop must
        do it here so consumers see tool-call telemetry in the
        JSONL log. Bug #3 in the workflow audit observed zero
        tool_call spans even when ReAct made real tool calls.

        Tolerant to missing fields: when `exec_result.tool_executions`
        is empty, no spans are emitted (no false-positive events).
        """
        executions = getattr(exec_result, "tool_executions", None) or ()
        if not executions:
            return

        for execution in executions:
            tool_name = str(getattr(execution, "tool_name", "") or "")
            if not tool_name:
                continue
            call_id = str(getattr(execution, "call_id", "") or tool_name)
            arguments = getattr(execution, "arguments", ()) or ()
            is_error = bool(getattr(execution, "is_error", False))
            result_summary = str(getattr(execution, "result_summary", "") or "")

            tc_span = emitter.span_start(
                SpanSubject.TOOL_CALL,
                name=f"tool.{tool_name}",
                attributes={
                    "tool_name": tool_name,
                    "call_id": call_id,
                    "arguments": tuple(arguments),
                },
            )
            emitter.span_complete(
                SpanSubject.TOOL_CALL,
                span_id=tc_span.span_id,
                name=f"tool.{tool_name}",
                attributes={
                    "tool_name": tool_name,
                    "call_id": call_id,
                    "result_summary": result_summary,
                    "is_error": is_error,
                },
            )

    def _select_planner_for_intent(self, intent: IntentResult) -> Any | None:
        """Classifier-driven Planner selection (Phase 3.D, Resolved #3).

        Maps :class:`AgentPattern` → concrete Planner instance using
        the three Phase-3 planners. Lazy-imports the planner classes
        so AgentLoop's import graph stays light when callers pass an
        explicit ``planner=`` and never hit this branch.

        Returns ``None`` when the pattern is unrecognised, leaving the
        skeleton path active.

        Subclasses may override to inject custom planners or honor an
        envelope's ``planner_overrides`` (Phase 4+ wiring).
        """
        from kaos_agents.config import AgentPattern
        from kaos_agents.planning.hierarchical_planner import HierarchicalPlanner
        from kaos_agents.planning.plan_execute_planner import PlanExecutePlanner
        from kaos_agents.planning.react_planner import ReActPlanner

        pattern = intent.pattern
        # Phase 5.D: forward the AgentLoop's KaosHook tuple into the
        # auto-selected planner so the inner kaos-llm-core program
        # (ReAct/RAG/Refine) sees the same hook tree the agent layer
        # does. A single OTelHook on the Runner now observes both
        # CallHooks/ProgramHooks events from the inner Programs AND
        # the agent's outer Span / IntentClassified / TurnSummary
        # events without manual wiring.
        if pattern == AgentPattern.CHAT:
            return ReActPlanner(
                model=self._default_planner_model,
                hooks=self._hooks,
                tools=self._tools or None,
            )
        if pattern == AgentPattern.PLAN:
            return PlanExecutePlanner()
        if pattern == AgentPattern.RESEARCH:
            # Default: one heuristic sub-agent. Phase 4+ wires a real
            # retrieval-sub-agent envelope (paper §4.2 / §6 RAG path).
            # Tools propagate via the sub-agent's own AgentLoop
            # construction inside HierarchicalPlanner._default_agent_loop_factory,
            # which receives the same tool tuple.
            return HierarchicalPlanner(tools=self._tools or None)
        return None

    @staticmethod
    def _sum_usage_from_collector(collector: EventCollector) -> InvocationUsage:
        """Roll up every :class:`UsageObserved` event in the collector.

        Phase 2 doesn't emit UsageObserved itself (no LLM calls in the
        skeleton path), but a stub planner that emits them via the
        active emitter will be picked up by this rollup, and Phase 3+
        planners will populate the field naturally.
        """
        total = ZERO_USAGE
        for event in collector.events:
            if isinstance(event, UsageObserved):
                total = total + InvocationUsage(
                    input_tokens=event.input_tokens,
                    output_tokens=event.output_tokens,
                    total_tokens=event.total_tokens,
                    cost_usd=event.cost_usd,
                )
        return total


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _unwrap_decision(value: Any) -> Any:
    """Unwrap a kaos-llm-core ``Invocation`` to its Decision payload.

    DEFECT-3 (May 2026): :class:`TerminationJudge.invoke` returns an
    :class:`Invocation` whose ``.output`` is the actual
    :class:`Decision`. Reading ``Invocation.kind`` returns ``None``;
    ``invocation.output.kind`` is the real value.

    The helper is defensive against test stubs that return a Decision
    directly (no ``.output`` attribute on a Decision pydantic model
    that *would* have one only because pydantic's free-form model
    config might allow extras). The probe order is:

      1. If ``value`` has a ``.kind`` attribute and is NOT an
         :class:`Invocation`-shaped object (which would have ``.output``
         and would expose ``.kind`` only via ``__getattr__`` if a
         :class:`pydantic.BaseModel` allows extras), return ``value``.
      2. Else if ``value.output`` is non-None and has ``.kind``, return
         ``value.output``.
      3. Else return ``value`` (best-effort; downstream getattrs
         tolerate missing attributes).
    """
    if value is None:
        return value
    # Probe by precedence: Invocation has ``.output``; Decision does not.
    output = getattr(value, "output", None)
    if output is not None and hasattr(output, "kind"):
        return output
    if hasattr(value, "kind"):
        return value
    if output is not None:
        return output
    return value


async def _maybe_await(value: Any) -> Any:
    """Await an awaitable; pass through anything else unchanged.

    Lets the planner protocol be flexible — sync ``plan()`` /
    ``execute()`` and async variants are both accepted by the loop.
    """
    if hasattr(value, "__await__"):
        return await value
    return value


def _default_run_id() -> str:
    return f"run_{uuid4().hex[:12]}"


def _new_session_id() -> str:
    return f"session_{uuid4().hex[:12]}"


__all__ = ["AgentLoop"]
