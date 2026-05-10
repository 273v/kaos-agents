"""Phase 4.D — AgentLoop integration with Memory + Termination + Escalation.

Verifies that:

1. When an EscalationPolicy is configured AND intent.requires_clarification,
   AgentLoop emits an EscalationRequired event AND records it on
   invocation.escalations.
2. When a TerminationJudge is configured, AgentLoop calls its invoke()
   after planner.execute and stamps extras["termination_decision_kind"].
3. When the TerminationJudge returns DEGRADED with a partial_result,
   invocation.output is updated to the partial.
4. When the TerminationJudge says should_escalate AND a policy is
   configured, AgentLoop emits a second EscalationRequired event.
5. When KnowledgeBase + PromotionPolicy + intent.matter_client +
   findings are all present, qualifying findings are promoted to the KB.
6. None of the above fires when the corresponding subsystem is None.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from kaos_agents.config import AgentPattern
from kaos_agents.escalation import EscalationKind, EscalationPolicy
from kaos_agents.events.escalation import EscalationRequired
from kaos_agents.intent import Ambiguity, AmbiguityKind, IntentResult
from kaos_agents.intent.types import Goal
from kaos_agents.loop.agent_loop import AgentLoop
from kaos_agents.memory.institutional import KnowledgeBase
from kaos_agents.memory.promotion import PromotionPolicy
from kaos_agents.termination.types import Decision, DecisionKind
from kaos_agents.triggers import Trigger
from kaos_agents.types import IntentType


def _intent(
    *,
    pattern: AgentPattern = AgentPattern.CHAT,
    requires_clarification: bool = False,
    confidence: float = 0.9,
    ambiguities: tuple[Ambiguity, ...] = (),
    matter_client: tuple[str, str] | None = None,
) -> IntentResult:
    return IntentResult(
        goal=Goal(
            statement="test goal",
            intent_type=IntentType.RESPOND,
            matter_client=matter_client,
        ),
        constraints=(),
        ambiguities=ambiguities,
        requires_clarification=requires_clarification,
        pattern=pattern,
        confidence=confidence,
        raw_input="test",
    )


class _StubExtractor:
    def __init__(self, intent: IntentResult) -> None:
        self._intent = intent

    async def invoke(self, **kwargs: Any) -> Any:
        return SimpleNamespace(
            output=self._intent,
            usage=SimpleNamespace(input_tokens=1, output_tokens=1, total_tokens=2, cost_usd=0.0),
        )


def _stub_extractor(intent: IntentResult) -> Any:
    return _StubExtractor(intent)


class _OkPlanner:
    """Planner that returns a fixed text result."""

    def __init__(self, text: str = "planner-result") -> None:
        self._text = text

    async def plan(self, intent: Any, memory: Any = None) -> Any:
        return SimpleNamespace(pattern="stub", goal=intent.goal.statement)

    async def execute(self, plan: Any, *, perceiver: Any = None, actor: Any = None) -> Any:
        return SimpleNamespace(text=self._text, output=self._text)


class _StubTerminationJudge:
    """Returns a pre-built Decision; tracks call count + last kwargs."""

    def __init__(self, decision: Decision) -> None:
        self._decision = decision
        self.call_count = 0
        self.last_kwargs: dict[str, Any] = {}

    async def invoke(self, **kwargs: Any) -> Decision:
        self.call_count += 1
        self.last_kwargs = kwargs
        return self._decision


@pytest.mark.unit
class TestEscalationOnClarification:
    async def test_clarification_emits_escalation_when_policy_configured(self) -> None:
        ambig = Ambiguity(
            kind=AmbiguityKind.UNKNOWN_REFERENCE,
            span=(0, 5),
            excerpt="this",
            preferred_clarification="Which contract are you referring to?",
        )
        intent = _intent(
            requires_clarification=True,
            confidence=0.3,
            ambiguities=(ambig,),
        )
        loop = AgentLoop(
            intent_extractor=_stub_extractor(intent),
            escalation_policy=EscalationPolicy(),
        )
        invocation = await loop.forward(trigger=Trigger.mcp("hi", session_id="s1"))

        # The loop emitted at least one EscalationRequired event.
        esc_events = [e for e in invocation.events if isinstance(e, EscalationRequired)]
        assert len(esc_events) >= 1
        assert esc_events[0].kind == EscalationKind.CLARIFICATION_NEEDED.value
        # And recorded it on the canonical bundle.
        assert len(invocation.escalations) >= 1
        assert isinstance(invocation.escalations[0], EscalationRequired)

    async def test_clarification_without_policy_falls_back_to_span_only(self) -> None:
        intent = _intent(requires_clarification=True, confidence=0.3)
        loop = AgentLoop(intent_extractor=_stub_extractor(intent))
        invocation = await loop.forward(trigger=Trigger.mcp("hi", session_id="s1"))

        # No EscalationRequired emitted (policy is None).
        esc_events = [e for e in invocation.events if isinstance(e, EscalationRequired)]
        assert esc_events == []
        # But the legacy clarification flag still fires.
        assert invocation.extras.get("clarification_required") is True


@pytest.mark.unit
class TestTerminationJudgeIntegration:
    async def test_judge_invoked_after_planner(self) -> None:
        decision = Decision(kind=DecisionKind.COMPLETE, is_complete=True)
        judge = _StubTerminationJudge(decision)
        loop = AgentLoop(
            intent_extractor=_stub_extractor(_intent()),
            planner=_OkPlanner(text="planner-output"),
            termination_judge=judge,
        )
        invocation = await loop.forward(trigger=Trigger.mcp("hi", session_id="s1"))

        assert judge.call_count == 1
        # Judge received the planner's partial_text.
        assert judge.last_kwargs.get("partial_text") == "planner-output"
        # extras stamped with the decision kind.
        assert invocation.extras["termination_decision_kind"] == DecisionKind.COMPLETE.value

    async def test_degraded_decision_swaps_in_partial_result(self) -> None:
        partial = "this is a long-enough partial result for degradation"
        decision = Decision(
            kind=DecisionKind.DEGRADED,
            is_complete=True,
            partial_result=partial,
        )
        judge = _StubTerminationJudge(decision)
        loop = AgentLoop(
            intent_extractor=_stub_extractor(_intent()),
            planner=_OkPlanner(text="short"),
            termination_judge=judge,
        )
        invocation = await loop.forward(trigger=Trigger.mcp("hi", session_id="s1"))
        # The longer of the two wins; partial is longer than "short".
        assert invocation.output == partial

    async def test_escalating_decision_emits_event(self) -> None:
        decision = Decision(
            kind=DecisionKind.LOOP_DETECTED,
            is_complete=True,
            should_escalate=True,
            feedback="loop detected",
        )
        judge = _StubTerminationJudge(decision)
        loop = AgentLoop(
            intent_extractor=_stub_extractor(_intent()),
            planner=_OkPlanner(text="x"),
            termination_judge=judge,
            escalation_policy=EscalationPolicy(),
        )
        invocation = await loop.forward(trigger=Trigger.mcp("hi", session_id="s1"))
        esc_events = [e for e in invocation.events if isinstance(e, EscalationRequired)]
        # At least one EscalationRequired (from the judge's escalate signal).
        assert any(e.kind == EscalationKind.LOOP_DETECTED.value for e in esc_events)


@pytest.mark.unit
class TestPromotionWiring:
    async def test_findings_promoted_when_kb_and_policy_configured(self) -> None:
        kb = KnowledgeBase()
        policy = PromotionPolicy(min_confidence=0.5, require_grounding=False)
        intent = _intent(matter_client=("matter-1", "client-1"))

        # Planner that stamps findings into the invocation extras.
        class _PromotingPlanner:
            async def plan(self, intent: Any, memory: Any = None) -> Any:
                return SimpleNamespace(pattern="stub")

            async def execute(self, plan: Any, *, perceiver: Any = None, actor: Any = None) -> Any:
                # Side-effect: write findings onto the active TurnInvocation.
                from kaos_agents.core.invocation import current_turn

                turn = current_turn()
                if turn is not None:
                    turn.extras["findings"] = (
                        SimpleNamespace(
                            statement="Tesla 2023 revenue was $96.8B",
                            confidence=0.92,
                            spans=(),
                            id="finding-1",
                        ),
                    )
                return SimpleNamespace(text="planner-result", output="planner-result")

        loop = AgentLoop(
            intent_extractor=_stub_extractor(intent),
            planner=_PromotingPlanner(),
            knowledge_base=kb,
            promotion_policy=policy,
        )
        invocation = await loop.forward(trigger=Trigger.mcp("hi", session_id="s1"))

        # The finding was promoted.
        assert invocation.extras.get("promoted_findings") == 1
        # And lives in the KB.
        from kaos_agents.memory.institutional import KBQuery

        result = kb.query(KBQuery(query_text="Tesla", matter_client=("matter-1", "client-1")))
        assert len(result.entries) == 1
        assert "Tesla" in result.entries[0].statement

    async def test_no_promotion_without_matter_client(self) -> None:
        kb = KnowledgeBase()
        policy = PromotionPolicy(min_confidence=0.5, require_grounding=False)
        # No matter_client on the intent.
        loop = AgentLoop(
            intent_extractor=_stub_extractor(_intent(matter_client=None)),
            planner=_OkPlanner(),
            knowledge_base=kb,
            promotion_policy=policy,
        )
        invocation = await loop.forward(trigger=Trigger.mcp("hi", session_id="s1"))
        assert "promoted_findings" not in invocation.extras

    async def test_no_promotion_without_policy(self) -> None:
        kb = KnowledgeBase()
        intent = _intent(matter_client=("m", "c"))
        loop = AgentLoop(
            intent_extractor=_stub_extractor(intent),
            planner=_OkPlanner(),
            knowledge_base=kb,
            promotion_policy=None,
        )
        invocation = await loop.forward(trigger=Trigger.mcp("hi", session_id="s1"))
        assert "promoted_findings" not in invocation.extras
