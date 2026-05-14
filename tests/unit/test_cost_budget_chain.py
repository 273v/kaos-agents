"""Unit tests for the Wave-4 cost/budget honesty chain.

Covers four follow-ups landed together (#170-#173-ish):

- **PA11** — ``ResearchAgent`` honors ``max_cost_usd`` in the RAG path.
- **PA12** — ``TurnInvocation`` accumulates cost through the typed
  :class:`CostBudget` value type, not a bare-float ``+=``.
- **PA13** — chat CLI cost cap is checked BEFORE the next LLM call
  (loop-condition gate), bounding overshoot to whatever the
  just-finished turn produced rather than allowing a full extra turn.
- PA14 lives in ``test_drain_paths_agree.py`` (already converted
  from xfail to passing assertions after PA14's consolidation).

These are unit tests with mocked LLM helpers. Live coverage runs
in the integration tier.
"""

from __future__ import annotations

import pytest
from kaos_core.types.enums import IsolationMode, StorageBackend
from kaos_core.vfs.core import VirtualFileSystem
from kaos_core.vfs.models import VFSConfig

from kaos_agents.core.invocation import TurnInvocation
from kaos_agents.patterns.research import ResearchAgent
from kaos_agents.types.budget import UNLIMITED_BUDGET, CostBudget

# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------


def _memory_vfs() -> VirtualFileSystem:
    return VirtualFileSystem(
        config=VFSConfig(
            default_backend=StorageBackend.MEMORY,
            isolation_mode=IsolationMode.GLOBAL,
        )
    )


# --------------------------------------------------------------------------
# PA12 — CostBudget threading on TurnInvocation
# --------------------------------------------------------------------------


@pytest.mark.unit
class TestPA12TurnInvocationCostBudget:
    """``TurnInvocation`` accumulates cost through :class:`CostBudget`.

    The public ``cost_usd: float`` attribute (wire-surface contract,
    Sprint-3 #10) stays in place — we just back it with the typed
    :class:`CostBudget` so the cap+spend value type is the source of
    truth and bare-float ``+=`` is no longer the only path.
    """

    def test_default_unlimited_budget(self) -> None:
        inv = TurnInvocation()
        assert inv.cost_budget is UNLIMITED_BUDGET
        assert inv.cost_usd == 0.0
        assert inv.budget_exceeded is False

    def test_spend_accumulates_through_typed_budget(self) -> None:
        inv = TurnInvocation()
        inv.spend(0.10)
        assert inv.cost_usd == pytest.approx(0.10)
        assert inv.cost_budget.spent_usd == pytest.approx(0.10)
        inv.spend(0.05)
        assert inv.cost_usd == pytest.approx(0.15)
        assert inv.cost_budget.spent_usd == pytest.approx(0.15)

    def test_bare_float_write_mirrors_into_budget(self) -> None:
        """Legacy callers that write ``inv.cost_usd`` directly still
        produce a coherent ``CostBudget`` underneath."""
        inv = TurnInvocation()
        inv.cost_usd = 0.25
        assert inv.cost_budget.spent_usd == pytest.approx(0.25)

    def test_budget_replacement_mirrors_into_float(self) -> None:
        """Writing the typed budget updates the float mirror."""
        inv = TurnInvocation()
        inv.cost_budget = CostBudget(total_usd=1.0, spent_usd=0.42)
        assert inv.cost_usd == pytest.approx(0.42)

    def test_add_child_accumulates_through_typed_budget(self) -> None:
        """:meth:`add_child` rolls a child invocation's cost into the
        parent via :meth:`CostBudget.spend`, not via ``+=`` on a
        float. The float mirror still reflects the running total."""
        parent = TurnInvocation()
        child_a = TurnInvocation(cost_usd=0.05)
        child_b = TurnInvocation(cost_usd=0.07)
        parent.add_child(child_a)
        parent.add_child(child_b)
        assert parent.cost_usd == pytest.approx(0.12)
        assert parent.cost_budget.spent_usd == pytest.approx(0.12)
        # ``children`` tuple records both
        assert len(parent.children) == 2

    def test_typed_total_equals_float_path(self) -> None:
        """The PA12 typed path reports the SAME totals as the
        pre-PA12 bare-float path. Parametrize across a synthetic
        sequence of child costs and assert the running float total
        and the running budget total never drift."""
        children = [TurnInvocation(cost_usd=c) for c in (0.01, 0.02, 0.03, 0.04, 0.05)]
        parent = TurnInvocation()
        # Snapshot the typed and float totals after each add_child
        # call. The two MUST agree at every step.
        float_total = 0.0
        for child in children:
            parent.add_child(child)
            float_total += child.cost_usd
            assert parent.cost_usd == pytest.approx(float_total), (
                f"float drift after adding child cost={child.cost_usd}: "
                f"parent.cost_usd={parent.cost_usd}, expected={float_total}"
            )
            assert parent.cost_budget.spent_usd == pytest.approx(float_total), (
                f"budget drift after adding child cost={child.cost_usd}: "
                f"budget.spent_usd={parent.cost_budget.spent_usd}, expected={float_total}"
            )
        # Final totals
        assert parent.cost_usd == pytest.approx(0.15)
        assert parent.cost_budget.spent_usd == pytest.approx(0.15)

    def test_finite_budget_exceeded_property(self) -> None:
        """When constructed with a finite cap, ``budget_exceeded``
        flips True once spend crosses it."""
        inv = TurnInvocation(cost_budget=CostBudget(total_usd=0.10))
        assert inv.budget_exceeded is False
        inv.spend(0.05)
        assert inv.budget_exceeded is False
        inv.spend(0.05)  # now at exactly 0.10 — at-cap
        assert inv.budget_exceeded is True


# --------------------------------------------------------------------------
# PA11 — ResearchAgent honors max_cost_usd in the RAG path
# --------------------------------------------------------------------------


@pytest.mark.unit
class TestPA11ResearchAgentCostCap:
    """``ResearchAgent(max_cost_usd=...)`` is honored in the RAG path.

    The cap is checked BEFORE every ``rag.invoke()`` call (initial +
    retry). When the headroom is too small for one more estimated
    call we emit :class:`EvidenceInsufficient` with a budget reason
    and return a partial response.
    """

    def test_construction_rejects_zero_max_cost(self) -> None:
        vfs = _memory_vfs()
        with pytest.raises(ValueError, match="must be > 0"):
            ResearchAgent(vfs, max_cost_usd=0.0)

    def test_construction_rejects_negative_max_cost(self) -> None:
        vfs = _memory_vfs()
        with pytest.raises(ValueError, match="must be > 0"):
            ResearchAgent(vfs, max_cost_usd=-0.5)

    def test_construction_accepts_none_max_cost(self) -> None:
        vfs = _memory_vfs()
        agent = ResearchAgent(vfs, max_cost_usd=None)
        assert agent._max_cost_usd is None

    def test_construction_accepts_positive_max_cost(self) -> None:
        vfs = _memory_vfs()
        agent = ResearchAgent(vfs, max_cost_usd=0.05)
        assert agent._max_cost_usd == pytest.approx(0.05)

    @pytest.mark.asyncio
    async def test_pre_call_gate_refuses_initial_dispatch_when_cap_too_small(
        self,
    ) -> None:
        """A tighter-than-per-call-estimate cap refuses to even
        dispatch the first RAG call. The expected output is an
        :class:`EvidenceInsufficient` event with a budget reason,
        not silent overshoot.
        """
        from kaos_agents.events import EventEmitter, EvidenceInsufficient
        from kaos_agents.memory.session import SessionMemory

        vfs = _memory_vfs()
        # Cap is below the conservative per-call estimate ($0.01),
        # so the pre-call gate fires immediately.
        agent = ResearchAgent(vfs, max_cost_usd=0.001)
        memory = SessionMemory("test-pa11-tight-cap")
        agent.load_document(memory, "doc:test", "Some test content for retrieval.")

        emitter = EventEmitter(session_id="test", run_id="test-run")
        events = []
        async for event in agent._handle_research_streaming(
            "What is in the document?", memory, {}, emitter
        ):
            events.append(event)

        # The pre-call gate must produce an EvidenceInsufficient
        # event. No rag-query span — we refused BEFORE dispatch.
        insufficient = [e for e in events if isinstance(e, EvidenceInsufficient)]
        assert len(insufficient) == 1, (
            f"Expected exactly one EvidenceInsufficient event, got "
            f"{len(insufficient)}. Events: {[type(e).__name__ for e in events]}"
        )
        assert "cost cap" in insufficient[0].reason.lower(), (
            f"EvidenceInsufficient reason missing budget-cap signal: {insufficient[0].reason!r}"
        )

    @pytest.mark.asyncio
    async def test_uncapped_path_unchanged(self) -> None:
        """``max_cost_usd=None`` preserves the historical behavior:
        the RAG path proceeds through to ``rag.invoke()`` and the
        pre-call gate is a no-op."""
        from kaos_agents.events import EventEmitter
        from kaos_agents.memory.session import SessionMemory

        vfs = _memory_vfs()
        agent = ResearchAgent(vfs, max_cost_usd=None)
        memory = SessionMemory("test-pa11-uncapped")
        agent.load_document(memory, "doc:test", "Some test content.")

        emitter = EventEmitter(session_id="test", run_id="test-run")
        events = []
        # Wrap in try/except — without a real LLM call this will fail
        # at ``require_llm_core()`` or downstream, but the point is
        # to confirm the pre-call BUDGET gate does NOT fire.
        try:
            async for event in agent._handle_research_streaming(
                "What is in the document?", memory, {}, emitter
            ):
                events.append(event)
                # Cap the loop so we don't wait on real LLM calls if
                # they somehow proceed in the test environment.
                if len(events) > 20:
                    break
        except Exception:
            # Any LLM-availability error is acceptable here; we're
            # checking that no pre-call budget refusal was emitted.
            pass

        # The pre-call budget gate must NOT have produced an
        # EvidenceInsufficient with a "cost cap" reason.
        from kaos_agents.events import EvidenceInsufficient

        budget_refusals = [
            e
            for e in events
            if isinstance(e, EvidenceInsufficient) and "cost cap" in e.reason.lower()
        ]
        assert budget_refusals == [], (
            "Uncapped ResearchAgent emitted a budget refusal: "
            f"{[r.reason for r in budget_refusals]}"
        )


# --------------------------------------------------------------------------
# PA13 — chat CLI cost cap is checked BEFORE the next LLM call
# --------------------------------------------------------------------------


@pytest.mark.unit
class TestPA13ChatPreDispatchCostGate:
    """The chat REPL session-state predicate is read BEFORE each
    user turn dispatches, not just after each turn completes.

    The unit-test scope here is on the predicate's semantics —
    integration coverage of the full REPL loop lives in
    ``tests/integration/test_cost_cap_honesty_live.py``.
    """

    def test_session_state_budget_exceeded_predicate_at_cap(self) -> None:
        """When ``cost_usd >= max_cost_usd`` the predicate returns
        True; the REPL loop reads this BEFORE dispatching the next
        turn (PA13) so no further LLM call fires."""
        from kaos_agents.cli.chat import _SessionState

        state = _SessionState(max_cost_usd=4.0)
        # After several turns the running spend hits the cap exactly.
        state.absorb(tokens=10_000, cost=4.0)
        assert state.budget_exceeded() is True

    def test_session_state_predicate_below_cap(self) -> None:
        from kaos_agents.cli.chat import _SessionState

        state = _SessionState(max_cost_usd=4.0)
        state.absorb(tokens=10_000, cost=3.99)
        # Strictly below — not yet exceeded; the next turn can fire.
        assert state.budget_exceeded() is False

    def test_session_state_predicate_disabled_with_none_cap(self) -> None:
        from kaos_agents.cli.chat import _SessionState

        state = _SessionState(max_cost_usd=None)
        state.absorb(tokens=10_000, cost=1000.0)
        assert state.budget_exceeded() is False

    def test_session_state_predicate_disabled_with_zero_cap(self) -> None:
        """A cap of 0.0 is the documented opt-out."""
        from kaos_agents.cli.chat import _SessionState

        state = _SessionState(max_cost_usd=0.0)
        state.absorb(tokens=10_000, cost=1000.0)
        assert state.budget_exceeded() is False

    def test_pa13_no_overshoot_step_simulation(self) -> None:
        """Simulate the PA13 contract: with a cap of $4, after the
        spend crosses the cap, the predicate fires BEFORE the next
        step would dispatch. The expected accumulated cost is
        bounded by "whatever the just-finished turn produced",
        which here is $4-something, NOT $5-something.

        Mirrors the example in the task spec: "AFTER 5 steps with
        cap of 4 we have used $4-something, not $5-something".
        """
        from kaos_agents.cli.chat import _SessionState

        cap = 4.0
        per_step_cost = 1.0
        state = _SessionState(max_cost_usd=cap)
        dispatched_steps = 0
        # Loop simulates the PA13 pre-dispatch gate: check the
        # predicate BEFORE absorbing the next turn's cost.
        max_iterations = 10
        for _ in range(max_iterations):
            if state.budget_exceeded():
                break  # PA13 pre-dispatch gate fires here
            # Otherwise dispatch the turn and absorb its cost
            state.absorb(tokens=1000, cost=per_step_cost)
            dispatched_steps += 1
        # Pre-PA13 (post-turn-only gate) we'd see 5 dispatched
        # steps ($5 cost) before the gate fired. With PA13's
        # pre-dispatch gate we see 4 dispatched steps ($4 cost)
        # because after step 4 the predicate returns True.
        assert dispatched_steps == 4, (
            f"Expected 4 dispatched steps under the PA13 pre-dispatch "
            f"gate (cap=$4, $1/step), got {dispatched_steps}. Pre-PA13 "
            f"behaviour would have allowed 5 steps to dispatch."
        )
        assert state.cost_usd == pytest.approx(4.0), (
            f"Accumulated cost overshoot: expected $4 (one step short "
            f"of cap), got ${state.cost_usd:.4f}. PA13's pre-dispatch "
            f"gate must bound spend to <= cap."
        )
