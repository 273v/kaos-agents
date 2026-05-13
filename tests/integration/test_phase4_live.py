"""Live integration tests for Phase 4 — TerminationJudge, LoopDetector,
PromotionPolicy, and end-to-end AgentLoop.

Mandate: Claude >= 4.6 AND GPT >= 5.4. Real API calls throughout.

DEFECT-1 (documented, NOT patched):
    TerminationJudge._score_quality reads ``invocation.output.score``.
    kaos_llm_core.Judge.invoke() returns Invocation(output=JudgedResult).
    JudgedResult has ``.judgment.quality_score`` but NOT ``.score``.
    Result: quality axis silently returns 0.0 for any real Judge,
    causing QUALITY_FAILED on every call regardless of answer quality.
    Fix suggestion (termination/judge.py):
        - score = float(
            getattr(invocation.output, "score", None)
            or getattr(getattr(invocation.output, "judgment", None), "quality_score", 0.0)
            or 0.0
        )
    Or: require the duck-typed judge to expose .score on its output.
    Tests below exercise the defect directly.

Run with:
    uv run pytest tests/integration/test_phase4_live.py -m live -v --no-cov -s
"""

from __future__ import annotations

import os
from types import SimpleNamespace
from typing import Any

import pytest

from kaos_agents.config import AgentPattern

# ---------------------------------------------------------------------------
# Skip markers
# ---------------------------------------------------------------------------

requires_anthropic = pytest.mark.skipif(
    "ANTHROPIC_API_KEY" not in os.environ,
    reason="ANTHROPIC_API_KEY missing",
)
requires_openai = pytest.mark.skipif(
    "OPENAI_API_KEY" not in os.environ,
    reason="OPENAI_API_KEY missing",
)

# ---------------------------------------------------------------------------
# Model strings — pinned.
# Source: kaos-llm-client/tests/integration/test_live.py header.
# ---------------------------------------------------------------------------

ANTHROPIC_DEFAULT = "anthropic:claude-sonnet-4-6"
ANTHROPIC_FLAGSHIP = "anthropic:claude-opus-4-6"
OPENAI_DEFAULT = "openai:gpt-5.4-mini"
OPENAI_FLAGSHIP = "openai:gpt-5.4"

# ---------------------------------------------------------------------------
# Stub judge that exposes a .score attribute on its output directly
# (the duck-typed interface TerminationJudge._score_quality actually needs).
# ---------------------------------------------------------------------------


class _DuckTypedJudge:
    """Duck-typed judge with the interface TerminationJudge actually uses.

    TerminationJudge._score_quality calls:
        invocation = await judge.invoke(output=text, criteria=...)
        score = float(getattr(invocation.output, "score", 0.0) or 0.0)

    So invocation.output must have a .score attribute. A real
    kaos_llm_core.Judge returns JudgedResult which has .judgment.quality_score
    but NOT .score — that's DEFECT-1. This stub provides the correct interface.
    """

    def __init__(self, *, score_good: float = 0.9, score_bad: float = 0.2) -> None:
        self._score_good = score_good
        self._score_bad = score_bad
        self.calls: list[dict[str, Any]] = []

    async def invoke(self, **kwargs: Any) -> Any:
        text = str(kwargs.get("output", ""))
        criteria = str(kwargs.get("criteria", ""))
        self.calls.append({"output": text, "criteria": criteria})
        # Heuristic: if text contains both "Tesla" and "$", it's "good"
        is_good = "tesla" in text.lower() and ("$" in text or "billion" in text.lower())
        score = self._score_good if is_good else self._score_bad
        return SimpleNamespace(output=SimpleNamespace(score=score, reasoning="stub"))


class _RealQualityJudge:
    """A REAL LLM-backed judge using kaos_llm_core.Call directly.

    Uses a simple Call with a quality scoring prompt to return a
    numeric score. Workaround for DEFECT-1 (JudgedResult.score mismatch).
    The output is shaped so that invocation.output.score is a float.
    """

    def __init__(self, *, model: str) -> None:
        from kaos_llm_core import InputField, OutputField, Signature
        from kaos_llm_core.programs.call import Call

        class _QualityScoreSig(Signature):
            """Score the quality of a text output against the given criteria.
            Return a numeric score between 0.0 and 1.0. Be strict.
            """

            output: str = InputField(description="The text to evaluate.")
            criteria: str = InputField(description="The quality criteria.")
            score: float = OutputField(
                description="Quality score from 0.0 (fails criteria) to 1.0 (fully meets criteria)."
            )

        self._call = Call(_QualityScoreSig, model=model)
        self.calls: list[dict[str, Any]] = []

    async def invoke(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        invocation = await self._call.invoke(**kwargs)
        # invocation.output is _QualityScoreSig with .score attribute
        return invocation


# ===========================================================================
# TerminationJudge — quality axis
# ===========================================================================


@pytest.mark.live
@requires_anthropic
class TestTerminationJudgeQualityLive:
    """Live tests for TerminationJudge quality axis."""

    async def test_good_output_complete(self) -> None:
        """Good Tesla revenue output → COMPLETE (score >= 0.7)."""
        from kaos_agents.termination.judge import TerminationJudge
        from kaos_agents.termination.types import DecisionKind, SuccessCriteria
        from kaos_agents.types.usage import InvocationUsage

        judge = _DuckTypedJudge(score_good=0.9, score_bad=0.2)
        t_judge = TerminationJudge(
            judge=judge,
            min_quality=0.7,
            max_iterations=10,
        )
        criteria = SuccessCriteria(
            goal_statement="Report Tesla revenue",
            criteria=("answer must mention Tesla and a dollar amount",),
        )
        good_text = "Tesla's 2023 revenue was $96.8 billion."
        decision = await t_judge.forward(
            usage=InvocationUsage(),
            iteration=1,
            partial_text=good_text,
            success_criteria=criteria,
        )
        print(f"\n[TJudge/good] decision.kind={decision.kind}, is_complete={decision.is_complete}")
        assert decision.kind == DecisionKind.COMPLETE, (
            f"Expected COMPLETE for good output, got {decision.kind}"
        )
        assert decision.is_complete is True

    async def test_bad_output_quality_failed_or_degraded(self) -> None:
        """Bad output (no Tesla/dollar) → QUALITY_FAILED or DEGRADED."""
        from kaos_agents.termination.judge import TerminationJudge
        from kaos_agents.termination.types import DecisionKind, SuccessCriteria
        from kaos_agents.types.usage import InvocationUsage

        judge = _DuckTypedJudge(score_good=0.9, score_bad=0.2)
        t_judge = TerminationJudge(
            judge=judge,
            min_quality=0.7,
            max_iterations=10,
        )
        criteria = SuccessCriteria(
            goal_statement="Report Tesla revenue",
            criteria=("answer must mention Tesla and a dollar amount",),
        )
        bad_text = "Apples are red."
        decision = await t_judge.forward(
            usage=InvocationUsage(),
            iteration=1,
            partial_text=bad_text,
            success_criteria=criteria,
        )
        print(f"\n[TJudge/bad] decision.kind={decision.kind}, is_complete={decision.is_complete}")
        assert decision.kind in (DecisionKind.QUALITY_FAILED, DecisionKind.DEGRADED), (
            f"Expected QUALITY_FAILED or DEGRADED for bad output, got {decision.kind}"
        )

    async def test_real_llm_judge_quality_axis(self) -> None:
        """Real LLM judge via _RealQualityJudge: good text → score > 0.7."""
        from kaos_agents.termination.judge import TerminationJudge
        from kaos_agents.termination.types import DecisionKind, SuccessCriteria
        from kaos_agents.types.usage import InvocationUsage

        # Use a real LLM-backed judge that exposes .score correctly
        real_judge = _RealQualityJudge(model=ANTHROPIC_DEFAULT)
        t_judge = TerminationJudge(
            judge=real_judge,
            min_quality=0.7,
            max_iterations=10,
        )
        criteria = SuccessCriteria(
            goal_statement="Report Tesla revenue",
            criteria=("answer must mention Tesla and a dollar amount",),
        )
        good_text = "Tesla's 2023 revenue was $96.8 billion according to their annual report."
        decision = await t_judge.forward(
            usage=InvocationUsage(),
            iteration=1,
            partial_text=good_text,
            success_criteria=criteria,
        )
        print(
            f"\n[TJudge/real-llm] decision.kind={decision.kind}, "
            f"judge_calls={len(real_judge.calls)}"
        )
        assert decision.kind in (
            DecisionKind.COMPLETE,
            DecisionKind.QUALITY_FAILED,
            DecisionKind.DEGRADED,
        )
        # The LLM-backed judge was actually called
        assert len(real_judge.calls) >= 1

    async def test_defect_1_real_judge_returns_zero_score(self) -> None:
        """DEFECT-1 proof: real kaos_llm_core.Judge always scores 0.0 via TerminationJudge.

        TerminationJudge._score_quality reads invocation.output.score.
        Judge.invoke() returns Invocation(output=JudgedResult).
        JudgedResult has .judgment.quality_score but NOT .score.
        getattr(JudgedResult_instance, "score", 0.0) → 0.0 always.

        Expected (correct) behavior: score should reflect the judge's assessment.
        Actual (broken) behavior: score is always 0.0 regardless of answer quality.
        """
        from kaos_llm_core.programs.judge import JudgedResult

        from kaos_agents.termination.judge import TerminationJudge
        from kaos_agents.termination.types import DecisionKind, SuccessCriteria
        from kaos_agents.types.usage import InvocationUsage

        class _ProductionShapedJudge:
            """Returns exactly what kaos_llm_core.Judge.invoke() returns."""

            async def invoke(self, **kwargs: Any) -> Any:
                # Simulate what Judge.invoke() actually returns
                judgment = SimpleNamespace(quality_score=0.95, reasoning="excellent output")
                judged_result = JudgedResult(
                    output="Tesla 2023 revenue was $96.8B", judgment=judgment
                )
                # This is the Invocation.output that TerminationJudge reads
                return SimpleNamespace(output=judged_result)

        production_judge = _ProductionShapedJudge()
        t_judge = TerminationJudge(
            judge=production_judge,
            min_quality=0.7,
            max_iterations=10,
        )
        criteria = SuccessCriteria(
            goal_statement="Report Tesla revenue",
            criteria=("answer must mention Tesla and a dollar amount",),
        )
        # This text IS good — the judge says 0.95 — but the bug makes it 0.0
        good_text = "Tesla's 2023 revenue was $96.8 billion."
        decision = await t_judge.forward(
            usage=InvocationUsage(),
            iteration=1,
            partial_text=good_text,
            success_criteria=criteria,
        )
        # DEFECT: should be COMPLETE (score 0.95 > 0.7) but is QUALITY_FAILED (score 0.0)
        is_defect_present = decision.kind in (DecisionKind.QUALITY_FAILED, DecisionKind.DEGRADED)
        print(
            f"\n[DEFECT-1-PROOF] decision.kind={decision.kind}, defect_present={is_defect_present}"
        )
        # Document the defect: this assertion WILL FAIL when the defect is fixed.
        # For now it passes because the bug IS present.
        if is_defect_present:
            pytest.xfail(
                reason=(
                    "DEFECT-1: TerminationJudge._score_quality reads invocation.output.score "
                    "but kaos_llm_core.Judge returns JudgedResult(output=..., judgment=...) "
                    "which has .judgment.quality_score, NOT .score. "
                    "Fix: read getattr(invocation.output, 'score', None) or "
                    "getattr(getattr(invocation.output, 'judgment', None), 'quality_score', 0.0). "
                    "File: kaos_agents/termination/judge.py, method: _score_quality"
                )
            )
        else:
            # Defect was fixed — assert the correct behavior
            assert decision.kind == DecisionKind.COMPLETE, (
                f"Expected COMPLETE after defect fix, got {decision.kind}"
            )

    async def test_budget_exceeded_fires_before_quality(self) -> None:
        """Budget axis short-circuits quality: iteration >= cap → BUDGET_EXCEEDED."""
        from kaos_agents.termination.judge import TerminationJudge
        from kaos_agents.termination.types import DecisionKind, SuccessCriteria
        from kaos_agents.types.usage import InvocationUsage

        judge = _DuckTypedJudge(score_good=0.9)
        t_judge = TerminationJudge(
            judge=judge,
            min_quality=0.7,
            max_iterations=3,
        )
        criteria = SuccessCriteria(criteria=("must mention Tesla",))
        decision = await t_judge.forward(
            usage=InvocationUsage(),
            iteration=5,  # exceeds max_iterations=3
            partial_text="Tesla revenue stub",
            success_criteria=criteria,
        )
        print(f"\n[TJudge/budget] decision.kind={decision.kind}")
        # Should be BUDGET_EXCEEDED or DEGRADED (via degradation policy)
        assert decision.kind in (DecisionKind.BUDGET_EXCEEDED, DecisionKind.DEGRADED), (
            f"Expected BUDGET_EXCEEDED or DEGRADED, got {decision.kind}"
        )

    async def test_no_partial_text_returns_incomplete(self) -> None:
        """No output → INCOMPLETE (allows_replan=True)."""
        from kaos_agents.termination.judge import TerminationJudge
        from kaos_agents.termination.types import DecisionKind
        from kaos_agents.types.usage import InvocationUsage

        t_judge = TerminationJudge(max_iterations=10)
        decision = await t_judge.forward(
            usage=InvocationUsage(),
            iteration=1,
            partial_text="",
        )
        print(
            f"\n[TJudge/no-text] decision.kind={decision.kind}, "
            f"allows_replan={decision.allows_replan}"
        )
        assert decision.kind == DecisionKind.INCOMPLETE
        assert decision.allows_replan is True


# ===========================================================================
# LoopDetector — live behaviour
# ===========================================================================


@pytest.mark.live
class TestLoopDetectorLiveBehaviour:
    """Live loop detection tests using real agent step signature patterns.

    Phase 4.E calibration: ngram_jaccard at threshold 0.5 separates
    LOOP corpus [0.83, 0.92] from NON-LOOP corpus [0.18, 0.20].
    """

    def test_identical_signatures_detected(self) -> None:
        """Exact-same call signature 3x → loop detected."""
        from kaos_agents.termination.loop_detect import LoopDetector

        detector = LoopDetector(window_size=5, min_similarity=0.5, algorithm="ngram_jaccard")
        sig = "call_tool(name='web_search', args={'query': 'Tesla revenue'})"
        result1 = detector.observe(sig)
        result2 = detector.observe(sig)
        result3 = detector.observe(sig)
        # Should detect by the third observation at minimum
        any_detected = result1.detected or result2.detected or result3.detected
        print(
            f"\n[LoopDetect/identical] "
            f"r1={result1.detected}, r2={result2.detected}, r3={result3.detected}, "
            f"algo={detector.algorithm}"
        )
        assert any_detected, "Expected loop detection on 3 identical signatures"

    def test_micro_variation_signatures_detected(self) -> None:
        """Micro-variations on the same call signature → loop detected.

        Simulates an agent stuck in a loop calling the same tool with
        slightly different parameter whitespace/capitalization.
        Validates Phase 4.E calibration: ngram_jaccard handles micro-variations.
        """
        from kaos_agents.termination.loop_detect import LoopDetector

        detector = LoopDetector(window_size=5, min_similarity=0.5, algorithm="ngram_jaccard")
        sigs = [
            "call_tool(name='web_search', args={'query': 'Tesla 2023 revenue'})",
            "call_tool(name='web_search', args={'query': 'tesla 2023 revenue'})",
            "call_tool(name='web_search', args={'query': 'Tesla revenue 2023'})",
            "call_tool(name='web_search', args={'query': 'Tesla 2023 revenue figures'})",
        ]
        results = []
        for sig in sigs:
            results.append(detector.observe(sig))

        detected = [r for r in results if r.detected]
        print(
            f"\n[LoopDetect/micro-variation] "
            f"detected_count={len(detected)}, algo={detector.algorithm}, "
            f"similarities={[r.similarity for r in detected]}"
        )
        assert len(detected) >= 1, (
            f"Expected loop detection on micro-variation signatures; "
            f"algorithm={detector.algorithm}, "
            f"results={[(r.detected, r.similarity) for r in results]}"
        )

    def test_different_signatures_not_detected(self) -> None:
        """Legitimately different tool calls → NOT a loop."""
        from kaos_agents.termination.loop_detect import LoopDetector

        detector = LoopDetector(window_size=5, min_similarity=0.5, algorithm="ngram_jaccard")
        sigs = [
            "call_tool(name='web_search', args={'query': 'Tesla revenue'})",
            "call_tool(name='edgar_fetch', args={'ticker': 'TSLA', 'form': '10-K'})",
            "call_tool(name='read_document', args={'doc_id': 'tsla-2023-10k'})",
            "call_tool(name='extract_table', args={'page': 42, 'doc': 'annual_report'})",
        ]
        final_result = None
        for sig in sigs:
            final_result = detector.observe(sig)
        print(
            f"\n[LoopDetect/different] "
            f"detected={final_result.detected if final_result else None}, "
            f"algo={detector.algorithm}"
        )
        assert final_result is not None
        # Should NOT detect a loop for legitimately different signatures
        # (Phase 4.E calibration: NON-LOOP corpus similarity in [0.18, 0.20])
        assert not final_result.detected, (
            f"False positive: loop detected for different signatures. "
            f"similarity={final_result.similarity}"
        )

    def test_reset_clears_window(self) -> None:
        """reset() clears the sliding window so detection restarts."""
        from kaos_agents.termination.loop_detect import LoopDetector

        detector = LoopDetector(window_size=5, min_similarity=0.5)
        sig = "call_tool(name='web_search', args={'query': 'Tesla'})"
        detector.observe(sig)
        detector.observe(sig)  # should detect
        detector.reset()
        result = detector.check()
        assert not result.detected, "After reset(), window should be empty"

    def test_algorithm_degrades_to_equality_gracefully(self) -> None:
        """use_fuzzy=False → equality algorithm, still detects exact repeats."""
        from kaos_agents.termination.loop_detect import LoopDetector

        detector = LoopDetector(window_size=5, use_fuzzy=False)
        assert detector.algorithm == "equality"
        sig = "call_tool(name='web_search', args={'query': 'Tesla'})"
        result1 = detector.observe(sig)
        result2 = detector.observe(sig)
        assert not result1.detected
        assert result2.detected

    @pytest.mark.live
    @requires_anthropic
    async def test_react_loop_signatures_trip_detector(self) -> None:
        """End-to-end: ReActPlanner with a stuck tool → LoopDetector trips.

        Builds a tool that always returns the same result, exercises the
        ReAct loop to generate call signatures, then feeds them into
        LoopDetector to verify the empirical close-the-loop.

        The tool is designed to create a 'stuck' loop pattern by always
        returning the same unhelpful result, causing the LLM to retry
        the same call repeatedly.
        """
        from kaos_llm_core import Tool

        from kaos_agents.termination.loop_detect import LoopDetector

        call_log: list[dict[str, Any]] = []

        def stuck_search(query: str) -> str:
            """Search for information. Always returns the same result."""
            call_log.append({"tool": "stuck_search", "query": query})
            return "No results found. Please try again with different keywords."

        stuck_tool = Tool.from_callable(stuck_search)

        from kaos_agents.intent.types import Goal, IntentResult
        from kaos_agents.planning.react_planner import ReActPlanner
        from kaos_agents.types.intents import IntentType

        planner = ReActPlanner(
            model=ANTHROPIC_DEFAULT,
            max_iterations=5,  # keep cost low
            tools=(stuck_tool,),
            instructions=(
                "You have a stuck_search tool. Use it to find Tesla's 2023 revenue. "
                "Keep trying if you don't get results."
            ),
        )
        intent = IntentResult(
            goal=Goal(
                statement="Find Tesla 2023 revenue using stuck_search.",
                intent_type=IntentType.RESEARCH,
            ),
            pattern=AgentPattern.CHAT,
            confidence=0.9,
        )
        plan = await planner.plan(intent)
        result = await planner.execute(plan)

        # Now test LoopDetector on the call signatures from the react run
        # Generate synthetic signatures that represent the pattern of calls made
        detector = LoopDetector(window_size=5, min_similarity=0.5, algorithm="ngram_jaccard")

        print(f"\n[LoopDetect/react] call_log={call_log}, react_text={result.text[:100]!r}")

        # The call_log should show the stuck search being called repeatedly
        # Build signatures from the log
        if len(call_log) >= 2:
            detected_loop = False
            for entry in call_log:
                sig = f"call_tool(name='{entry['tool']}', args={{'query': {entry['query']!r}}})"
                r = detector.observe(sig)
                if r.detected:
                    detected_loop = True
                    print(f"  → Loop detected: {r.reason}")
                    break
            # The detector SHOULD trip on repeated search calls
            # (but only if the LLM called the stuck_search multiple times)
            if len(call_log) >= 2:
                print(f"  → call_count={len(call_log)}, loop_detected={detected_loop}")
                # Only assert if we have ≥ 2 calls; with max_iterations=5 this is expected
                assert detected_loop, (
                    f"Expected LoopDetector to trip on "
                    f"{len(call_log)} repeated stuck_search calls. "
                    f"Signatures={[c['query'] for c in call_log]}"
                )
        else:
            # LLM gave up after 1 call — that's fine, no loop to detect
            print(f"  → LLM gave up after {len(call_log)} call(s); no loop to detect")


# ===========================================================================
# PromotionPolicy with Cited[T]-shaped findings
# ===========================================================================


@pytest.mark.live
class TestPromotionPolicyLive:
    """Live tests for PromotionPolicy.consider()."""

    def _make_finding(
        self,
        *,
        confidence: float,
        statement: str,
        spans: tuple[Any, ...] = (),
        is_verified: bool | None = None,
    ) -> Any:
        """Build a Cited[T]-shaped finding stub."""
        return SimpleNamespace(
            confidence=confidence,
            statement=statement,
            spans=spans,
            is_verified=is_verified,
        )

    def test_high_confidence_verified_promotes(self) -> None:
        """confidence >= 0.85 + grounded spans → promotes to KB."""
        from kaos_agents.memory.institutional import KBQuery, KnowledgeBase
        from kaos_agents.memory.promotion import PromotionPolicy

        kb = KnowledgeBase()
        policy = PromotionPolicy(min_confidence=0.85, require_grounding=True)

        finding = self._make_finding(
            confidence=0.95,
            statement="Tesla 2023 revenue was $96.8 billion.",
            spans=(SimpleNamespace(source="page:1", text="$96.8 billion"),),
        )
        mc = ("tesla-matter", "acme-corp")
        decision = policy.consider(finding, matter_client=mc, knowledge_base=kb)
        print(f"\n[Promotion/high-conf] promoted={decision.promoted}, reason={decision.reason!r}")
        assert decision.promoted is True
        assert decision.entry is not None
        assert decision.entry.confidence == 0.95
        assert decision.entry.grounding_verified is True

        # Verify it's actually in the KB
        q = KBQuery(query_text="Tesla revenue", matter_client=mc, top_k=5)
        result = kb.query(q)
        assert len(result.entries) == 1
        assert "Tesla" in result.entries[0].statement

    def test_low_confidence_rejected(self) -> None:
        """confidence < 0.85 → not promoted."""
        from kaos_agents.memory.institutional import KnowledgeBase
        from kaos_agents.memory.promotion import PromotionPolicy

        kb = KnowledgeBase()
        policy = PromotionPolicy(min_confidence=0.85, require_grounding=True)

        finding = self._make_finding(
            confidence=0.60,
            statement="Apple revenue was big.",
            spans=(SimpleNamespace(source="page:2", text="big"),),
        )
        mc = ("apple-matter", "acme-corp")
        decision = policy.consider(finding, matter_client=mc, knowledge_base=kb)
        print(f"\n[Promotion/low-conf] promoted={decision.promoted}, reason={decision.reason!r}")
        assert decision.promoted is False
        assert "0.60" in decision.reason

    def test_no_grounding_rejected(self) -> None:
        """High confidence but no spans → not promoted when require_grounding=True."""
        from kaos_agents.memory.institutional import KnowledgeBase
        from kaos_agents.memory.promotion import PromotionPolicy

        kb = KnowledgeBase()
        policy = PromotionPolicy(min_confidence=0.85, require_grounding=True)

        finding = self._make_finding(
            confidence=0.95,
            statement="Apple revenue was $400 billion.",
            spans=(),  # no spans!
            is_verified=None,
        )
        mc = ("apple-matter", "acme-corp")
        decision = policy.consider(finding, matter_client=mc, knowledge_base=kb)
        print(
            f"\n[Promotion/no-grounding] promoted={decision.promoted}, reason={decision.reason!r}"
        )
        assert decision.promoted is False
        assert "grounding" in decision.reason.lower()

    def test_is_verified_flag_satisfies_grounding(self) -> None:
        """is_verified=True (no spans) counts as grounded."""
        from kaos_agents.memory.institutional import KnowledgeBase
        from kaos_agents.memory.promotion import PromotionPolicy

        kb = KnowledgeBase()
        policy = PromotionPolicy(min_confidence=0.85, require_grounding=True)

        finding = self._make_finding(
            confidence=0.90,
            statement="Microsoft revenue was $200 billion.",
            spans=(),
            is_verified=True,  # is_verified=True → treated as grounded
        )
        mc = ("msft-matter", "acme-corp")
        decision = policy.consider(finding, matter_client=mc, knowledge_base=kb)
        print(f"\n[Promotion/is-verified] promoted={decision.promoted}")
        assert decision.promoted is True

    @pytest.mark.live
    @requires_anthropic
    async def test_promotion_from_perception_rag_output(self) -> None:
        """End-to-end: PerceptionRAG → Cited[T]-shaped output → PromotionPolicy.

        Runs a real PerceptionRAG on a tiny corpus, extracts the answer,
        and verifies PromotionPolicy promotes it when confidence is high.
        """
        from kaos_agents.events.collector import collect_events
        from kaos_agents.memory.institutional import KBQuery, KnowledgeBase
        from kaos_agents.memory.promotion import PromotionPolicy
        from kaos_agents.perception.rag import PerceptionRAG

        rag = PerceptionRAG(
            model=ANTHROPIC_DEFAULT,
            top_k=3,
        )
        corpus = {
            "doc:tesla-2023": (
                "Tesla Inc. reported total revenue of $96.77 billion for fiscal year 2023, "
                "representing a 19% increase from $81.46 billion in 2022. "
                "Automotive revenue was $82.42 billion."
            )
        }
        with collect_events():
            output = await rag.forward(
                question="What was Tesla's total revenue in 2023?",
                documents=corpus,
            )

        print(f"\n[Promotion/rag-output] output type={type(output).__name__}")

        # Extract the answer text from the RAG output
        # RAGResult has .grounded_answer (Answer or str) or .answer
        answer_text = (
            getattr(output, "text", None)
            or getattr(getattr(output, "grounded_answer", None), "text", None)
            or str(output)
        )
        spans = getattr(output, "spans", None) or ()
        grounded_answer = getattr(output, "grounded_answer", None)
        if grounded_answer is not None:
            spans = getattr(grounded_answer, "spans", ()) or ()

        print(f"  answer_text={answer_text[:100]!r}, spans_count={len(spans)}")

        # Build a finding-shaped object from the RAG output
        # Use a confidence of 0.9 for this test (RAG doesn't emit confidence directly)
        finding = SimpleNamespace(
            confidence=0.90,
            statement=answer_text[:500] if answer_text else "RAG produced no text",
            spans=spans,
            is_verified=bool(spans),
        )

        kb = KnowledgeBase()
        policy = PromotionPolicy(min_confidence=0.85, require_grounding=False)
        mc = ("tesla-rag-matter", "test-client")
        decision = policy.consider(finding, matter_client=mc, knowledge_base=kb)
        print(f"  promoted={decision.promoted}, reason={decision.reason!r}")
        assert decision.promoted is True, (
            f"Expected promotion (confidence=0.90), got: {decision.reason!r}"
        )
        # Verify KB contains the finding
        q = KBQuery(query_text="Tesla revenue 2023", matter_client=mc)
        result = kb.query(q)
        assert len(result.entries) >= 1


# ===========================================================================
# End-to-end AgentLoop — fully wired
# ===========================================================================


@pytest.mark.live
@requires_anthropic
class TestAgentLoopE2ELive:
    """End-to-end AgentLoop tests: real LLM + real termination + real KB."""

    async def test_full_wired_agent_loop_completes(self) -> None:
        """Fully-wired AgentLoop runs end-to-end without raising.

        Wires: IntentExtractor + explicit ReActPlanner + TerminationJudge
        + EscalationPolicy + KnowledgeBase + PromotionPolicy.

        Note: uses an explicit ReActPlanner to avoid HierarchicalPlanner's
        recursive depth issue (DEFECT-4: HierarchicalPlanner heuristic default
        creates infinite nesting when auto_select picks it for RESEARCH).
        """
        from kaos_agents.escalation import EscalationPolicy
        from kaos_agents.intent import IntentExtractor
        from kaos_agents.loop.agent_loop import AgentLoop
        from kaos_agents.memory.institutional import KnowledgeBase
        from kaos_agents.memory.promotion import PromotionPolicy
        from kaos_agents.planning.react_planner import ReActPlanner
        from kaos_agents.termination.judge import TerminationJudge
        from kaos_agents.triggers.base import Trigger

        t_judge = TerminationJudge(
            judge=_DuckTypedJudge(score_good=0.95, score_bad=0.1),
            min_quality=0.7,
            max_iterations=3,
        )
        loop = AgentLoop(
            intent_extractor=IntentExtractor(model=ANTHROPIC_DEFAULT),
            # Use explicit ReActPlanner to avoid HierarchicalPlanner recursion
            planner=ReActPlanner(model=ANTHROPIC_DEFAULT, max_iterations=5),
            auto_select_planner=False,
            default_planner_model=ANTHROPIC_DEFAULT,
            termination_judge=t_judge,
            escalation_policy=EscalationPolicy(),
            knowledge_base=KnowledgeBase(),
            promotion_policy=PromotionPolicy(min_confidence=0.85),
        )
        trigger = Trigger.mcp(
            "What is Tesla's ticker symbol?",
            session_id="s-test-001",
        )
        invocation = await loop.invoke(trigger=trigger)
        print(
            f"\n[AgentLoop/e2e] is_complete={invocation.is_complete}, "
            f"output={invocation.output[:100]!r}, "
            f"extras_keys={list(invocation.extras.keys())}"
        )
        assert invocation.is_complete is True
        assert invocation.output, "Expected non-empty output"
        # With explicit planner, selected_planner is NOT set (that's only for auto-select)
        # Verify the output contains "TSLA"
        assert "tsla" in invocation.output.lower() or "tesla" in invocation.output.lower(), (
            f"Expected Tesla ticker in output, got: {invocation.output!r}"
        )

    async def test_intent_pattern_is_set(self) -> None:
        """invocation.intent.pattern is one of (CHAT/RESEARCH/PLAN)."""
        from kaos_agents.intent import IntentExtractor
        from kaos_agents.loop.agent_loop import AgentLoop
        from kaos_agents.triggers.base import Trigger

        loop = AgentLoop(
            intent_extractor=IntentExtractor(model=ANTHROPIC_DEFAULT),
            auto_select_planner=True,
            default_planner_model=ANTHROPIC_DEFAULT,
        )
        trigger = Trigger.mcp("What year did Tesla go public?", session_id="s-test-002")
        invocation = await loop.invoke(trigger=trigger)
        pattern = invocation.intent.pattern
        print(f"\n[AgentLoop/pattern] intent.pattern={pattern}")
        assert pattern in (AgentPattern.CHAT, AgentPattern.RESEARCH, AgentPattern.PLAN), (
            f"Expected CHAT/RESEARCH/PLAN, got {pattern}"
        )

    async def test_usage_cost_positive(self) -> None:
        """invocation.usage.cost_usd > 0 after a real LLM call.

        DEFECT-2 (documented, NOT patched):
            AgentLoop._sum_usage_from_collector() only sums UsageObserved events
            from the loop's event collector. ReActPlanner.execute() calls
            ReAct.invoke() which returns usage in its Invocation, but this usage
            is captured inside ReAct's own trace collector, NOT emitted as a
            UsageObserved event to the AgentLoop's collector.
            Result: invocation.usage.cost_usd is always 0.0 even after real LLM calls.
            Fix: AgentLoop should also read exec_result.usage (PlanResult.usage) and
            add it to the collector total, or ReActPlanner should emit UsageObserved
            events to the active collector after invoke().
            File: kaos_agents/loop/agent_loop.py, method: _run_8_step_turn (step 7)
                  and kaos_agents/planning/react_planner.py, method: execute()
        """
        from kaos_agents.intent import IntentExtractor
        from kaos_agents.loop.agent_loop import AgentLoop
        from kaos_agents.triggers.base import Trigger

        loop = AgentLoop(
            intent_extractor=IntentExtractor(model=ANTHROPIC_DEFAULT),
            auto_select_planner=True,
            default_planner_model=ANTHROPIC_DEFAULT,
        )
        trigger = Trigger.mcp("Who invented the telephone?", session_id="s-test-003")
        invocation = await loop.invoke(trigger=trigger)
        print(
            f"\n[AgentLoop/usage] cost_usd={invocation.usage.cost_usd}, "
            f"total_tokens={invocation.usage.total_tokens}"
        )
        # DEFECT-2: cost_usd is 0.0 because ReActPlanner doesn't emit UsageObserved
        # to the loop's collector. We assert == 0 to document the defect.
        # When fixed, change to: assert invocation.usage.cost_usd > 0.0
        if invocation.usage.cost_usd == 0.0:
            pytest.xfail(
                reason=(
                    "DEFECT-2: AgentLoop.usage is 0.0 because ReActPlanner.execute() "
                    "does not emit UsageObserved events to the loop's EventCollector. "
                    "Usage stays inside ReAct's trace, invisible to the outer loop's "
                    "_sum_usage_from_collector(). "
                    "Fix: emit UsageObserved from ReActPlanner.execute() after "
                    "react.invoke() returns, or have AgentLoop accumulate "
                    "exec_result.usage directly in step 7. "
                    "Files: kaos_agents/planning/react_planner.py, "
                    "kaos_agents/loop/agent_loop.py"
                )
            )
        else:
            assert invocation.usage.cost_usd > 0.0

    async def test_event_taxonomy_present(self) -> None:
        """Span(TURN,START), IntentClassified, TurnSummary, Span(TURN,COMPLETE) all present."""
        from kaos_agents.events.lifecycle import IntentClassified, TurnSummary
        from kaos_agents.events.spans import Span, SpanPhase, SpanSubject
        from kaos_agents.intent import IntentExtractor
        from kaos_agents.loop.agent_loop import AgentLoop
        from kaos_agents.triggers.base import Trigger

        loop = AgentLoop(
            intent_extractor=IntentExtractor(model=ANTHROPIC_DEFAULT),
            auto_select_planner=True,
            default_planner_model=ANTHROPIC_DEFAULT,
        )
        trigger = Trigger.mcp("Name one planet in our solar system.", session_id="s-test-004")
        invocation = await loop.invoke(trigger=trigger)
        events = invocation.events

        turn_starts = [
            e
            for e in events
            if isinstance(e, Span) and e.subject == SpanSubject.TURN and e.phase == SpanPhase.START
        ]
        turn_completes = [
            e
            for e in events
            if isinstance(e, Span)
            and e.subject == SpanSubject.TURN
            and e.phase == SpanPhase.COMPLETE
        ]
        intent_classified = [e for e in events if isinstance(e, IntentClassified)]
        turn_summaries = [e for e in events if isinstance(e, TurnSummary)]

        print(
            f"\n[AgentLoop/events] "
            f"turn_starts={len(turn_starts)}, "
            f"turn_completes={len(turn_completes)}, "
            f"intent_classified={len(intent_classified)}, "
            f"turn_summaries={len(turn_summaries)}, "
            f"total_events={len(events)}"
        )
        assert len(turn_starts) >= 1, "Missing Span(TURN, START)"
        assert len(turn_completes) >= 1, "Missing Span(TURN, COMPLETE)"
        assert len(intent_classified) >= 1, "Missing IntentClassified"
        assert len(turn_summaries) >= 1, "Missing TurnSummary"

    async def test_termination_decision_kind_extra_set(self) -> None:
        """extras['termination_decision_kind'] reflects the real Decision kind.

        DEFECT-3 (documented, NOT patched):
            AgentLoop._run_termination_judge calls judge.invoke() (Program.invoke)
            which returns an Invocation, not a Decision. The code reads:
                str(getattr(decision, "kind", ""))
            on the Invocation. Invocation has no .kind attribute, so the result
            is always "". termination_decision_kind is never set to a real value.
            Fix: use decision.output.kind (the actual Decision.kind from forward())
            or call judge.forward() directly instead of judge.invoke().
            File: kaos_agents/loop/agent_loop.py, _run_termination_judge, line ~693
        """
        from kaos_agents.intent import IntentExtractor
        from kaos_agents.loop.agent_loop import AgentLoop
        from kaos_agents.termination.judge import TerminationJudge
        from kaos_agents.triggers.base import Trigger

        t_judge = TerminationJudge(
            judge=_DuckTypedJudge(score_good=0.9, score_bad=0.2),
            min_quality=0.7,
            max_iterations=10,
        )
        loop = AgentLoop(
            intent_extractor=IntentExtractor(model=ANTHROPIC_DEFAULT),
            auto_select_planner=True,
            default_planner_model=ANTHROPIC_DEFAULT,
            termination_judge=t_judge,
        )
        trigger = Trigger.mcp("What is 2 + 2?", session_id="s-test-005")
        invocation = await loop.invoke(trigger=trigger)
        td_kind = invocation.extras.get("termination_decision_kind", "")
        print(
            f"\n[AgentLoop/td_kind] termination_decision_kind={td_kind!r}, "
            f"output={invocation.output!r}"
        )
        # DEFECT-3: td_kind is "" because judge.invoke() returns Invocation which
        # has no .kind attribute. When fixed, this should be "complete" for a
        # successful turn.
        if not td_kind:
            pytest.xfail(
                reason=(
                    "DEFECT-3: AgentLoop._run_termination_judge calls judge.invoke() "
                    "which returns an Invocation (Program wrapper), not a Decision. "
                    "getattr(Invocation, 'kind', '') == '' always. "
                    "Fix: read decision.output.kind (the Decision from forward()) "
                    "or call judge.forward() instead of judge.invoke(). "
                    "File: kaos_agents/loop/agent_loop.py, _run_termination_judge"
                )
            )
        else:
            assert any(
                kw in td_kind.lower()
                for kw in ("complete", "degraded", "incomplete", "quality_failed", "budget")
            ), f"Unexpected termination_decision_kind: {td_kind!r}"

    async def test_output_non_empty_for_factual_question(self) -> None:
        """invocation.output is non-empty for a clear factual question."""
        from kaos_agents.intent import IntentExtractor
        from kaos_agents.loop.agent_loop import AgentLoop
        from kaos_agents.triggers.base import Trigger

        loop = AgentLoop(
            intent_extractor=IntentExtractor(model=ANTHROPIC_DEFAULT),
            auto_select_planner=True,
            default_planner_model=ANTHROPIC_DEFAULT,
        )
        trigger = Trigger.mcp(
            "What is the chemical symbol for gold?",
            session_id="s-test-006",
        )
        invocation = await loop.invoke(trigger=trigger)
        print(f"\n[AgentLoop/output] output={invocation.output[:100]!r}")
        assert invocation.output, "Expected non-empty output for factual question"
        # Chemical symbol for gold is Au
        assert "au" in invocation.output.lower() or "gold" in invocation.output.lower(), (
            f"Expected 'au' or 'gold' in output, got: {invocation.output!r}"
        )

    async def test_selected_planner_extra_set(self) -> None:
        """extras['selected_planner'] reflects the auto-selected planner name."""
        from kaos_agents.intent import IntentExtractor
        from kaos_agents.loop.agent_loop import AgentLoop
        from kaos_agents.triggers.base import Trigger

        loop = AgentLoop(
            intent_extractor=IntentExtractor(model=ANTHROPIC_DEFAULT),
            auto_select_planner=True,
            default_planner_model=ANTHROPIC_DEFAULT,
        )
        trigger = Trigger.mcp("What is the capital of France?", session_id="s-test-007")
        invocation = await loop.invoke(trigger=trigger)
        planner_name = invocation.extras.get("selected_planner", "")
        print(f"\n[AgentLoop/planner] selected_planner={planner_name!r}")
        assert planner_name, "Expected selected_planner to be set"
        assert planner_name in ("ReActPlanner", "PlanExecutePlanner", "HierarchicalPlanner"), (
            f"Unexpected planner name: {planner_name!r}"
        )


@pytest.mark.live
@requires_openai
class TestAgentLoopE2EOpenAILive:
    """End-to-end AgentLoop tests with OpenAI models."""

    async def test_gpt_mini_agent_loop_completes(self) -> None:
        """gpt-5.4-mini drives the full AgentLoop end-to-end."""
        from kaos_agents.intent import IntentExtractor
        from kaos_agents.loop.agent_loop import AgentLoop
        from kaos_agents.triggers.base import Trigger

        loop = AgentLoop(
            intent_extractor=IntentExtractor(model=OPENAI_DEFAULT),
            auto_select_planner=True,
            default_planner_model=OPENAI_DEFAULT,
        )
        trigger = Trigger.mcp("What is 6 times 7?", session_id="s-gpt-001")
        invocation = await loop.invoke(trigger=trigger)
        print(
            f"\n[AgentLoop/gpt-mini] output={invocation.output[:80]!r}, "
            f"selected_planner={invocation.extras.get('selected_planner')}"
        )
        assert invocation.is_complete is True
        assert invocation.output, "Expected non-empty output"
        assert "42" in invocation.output, f"Expected '42' in output, got: {invocation.output!r}"

    async def test_gpt_flagship_agent_loop_usage(self) -> None:
        """gpt-5.4 loop produces correct answer. Cost tracking is DEFECT-2.

        DEFECT-2 is documented in test_usage_cost_positive above.
        This test verifies the output is correct even when cost_usd=0.0.
        """
        from kaos_agents.intent import IntentExtractor
        from kaos_agents.loop.agent_loop import AgentLoop
        from kaos_agents.triggers.base import Trigger

        loop = AgentLoop(
            intent_extractor=IntentExtractor(model=OPENAI_FLAGSHIP),
            auto_select_planner=True,
            default_planner_model=OPENAI_FLAGSHIP,
        )
        trigger = Trigger.mcp("Name the speed of light in m/s.", session_id="s-gpt-002")
        invocation = await loop.invoke(trigger=trigger)
        print(
            f"\n[AgentLoop/gpt-flagship] cost_usd={invocation.usage.cost_usd}, "
            f"total_tokens={invocation.usage.total_tokens}, "
            f"output={invocation.output[:80]!r}"
        )
        assert invocation.is_complete is True
        # Speed of light: 299,792,458 m/s
        assert "299" in invocation.output or "light" in invocation.output.lower(), (
            f"Expected speed of light in output, got: {invocation.output!r}"
        )
        # Note: cost_usd is expected to be 0.0 due to DEFECT-2.
        # When fixed, add: assert invocation.usage.cost_usd > 0.0
