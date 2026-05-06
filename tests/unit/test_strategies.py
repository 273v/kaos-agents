"""Tests for planning strategies — composition logic over primitives."""

from __future__ import annotations

from kaos_agents.planning.strategies.adaptive import _assess_complexity


class TestAdaptiveComplexity:
    """Test the heuristic complexity assessor."""

    def test_short_goal_is_simple(self):
        score = _assess_complexity("Search for X", n_tools=3)
        assert score >= 0.6, f"Short goal should be simple, got {score}"

    def test_long_complex_goal(self):
        score = _assess_complexity(
            "First search the eCFR for regulations, then search EDGAR for filings, "
            "then compare the two and write a comprehensive analysis",
            n_tools=5,
        )
        assert score < 0.6, f"Complex goal should be complex, got {score}"

    def test_then_keyword_reduces_confidence(self):
        simple = _assess_complexity("Search eCFR", n_tools=3)
        complex_ = _assess_complexity("Search eCFR then analyze results", n_tools=3)
        assert complex_ < simple

    def test_few_tools_increases_confidence(self):
        many = _assess_complexity("Do something", n_tools=10)
        few = _assess_complexity("Do something", n_tools=1)
        assert few >= many

    def test_multiple_sentences_reduce_confidence(self):
        one = _assess_complexity("Search for regulations", n_tools=3)
        multi = _assess_complexity(
            "Search for regulations. Then analyze them. Finally summarize.", n_tools=3
        )
        assert multi < one

    def test_score_bounded(self):
        """Score should always be in [0.1, 0.95]."""
        # Very complex
        score = _assess_complexity(
            "First do A, then B, after that C. Compare and cross-reference "
            "multiple comprehensive analyses step by step.",
            n_tools=20,
        )
        assert 0.1 <= score <= 0.95

        # Very simple
        score = _assess_complexity("Hi", n_tools=0)
        assert 0.1 <= score <= 0.95

    def test_custom_word_threshold(self):
        """Configurable word threshold affects complexity assessment."""
        # With default threshold (15), this is "short"
        default = _assess_complexity("one two three four five", n_tools=3)
        # With threshold=3, this is "long"
        strict = _assess_complexity("one two three four five", n_tools=3, simple_word_threshold=3)
        assert strict < default


class TestRollingBuildResult:
    def test_build_result(self):
        from kaos_agents.planning.strategies.rolling import _build_result
        from kaos_agents.types.plan import PlanBudget, StopReason

        budget = PlanBudget()
        budget.record_step(cost_usd=0.01, tokens=100)
        result = _build_result([], {"s1": "ok"}, 1, budget, StopReason.SUCCESS)
        assert result.stop_reason == StopReason.SUCCESS
        assert result.steps_executed == 1
        assert result.total_cost_usd == 0.01
        assert result.step_results == {"s1": "ok"}

    def test_build_result_failure(self):
        from kaos_agents.planning.strategies.rolling import _build_result
        from kaos_agents.types.plan import PlanBudget, StopReason

        budget = PlanBudget()
        result = _build_result([], {}, 0, budget, StopReason.FAILURE)
        assert result.stop_reason == StopReason.FAILURE
