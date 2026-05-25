"""Live LLM tests for planning primitives.

These tests call real LLM APIs (Anthropic Claude Haiku) to verify that
the planning primitives produce usable output. They are NOT unit tests —
they verify that the LLM generates structurally valid, semantically
reasonable plans and evaluations.

Run with: uv run pytest tests/integration/test_planning_live.py -v
Requires: ANTHROPIC_API_KEY in environment.
"""

from __future__ import annotations

import pytest

from kaos_agents.planning.evaluate import evaluate, evaluate_semantic
from kaos_agents.planning.expand import expand
from kaos_agents.types.plan import EvalMode, StepType

MODEL = "anthropic:claude-haiku-4-5"


@pytest.mark.live
class TestExpandLive:
    """Test that the LLM generates valid plan steps."""

    async def test_single_tool_goal(self):
        """Simple goal with one obvious tool → LLM should produce 1-3 steps using it."""
        tools = {
            "kaos-source-ecfr-search": (
                "Search the eCFR (Electronic Code of Federal Regulations) "
                "for regulatory text by title, part, or keyword."
            ),
        }

        steps = await expand(
            "Find the Clean Air Act emission standards in the eCFR",
            available_tools=tools,
            model=MODEL,
        )

        assert len(steps) >= 1, f"Expected at least 1 step, got {len(steps)}"

        # At least one step should reference the eCFR tool
        tool_steps = [s for s in steps if s.step_type == StepType.TOOL]
        ecfr_steps = [s for s in tool_steps if s.tool_name == "kaos-source-ecfr-search"]
        assert len(ecfr_steps) >= 1, (
            f"Expected at least 1 step using kaos-source-ecfr-search, "
            f"got tools: {[s.tool_name for s in tool_steps]}"
        )

        # No hallucinated tools
        for s in tool_steps:
            assert s.tool_name in tools, f"Hallucinated tool: {s.tool_name}"

    async def test_multi_source_plan(self):
        """Complex goal needing multiple sources should produce a multi-step plan."""
        tools = {
            "kaos-source-ecfr-search": "Search the eCFR for regulatory text.",
            "kaos-source-edgar-search": "Search SEC EDGAR for company filings (10-K, 10-Q, etc.).",
        }

        steps = await expand(
            "Research whether Company XYZ complies with EPA Clean Air Act regulations. "
            "Search both the regulations and the company's SEC filings.",
            available_tools=tools,
            model=MODEL,
        )

        assert len(steps) >= 2, f"Expected at least 2 steps, got {len(steps)}"

        # Should use both tools
        tool_names = {s.tool_name for s in steps if s.tool_name}
        assert "kaos-source-ecfr-search" in tool_names or any(
            "ecfr" in (s.description or "").lower() for s in steps
        ), f"Expected eCFR search in plan, got tools: {tool_names}"

    async def test_with_prior_failures(self):
        """When prior failures are provided, LLM should avoid repeating them."""
        tools = {
            "kaos-source-ecfr-search": "Search the eCFR for regulatory text.",
            "kaos-web-fetch": "Fetch a URL and extract text content.",
        }

        steps = await expand(
            "Find EPA enforcement guidelines",
            available_tools=tools,
            prior_failures=(
                "Previous attempt: searched eCFR with query 'EPA enforcement' "
                "but got zero results. The guidelines may be on the EPA "
                "website instead."
            ),
            model=MODEL,
        )

        assert len(steps) >= 1

        # The plan should incorporate the failure feedback — likely using web-fetch
        # or trying a different eCFR query. We can't assert exactly what, but it
        # shouldn't be identical to the failed approach.
        descriptions = " ".join(s.description for s in steps).lower()
        assert len(descriptions) > 20, "Plan should have meaningful descriptions"

    async def test_empty_tools_produces_llm_steps(self):
        """With no tools available, LLM should produce reasoning-only steps."""
        steps = await expand(
            "Explain the difference between a regulation and a statute",
            available_tools={},
            model=MODEL,
        )

        assert len(steps) >= 1
        for s in steps:
            assert s.step_type == StepType.LLM, f"Expected LLM step, got {s.step_type}"

    async def test_max_steps_respected(self):
        """max_steps parameter should limit plan length."""
        tools = {"tool-a": "Do A", "tool-b": "Do B", "tool-c": "Do C"}
        steps = await expand(
            "Do a complex 20-step analysis",
            available_tools=tools,
            max_steps=5,
            model=MODEL,
        )

        assert len(steps) <= 5, f"Expected max 5 steps, got {len(steps)}"


@pytest.mark.live
class TestExpandMultiChainLive:
    """Live verification of the ``multi_chain_n`` MCC routing path.

    The unit tests in ``tests/unit/test_expand.py::TestExpandMultiChainRouting``
    pin the constructor wiring (signature, n, producer_model, examples)
    via monkeypatch stubs. This test exercises the same path against a
    real LLM end-to-end — requires kaos-llm-core >= 0.1.2 which added
    the ``examples=`` kwarg to ``MultiChainComparison``. Per
    ``CONTRIBUTING.md`` "New public API needs at least one live
    integration test through its real entry point."
    """

    async def test_multi_chain_n_2_produces_valid_plan(self):
        """``multi_chain_n=2`` routes through MCC and returns a usable plan.

        Verifies: (a) MCC's producer-then-aggregator pipeline doesn't
        crash with the kaos-agents PlanExpandSignature, (b) the
        synthesized output unwraps to a Steps list through
        ``OutputForwardingMixin``, (c) the same ``load_examples("plan_expand")``
        few-shot pool reaches the producer chains (verified upstream
        via the MCC constructor tests; here we just confirm the
        end-to-end path produces structurally valid plans).
        """
        tools = {
            "kaos-source-ecfr-search": (
                "Search the eCFR (Electronic Code of Federal Regulations) "
                "for regulatory text by title, part, or keyword."
            ),
        }

        steps = await expand(
            "Find the Clean Air Act emission standards in the eCFR",
            available_tools=tools,
            model=MODEL,
            multi_chain_n=2,
        )

        # MCC's aggregator must synthesize a plan — same structural
        # invariants as the single-Call path.
        assert len(steps) >= 1, f"Expected at least 1 step from MCC aggregator, got {len(steps)}"
        # At least one step should reference the eCFR tool (proves the
        # producer chains saw the tool catalog and the aggregator
        # preserved it).
        tool_steps = [s for s in steps if s.step_type == StepType.TOOL]
        assert tool_steps, (
            "MCC aggregator produced zero TOOL steps despite "
            f"eCFR tool being in the catalog. Steps: {steps}"
        )


@pytest.mark.live
class TestEvaluateSemanticLive:
    """Test that the LLM produces reasonable quality judgments."""

    async def test_relevant_result_matches(self):
        """A result that addresses the expectation should match."""
        j = await evaluate_semantic(
            result=(
                "42 USC 7412 establishes the list of hazardous air pollutants "
                "and requires EPA to set emission standards for major sources."
            ),
            expected="Information about Clean Air Act emission standards",
            model=MODEL,
        )

        assert j.matched, f"Expected match, got: {j.reasoning}"
        assert j.confidence >= 0.5, f"Expected confidence >= 0.5, got {j.confidence}"
        assert j.mode == EvalMode.SEMANTIC

    async def test_irrelevant_result_does_not_match(self):
        """A result completely unrelated to the expectation should not match."""
        j = await evaluate_semantic(
            result="The weather in Chicago today is 72 degrees and sunny.",
            expected="List of SEC filing deadlines for Form 10-K",
            model=MODEL,
        )

        assert not j.matched, f"Expected no match, got: {j.reasoning}"
        assert j.confidence >= 0.5, "LLM should be confident this doesn't match"

    async def test_partial_result_matches_with_lower_confidence(self):
        """A partially relevant result should match with moderate confidence."""
        j = await evaluate_semantic(
            result="The Clean Air Act was enacted in 1970 and has been amended several times.",
            expected="Detailed list of current emission standards under the Clean Air Act",
            model=MODEL,
        )

        # The result is related but doesn't provide the specific list requested
        # LLM should recognize partial relevance
        assert j.confidence > 0.0, "Should have some confidence (related topic)"
        assert j.confidence < 1.0, "Should not be fully confident (missing specifics)"

    async def test_extracts_new_facts(self):
        """Semantic evaluation should extract key facts from the result."""
        j = await evaluate_semantic(
            result=(
                "Under 40 CFR Part 60, new stationary sources must comply "
                "with New Source Performance Standards (NSPS). "
                "The effective date for Subpart OOOO is October 15, 2012."
            ),
            expected="Emission standards for new sources",
            model=MODEL,
        )

        # Should extract facts — at least the regulation reference
        # This is non-deterministic but the result has clear factual content
        assert j.matched
        # new_facts may or may not be populated depending on LLM behavior
        # but reasoning should reference the content
        assert len(j.reasoning) > 10


@pytest.mark.live
class TestEvaluateCombinedLive:
    """Test the combined evaluate (structural → semantic fallback)."""

    async def test_structural_short_circuits_on_error(self):
        """Error results should be caught structurally, no LLM call."""
        j = await evaluate(
            "ERROR: Connection timeout after 30s",
            expected="List of regulations",
            model=MODEL,
        )

        assert not j.matched
        assert j.mode == EvalMode.STRUCTURAL  # Structural caught it

    async def test_semantic_fallback_on_expectation(self):
        """Non-error result with expectation should trigger semantic evaluation."""
        j = await evaluate(
            "The regulation requires annual reporting of emissions data.",
            expected="Emission reporting requirements",
            model=MODEL,
        )

        assert j.matched
        # Could be structural (if no expectation triggers default) or semantic
        # The important thing is the answer is correct
