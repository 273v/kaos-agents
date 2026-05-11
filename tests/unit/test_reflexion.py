"""Unit tests for kaos_agents.patterns.reflexion.

Deterministic — exercises the loop control flow with a stub critic
and a stub inner agent. Live LLM integration is tested separately.
"""

from __future__ import annotations

import asyncio
from typing import cast

import pytest

from kaos_agents.base.agent import KaosAgent
from kaos_agents.patterns.reflexion import (
    CritiqueResult,
    ReflexionCritic,
    ReflexionLoop,
    ReflexionTrace,
    _format_feedback,
)
from kaos_agents.types import IntentResult, IntentType
from kaos_agents.types.response import AgentResponse

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_response(text: str) -> AgentResponse:
    return AgentResponse.create(
        text=text,
        intent=IntentResult(intent=IntentType.RESPOND, confidence=1.0, reasoning="stub"),
        tool_calls=(),
        turn_number=1,
        tokens_used=0,
    )


class _StubInnerAgent:
    """Records every `turn` call and returns canned responses."""

    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self.calls: list[tuple[str, str, str]] = []  # (msg, sess, extra)

    async def turn(
        self,
        message: str,
        session_id: str,
        *,
        extra_instruction: str = "",
    ) -> AgentResponse:
        self.calls.append((message, session_id, extra_instruction))
        if not self._responses:
            return _make_response("(no canned response left)")
        return _make_response(self._responses.pop(0))


class _StubCritic:
    """Returns canned CritiqueResults in order."""

    def __init__(self, results: list[CritiqueResult]) -> None:
        self._results = list(results)
        self.critique_count = 0

    async def critique(self, question: str, output: str) -> CritiqueResult:
        self.critique_count += 1
        if not self._results:
            raise AssertionError("StubCritic ran out of canned results")
        return self._results.pop(0)


# ---------------------------------------------------------------------------
# ReflexionCritic construction
# ---------------------------------------------------------------------------


def _trace_from_metadata(response) -> dict | None:
    """Look up the reflexion_trace entry in AgentResponse.metadata (a tuple of key/value tuples)."""
    for k, v in response.metadata or ():
        if k == "reflexion_trace":
            return v
    return None


class TestReflexionCriticConstruction:
    def test_empty_rubric_rejected(self) -> None:
        with pytest.raises(ValueError, match="rubric"):
            ReflexionCritic(rubric="")
        with pytest.raises(ValueError, match="rubric"):
            ReflexionCritic(rubric="   ")

    def test_threshold_out_of_range(self) -> None:
        with pytest.raises(ValueError, match="threshold"):
            ReflexionCritic(rubric="x", threshold=1.5)
        with pytest.raises(ValueError, match="threshold"):
            ReflexionCritic(rubric="x", threshold=-0.1)

    def test_defaults(self) -> None:
        critic = ReflexionCritic(rubric="be good")
        assert critic.threshold == 0.7
        assert "sonnet" in critic.model.lower()


# ---------------------------------------------------------------------------
# CritiqueResult + ReflexionTrace
# ---------------------------------------------------------------------------


class TestCritiqueResult:
    def test_frozen(self) -> None:
        r = CritiqueResult(score=0.9, approved=True, feedback="", reasoning="ok")
        with pytest.raises((AttributeError, TypeError)):
            r.score = 0.5  # ty: ignore[invalid-assignment]


# ---------------------------------------------------------------------------
# ReflexionLoop control flow
# ---------------------------------------------------------------------------


class TestReflexionLoop:
    def test_max_iterations_must_be_positive(self) -> None:
        inner = _StubInnerAgent(["x"])
        critic = _StubCritic([CritiqueResult(1.0, True, "", "fine")])
        with pytest.raises(ValueError, match="max_iterations"):
            ReflexionLoop(
                cast("KaosAgent", inner),
                cast("ReflexionCritic", critic),
                max_iterations=0,
            )

    def test_accepts_first_iteration_on_high_score(self) -> None:
        inner = _StubInnerAgent(["good answer"])
        critic = _StubCritic([CritiqueResult(0.95, True, "", "satisfies rubric")])
        loop = ReflexionLoop(
            cast("KaosAgent", inner),
            cast("ReflexionCritic", critic),
            max_iterations=3,
        )
        response = asyncio.run(loop.turn("question", "session-1"))
        assert response.text == "good answer"
        assert critic.critique_count == 1
        assert len(inner.calls) == 1
        # First call should have no feedback
        assert inner.calls[0][2] == ""
        # Trace shape
        trace = _trace_from_metadata(response)
        assert trace is not None
        assert trace["accepted"] is True
        assert trace["final_iteration"] == 1

    def test_retries_with_feedback_until_approved(self) -> None:
        inner = _StubInnerAgent(["weak first", "better second"])
        critic = _StubCritic(
            [
                CritiqueResult(0.3, False, "Add a citation", "missing source"),
                CritiqueResult(0.9, True, "", "now cites Foo"),
            ]
        )
        loop = ReflexionLoop(
            cast("KaosAgent", inner),
            cast("ReflexionCritic", critic),
            max_iterations=3,
        )
        response = asyncio.run(loop.turn("question", "session-1"))
        assert response.text == "better second"
        assert critic.critique_count == 2
        # Second call should have feedback from the first iteration —
        # prepended into the message itself (the inner agent has no
        # extra_instruction kwarg surface, so _call_inner wraps the
        # critique into the user message).
        second_message = inner.calls[1][0]
        assert "Add a citation" in second_message
        assert "CRITIQUE FROM PRIOR ITERATION" in second_message
        assert "question" in second_message  # original message preserved
        # Trace records both iterations
        trace = _trace_from_metadata(response)
        assert trace is not None
        assert trace["accepted"] is True
        assert trace["final_iteration"] == 2
        assert len(trace["iterations"]) == 2

    def test_returns_best_after_max_iterations(self) -> None:
        # Three iterations: scores 0.2, 0.5, 0.4 — none approved (threshold 0.7)
        # but iteration 2 was best, so it should be returned.
        inner = _StubInnerAgent(["v1", "v2", "v3"])
        critic = _StubCritic(
            [
                CritiqueResult(0.2, False, "fix A", "low"),
                CritiqueResult(0.5, False, "fix B", "medium"),
                CritiqueResult(0.4, False, "fix C", "low again"),
            ]
        )
        loop = ReflexionLoop(
            cast("KaosAgent", inner),
            cast("ReflexionCritic", critic),
            max_iterations=3,
        )
        response = asyncio.run(loop.turn("question", "session-1"))
        # Best response was "v2" (score 0.5)
        assert response.text == "v2"
        trace = _trace_from_metadata(response)
        assert trace is not None
        assert trace["accepted"] is False
        assert trace["final_iteration"] == 3
        assert len(trace["iterations"]) == 3

    def test_single_iteration_disables_loop(self) -> None:
        # max_iterations=1 — one call, no retry, accept whatever critic says.
        inner = _StubInnerAgent(["only answer"])
        critic = _StubCritic([CritiqueResult(0.1, False, "bad", "low")])
        loop = ReflexionLoop(
            cast("KaosAgent", inner),
            cast("ReflexionCritic", critic),
            max_iterations=1,
        )
        response = asyncio.run(loop.turn("q", "s"))
        # Even though critique rejected, single-iteration loops surface
        # the best (only) attempt.
        assert response.text == "only answer"
        assert critic.critique_count == 1
        trace = _trace_from_metadata(response)
        assert trace is not None
        assert trace["accepted"] is False


# ---------------------------------------------------------------------------
# _format_feedback
# ---------------------------------------------------------------------------


class TestFormatFeedback:
    def test_empty_history(self) -> None:
        assert _format_feedback([]) == ""

    def test_filters_blank_entries(self) -> None:
        assert _format_feedback(["", "  ", ""]) == ""

    def test_renders_each_critique(self) -> None:
        text = _format_feedback(["fix A", "fix B"])
        assert "fix A" in text
        assert "fix B" in text
        # Iteration markers present
        assert "Iteration 1" in text
        assert "Iteration 2" in text


# ---------------------------------------------------------------------------
# ReflexionTrace
# ---------------------------------------------------------------------------


class TestReflexionTrace:
    def test_default_empty(self) -> None:
        trace = ReflexionTrace()
        assert trace.iterations == ()
        assert trace.final_iteration == 0
        assert trace.accepted is False
