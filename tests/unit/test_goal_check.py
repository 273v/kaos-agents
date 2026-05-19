"""Tests for `kaos_agents.planning.goal_check`.

PRD `kaos-modules/docs/internal/agentic-loop-auto-elevation-plan.md`
§3.2. Pins:

- Three-way discriminated union (satisfied / needs_more_work /
  insufficient_evidence) round-trips through the codec correctly.
- `check_goal` returns Satisfied on confident "this answered the
  question" stub.
- `check_goal` returns NeedsMoreWork on stub that signals continuation.
- `check_goal` returns InsufficientEvidence on stub that signals
  refusal.
- On provider exception / missing [llm] extra, the checker DEFAULTS
  to NeedsMoreWork (NEVER to Satisfied — that would silently ship a
  bad answer).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from unittest.mock import patch

import pytest

from kaos_agents.planning.goal_check import (
    GoalCheckInsufficientEvidence,
    GoalCheckNeedsMoreWork,
    GoalCheckOutcome,
    GoalCheckSatisfied,
    check_goal,
)

pytestmark = pytest.mark.unit


# ─── Stub helpers ────────────────────────────────────────────────────


@dataclass
class _FakeUsage:
    cost_usd: float = 0.0001


@dataclass
class _FakeOutput:
    result: Any  # one of the three GoalCheckResult shapes


@dataclass
class _FakeInvocation:
    output: _FakeOutput
    usage: _FakeUsage


def _stub_invoke(result: Any, cost_usd: float = 0.0001) -> Any:
    """Patchable replacement for ``Call.invoke``."""

    async def _impl(self: Any, **_kwargs: Any) -> _FakeInvocation:
        return _FakeInvocation(
            output=_FakeOutput(result=result),
            usage=_FakeUsage(cost_usd=cost_usd),
        )

    return _impl


# ─── Value-type round-trips ──────────────────────────────────────────


class TestDiscriminatedUnion:
    """The three-way union is Pydantic-discriminated on ``kind``."""

    def test_satisfied_shape(self) -> None:
        v = GoalCheckSatisfied(confidence=0.95, rationale="ok")
        assert v.kind == "satisfied"
        assert v.confidence == pytest.approx(0.95)

    def test_needs_more_work_shape(self) -> None:
        v = GoalCheckNeedsMoreWork(
            next_action="search SCOTUS directly",
            confidence=0.7,
            rationale="agent gave up too early",
        )
        assert v.kind == "needs_more_work"
        assert "SCOTUS" in v.next_action

    def test_insufficient_evidence_shape(self) -> None:
        v = GoalCheckInsufficientEvidence(
            missing="no Delaware case law on this fact pattern",
            rationale="we looked, it isn't there",
        )
        assert v.kind == "insufficient_evidence"

    def test_kind_is_literal_unbiased(self) -> None:
        """The three shapes are distinguishable by ``kind`` alone."""
        s = GoalCheckSatisfied(confidence=1.0, rationale="x")
        n = GoalCheckNeedsMoreWork(next_action="x", confidence=0.5, rationale="x")
        i = GoalCheckInsufficientEvidence(missing="x", rationale="x")
        kinds = {s.kind, n.kind, i.kind}
        assert kinds == {"satisfied", "needs_more_work", "insufficient_evidence"}


# ─── GoalCheckOutcome wrapper ────────────────────────────────────────


class TestGoalCheckOutcome:
    """The AgenticLoop-facing record with cost + latency + iteration."""

    def test_outcome_carries_observability_fields(self) -> None:
        oc = GoalCheckOutcome(
            result=GoalCheckSatisfied(confidence=0.9, rationale="ok"),
            cost_usd=0.0001,
            latency_ms=123.4,
            iteration=2,
        )
        assert oc.kind == "satisfied"
        assert oc.satisfied is True
        assert oc.is_terminal is True
        assert oc.cost_usd == pytest.approx(0.0001)
        assert oc.iteration == 2

    def test_needs_more_work_is_not_terminal(self) -> None:
        oc = GoalCheckOutcome(
            result=GoalCheckNeedsMoreWork(next_action="x", confidence=0.5, rationale="x"),
            cost_usd=0.0001,
            latency_ms=100.0,
            iteration=1,
        )
        assert oc.needs_more_work is True
        assert oc.is_terminal is False

    def test_insufficient_evidence_is_terminal(self) -> None:
        oc = GoalCheckOutcome(
            result=GoalCheckInsufficientEvidence(missing="x", rationale="x"),
            cost_usd=0.0001,
            latency_ms=100.0,
            iteration=1,
        )
        assert oc.insufficient_evidence is True
        assert oc.is_terminal is True


# ─── check_goal happy paths ──────────────────────────────────────────


@pytest.mark.asyncio
async def test_check_goal_returns_satisfied_when_critic_confident() -> None:
    stub = GoalCheckSatisfied(
        confidence=0.92,
        rationale="Agent answered with citations + tool-grounded facts.",
    )
    with patch("kaos_llm_core.Call.invoke", new=_stub_invoke(stub)):
        oc = await check_goal(
            user_message="Find recent SCOTUS opinions on Chevron deference.",
            agent_response="The most recent SCOTUS opinion is Loper Bright v. Raimondo (2024)...",
            tool_calls_made=[
                {
                    "name": "kaos-source-fr-search",
                    "is_error": False,
                    "summary_excerpt": "Found 5 results.",
                }
            ],
            elevation_trail=["web"],
            available_groups=["web", "documents", "citations"],
            iteration=1,
        )
    assert oc.satisfied
    assert oc.is_terminal
    assert isinstance(oc.result, GoalCheckSatisfied)
    assert oc.result.confidence == pytest.approx(0.92)


@pytest.mark.asyncio
async def test_check_goal_returns_needs_more_work_with_next_action() -> None:
    stub = GoalCheckNeedsMoreWork(
        next_action="request the web tool group to search SCOTUS directly",
        confidence=0.75,
        rationale="Agent apologized for lacking web access; the web group is available.",
    )
    with patch("kaos_llm_core.Call.invoke", new=_stub_invoke(stub)):
        oc = await check_goal(
            user_message="Find recent SCOTUS opinions on Chevron deference.",
            agent_response=(
                "I don't have web access in this session, so I can't check for recent opinions."
            ),
            tool_calls_made=[],
            elevation_trail=[],
            available_groups=["web", "documents", "citations"],
        )
    assert oc.needs_more_work
    assert not oc.is_terminal
    assert isinstance(oc.result, GoalCheckNeedsMoreWork)
    assert "web" in oc.result.next_action.lower()


@pytest.mark.asyncio
async def test_check_goal_returns_insufficient_evidence_when_corpus_lacks() -> None:
    stub = GoalCheckInsufficientEvidence(
        missing="no Delaware case law on this fact pattern in CourtListener or Lexis connectors",
        rationale="The agent looked in all reasonable sources; the case law isn't there.",
    )
    with patch("kaos_llm_core.Call.invoke", new=_stub_invoke(stub)):
        oc = await check_goal(
            user_message="Find a DE Chancery opinion involving X exact fact pattern.",
            agent_response="I searched CourtListener + the FR; no opinion matches.",
            tool_calls_made=[
                {
                    "name": "kaos-source-fr-search",
                    "is_error": False,
                    "summary_excerpt": "0 results.",
                }
            ],
            elevation_trail=["web"],
            available_groups=["web", "documents"],
            iteration=2,
        )
    assert oc.insufficient_evidence
    assert oc.is_terminal


# ─── check_goal failure modes ────────────────────────────────────────


@pytest.mark.asyncio
async def test_provider_error_defaults_to_needs_more_work() -> None:
    """On any exception, the checker MUST default to needs_more_work —
    NEVER to satisfied (false satisfaction silently ships a bad answer)."""

    async def _raise(self: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("provider 503")

    with patch("kaos_llm_core.Call.invoke", new=_raise):
        oc = await check_goal(
            user_message="anything",
            agent_response="anything",
        )
    assert oc.needs_more_work, (
        f"Critic on provider error MUST default to needs_more_work, not satisfied. Got {oc.kind!r}."
    )
    assert not oc.satisfied
    assert oc.cost_usd == 0.0
    assert "Critic call failed" in oc.result.rationale


@pytest.mark.asyncio
async def test_omitted_inputs_default_to_safe_values() -> None:
    """All inputs except ``user_message`` + ``agent_response`` are
    optional; the checker handles missing context gracefully."""
    stub = GoalCheckSatisfied(confidence=0.5, rationale="basic")
    captured: dict[str, Any] = {}

    async def _capture(self: Any, **kwargs: Any) -> _FakeInvocation:
        captured.update(kwargs)
        return _FakeInvocation(output=_FakeOutput(result=stub), usage=_FakeUsage())

    with patch("kaos_llm_core.Call.invoke", new=_capture):
        await check_goal(
            user_message="hi",
            agent_response="hello",
        )

    assert captured["tool_calls_made"] == []
    assert captured["elevation_trail"] == []
    assert captured["available_groups"] == []
    assert captured["iteration"] == 1


@pytest.mark.asyncio
async def test_iteration_is_passed_through() -> None:
    """The Critic's iteration-aware decision rule (see Signature docstring)
    needs the current iteration count to decide between needs_more_work
    and insufficient_evidence on repeated stalls."""
    stub = GoalCheckSatisfied(confidence=0.9, rationale="ok")
    captured: dict[str, Any] = {}

    async def _capture(self: Any, **kwargs: Any) -> _FakeInvocation:
        captured.update(kwargs)
        return _FakeInvocation(output=_FakeOutput(result=stub), usage=_FakeUsage())

    with patch("kaos_llm_core.Call.invoke", new=_capture):
        oc = await check_goal(
            user_message="x",
            agent_response="y",
            iteration=3,
        )

    assert captured["iteration"] == 3
    assert oc.iteration == 3


# ─── Header-then-stop deterministic pre-check (P0-4 #436) ────────────


class TestHeaderThenStopPreCheck:
    """The deterministic pre-check short-circuits before LLM call when
    the agent's response ends near a deliverable heading with no body.

    Closes 2026-05-19 NDA matrix Persona #4 (header-and-quit shape).
    The Critic Signature docstring covers the long-tail wording the
    regex can't recognize; this regex covers the easy 80%.
    """

    @pytest.mark.asyncio
    async def test_header_only_short_circuits_to_needs_more_work(self) -> None:
        """An H2 naming a deliverable + zero body lines → needs_more_work."""

        async def _should_not_invoke(self: Any, **_kwargs: Any) -> Any:
            raise AssertionError("LLM critic was called even though pre-check should have fired")

        with patch("kaos_llm_core.Call.invoke", new=_should_not_invoke):
            oc = await check_goal(
                user_message="Please produce a governing-law review table for 5 NDAs.",
                agent_response="## Governing-Law Review — 5 NDAs\n",
            )

        assert oc.kind == "needs_more_work"
        assert oc.cost_usd == 0.0  # no LLM spend
        assert isinstance(oc.result, GoalCheckNeedsMoreWork)
        assert "Governing-Law Review" in oc.result.next_action

    @pytest.mark.asyncio
    async def test_header_with_preamble_and_no_rows_short_circuits(self) -> None:
        """Preamble sentence + named-deliverable H3 + nothing → catches the
        2026-05-19 Persona #4 shape exactly."""

        async def _should_not_invoke(self: Any, **_kwargs: Any) -> Any:
            raise AssertionError("LLM critic called but pre-check should fire")

        body = (
            "## Governing-Law Review — 5 NDAs\n\n"
            "Below is the governing-law review of the five attached NDAs.\n\n"
            "### CSV-ready table\n"
        )
        with patch("kaos_llm_core.Call.invoke", new=_should_not_invoke):
            oc = await check_goal(
                user_message="produce a CSV-ready table",
                agent_response=body,
            )

        assert oc.kind == "needs_more_work"
        assert isinstance(oc.result, GoalCheckNeedsMoreWork)
        assert "CSV-ready table" in oc.result.next_action

    @pytest.mark.asyncio
    async def test_header_with_substantive_body_does_not_short_circuit(self) -> None:
        """A heading followed by real rows must NOT trip the pre-check."""
        stub = GoalCheckSatisfied(confidence=0.9, rationale="grounded")
        body = (
            "## Governing-Law Review — 5 NDAs\n\n"
            "| NDA | Governing Law | Forum |\n"
            "|---|---|---|\n"
            "| EMNA Mutual NDA | Delaware | Delaware courts |\n"
            "| MNDA - Acme | New York | New York courts |\n"
            "| MNDA - BI | California | Bay Area |\n"
            "| MNDA - CC | Texas | Dallas |\n"
            "| MNDA - DynaMo | Massachusetts | Boston |\n"
        )
        with patch("kaos_llm_core.Call.invoke", new=_stub_invoke(stub)):
            oc = await check_goal(
                user_message="produce a governing-law review table",
                agent_response=body,
            )

        assert oc.kind == "satisfied"

    @pytest.mark.asyncio
    async def test_non_deliverable_heading_does_not_short_circuit(self) -> None:
        """A heading that doesn't name a deliverable doesn't trip the check.

        ``## Introduction`` doesn't promise a table/list/summary, so even
        ending the response there is the LLM critic's call to make, not
        the regex's.
        """
        stub = GoalCheckNeedsMoreWork(next_action="elaborate", confidence=0.5, rationale="thin")
        with patch("kaos_llm_core.Call.invoke", new=_stub_invoke(stub)):
            oc = await check_goal(
                user_message="what is the contract about",
                agent_response="## Introduction\n",
            )

        # Whatever the LLM says — the deterministic pre-check did NOT fire,
        # so the stub's NeedsMoreWork must propagate. (If the pre-check
        # had fired the rationale would be the deterministic one.)
        assert oc.kind == "needs_more_work"
        assert oc.result.rationale == "thin"

    @pytest.mark.asyncio
    async def test_empty_response_does_not_short_circuit(self) -> None:
        """Edge case: empty / whitespace-only response is the LLM's call."""
        stub = GoalCheckNeedsMoreWork(
            next_action="answer the question", confidence=0.9, rationale="empty"
        )
        with patch("kaos_llm_core.Call.invoke", new=_stub_invoke(stub)):
            oc = await check_goal(user_message="x", agent_response="   \n  ")

        assert oc.kind == "needs_more_work"
        assert oc.result.rationale == "empty"
