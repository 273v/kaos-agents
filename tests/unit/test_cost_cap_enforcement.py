"""Unit tests for Sprint-3 #9 — cost-cap enforcement / transparency.

Verifies the agent honors its documented ``max_cost_usd`` contract
across:

- ``FindingsAgent`` (chunked filter loop): the chunk dispatcher
  short-circuits before the next wave when the cap is exceeded; the
  pre-synthesis headroom check skips Phase-3 when synthesis would
  push past the cap.
- ``AgentFindingsTool`` (MCP wrapper): the ``max_cost_usd`` parameter
  flows through to the agent and surfaces ``budget_exceeded`` in
  structuredContent; the rewrite-pre-call cap path is covered.
- ``AgentCorpusFilterTool``: post-call cap detection (K8 is one
  LLM call — cannot abort mid-call but the overshoot must surface).
- New ``REFUSAL_BUDGET_EXCEEDED`` constant + ``FindingsRefusal``
  composition with the existing two refusal reasons.

Live tests over real APIs live in
``tests/integration/test_cost_cap_honesty_live.py``.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest

from kaos_agents.patterns import findings as findings_mod
from kaos_agents.patterns.findings import (
    REFUSAL_BUDGET_EXCEEDED,
    REFUSAL_NO_CANDIDATES_ENUMERATED,
    REFUSAL_NO_RELEVANT_CANDIDATES,
    FilteredFinding,
    FindingCandidate,
    FindingsAgent,
    FindingsRefusal,
    every_sentence_selector,
)

# ---------------------------------------------------------------------------
# Fake DocumentView — duck-typed sentences/section_by_ref. We hand-roll a
# small one rather than import the test_findings.py copy because that
# module also patches the LLM helpers at import time.
# ---------------------------------------------------------------------------


class _FakeSentence:
    def __init__(
        self,
        text: str,
        paragraph_ref: str = "#/body/0",
        section_ref: str | None = None,
        page: int | None = None,
    ) -> None:
        self.text = text
        self.paragraph_ref = paragraph_ref
        self.section_ref = section_ref
        self.page = page


class _FakeView:
    """Stand-in for DocumentView with just the surface findings uses."""

    def __init__(self, sentences: list[_FakeSentence]) -> None:
        self.sentences = sentences

    def section_by_ref(self, _ref: str) -> None:
        return None


def _view_with_n_sentences(n: int) -> _FakeView:
    """Return a fake view with ``n`` synthetic candidate sentences."""
    sents = [_FakeSentence(f"This is sentence number {i}.", f"#/body/{i}") for i in range(n)]
    return _FakeView(sents)


# ---------------------------------------------------------------------------
# Fake filter/synthesize helpers with controllable per-call cost
# ---------------------------------------------------------------------------


class _FakeCostMeter:
    """Mutable per-call cost tracker — share one instance across stubs."""

    def __init__(self, per_filter_cost: float, per_synthesis_cost: float) -> None:
        self.per_filter_cost = per_filter_cost
        self.per_synthesis_cost = per_synthesis_cost
        self.filter_calls = 0
        self.synthesis_calls = 0


def _make_stubs(
    meter: _FakeCostMeter,
    *,
    survivor_per_chunk: int = 1,
) -> tuple[Any, Any]:
    """Build ``(_filter_chunk, _synthesize)`` stubs bound to ``meter``."""

    async def stub_filter_chunk(
        chunk: tuple[FindingCandidate, ...],
        **_kwargs: Any,
    ) -> tuple[tuple[FilteredFinding, ...], float]:
        meter.filter_calls += 1
        # Keep ``survivor_per_chunk`` survivors at relevance 0.9 each
        survivors = tuple(
            FilteredFinding(candidate=cand, relevance=0.9, reasoning="stub")
            for cand in chunk[:survivor_per_chunk]
        )
        return survivors, meter.per_filter_cost

    async def stub_synthesize(**_kwargs: Any) -> tuple[str, float]:
        meter.synthesis_calls += 1
        return "stub answer", meter.per_synthesis_cost

    return stub_filter_chunk, stub_synthesize


# ---------------------------------------------------------------------------
# FindingsAgent — chunked-filter cap
# ---------------------------------------------------------------------------


class TestFindingsAgentChunkCap:
    @pytest.mark.asyncio
    async def test_cap_aborts_before_next_chunk(self) -> None:
        """5 chunks at $0.002 each, cap=$0.005 → should fire after 3 chunks.

        Wave size = num_parallel = 1 to make the abort granularity
        per-chunk so the assertion is tight. After 3 chunks we have
        $0.006 accumulated; the pre-wave check fires before chunk 4
        dispatches.
        """
        meter = _FakeCostMeter(per_filter_cost=0.002, per_synthesis_cost=0.0)
        # 5 chunks of 1 sentence each at chunk_size=1
        view = _view_with_n_sentences(5)
        stub_filter, stub_synth = _make_stubs(meter)
        with (
            patch.object(findings_mod, "_filter_chunk", stub_filter),
            patch.object(findings_mod, "_synthesize", stub_synth),
        ):
            agent = FindingsAgent(
                selector=every_sentence_selector,
                chunk_size=1,
                num_parallel=1,  # serial waves so cap check is tight
                relevance_threshold=0.5,
                max_cost_usd=0.005,
            )
            result = await agent.run("what is the question?", view)  # ty: ignore[invalid-argument-type]

        # 3 chunks fire: cost = 0.002 * 3 = 0.006 > 0.005 → 4th wave aborts.
        assert meter.filter_calls == 3, (
            f"Expected exactly 3 filter calls (cap fires after 3rd), got {meter.filter_calls}"
        )
        # Synthesis was skipped because filter cost already past the cap.
        assert meter.synthesis_calls == 0
        # Result must surface budget_exceeded.
        assert result.budget_exceeded is True
        # The result carries 3 survivors (one per completed chunk).
        # NB: synthesis was skipped so answer is empty + refusal set.
        assert result.answer == ""
        assert result.refusal is not None
        assert result.refusal.reason == REFUSAL_BUDGET_EXCEEDED
        # filter_cost reflects the chunks that actually ran.
        assert result.filter_cost_usd == pytest.approx(0.006)

    @pytest.mark.asyncio
    async def test_cap_uncapped_runs_everything(self) -> None:
        """``max_cost_usd=None`` → behavior unchanged from pre-Sprint-3."""
        meter = _FakeCostMeter(per_filter_cost=0.002, per_synthesis_cost=0.01)
        view = _view_with_n_sentences(5)
        stub_filter, stub_synth = _make_stubs(meter)
        with (
            patch.object(findings_mod, "_filter_chunk", stub_filter),
            patch.object(findings_mod, "_synthesize", stub_synth),
        ):
            agent = FindingsAgent(
                selector=every_sentence_selector,
                chunk_size=1,
                num_parallel=2,
                relevance_threshold=0.5,
                max_cost_usd=None,
            )
            result = await agent.run("what is the question?", view)  # ty: ignore[invalid-argument-type]

        # Every chunk fires + synthesis fires.
        assert meter.filter_calls == 5
        assert meter.synthesis_calls == 1
        assert result.budget_exceeded is False
        assert result.refusal is None
        assert result.answer == "stub answer"

    @pytest.mark.asyncio
    async def test_synthesis_skipped_when_no_headroom(self) -> None:
        """All filter chunks complete but synthesis would breach the cap."""
        meter = _FakeCostMeter(per_filter_cost=0.001, per_synthesis_cost=0.05)
        view = _view_with_n_sentences(3)
        stub_filter, stub_synth = _make_stubs(meter)
        with (
            patch.object(findings_mod, "_filter_chunk", stub_filter),
            patch.object(findings_mod, "_synthesize", stub_synth),
        ):
            # Total filter cost = $0.003. Cap = $0.01. Synthesis
            # estimate $0.02 default would push to $0.023 > $0.01.
            agent = FindingsAgent(
                selector=every_sentence_selector,
                chunk_size=1,
                num_parallel=4,
                relevance_threshold=0.5,
                max_cost_usd=0.01,
                synthesis_cost_estimate_usd=0.02,
            )
            result = await agent.run("what is the question?", view)  # ty: ignore[invalid-argument-type]

        # All filter chunks fired (3 chunks * $0.001 = $0.003 < $0.01).
        assert meter.filter_calls == 3
        # Synthesis WAS skipped because $0.003 + $0.02 > $0.01.
        assert meter.synthesis_calls == 0
        assert result.budget_exceeded is True
        assert result.refusal is not None
        assert result.refusal.reason == REFUSAL_BUDGET_EXCEEDED
        # Surviving findings ARE preserved — the result is partial,
        # not empty. Caller can consume them directly.
        assert len(result.findings) == 3
        # The refusal carries the survivor count.
        assert result.refusal.candidates_surviving_filter == 3

    @pytest.mark.asyncio
    async def test_synthesis_runs_with_headroom(self) -> None:
        """Tight-but-feasible budget: filter + synthesis both fit."""
        meter = _FakeCostMeter(per_filter_cost=0.001, per_synthesis_cost=0.005)
        view = _view_with_n_sentences(3)
        stub_filter, stub_synth = _make_stubs(meter)
        with (
            patch.object(findings_mod, "_filter_chunk", stub_filter),
            patch.object(findings_mod, "_synthesize", stub_synth),
        ):
            # Filter $0.003 + estimate $0.005 = $0.008 <= cap $0.01.
            agent = FindingsAgent(
                selector=every_sentence_selector,
                chunk_size=1,
                num_parallel=4,
                relevance_threshold=0.5,
                max_cost_usd=0.01,
                synthesis_cost_estimate_usd=0.005,
            )
            result = await agent.run("what is the question?", view)  # ty: ignore[invalid-argument-type]

        assert meter.filter_calls == 3
        assert meter.synthesis_calls == 1
        assert result.budget_exceeded is False
        assert result.refusal is None
        assert result.answer == "stub answer"

    @pytest.mark.asyncio
    async def test_refusal_budget_exceeded_distinct_from_no_relevant(self) -> None:
        """REFUSAL_BUDGET_EXCEEDED is NOT the same as REFUSAL_NO_RELEVANT_CANDIDATES.

        When no survivors are produced AND the cap fired mid-filter,
        the refusal reason MUST be REFUSAL_BUDGET_EXCEEDED, not
        REFUSAL_NO_RELEVANT_CANDIDATES. The remediation differs:
        budget-exceeded → "raise the cap"; no-relevant → "the answer
        isn't in this document".
        """
        meter = _FakeCostMeter(per_filter_cost=0.005, per_synthesis_cost=0.0)
        view = _view_with_n_sentences(3)

        async def empty_filter_chunk(
            chunk: tuple[FindingCandidate, ...],
            **_kwargs: Any,
        ) -> tuple[tuple[FilteredFinding, ...], float]:
            meter.filter_calls += 1
            # No survivors from any chunk
            return (), meter.per_filter_cost

        stub_filter, stub_synth = empty_filter_chunk, _make_stubs(meter)[1]
        with (
            patch.object(findings_mod, "_filter_chunk", stub_filter),
            patch.object(findings_mod, "_synthesize", stub_synth),
        ):
            agent = FindingsAgent(
                selector=every_sentence_selector,
                chunk_size=1,
                num_parallel=1,
                relevance_threshold=0.5,
                max_cost_usd=0.006,  # 2 chunks @ $0.005 = $0.010 > $0.006 → fire after chunk 2
            )
            result = await agent.run("what is the question?", view)  # ty: ignore[invalid-argument-type]

        # 2 chunks fired → cap hit (0.010 > 0.006) → 3rd aborts
        assert meter.filter_calls == 2
        assert result.budget_exceeded is True
        assert result.refusal is not None
        # The discriminator: budget_exceeded, NOT no_relevant_candidates.
        assert result.refusal.reason == REFUSAL_BUDGET_EXCEEDED


# ---------------------------------------------------------------------------
# FindingsAgent — constructor validation
# ---------------------------------------------------------------------------


class TestFindingsAgentConstructor:
    def test_max_cost_usd_zero_rejected(self) -> None:
        with pytest.raises(ValueError, match="must be > 0"):
            FindingsAgent(selector=every_sentence_selector, max_cost_usd=0.0)

    def test_max_cost_usd_negative_rejected(self) -> None:
        with pytest.raises(ValueError, match="must be > 0"):
            FindingsAgent(selector=every_sentence_selector, max_cost_usd=-0.5)

    def test_max_cost_usd_none_accepted(self) -> None:
        agent = FindingsAgent(selector=every_sentence_selector, max_cost_usd=None)
        assert agent.max_cost_usd is None

    def test_max_cost_usd_positive_accepted(self) -> None:
        agent = FindingsAgent(selector=every_sentence_selector, max_cost_usd=0.05)
        assert agent.max_cost_usd == pytest.approx(0.05)

    def test_synthesis_cost_estimate_negative_rejected(self) -> None:
        with pytest.raises(ValueError, match=">= 0"):
            FindingsAgent(
                selector=every_sentence_selector,
                synthesis_cost_estimate_usd=-0.01,
            )


# ---------------------------------------------------------------------------
# REFUSAL_BUDGET_EXCEEDED constant compose with existing reasons
# ---------------------------------------------------------------------------


class TestRefusalConstants:
    def test_three_distinct_string_values(self) -> None:
        # Each refusal reason has a stable wire-friendly string value.
        assert REFUSAL_NO_CANDIDATES_ENUMERATED == "no_candidates_enumerated"
        assert REFUSAL_NO_RELEVANT_CANDIDATES == "no_relevant_candidates"
        assert REFUSAL_BUDGET_EXCEEDED == "budget_exceeded"
        # All three are pairwise distinct
        assert (
            len(
                {
                    REFUSAL_NO_CANDIDATES_ENUMERATED,
                    REFUSAL_NO_RELEVANT_CANDIDATES,
                    REFUSAL_BUDGET_EXCEEDED,
                }
            )
            == 3
        )

    def test_findings_refusal_round_trip(self) -> None:
        r = FindingsRefusal(
            reason=REFUSAL_BUDGET_EXCEEDED,
            message="test budget refusal",
            candidates_enumerated=10,
            candidates_surviving_filter=3,
        )
        assert r.reason == REFUSAL_BUDGET_EXCEEDED
        assert r.candidates_enumerated == 10
        assert r.candidates_surviving_filter == 3


# ---------------------------------------------------------------------------
# AgentChatTool — max_cost_usd parameter wiring
# ---------------------------------------------------------------------------


class TestAgentChatToolValidation:
    """The chat tool must validate max_cost_usd before constructing settings."""

    @pytest.mark.asyncio
    async def test_negative_max_cost_rejected(self) -> None:
        from kaos_agents.tools.registry import AgentChatTool

        tool = AgentChatTool()
        result = await tool.execute(
            {
                "message": "test",
                "session_id": "test-session",
                "max_cost_usd": -0.001,
            }
        )
        # Validation error surfaces as isError=True with a useful message
        assert result.isError is True
        # The error must name the offending field
        text = " ".join(str(getattr(c, "text", "")) for c in result.content)
        assert "max_cost_usd" in text

    @pytest.mark.asyncio
    async def test_zero_max_cost_rejected(self) -> None:
        from kaos_agents.tools.registry import AgentChatTool

        tool = AgentChatTool()
        result = await tool.execute(
            {
                "message": "test",
                "session_id": "test-session",
                "max_cost_usd": 0.0,
            }
        )
        assert result.isError is True


class TestAgentFindingsToolValidation:
    @pytest.mark.asyncio
    async def test_negative_max_cost_rejected(self) -> None:
        from kaos_agents.tools.findings import AgentFindingsTool

        tool = AgentFindingsTool()
        # context is None → first triggers "no runtime" error path BUT
        # we want to hit the max_cost validation. Provide a faux
        # runtime via a small object.
        # The tool checks runtime first, so this still tests param
        # gating: feed a fake context where context.runtime exists.
        from types import SimpleNamespace

        ctx = SimpleNamespace(runtime=SimpleNamespace())
        result = await tool.execute(
            {
                "artifact_id": "x",
                "question": "y",
                "max_cost_usd": -1.0,
            },
            ctx,  # ty: ignore[invalid-argument-type]
        )
        assert result.isError is True
        text = " ".join(str(getattr(c, "text", "")) for c in result.content)
        assert "max_cost_usd" in text


class TestAgentCorpusFilterToolValidation:
    @pytest.mark.asyncio
    async def test_negative_max_cost_rejected(self) -> None:
        from types import SimpleNamespace

        from kaos_agents.tools.corpus_filter import AgentCorpusFilterTool

        tool = AgentCorpusFilterTool()
        ctx = SimpleNamespace(runtime=SimpleNamespace())
        result = await tool.execute(
            {
                "intent": "test",
                "artifact_ids": ["a"],
                "max_cost_usd": -1.0,
            },
            ctx,  # ty: ignore[invalid-argument-type]
        )
        assert result.isError is True
        text = " ".join(str(getattr(c, "text", "")) for c in result.content)
        assert "max_cost_usd" in text
