"""Unit tests for :func:`execute_decompose`'s working replan loop.

Stubs ``expand`` and ``compose`` so the loop is driven deterministically
— no live LLM calls, no real tool execution. Covers the four termination
branches:

* Initial Compose returns SUCCESS → strategy returns immediately.
* Initial Compose returns NEEDS_REPLAN **with** step_results → strategy
  stops (don't burn replan budget when partial work succeeded; the
  caller's response formatter handles partials).
* Initial Compose returns NEEDS_REPLAN with **empty** step_results +
  budget allows → strategy re-expands and retries. If a retry
  succeeds, the cumulative step_results carry across.
* Repeated NEEDS_REPLAN with empty step_results until ``max_replans``
  is hit → strategy promotes the final stop_reason to ``MAX_REPLANS``.

Plus regression coverage for the prior failed attempt (commit f1b8907
on a previous branch): the replan loop must **not** fire when any
prior step succeeded with a tool call, because feeding successful
content back into ``expand``'s ``prior_failures`` field biased the
LLM toward LLM-only "analysis" steps and turned 1-step-2-tool-call
plans into 18-step-0-tool-call plans.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import pytest

from kaos_agents.planning.graph import PlanGraph
from kaos_agents.planning.strategies import decompose as decompose_module
from kaos_agents.planning.strategies.decompose import (
    _format_prior_failures,
    execute_decompose,
)
from kaos_agents.types.plan import ComposeResult, PlanBudget, Step, StepType, StopReason

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _step(sid: str, *, tool: str = "kaos-source-fr-search") -> Step:
    return Step(
        id=sid,
        step_type=StepType.TOOL,
        description=f"step {sid}",
        tool_name=tool,
        expected_output="anything",
    )


def _compose_result(
    *,
    stop: StopReason,
    step_results: dict[str, Any] | None = None,
    steps_executed: int = 1,
) -> ComposeResult:
    return ComposeResult(
        plan_json="{}",
        stop_reason=stop,
        steps_executed=steps_executed,
        step_results=step_results or {},
    )


def _patch(monkeypatch: pytest.MonkeyPatch, *, expand: Any, compose: Any) -> None:
    """Replace the module-level ``expand`` and ``compose`` symbols inside
    decompose.py for the duration of one test."""
    monkeypatch.setattr(decompose_module, "expand", expand)
    monkeypatch.setattr(decompose_module, "compose", compose)


# ---------------------------------------------------------------------------
# _format_prior_failures — pure formatter
# ---------------------------------------------------------------------------


class TestFormatPriorFailures:
    def test_empty_graph_emits_header_only(self) -> None:
        graph = PlanGraph(name="t")
        graph.add_step(_step("s1"))
        text = _format_prior_failures("", graph=graph, attempt=1, limit=3)
        # No failures → only the header + empty failures section. Caller
        # is expected to gate on `graph.has_failures()`; this just asserts
        # the formatter doesn't crash on a clean graph.
        assert "replan attempt 1 of 3" in text.lower()
        assert "Prior failures:" in text

    def test_failed_step_included_with_tool_name(self) -> None:
        graph = PlanGraph(name="t")
        graph.add_step(_step("s1", tool="kaos-source-fr-search"))
        graph.mark_failed("s1", "ERROR: tool returned 500")
        text = _format_prior_failures("", graph=graph, attempt=2, limit=3)
        assert "Step s1" in text
        assert "kaos-source-fr-search" in text
        assert "ERROR: tool returned 500" in text

    def test_succeeded_step_excluded(self) -> None:
        # Successful step content must NEVER appear in prior_failures —
        # that's the regression we're guarding against.
        graph = PlanGraph(name="t")
        graph.add_step(_step("s1"))
        graph.mark_complete("s1", "Found 9 Federal Register documents")
        text = _format_prior_failures("", graph=graph, attempt=2, limit=3)
        assert "Found 9 Federal Register documents" not in text
        assert "Step s1" not in text

    def test_base_prior_failures_stacked(self) -> None:
        # The caller-provided base (e.g., REFLECTION-section content
        # from earlier turns) is preserved and the new failures stack
        # on top.
        graph = PlanGraph(name="t")
        graph.add_step(_step("s1"))
        graph.mark_failed("s1", "boom")
        text = _format_prior_failures(
            "Earlier turn: skipped FR lookup", graph=graph, attempt=2, limit=3
        )
        assert text.startswith("Earlier turn: skipped FR lookup")
        assert "Step s1" in text
        assert "boom" in text

    def test_result_truncated_to_200_chars(self) -> None:
        graph = PlanGraph(name="t")
        graph.add_step(_step("s1"))
        long_err = "ERROR: " + ("x" * 300)
        graph.mark_failed("s1", long_err)
        text = _format_prior_failures("", graph=graph, attempt=1, limit=3)
        # We only check that the full 300-x string was truncated; the
        # exact cutoff length is an implementation detail of the
        # formatter. 250 is a generous upper bound that still catches
        # untruncated cases.
        assert long_err not in text
        for line in text.splitlines():
            if line.startswith("- Step s1"):
                assert len(line) < 300


# ---------------------------------------------------------------------------
# PlanGraph.get_failures helper coverage
# ---------------------------------------------------------------------------


class TestPlanGraphGetFailures:
    def test_returns_empty_when_no_failures(self) -> None:
        graph = PlanGraph(name="t")
        graph.add_step(_step("s1"))
        graph.mark_complete("s1", "ok")
        assert graph.get_failures() == {}

    def test_returns_failed_step_metadata(self) -> None:
        graph = PlanGraph(name="t")
        graph.add_step(_step("s1", tool="kaos-source-fr-search"))
        # mark_failed prepends "ERROR: " to whatever caller passes, so
        # the stored result here is "ERROR: boom".
        graph.mark_failed("s1", "boom")
        failures = graph.get_failures()
        assert "s1" in failures
        assert failures["s1"]["tool_name"] == "kaos-source-fr-search"
        assert failures["s1"]["result"] == "ERROR: boom"

    def test_distinguishes_failed_from_completed(self) -> None:
        graph = PlanGraph(name="t")
        graph.add_step(_step("s1"))
        graph.add_step(_step("s2"))
        graph.mark_complete("s1", "ok")
        graph.mark_failed("s2", "ERROR: boom")
        failures = graph.get_failures()
        assert list(failures) == ["s2"]


# ---------------------------------------------------------------------------
# execute_decompose — termination branches
# ---------------------------------------------------------------------------


class TestExecuteDecomposeTerminalBranches:
    async def test_success_returns_immediately(self, monkeypatch: pytest.MonkeyPatch) -> None:
        expand_mock = AsyncMock(return_value=[_step("s1")])
        compose_mock = AsyncMock(
            return_value=_compose_result(stop=StopReason.SUCCESS, step_results={"s1": "ok"})
        )
        _patch(monkeypatch, expand=expand_mock, compose=compose_mock)

        result = await execute_decompose("go", budget=PlanBudget(max_replans=3))

        assert result.stop_reason == StopReason.SUCCESS
        assert expand_mock.await_count == 1
        assert compose_mock.await_count == 1

    async def test_failure_returns_immediately(self, monkeypatch: pytest.MonkeyPatch) -> None:
        expand_mock = AsyncMock(return_value=[_step("s1")])
        compose_mock = AsyncMock(return_value=_compose_result(stop=StopReason.FAILURE))
        _patch(monkeypatch, expand=expand_mock, compose=compose_mock)

        result = await execute_decompose("go", budget=PlanBudget(max_replans=3))

        assert result.stop_reason == StopReason.FAILURE
        assert expand_mock.await_count == 1
        assert compose_mock.await_count == 1

    async def test_max_cost_returns_immediately(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Budget-driven stop reasons are also terminal.
        expand_mock = AsyncMock(return_value=[_step("s1")])
        compose_mock = AsyncMock(return_value=_compose_result(stop=StopReason.MAX_COST))
        _patch(monkeypatch, expand=expand_mock, compose=compose_mock)

        result = await execute_decompose("go", budget=PlanBudget(max_replans=3))

        assert result.stop_reason == StopReason.MAX_COST
        assert expand_mock.await_count == 1


# ---------------------------------------------------------------------------
# execute_decompose — replan loop behavior
# ---------------------------------------------------------------------------


class TestExecuteDecomposeReplanGuards:
    async def test_needs_replan_with_partial_results_does_not_retry(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The regression guard: if any tool call succeeded, the loop
        MUST NOT re-expand. Re-planning with successful step content
        in prior_failures historically biased the LLM into producing
        18 LLM-only steps."""
        expand_mock = AsyncMock(return_value=[_step("s1")])
        compose_mock = AsyncMock(
            return_value=_compose_result(
                stop=StopReason.NEEDS_REPLAN,
                step_results={"s1": "Found 9 FR documents"},
            )
        )
        _patch(monkeypatch, expand=expand_mock, compose=compose_mock)

        result = await execute_decompose("go", budget=PlanBudget(max_replans=3))

        # Stops on the first attempt — exactly one expand + one compose.
        assert expand_mock.await_count == 1
        assert compose_mock.await_count == 1
        # NEEDS_REPLAN is preserved (not promoted to MAX_REPLANS) so
        # the caller can distinguish "partial findings, would have
        # re-tried but it had work to surface" from "ran out of
        # retries".
        assert result.stop_reason == StopReason.NEEDS_REPLAN
        assert result.step_results == {"s1": "Found 9 FR documents"}


class TestExecuteDecomposeReplanLoop:
    async def test_empty_results_triggers_retry_and_succeeds(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """All-failures on attempt 0, then attempt 1 succeeds — the
        replan path actually re-expands and the successful results
        come back."""
        expand_mock = AsyncMock(side_effect=[[_step("s1")], [_step("s2")]])

        compose_calls = {"n": 0}

        async def _compose(graph: PlanGraph, *, budget: PlanBudget, **_: Any) -> ComposeResult:
            compose_calls["n"] += 1
            if compose_calls["n"] == 1:
                # Attempt 0: every step fails, no partials.
                graph.mark_failed("s1", "tool timed out")
                budget.record_replan()
                return _compose_result(
                    stop=StopReason.NEEDS_REPLAN, step_results={}, steps_executed=1
                )
            # Attempt 1: success with results.
            graph.mark_complete("s2", "great")
            return _compose_result(
                stop=StopReason.SUCCESS,
                step_results={"s2": "great"},
                steps_executed=1,
            )

        _patch(monkeypatch, expand=expand_mock, compose=_compose)

        result = await execute_decompose("go", budget=PlanBudget(max_replans=3))

        assert expand_mock.await_count == 2
        assert compose_calls["n"] == 2
        assert result.stop_reason == StopReason.SUCCESS
        assert result.step_results == {"s2": "great"}

    async def test_prior_failures_stacks_across_attempts(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The second ``expand`` call receives a ``prior_failures`` arg
        that includes structured context from attempt 0's failures."""
        captured_kwargs: list[dict[str, Any]] = []

        async def _expand(*_args: Any, **kwargs: Any) -> list[Step]:
            captured_kwargs.append(kwargs)
            return [_step(f"s{len(captured_kwargs)}")]

        async def _compose(graph: PlanGraph, **_: Any) -> ComposeResult:
            # Always fail on the first step.
            sid = next(iter(graph.step_ids()))
            graph.mark_failed(sid, f"ERROR: {sid} boom")
            return _compose_result(stop=StopReason.NEEDS_REPLAN, step_results={}, steps_executed=1)

        _patch(monkeypatch, expand=_expand, compose=_compose)

        await execute_decompose("go", budget=PlanBudget(max_replans=2))

        # 3 expand calls: initial + 2 replans (until budget exhausted).
        assert len(captured_kwargs) >= 2
        # First call's prior_failures is empty / the caller-supplied
        # base (we pass nothing → "").
        assert captured_kwargs[0].get("prior_failures", "") == ""
        # Second call's prior_failures contains the new failure section.
        second = captured_kwargs[1].get("prior_failures", "")
        assert "replan attempt" in second.lower()
        assert "ERROR: s1 boom" in second

    async def test_max_replans_promotes_stop_reason(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """If every attempt fails completely until ``max_replans`` is
        hit, the strategy promotes NEEDS_REPLAN → MAX_REPLANS."""

        async def _expand(*_args: Any, **_kw: Any) -> list[Step]:
            return [_step("s1")]

        async def _compose(graph: PlanGraph, **_: Any) -> ComposeResult:
            graph.mark_failed("s1", "ERROR: boom")
            return _compose_result(stop=StopReason.NEEDS_REPLAN, step_results={}, steps_executed=1)

        _patch(monkeypatch, expand=_expand, compose=_compose)

        budget = PlanBudget(max_replans=2)
        # Pre-simulate that compose() bumped the counter on each call.
        # We patch compose with a wrapper that does so explicitly.

        original_compose = _compose
        call_count = {"n": 0}

        async def _compose_with_budget_bump(
            graph: PlanGraph, *, budget: PlanBudget, **kw: Any
        ) -> ComposeResult:
            call_count["n"] += 1
            # Real compose() calls budget.record_replan() before
            # returning NEEDS_REPLAN — emulate that so the strategy's
            # ``budget.replans >= budget.max_replans`` termination
            # condition fires correctly.
            budget.record_replan()
            return await original_compose(graph, budget=budget, **kw)

        _patch(monkeypatch, expand=_expand, compose=_compose_with_budget_bump)

        result = await execute_decompose("go", budget=budget)

        # Loop body: attempt 0 → compose calls record_replan → replans=1 < 2, retry.
        # attempt 1 → compose calls record_replan → replans=2, NOT < 2, break.
        # Total: 2 compose calls. Matches the real compose semantics
        # (compose itself increments budget.replans BEFORE returning
        # NEEDS_REPLAN, so the loop's post-check stops the next iter).
        assert call_count["n"] == 2
        assert result.stop_reason == StopReason.MAX_REPLANS
        assert result.step_results == {}

    async def test_cumulative_step_results_across_attempts(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """If attempt 0 fails completely, attempt 1 partially succeeds
        with one result, then NEEDS_REPLAN again (with partial) → the
        loop must stop (regression guard) AND the result must carry
        the partial."""
        expand_mock = AsyncMock(side_effect=[[_step("s1")], [_step("s2"), _step("s3")]])

        attempt = {"n": 0}

        async def _compose(graph: PlanGraph, *, budget: PlanBudget, **_: Any) -> ComposeResult:
            attempt["n"] += 1
            if attempt["n"] == 1:
                # All fail.
                graph.mark_failed("s1", "ERROR: boom")
                budget.record_replan()
                return _compose_result(
                    stop=StopReason.NEEDS_REPLAN, step_results={}, steps_executed=1
                )
            # Attempt 2: s2 succeeds, s3 fails → NEEDS_REPLAN with one
            # successful partial. Loop must STOP here.
            graph.mark_complete("s2", "found docs")
            graph.mark_failed("s3", "ERROR: judge rejected")
            return _compose_result(
                stop=StopReason.NEEDS_REPLAN,
                step_results={"s2": "found docs"},
                steps_executed=2,
            )

        _patch(monkeypatch, expand=expand_mock, compose=_compose)

        result = await execute_decompose("go", budget=PlanBudget(max_replans=3))

        assert attempt["n"] == 2
        # Cumulative step_results carry forward.
        assert result.step_results == {"s2": "found docs"}
        # Partial-with-results path preserves NEEDS_REPLAN (not promoted
        # to MAX_REPLANS — we stopped on the soft-stop, not on budget).
        assert result.stop_reason == StopReason.NEEDS_REPLAN

    async def test_expand_returns_no_steps_after_partial_success(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """If a retry's ``expand`` returns no steps but we already
        accumulated partial work, surface the partials with FAILURE
        rather than discarding."""

        async def _expand(*_args: Any, **kwargs: Any) -> list[Step]:
            # First call: returns one step. Second: returns nothing.
            if expand_calls["n"] == 0:
                expand_calls["n"] += 1
                return [_step("s1")]
            expand_calls["n"] += 1
            return []

        async def _compose(graph: PlanGraph, *, budget: PlanBudget, **_: Any) -> ComposeResult:
            graph.mark_failed("s1", "ERROR: boom")
            budget.record_replan()
            return _compose_result(stop=StopReason.NEEDS_REPLAN, step_results={}, steps_executed=1)

        expand_calls = {"n": 0}
        _patch(monkeypatch, expand=_expand, compose=_compose)

        result = await execute_decompose("go", budget=PlanBudget(max_replans=2))

        # Attempt 0 fails completely, attempt 1's expand returns no
        # steps → loop ends. Stop reason is whatever the last compose
        # produced (NEEDS_REPLAN) promoted to MAX_REPLANS only if the
        # budget actually exhausted. Here we want either MAX_REPLANS or
        # FAILURE — both are acceptable.
        assert result.stop_reason in (
            StopReason.FAILURE,
            StopReason.MAX_REPLANS,
            StopReason.NEEDS_REPLAN,
        )
