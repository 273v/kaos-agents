"""Tests for ``_format_plan_response`` in ``patterns/plan_execute.py``.

Covers the four formatter branches that decide what the SPA actually
displays at the end of a plan-execute turn:

1. SUCCESS + non-empty step_results → synthesized findings.
2. SUCCESS + empty step_results → "no results were produced" notice.
3. non-SUCCESS + non-empty step_results → "stopped early ... partial
   findings" banner + synthesized findings. This is the branch the
   R1-REAL v2 matrix Tests 3 & 7 (long-horizon FR/EDGAR plans) regressed
   on — pre-fix it emitted "Plan execution stopped: needs_replan.
   Completed N steps." with zero findings.
4. non-SUCCESS + empty step_results → bare "Plan execution stopped"
   status (no-progress case).
"""

from __future__ import annotations

from kaos_agents.patterns.plan_execute import _format_plan_response
from kaos_agents.types.plan import ComposeResult, StopReason


def _result(
    stop_reason: StopReason,
    *,
    step_results: dict[str, str] | None = None,
    steps_executed: int = 0,
) -> ComposeResult:
    return ComposeResult(
        plan_json="{}",
        stop_reason=stop_reason,
        steps_executed=steps_executed,
        step_results=step_results or {},
    )


class TestFormatPlanResponseSuccess:
    def test_success_with_results_synthesizes_findings(self) -> None:
        result = _result(
            StopReason.SUCCESS,
            step_results={"step-1": "Found 9 Federal Register documents"},
            steps_executed=1,
        )

        text = _format_plan_response(result)

        assert "Plan completed with the following results" in text
        assert "step-1" in text
        assert "Found 9 Federal Register documents" in text
        # Success branch must not prepend the "stopped early" banner.
        assert "stopped early" not in text

    def test_success_with_empty_results_emits_terse_notice(self) -> None:
        result = _result(StopReason.SUCCESS, step_results={}, steps_executed=0)

        text = _format_plan_response(result)

        assert text == "Plan completed but no results were produced."


class TestFormatPlanResponseStoppedEarlyWithPartials:
    """The R1-REAL regression branch — most important to preserve."""

    def test_needs_replan_with_partials_emits_banner_and_findings(self) -> None:
        result = _result(
            StopReason.NEEDS_REPLAN,
            step_results={
                "step-1-2fcdae": (
                    'Found 9 Federal Register document(s), showing 9. {"results": '
                    '[{"document_number": "2024-30494", "title": "EDGAR Filer '
                    'Access and Account Management"}]}'
                ),
                "step-2-5d7ab4": "EDGAR Filer Access and Account Management (Rule)",
            },
            steps_executed=3,
        )

        text = _format_plan_response(result)

        # Banner with the stop reason + step count.
        assert "_Plan stopped early (reason: needs_replan, 3 step(s) completed)" in text
        assert "Partial findings:" in text
        # And the actual findings the agent produced — the regression
        # we're guarding against silently dropped these.
        assert "step-1-2fcdae" in text
        assert "EDGAR Filer Access" in text
        assert "step-2-5d7ab4" in text

    def test_max_replans_with_partials_emits_banner_and_findings(self) -> None:
        result = _result(
            StopReason.MAX_REPLANS,
            step_results={"step-1": "Found 10 SEC cybersecurity rules from 2024"},
            steps_executed=2,
        )

        text = _format_plan_response(result)

        assert "_Plan stopped early (reason: max_replans, 2 step(s) completed)" in text
        assert "Found 10 SEC cybersecurity rules" in text

    def test_max_cost_with_partials_emits_banner_and_findings(self) -> None:
        result = _result(
            StopReason.MAX_COST,
            step_results={"step-1": "First batch of findings"},
            steps_executed=1,
        )

        text = _format_plan_response(result)

        assert "_Plan stopped early (reason: max_cost, 1 step(s) completed)" in text
        assert "First batch of findings" in text

    def test_failure_with_partials_still_emits_banner_and_findings(self) -> None:
        # Even on hard FAILURE, if the agent got some work done before
        # bailing, the user should see it.
        result = _result(
            StopReason.FAILURE,
            step_results={"step-1": "Partial result from step 1 before failure"},
            steps_executed=1,
        )

        text = _format_plan_response(result)

        assert "_Plan stopped early (reason: failure, 1 step(s) completed)" in text
        assert "Partial result from step 1 before failure" in text


class TestFormatPlanResponseStoppedEarlyEmpty:
    def test_needs_replan_with_no_results_emits_bare_status(self) -> None:
        result = _result(StopReason.NEEDS_REPLAN, step_results={}, steps_executed=0)

        text = _format_plan_response(result)

        assert text == "Plan execution stopped: needs_replan. Completed 0 steps."
        # Bare branch must NOT carry the partial-findings banner.
        assert "Partial findings" not in text

    def test_failure_with_no_results_emits_bare_status(self) -> None:
        result = _result(StopReason.FAILURE, step_results={}, steps_executed=0)

        text = _format_plan_response(result)

        assert text == "Plan execution stopped: failure. Completed 0 steps."

    def test_max_steps_with_no_results_emits_bare_status(self) -> None:
        result = _result(StopReason.MAX_STEPS, step_results={}, steps_executed=20)

        text = _format_plan_response(result)

        assert text == "Plan execution stopped: max_steps. Completed 20 steps."


class TestFormatPlanResponseSkipsErrorOnlyResults:
    """When all step_results are error strings, the synthesizer should
    fall back to its "results were empty or all errored" path rather
    than dumping ERROR: blobs to the user."""

    def test_success_with_only_error_results(self) -> None:
        result = _result(
            StopReason.SUCCESS,
            step_results={"step-1": "ERROR: tool timed out"},
            steps_executed=1,
        )

        text = _format_plan_response(result)

        assert text == "Plan completed but results were empty or all errored."

    def test_stopped_early_with_only_error_results(self) -> None:
        result = _result(
            StopReason.NEEDS_REPLAN,
            step_results={"step-1": "ERROR: tool returned 500"},
            steps_executed=1,
        )

        text = _format_plan_response(result)

        # Banner still emits so the user sees why we stopped.
        assert "_Plan stopped early (reason: needs_replan, 1 step(s) completed)" in text
        # But the synthesizer correctly skips the error blob.
        assert "Plan completed but results were empty or all errored." in text
        assert "ERROR:" not in text
