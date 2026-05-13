"""Unit tests for kaos_agents.runtime.invariants."""

from __future__ import annotations

import pytest

from kaos_agents.runtime.invariants import (
    StepInvariant,
    check_invariants,
    first_hard_violation,
    render_violations_for_retry,
)

# ---------------------------------------------------------------------------
# StepInvariant — construction + invariants
# ---------------------------------------------------------------------------


class TestStepInvariantConstruction:
    def test_empty_name_rejected(self) -> None:
        with pytest.raises(ValueError, match="name must be non-empty"):
            StepInvariant(name="", predicate=lambda _: True, message_on_violation="x")

    def test_whitespace_name_rejected(self) -> None:
        with pytest.raises(ValueError, match="name must be non-empty"):
            StepInvariant(name="   ", predicate=lambda _: True, message_on_violation="x")

    def test_empty_message_rejected(self) -> None:
        with pytest.raises(ValueError, match="message_on_violation"):
            StepInvariant(name="x", predicate=lambda _: True, message_on_violation="")

    def test_unknown_severity_rejected(self) -> None:
        with pytest.raises(ValueError, match="severity"):
            StepInvariant(
                name="x",
                predicate=lambda _: True,
                message_on_violation="m",
                severity="wat",
            )

    def test_default_severity_is_hard(self) -> None:
        inv = StepInvariant(name="x", predicate=lambda _: True, message_on_violation="m")
        assert inv.severity == "hard"

    def test_frozen(self) -> None:
        inv = StepInvariant(name="x", predicate=lambda _: True, message_on_violation="m")
        with pytest.raises((AttributeError, TypeError)):
            inv.name = "y"  # ty: ignore[invalid-assignment]


# ---------------------------------------------------------------------------
# evaluate — happy path + error path
# ---------------------------------------------------------------------------


class TestEvaluate:
    def test_passing_predicate_returns_passed(self) -> None:
        inv = StepInvariant(
            name="non_empty",
            predicate=lambda o: bool(o),
            message_on_violation="output must be truthy",
        )
        result = inv.evaluate("hello")
        assert result.passed
        assert result.error is None

    def test_failing_predicate_returns_not_passed(self) -> None:
        inv = StepInvariant(
            name="non_empty",
            predicate=lambda o: bool(o),
            message_on_violation="output must be truthy",
        )
        result = inv.evaluate("")
        assert not result.passed
        assert result.error is None

    def test_raising_predicate_does_not_propagate(self) -> None:
        def broken(_: object) -> bool:
            raise RuntimeError("oops")

        inv = StepInvariant(
            name="broken_predicate",
            predicate=broken,
            message_on_violation="x",
        )
        result = inv.evaluate("anything")
        assert not result.passed
        assert result.error is not None
        assert "RuntimeError" in result.error
        assert "oops" in result.error

    def test_non_bool_truthy_return_handled(self) -> None:
        # Predicate returns an int; the bool() coercion makes 1 → True, 0 → False.
        inv = StepInvariant(
            name="non_empty_str",
            predicate=lambda o: bool(len(str(o))),
            message_on_violation="x",
        )
        assert inv.evaluate("hi").passed
        assert not inv.evaluate("").passed


# ---------------------------------------------------------------------------
# InvariantResult.message
# ---------------------------------------------------------------------------


class TestResultMessage:
    def test_passing_result_has_empty_message(self) -> None:
        inv = StepInvariant(name="x", predicate=lambda _: True, message_on_violation="m")
        assert inv.evaluate("ok").message == ""

    def test_failing_result_uses_violation_message(self) -> None:
        inv = StepInvariant(name="x", predicate=lambda _: False, message_on_violation="fix it")
        assert "fix it" in inv.evaluate("ignored").message

    def test_predicate_error_appended_to_message(self) -> None:
        def broken(_: object) -> bool:
            raise ValueError("bad input")

        inv = StepInvariant(
            name="x",
            predicate=broken,
            message_on_violation="must satisfy X",
        )
        msg = inv.evaluate("anything").message
        assert "must satisfy X" in msg
        assert "ValueError" in msg
        assert "bad input" in msg


# ---------------------------------------------------------------------------
# check_invariants — list comprehension wrapper
# ---------------------------------------------------------------------------


def _has_findings(s: object) -> bool:
    return "## Findings" in str(s)


def _has_summary(s: object) -> bool:
    return "## Summary" in str(s)


class TestCheckInvariants:
    def test_empty_tuple_returns_empty_list(self) -> None:
        assert check_invariants("anything", ()) == []

    def test_all_pass(self) -> None:
        invs = (
            StepInvariant(name="has_findings", predicate=_has_findings, message_on_violation="x"),
            StepInvariant(name="has_summary", predicate=_has_summary, message_on_violation="x"),
        )
        memo = "## Summary\nfoo\n## Findings\nbar"
        results = check_invariants(memo, invs)
        assert all(r.passed for r in results)

    def test_partial_failure(self) -> None:
        invs = (
            StepInvariant(
                name="has_findings",
                predicate=_has_findings,
                message_on_violation="Add ## Findings",
            ),
            StepInvariant(
                name="has_summary",
                predicate=_has_summary,
                message_on_violation="Add ## Summary",
            ),
        )
        memo_summary_only = "## Summary\nbar"
        results = check_invariants(memo_summary_only, invs)
        assert results[0].passed is False  # has_findings
        assert results[1].passed is True  # has_summary


# ---------------------------------------------------------------------------
# first_hard_violation — runner uses this to decide on retry
# ---------------------------------------------------------------------------


class TestFirstHardViolation:
    def test_none_when_all_pass(self) -> None:
        invs = (StepInvariant(name="ok", predicate=lambda _: True, message_on_violation="x"),)
        results = check_invariants("output", invs)
        assert first_hard_violation(results) is None

    def test_returns_first_hard(self) -> None:
        invs = (
            StepInvariant(
                name="failing_hard",
                predicate=lambda _: False,
                message_on_violation="hard fail",
                severity="hard",
            ),
        )
        results = check_invariants("output", invs)
        violation = first_hard_violation(results)
        assert violation is not None
        assert violation.invariant.name == "failing_hard"

    def test_soft_violation_does_not_count(self) -> None:
        invs = (
            StepInvariant(
                name="failing_soft",
                predicate=lambda _: False,
                message_on_violation="soft warn",
                severity="soft",
            ),
        )
        results = check_invariants("output", invs)
        assert first_hard_violation(results) is None


# ---------------------------------------------------------------------------
# render_violations_for_retry — prompt builder
# ---------------------------------------------------------------------------


class TestRenderViolations:
    def test_empty_when_no_failures(self) -> None:
        invs = (StepInvariant(name="passing", predicate=lambda _: True, message_on_violation="x"),)
        assert render_violations_for_retry(check_invariants("ok", invs)) == ""

    def test_includes_each_failure(self) -> None:
        invs = (
            StepInvariant(
                name="needs_section",
                predicate=lambda _: False,
                message_on_violation="Add ## Findings",
            ),
            StepInvariant(
                name="needs_citation",
                predicate=lambda _: False,
                message_on_violation="Cite at least one source",
            ),
        )
        prompt = render_violations_for_retry(check_invariants("output", invs))
        assert "Add ## Findings" in prompt
        assert "Cite at least one source" in prompt
        assert "[needs_section]" in prompt
        assert "[needs_citation]" in prompt

    def test_soft_marker(self) -> None:
        invs = (
            StepInvariant(
                name="soft_one",
                predicate=lambda _: False,
                message_on_violation="warn",
                severity="soft",
            ),
        )
        prompt = render_violations_for_retry(check_invariants("output", invs))
        assert "(warning)" in prompt


# ---------------------------------------------------------------------------
# Realistic example: the S10 memo invariant
# ---------------------------------------------------------------------------


class TestRealisticS10Example:
    def test_memo_structure_invariant(self) -> None:
        # Simulates the S10 surface test's "memo must have ## Findings"
        # check that today is hand-written in _assert_memo_structure.
        # With invariants, it's a one-liner.
        memo_invariant = StepInvariant(
            name="memo_has_findings_section",
            predicate=lambda output: "## Findings" in str(output),
            message_on_violation=(
                "The memo must contain a '## Findings' section header. Add the section and retry."
            ),
        )
        # Good memo
        good = "## Summary\nfoo\n\n## Findings\nbar"
        assert memo_invariant.evaluate(good).passed
        # Bad memo — missing the section
        bad = "## Summary\nfoo only"
        result = memo_invariant.evaluate(bad)
        assert not result.passed
        assert "Findings" in result.message
