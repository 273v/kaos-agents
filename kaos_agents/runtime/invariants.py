"""Step invariants — declarative output contracts with automatic retry.

A ``StepInvariant`` is a typed contract a step's output must satisfy.
When the predicate fails, the Runner re-prompts the LLM with the
violation message in the prompt and accepts a single retry, then
surfaces an ``InvariantViolation`` event if the retry also fails.

Generic safety primitive — replaces ad-hoc "if the output is too
short, retry" branches strewn across patterns. The S10 surface tests'
"memo must contain ``## Findings``" check is the canonical example:
today it's hand-written into each test's `_assert_memo_structure`;
with this primitive it's declared once on the step.

Usage::

    structure_invariant = StepInvariant(
        name="memo_has_findings_section",
        predicate=lambda output: "## Findings" in str(output),
        message_on_violation=(
            "The memo must contain a '## Findings' section header. "
            "Add the section and retry."
        ),
    )

    runner = Runner(agent, step_invariants=(structure_invariant,))

Invariants are pure functions of the step output. They don't have
side effects, don't issue LLM calls, and don't take additional
context — those concerns belong in the agent's own validation /
retry logic. Keeping invariants pure lets us run them deterministically
across replay + diff.

Composition: multiple invariants are conjunctive — every invariant
must pass for the step to be accepted. Pass a tuple; order doesn't
matter semantically (all run; failure messages accumulate).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class StepInvariant:
    """A declarative output contract.

    Attributes:
        name: Short stable identifier used in
            :class:`InvariantViolation` events and in audit logs.
            Snake-case convention.
        predicate: Pure function ``(output) -> bool``. Returns True
            when the output satisfies the invariant. Must not raise;
            the runner treats any raised exception as a violation
            with the exception message in the violation report.
        message_on_violation: Human-readable / model-readable
            explanation of what was expected. Goes into the prompt
            on retry so the model knows what to fix. Phrase as an
            instruction: "Add the X" rather than "Output lacks X."
        severity: ``"hard"`` (default) means a violation triggers a
            retry then surfaces an ``InvariantViolation`` event;
            ``"soft"`` means the violation is reported as a warning
            but the output is still accepted.
    """

    name: str
    predicate: Callable[[object], bool]
    message_on_violation: str
    severity: str = "hard"

    def __post_init__(self) -> None:
        if not self.name or not self.name.strip():
            raise ValueError("StepInvariant.name must be non-empty.")
        if not self.message_on_violation.strip():
            raise ValueError("StepInvariant.message_on_violation must be non-empty.")
        if self.severity not in ("hard", "soft"):
            raise ValueError(
                f"StepInvariant.severity must be 'hard' or 'soft', got {self.severity!r}."
            )

    def evaluate(self, output: object) -> InvariantResult:
        """Run the predicate on ``output``. Never raises.

        Returns an :class:`InvariantResult` indicating pass/fail
        and (on fail) the appropriate violation message. If the
        predicate itself raises, the result is a failure with the
        exception message appended.
        """
        try:
            ok = bool(self.predicate(output))
        except Exception as exc:
            return InvariantResult(
                invariant=self,
                passed=False,
                error=f"predicate raised {type(exc).__name__}: {exc}",
            )
        return InvariantResult(invariant=self, passed=ok)


@dataclass(frozen=True, slots=True)
class InvariantResult:
    """Outcome of running a single :class:`StepInvariant` against an output."""

    invariant: StepInvariant
    passed: bool
    error: str | None = None

    @property
    def message(self) -> str:
        """Violation report — message + (optional) predicate error."""
        if self.passed:
            return ""
        if self.error:
            return f"{self.invariant.message_on_violation} (predicate error: {self.error})"
        return self.invariant.message_on_violation


def check_invariants(
    output: object, invariants: tuple[StepInvariant, ...]
) -> list[InvariantResult]:
    """Run every invariant against ``output``. Returns all results.

    Callers typically filter on ``.passed`` to find violations.
    Empty tuple → empty list (the no-invariants case).
    """
    return [inv.evaluate(output) for inv in invariants]


def first_hard_violation(
    results: list[InvariantResult],
) -> InvariantResult | None:
    """Return the first hard-severity failure, or None.

    Used by the Runner to decide whether to retry: a hard violation
    triggers retry; soft violations are reported but don't block.
    """
    for r in results:
        if not r.passed and r.invariant.severity == "hard":
            return r
    return None


def render_violations_for_retry(results: list[InvariantResult]) -> str:
    """Build a single retry-prompt string from a list of violations.

    Includes only failures. Used by the Runner to append to the
    prompt on retry so the model knows what to fix.

    Returns the empty string when nothing failed (caller can
    unconditionally append).
    """
    failures = [r for r in results if not r.passed]
    if not failures:
        return ""
    lines = ["The previous output violated the following requirements:"]
    for r in failures:
        marker = "" if r.invariant.severity == "hard" else " (warning)"
        lines.append(f"- [{r.invariant.name}]{marker}: {r.message}")
    lines.append("")
    lines.append("Produce a new output that satisfies every requirement.")
    return "\n".join(lines)


__all__ = [
    "InvariantResult",
    "StepInvariant",
    "check_invariants",
    "first_hard_violation",
    "render_violations_for_retry",
]
