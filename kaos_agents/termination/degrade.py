"""DegradationPolicy — paper §8.6 graceful degradation.

When a turn cannot fully complete (budget exhausted, quality below
threshold, partial information), the policy decides whether to:

* return a partial result with a degradation note,
* escalate to the user, or
* fail outright.

Phase 4.B baseline: a simple "accept partial when budget exceeded
AND >= ``min_partial_chars`` produced" rule. Quality failures default
to *not* accepted (an unverified bad answer is worse than no answer)
but the policy is overridable per-instance.

The policy never inspects the partial text's contents — that's the
quality axis's job. It only checks size and the kind of failure that
prompted degradation.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class DegradationOutcome:
    """Verdict from :meth:`DegradationPolicy.evaluate`.

    ``accept`` is the headline. When ``True``, ``partial`` is the
    text the AgentLoop should surface to the user (verbatim from the
    input — the policy doesn't rewrite). ``reason`` is a one-line
    explanation suitable for the Decision's ``feedback`` field.
    """

    accept: bool
    partial: str = ""
    reason: str = ""


class DegradationPolicy:
    """Decides whether a partial result is acceptable to surface.

    Constructor kwargs:

      min_partial_chars: floor on partial-text length below which
        degradation is refused (a 5-character "ok" is not a useful
        partial answer). Default 32.
      accept_on_budget_exhaustion: if True, budget-related kinds
        accept the partial when it meets the size floor. Default True
        — paper §8.6 says budget exhaustion is the canonical
        degradation case.
      accept_on_quality_failure: if True, quality-failure kinds
        accept the partial. Default False — an unverified low-quality
        answer is generally worse than escalating.
    """

    def __init__(
        self,
        *,
        min_partial_chars: int = 32,
        accept_on_budget_exhaustion: bool = True,
        accept_on_quality_failure: bool = False,
    ) -> None:
        self._min_chars = int(min_partial_chars)
        self._accept_budget = bool(accept_on_budget_exhaustion)
        self._accept_quality = bool(accept_on_quality_failure)

    @property
    def min_partial_chars(self) -> int:
        return self._min_chars

    def evaluate(
        self,
        *,
        kind: str,
        partial_text: str = "",
    ) -> DegradationOutcome:
        """Return whether ``partial_text`` is acceptable for ``kind``.

        ``kind`` is the string value of a
        :class:`~kaos_agents.termination.types.DecisionKind` (e.g.
        ``"budget_exceeded"`` or ``"quality_failed"``). Unknown kinds
        are refused — the policy is intentionally explicit about what
        it accepts.
        """
        if not partial_text or len(partial_text) < self._min_chars:
            return DegradationOutcome(
                accept=False,
                reason=(
                    f"partial output too short ({len(partial_text)} < {self._min_chars} chars)"
                ),
            )
        if kind == "budget_exceeded" and self._accept_budget:
            return DegradationOutcome(
                accept=True,
                partial=partial_text,
                reason="budget exhausted; accepting partial result",
            )
        if kind == "quality_failed" and self._accept_quality:
            return DegradationOutcome(
                accept=True,
                partial=partial_text,
                reason="quality below threshold; accepting partial result",
            )
        return DegradationOutcome(
            accept=False,
            reason=f"degradation not accepted for kind={kind}",
        )


__all__ = ["DegradationOutcome", "DegradationPolicy"]
