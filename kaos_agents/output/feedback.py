"""Pure feedback formatters — turn DeliverableVerdict into prompt text.

Three strategies, one pure function each. Output is markdown text
suitable for injection into ``MemoryType.REFLECTION`` for the next
producer iteration. No I/O, no LLM calls.

* :func:`format_gap_list` — bulleted list of failed criteria + a
  one-line excerpt of the judge's reasoning.
* :func:`format_gap_narrative` — short paragraph synthesising the
  failures into prose. (LLM-free; templated from the verdict's own
  reasoning.)
* :func:`format_gap_feedback` — strategy dispatcher. Default
  ``"gap_list"`` because it's the most directly actionable shape;
  ``"narrative"`` is for callers that prefer prose.

The feedback text is what the producer sees on its next iteration
under :class:`RefineDeliverable`. The agent's existing recall()
machinery pulls REFLECTION items into the producer's context per
the C4-fixed cost-attribution path.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from kaos_agents.output.critic import DeliverableVerdict

FeedbackStrategy = Literal["gap_list", "narrative", "hybrid"]


_REASONING_PREVIEW_CHARS = 200
"""Per-criterion reasoning excerpt length for gap_list. Long enough to
be useful, short enough to not blow the producer's REFLECTION budget."""


def format_gap_feedback(
    verdict: DeliverableVerdict,
    *,
    strategy: FeedbackStrategy = "gap_list",
) -> str:
    """Render a verdict's failures as feedback text.

    Args:
        verdict: A :class:`DeliverableVerdict` from
            :class:`RubricDeliverableCritic` or any other Critic
            implementation.
        strategy: One of ``"gap_list"`` (default, bulleted),
            ``"narrative"`` (short paragraph), ``"hybrid"`` (list +
            narrative summary). Empty verdicts (all passed) return a
            short "no gaps identified" string regardless of strategy.

    Returns:
        Markdown feedback text. Always non-empty; safe to inject into
        REFLECTION memory.
    """
    if verdict.n_failed == 0 and verdict.n_judge_unavailable == 0:
        return _format_all_passed(verdict)
    if strategy == "narrative":
        return format_gap_narrative(verdict)
    if strategy == "hybrid":
        return format_gap_list(verdict) + "\n\n" + format_gap_narrative(verdict)
    return format_gap_list(verdict)


def format_gap_list(verdict: DeliverableVerdict) -> str:
    """Bulleted list of failed criteria + reasoning excerpts.

    Format::

        # Gaps identified

        - **C-001** (anti-assignment clause): The deliverable does
          not identify Section 14.2 of the Northland MSA.
        - **C-003** (reverse-triangular merger analysis): The
          deliverable does not analyze the assignment-by-operation-
          of-law question.

        Total: 2 of 5 criteria failed.
    """
    lines: list[str] = ["# Gaps identified", ""]
    for cr in verdict.failed_criteria:
        excerpt = cr.reasoning[:_REASONING_PREVIEW_CHARS].rstrip()
        if len(cr.reasoning) > _REASONING_PREVIEW_CHARS:
            excerpt += "..."
        lines.append(f"- **{cr.criterion_id}**: {excerpt}")
    if verdict.unavailable_criteria:
        lines.extend(["", "## Judge-unavailable criteria"])
        for cr in verdict.unavailable_criteria:
            lines.append(f"- {cr.criterion_id}: {cr.reasoning[:_REASONING_PREVIEW_CHARS]}")
    total = len(verdict.per_criterion)
    lines.extend(
        [
            "",
            f"Total: {verdict.n_failed} of {total} criteria failed "
            f"(weighted pass rate: {verdict.weighted_pass_rate:.1%}).",
        ]
    )
    return "\n".join(lines)


def format_gap_narrative(verdict: DeliverableVerdict) -> str:
    """Short paragraph synthesising the failures.

    Templated; no LLM call. Format::

        Two criteria failed: C-001 (anti-assignment clause) and C-003
        (reverse-triangular merger analysis). The deliverable's
        coverage is at 60% (3 of 5). Address the named gaps in the
        next iteration.
    """
    if verdict.n_failed == 0 and verdict.n_judge_unavailable == 0:
        return _format_all_passed(verdict)
    failed_ids = ", ".join(c.criterion_id for c in verdict.failed_criteria) or "(none)"
    unavailable_ids = ", ".join(c.criterion_id for c in verdict.unavailable_criteria) or ""
    pieces: list[str] = []
    if verdict.failed_criteria:
        pieces.append(
            f"{verdict.n_failed} criteria failed: {failed_ids}. "
            f"The deliverable's weighted pass rate is "
            f"{verdict.weighted_pass_rate:.1%}."
        )
    if unavailable_ids:
        pieces.append(
            f"The judge could not render a verdict on {verdict.n_judge_unavailable} "
            f"criteria ({unavailable_ids}); these are likely transient infra "
            "failures, not deliverable gaps."
        )
    pieces.append("Address the named gaps in the next iteration.")
    return " ".join(pieces)


def _format_all_passed(verdict: DeliverableVerdict) -> str:
    """All-clear message; default for verdicts with zero failures."""
    return (
        "No gaps identified. "
        f"All {verdict.n_passed} criteria passed "
        f"(weighted pass rate: {verdict.weighted_pass_rate:.1%})."
    )


__all__ = [
    "FeedbackStrategy",
    "format_gap_feedback",
    "format_gap_list",
    "format_gap_narrative",
]
