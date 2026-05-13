"""Structural diff of two :class:`RecordedRun` instances.

The diff answers "what changed between these two captured runs?" —
useful for regression testing when one is a baseline and the other
is the new behavior after a code change.

Three layers of comparison, surfaced as fields on :class:`RunDiff`:

1. **Summary-level** — event-count delta, turn-count delta, total-cost
   delta. Quick "did anything change?" signal.
2. **Histogram-level** — per-event-type delta. Catches "this run
   emitted 3 TextDeltas where the baseline had 8" without needing
   to walk the event sequence.
3. **Sequence-level** — first divergence point (event index where
   the two sequences first disagree on event type). Useful for
   bisection.

Final-answer text comparison is exposed separately as ``text_diff``
so callers can opt into it; it's not part of the "are they equal"
question because identical event sequences with byte-for-byte
identical TurnSummary.text round-trip to equal text_diff but
inevitably differ when prose generation has any randomness.
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from kaos_agents.base.event import KaosEvent
    from kaos_agents.replay.recorder import RecordedRun


@dataclass(frozen=True, slots=True)
class EventDelta:
    """One row of a per-type histogram delta."""

    event_type: str
    """Stable event-type discriminator (e.g. ``"text_delta"``, ``"span"``)."""

    baseline_count: int
    candidate_count: int

    @property
    def delta(self) -> int:
        return self.candidate_count - self.baseline_count


@dataclass(frozen=True, slots=True)
class RunDiff:
    """Result of comparing two recorded runs.

    Equality is summary-level: ``RunDiff.is_equivalent`` is True when
    event count, turn count, and per-type histogram all match exactly.
    Two runs may have ``is_equivalent=True`` but different prose text —
    inspect ``text_diff`` if you care about byte-level wording.

    Attributes:
        baseline: The reference run (typically the known-good capture).
        candidate: The new run under comparison.
        event_count_delta: ``candidate.summary.event_count - baseline.summary.event_count``.
        turn_count_delta: same for turns.
        cost_delta_usd: same for total cost.
        type_deltas: One :class:`EventDelta` per event type that appears
            in either run. Ordered by absolute delta magnitude, descending.
        first_divergence_index: Index of the first event where the
            type discriminator differs between baseline and candidate.
            ``None`` when sequences match up to ``min(len(a), len(b))``.
        text_diff: Unified-diff string of the final-answer texts.
            Empty string when both runs produced the same text or
            when neither produced any TurnSummary.
        is_equivalent: Quick "no relevant change" signal.
    """

    baseline: RecordedRun
    candidate: RecordedRun
    event_count_delta: int
    turn_count_delta: int
    cost_delta_usd: float
    type_deltas: tuple[EventDelta, ...]
    first_divergence_index: int | None
    text_diff: str
    is_equivalent: bool = field(default=False)

    def to_summary(self) -> str:
        """Render a one-paragraph human summary of the diff."""
        bl = self.baseline.label or "baseline"
        cd = self.candidate.label or "candidate"
        lines = [f"RunDiff: {bl} vs {cd}"]
        lines.append(
            f"  events: {self.baseline.summary.event_count} → "
            f"{self.candidate.summary.event_count} ({self.event_count_delta:+d})"
        )
        lines.append(
            f"  turns: {self.baseline.summary.turn_count} → "
            f"{self.candidate.summary.turn_count} ({self.turn_count_delta:+d})"
        )
        lines.append(
            f"  cost: ${self.baseline.summary.total_cost_usd:.4f} → "
            f"${self.candidate.summary.total_cost_usd:.4f} "
            f"({self.cost_delta_usd:+.4f})"
        )
        if self.first_divergence_index is not None:
            lines.append(f"  first divergence at event #{self.first_divergence_index}")
        nonzero_types = [d for d in self.type_deltas if d.delta != 0]
        if nonzero_types:
            lines.append("  type changes:")
            for d in nonzero_types[:10]:
                lines.append(
                    f"    {d.event_type}: {d.baseline_count} → {d.candidate_count} ({d.delta:+d})"
                )
        lines.append(f"  equivalent: {self.is_equivalent}")
        return "\n".join(lines)


def diff_runs(
    baseline: RecordedRun,
    candidate: RecordedRun,
    *,
    include_text_diff: bool = True,
) -> RunDiff:
    """Compare two recorded runs structurally.

    Args:
        baseline: The reference run.
        candidate: The new run.
        include_text_diff: When True (default), compute a unified
            text diff of the final-answer text. Set False on very
            long answers where the diff cost outweighs the value.

    Returns:
        A frozen :class:`RunDiff`.
    """
    bs = baseline.summary
    cs = candidate.summary

    # Summary-level deltas
    event_delta = cs.event_count - bs.event_count
    turn_delta = cs.turn_count - bs.turn_count
    cost_delta = cs.total_cost_usd - bs.total_cost_usd

    # Histogram deltas — union of all types in either run
    all_types: set[str] = set(bs.type_histogram) | set(cs.type_histogram)
    type_deltas = tuple(
        sorted(
            (
                EventDelta(
                    event_type=t,
                    baseline_count=bs.type_histogram.get(t, 0),
                    candidate_count=cs.type_histogram.get(t, 0),
                )
                for t in all_types
            ),
            key=lambda d: abs(d.delta),
            reverse=True,
        )
    )

    # Sequence-level: first divergence by event type
    first_div = _first_divergence(baseline.events, candidate.events)

    # Text diff (opt-out)
    text_diff = _text_diff(bs.final_answer_text, cs.final_answer_text) if include_text_diff else ""

    is_equivalent = (
        event_delta == 0
        and turn_delta == 0
        and first_div is None
        and all(d.delta == 0 for d in type_deltas)
    )

    return RunDiff(
        baseline=baseline,
        candidate=candidate,
        event_count_delta=event_delta,
        turn_count_delta=turn_delta,
        cost_delta_usd=cost_delta,
        type_deltas=type_deltas,
        first_divergence_index=first_div,
        text_diff=text_diff,
        is_equivalent=is_equivalent,
    )


def _first_divergence(a: tuple[KaosEvent, ...], b: tuple[KaosEvent, ...]) -> int | None:
    """Return the index of the first event whose type differs, or None.

    Compares by the registered event-type string (stable across
    serde round-trips). A length mismatch where one sequence is a
    prefix of the other returns the index just past the shorter one's
    end.
    """
    for i, (ea, eb) in enumerate(zip(a, b, strict=False)):
        try:
            type_a = type(ea).event_type()
        except AttributeError:
            type_a = ea.__class__.__name__.lower()
        try:
            type_b = type(eb).event_type()
        except AttributeError:
            type_b = eb.__class__.__name__.lower()
        if type_a != type_b:
            return i
    # No divergence within the shared prefix
    if len(a) != len(b):
        return min(len(a), len(b))
    return None


def _text_diff(a: str, b: str) -> str:
    """Unified diff of two final-answer texts. Empty string if equal."""
    if a == b:
        return ""
    diff_lines = difflib.unified_diff(
        a.splitlines(keepends=False),
        b.splitlines(keepends=False),
        fromfile="baseline",
        tofile="candidate",
        lineterm="",
        n=3,
    )
    return "\n".join(diff_lines)


__all__ = [
    "EventDelta",
    "RunDiff",
    "diff_runs",
]
