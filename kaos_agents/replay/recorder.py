"""Recorder — capture event streams into ``RecordedRun`` artifacts.

The :class:`RecordedRun` value type bundles an ordered tuple of events
plus computed summary metadata (turn count, final answer text, total
cost). Pair with :func:`record_events` to capture a ``Runner.run``
iterator without modifying the runner itself; pair with
:func:`save_run` / :func:`load_run` for JSONL persistence.

All summary fields are computed once at construction so diffing or
inspecting a loaded run doesn't re-walk the event list.
"""

from __future__ import annotations

import contextlib
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from kaos_agents.events.serde import deserialize_event_json, serialize_event_json

if TYPE_CHECKING:
    from kaos_agents.base.event import KaosEvent


@dataclass(frozen=True, slots=True)
class RunSummary:
    """Computed summary of a recorded run — cheap to compute, cheap to diff."""

    event_count: int
    type_histogram: dict[str, int]
    """Map of event-type name → count, e.g. ``{"text_delta": 42, "span": 8}``."""
    total_cost_usd: float
    """Sum of ``cost_usd`` across ``UsageObserved`` events."""
    final_answer_text: str
    """Concatenated text from ``TurnSummary`` events (last-write wins
    when multiple turns; empty string if no TurnSummary emitted)."""
    turn_count: int
    """Number of ``TurnSummary`` events seen."""


@dataclass(frozen=True, slots=True)
class RecordedRun:
    """A captured event stream + computed summary.

    Frozen — the events tuple is immutable. ``summary`` is computed
    once at construction. Round-trips through :func:`save_run` and
    :func:`load_run` losslessly.

    Attributes:
        events: Ordered tuple of every event emitted during the run.
        summary: Pre-computed summary metrics.
        session_id: Optional identifier (typically the agent session id).
            None for ad-hoc captures.
        label: Optional human-readable name. Surfaces in diff reports.
    """

    events: tuple[KaosEvent, ...]
    summary: RunSummary
    session_id: str | None = None
    label: str | None = None


def _build_summary(events: tuple[KaosEvent, ...]) -> RunSummary:
    """Compute the :class:`RunSummary` from an event list. Pure function."""
    histogram: dict[str, int] = {}
    total_cost = 0.0
    answer_parts: list[str] = []
    turn_count = 0

    for event in events:
        # Type histogram — use the registered serde type string (the
        # snake_case wire format) for stability across round-trips.
        try:
            type_name = type(event).event_type()
        except AttributeError:
            type_name = event.__class__.__name__.lower()
        histogram[str(type_name)] = histogram.get(str(type_name), 0) + 1

        # Cost accumulation — UsageObserved events carry it.
        cost = getattr(event, "cost_usd", None)
        if cost is not None:
            with contextlib.suppress(TypeError, ValueError):
                total_cost += float(cost)

        # Turn summary detection
        cls = event.__class__.__name__
        if cls == "TurnSummary":
            turn_count += 1
            text = getattr(event, "text", "")
            if text:
                answer_parts.append(str(text))

    return RunSummary(
        event_count=len(events),
        type_histogram=histogram,
        total_cost_usd=total_cost,
        final_answer_text="\n".join(answer_parts),
        turn_count=turn_count,
    )


async def record_events(
    events: AsyncIterator[KaosEvent],
    *,
    session_id: str | None = None,
    label: str | None = None,
) -> RecordedRun:
    """Drain an event iterator into a :class:`RecordedRun`.

    Usage::

        captured = await record_events(runner.run(message, session_id))
        # captured is a frozen RecordedRun; save_run(captured, path) to persist.

    Args:
        events: The ``Runner.run`` iterator (or any async iterator
            of ``KaosEvent``).
        session_id: Optional identifier for the captured run.
        label: Optional human-readable name.

    Returns:
        A :class:`RecordedRun` carrying the full event list + summary.
    """
    collected: list[KaosEvent] = []
    async for event in events:
        collected.append(event)
    events_tuple = tuple(collected)
    return RecordedRun(
        events=events_tuple,
        summary=_build_summary(events_tuple),
        session_id=session_id,
        label=label,
    )


def save_run(run: RecordedRun, path: Path | str) -> None:
    """Persist a :class:`RecordedRun` to a JSONL file.

    Each line is one event in the same format as
    :func:`~kaos_agents.events.serde.serialize_event_json`. The first
    line is a metadata header (``{"_kind": "header", ...}``) so
    :func:`load_run` can recover ``session_id`` / ``label``.

    The header line is identified by its ``_kind`` discriminator and
    is NOT a serialized event — :func:`load_run` filters it before
    deserializing the remaining lines.
    """
    import json

    p = Path(path)
    with p.open("w", encoding="utf-8") as fh:
        header = {
            "_kind": "header",
            "session_id": run.session_id,
            "label": run.label,
            "event_count": run.summary.event_count,
        }
        fh.write(json.dumps(header, separators=(",", ":")))
        fh.write("\n")
        for event in run.events:
            fh.write(serialize_event_json(event))
            fh.write("\n")


def load_run(path: Path | str) -> RecordedRun:
    """Load a :class:`RecordedRun` from a JSONL file saved by :func:`save_run`.

    Skips the first-line header and deserializes every remaining line
    into a typed event via the registry-backed
    :func:`~kaos_agents.events.serde.deserialize_event_json`.
    """
    import json

    p = Path(path)
    session_id: str | None = None
    label: str | None = None
    events: list[KaosEvent] = []

    with p.open("r", encoding="utf-8") as fh:
        first_line = True
        for raw in fh:
            raw = raw.strip()
            if not raw:
                continue
            if first_line:
                first_line = False
                try:
                    obj = json.loads(raw)
                    if obj.get("_kind") == "header":
                        session_id = obj.get("session_id")
                        label = obj.get("label")
                        continue
                    # Backward compat: no header → treat as event.
                except (ValueError, TypeError):
                    pass
            try:
                events.append(deserialize_event_json(raw))
            except Exception:
                # Malformed line — skip with a debug log. Don't blow
                # up the whole load; partial replay is better than none.
                continue

    events_tuple = tuple(events)
    return RecordedRun(
        events=events_tuple,
        summary=_build_summary(events_tuple),
        session_id=session_id,
        label=label,
    )


__all__ = [
    "RecordedRun",
    "RunSummary",
    "load_run",
    "record_events",
    "save_run",
]
