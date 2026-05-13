"""Unit tests for kaos_agents.replay (recorder + diff + replayer)."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from kaos_agents.events import (
    SpanSubject,
    TextDelta,
    TurnSummary,
    UsageObserved,
)
from kaos_agents.events.emitter import EventEmitter
from kaos_agents.replay import (
    RecordedRun,
    diff_runs,
    load_run,
    record_events,
    replay_run,
    save_run,
)

# ---------------------------------------------------------------------------
# Helpers — synthetic event streams
# ---------------------------------------------------------------------------


async def _events_from(events: list) -> AsyncIterator:
    for e in events:
        yield e


def _baseline_stream() -> list:
    """Canonical 6-event stream built with EventEmitter."""
    em = EventEmitter(session_id="sess-1", run_id="run-1")
    span_start = em.span_start(SpanSubject.TURN, name="turn.1", attributes={"turn_number": 1})
    delta_1 = em.emit(TextDelta, content="hello")
    delta_2 = em.emit(TextDelta, content=" world")
    usage = em.emit(
        UsageObserved,
        input_tokens=10,
        output_tokens=5,
        total_tokens=15,
        cost_usd=0.001,
    )
    span_complete = em.span_complete(
        SpanSubject.TURN, span_id=span_start.span_id, duration_ms=100.0
    )
    summary = em.emit(
        TurnSummary,
        text="hello world",
        turn_number=1,
        tokens_used=15,
        cost_usd=0.001,
    )
    return [span_start, delta_1, delta_2, usage, span_complete, summary]


def _async_collect(stream: AsyncIterator) -> list:
    async def _go() -> list:
        return [x async for x in stream]

    return asyncio.run(_go())


# ---------------------------------------------------------------------------
# record_events + RecordedRun
# ---------------------------------------------------------------------------


class TestRecordEvents:
    def test_captures_every_event(self) -> None:
        events = _baseline_stream()
        run = asyncio.run(record_events(_events_from(events)))
        assert run.summary.event_count == len(events)

    def test_summary_counts_turns(self) -> None:
        events = _baseline_stream()
        run = asyncio.run(record_events(_events_from(events)))
        assert run.summary.turn_count == 1

    def test_summary_final_answer_text(self) -> None:
        events = _baseline_stream()
        run = asyncio.run(record_events(_events_from(events)))
        assert run.summary.final_answer_text == "hello world"

    def test_session_id_and_label_preserved(self) -> None:
        run = asyncio.run(
            record_events(
                _events_from(_baseline_stream()),
                session_id="sess-1",
                label="baseline-run",
            )
        )
        assert run.session_id == "sess-1"
        assert run.label == "baseline-run"

    def test_empty_stream(self) -> None:
        run = asyncio.run(record_events(_events_from([])))
        assert run.summary.event_count == 0
        assert run.summary.turn_count == 0
        assert run.summary.final_answer_text == ""

    def test_recorded_run_is_frozen(self) -> None:
        run = asyncio.run(record_events(_events_from(_baseline_stream())))
        with pytest.raises((AttributeError, TypeError)):
            run.events = ()  # ty: ignore[invalid-assignment]


# ---------------------------------------------------------------------------
# save_run + load_run round-trip
# ---------------------------------------------------------------------------


class TestSaveLoadRoundTrip:
    def test_basic_roundtrip(self, tmp_path: Path) -> None:
        original = asyncio.run(
            record_events(
                _events_from(_baseline_stream()),
                session_id="sess-1",
                label="baseline",
            )
        )
        path = tmp_path / "run.jsonl"
        save_run(original, path)
        loaded = load_run(path)

        assert loaded.summary.event_count == original.summary.event_count
        assert loaded.summary.turn_count == original.summary.turn_count
        assert loaded.summary.final_answer_text == original.summary.final_answer_text
        assert loaded.session_id == "sess-1"
        assert loaded.label == "baseline"

    def test_empty_run_roundtrip(self, tmp_path: Path) -> None:
        original = asyncio.run(record_events(_events_from([])))
        path = tmp_path / "empty.jsonl"
        save_run(original, path)
        loaded = load_run(path)
        assert loaded.summary.event_count == 0

    def test_file_format_is_jsonl(self, tmp_path: Path) -> None:
        run = asyncio.run(record_events(_events_from(_baseline_stream())))
        path = tmp_path / "format.jsonl"
        save_run(run, path)
        content = path.read_text()
        import json

        lines = [line for line in content.splitlines() if line.strip()]
        assert len(lines) == 1 + run.summary.event_count  # header + events
        for line in lines:
            json.loads(line)


# ---------------------------------------------------------------------------
# diff_runs
# ---------------------------------------------------------------------------


class TestDiffRuns:
    def test_identical_runs_equivalent(self) -> None:
        baseline = asyncio.run(record_events(_events_from(_baseline_stream())))
        candidate = asyncio.run(record_events(_events_from(_baseline_stream())))
        diff = diff_runs(baseline, candidate)
        assert diff.is_equivalent
        assert diff.event_count_delta == 0
        assert diff.turn_count_delta == 0
        assert diff.cost_delta_usd == pytest.approx(0.0)
        assert diff.first_divergence_index is None
        assert diff.text_diff == ""

    def test_extra_event_in_candidate(self) -> None:
        baseline = asyncio.run(record_events(_events_from(_baseline_stream())))
        em = EventEmitter(session_id="sess-1", run_id="run-1")
        extra = em.emit(TextDelta, content="extra")
        candidate = asyncio.run(record_events(_events_from([*_baseline_stream(), extra])))
        diff = diff_runs(baseline, candidate)
        assert not diff.is_equivalent
        assert diff.event_count_delta == 1
        # The TextDelta histogram should show +1
        td_delta = next(d for d in diff.type_deltas if d.event_type == "text_delta")
        assert td_delta.delta == 1

    def test_type_divergence_at_index(self) -> None:
        baseline = asyncio.run(record_events(_events_from(_baseline_stream())))
        em = EventEmitter(session_id="sess-1", run_id="run-1")
        mutated = _baseline_stream()
        # Replace TextDelta at index 1 with a different type
        mutated[1] = em.emit(
            UsageObserved,
            input_tokens=0,
            output_tokens=0,
            total_tokens=0,
            cost_usd=0.0,
        )
        candidate = asyncio.run(record_events(_events_from(mutated)))
        diff = diff_runs(baseline, candidate)
        assert diff.first_divergence_index == 1

    def test_text_diff_when_answers_differ(self) -> None:
        baseline = asyncio.run(record_events(_events_from(_baseline_stream())))
        em = EventEmitter(session_id="sess-1", run_id="run-1")
        mutated = _baseline_stream()
        mutated[-1] = em.emit(
            TurnSummary,
            text="goodbye world",
            turn_number=1,
            tokens_used=15,
            cost_usd=0.001,
        )
        candidate = asyncio.run(record_events(_events_from(mutated)))
        diff = diff_runs(baseline, candidate)
        assert diff.text_diff
        assert "hello" in diff.text_diff
        assert "goodbye" in diff.text_diff

    def test_text_diff_can_be_opted_out(self) -> None:
        baseline = asyncio.run(record_events(_events_from(_baseline_stream())))
        em = EventEmitter(session_id="sess-1", run_id="run-1")
        mutated = _baseline_stream()
        mutated[-1] = em.emit(
            TurnSummary,
            text="completely different",
            turn_number=1,
            tokens_used=15,
            cost_usd=0.001,
        )
        candidate = asyncio.run(record_events(_events_from(mutated)))
        diff = diff_runs(baseline, candidate, include_text_diff=False)
        assert diff.text_diff == ""

    def test_summary_renders(self) -> None:
        baseline = asyncio.run(record_events(_events_from(_baseline_stream()), label="bl"))
        em = EventEmitter(session_id="sess-1", run_id="run-1")
        candidate_stream = [*_baseline_stream(), em.emit(TextDelta, content="x")]
        candidate = asyncio.run(record_events(_events_from(candidate_stream), label="cd"))
        diff = diff_runs(baseline, candidate)
        rendered = diff.to_summary()
        assert "bl" in rendered
        assert "cd" in rendered
        assert "events:" in rendered


# ---------------------------------------------------------------------------
# replay_run — re-emit captured events
# ---------------------------------------------------------------------------


class TestReplayRun:
    def test_replay_yields_every_event_in_order(self) -> None:
        run = asyncio.run(record_events(_events_from(_baseline_stream())))
        replayed = _async_collect(replay_run(run))
        assert len(replayed) == len(run.events)
        for orig, copy in zip(run.events, replayed, strict=True):
            assert type(orig) is type(copy)

    def test_replay_empty_run(self) -> None:
        empty_summary = asyncio.run(record_events(_events_from([]))).summary
        empty = RecordedRun(events=(), summary=empty_summary)
        result = _async_collect(replay_run(empty))
        assert result == []
