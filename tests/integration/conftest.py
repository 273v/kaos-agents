"""Integration-test conftest — live-call telemetry recording.

Installs an autouse fixture that wraps every ``@pytest.mark.live``
test in :func:`record_live_test`, persisting one JSONL per test
under ``tests/integration/runs/<YYYY-MM-DD>/<sanitized-nodeid>.jsonl``
plus appending a one-line summary to ``tests/integration/runs/INDEX.jsonl``.

See ``tests/integration/_recorder.py`` for the recorder mechanics and
the rationale (audit trail for LLM-driven decisions in regulated-
industry contexts).

Opt-out: set ``KAOS_TESTS_NO_RECORD=1`` to disable, e.g. when
profiling and the recorder's overhead distorts measurements. The
recorder itself is non-intrusive on the hot path (a thin wrapper on
``Call._execute`` that appends one reference to a list) — overhead
is typically < 5ms across an entire test even with dozens of calls.

Two pytest pieces work together:

1. ``pytest_runtest_makereport`` (hook) — stashes the outcome of
   each test phase (``setup``, ``call``, ``teardown``) onto
   ``item.stash`` so the fixture's finally block can read it.
   Without this, ``@pytest.mark.live`` fixtures cannot see test-body
   assertion failures (pytest captures them for reporting before the
   fixture's ``yield`` re-raises).
2. ``_record_live_calls`` (autouse fixture) — the actual recorder
   wrapper. After ``yield`` returns, reads the stashed outcome and
   forwards it to :func:`record_live_test`.
"""

from __future__ import annotations

import contextlib
import datetime
import json
import os
import time
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest
import pytest_asyncio

from tests.integration._recorder import record_live_test

# Stash key for the test-phase outcome. ``StashKey`` is pytest's
# typed scratchpad on ``Item.stash``.
_OUTCOME_KEY = pytest.StashKey[dict[str, str]]()

# P1.3 — per-test summary line under runs/<date>/SUMMARY.jsonl.
# One line per @pytest.mark.live test with the canonical telemetry the
# corpus-stress design doc names (cost, n_llm_calls, n_tool_calls,
# tool_names, judge verdict). Diff-able cost over time without
# grepping the per-call JSONL files.
_SUMMARY_FILENAME = "SUMMARY.jsonl"


def _today_dir() -> Path:
    """Today's run subdirectory under tests/integration/runs/."""
    here = Path(__file__).parent
    today = datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%d")
    return here / "runs" / today


@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item, call):  # type: ignore[no-untyped-def]
    """Capture per-phase outcome so the autouse fixture can see test-body failures.

    pytest's autouse fixture body doesn't see assertion failures
    that fire inside the test body — pytest catches them for
    reporting first. This hook stashes the per-phase outcome
    (``"passed"``, ``"failed"``, ``"skipped"``) on the test item;
    the fixture's finally block reads it back.
    """
    outcome = yield
    rep = outcome.get_result()
    stash = item.stash.setdefault(_OUTCOME_KEY, {})
    stash[rep.when] = rep.outcome
    if rep.failed and rep.longreprtext:
        stash["longrepr"] = rep.longreprtext[:4000]


def _read_outcome(node) -> tuple[str, str | None]:  # type: ignore[no-untyped-def]
    """Read the test's call-phase outcome from the stash. Defaults to passed."""
    stash = node.stash.get(_OUTCOME_KEY, {})
    # "call" is the actual test body; setup/teardown are separate phases.
    outcome = stash.get("call") or stash.get("setup") or "passed"
    longrepr = stash.get("longrepr")
    return outcome, longrepr


@pytest.fixture
def judge_state() -> dict[str, Any]:
    """Per-test scratchpad for the Opus-as-judge layer (P1.1 / #482).

    A test that wires the judge calls
    ``assert_judge_passes(..., judge_state=judge_state)``; the helper
    mutates this dict with the judge label / confidence / cost /
    reasoning. The autouse recorder then folds the dict into the
    per-test SUMMARY.jsonl line.

    Empty dict for tests that never invoke the judge — the
    summary-writer treats that as "no judge invoked" and omits the
    ``judge`` key from the record.
    """
    return {}


def _write_summary_line(
    *,
    out_dir: Path,
    test_nodeid: str,
    outcome: str,
    elapsed_s: float,
    capture: Any,
    judge_state: dict[str, Any],
) -> None:
    """Append one per-test summary line to ``runs/<date>/SUMMARY.jsonl``.

    Shape (per the corpus-stress follow-up plan, §2.3 P1.3):

        {
          "test_nodeid": "...",
          "outcome": "passed|failed|skipped",
          "elapsed_s": 12.3,
          "cost_usd": 0.034,          # sum of all LLM calls in the test
          "n_llm_calls": 7,
          "n_tool_calls": 5,
          "n_tool_errors": 0,
          "tool_names": [...],
          "judge": {                  # only when judge_state populated
            "label": "grounded",
            "confidence": 0.95,
            "cost_usd": 0.045,
            "reasoning": "..."
          }
        }

    Best-effort: never raises. Telemetry must not break a test.
    """
    try:
        # Aggregate LLM-side spend from the recorder's captured
        # invocations + any stitched subprocess records.
        total_cost = 0.0
        n_llm_calls = 0
        for inv in getattr(capture, "invocations", ()) or ():
            n_llm_calls += 1
            usage = getattr(inv, "usage", None)
            if usage is not None:
                with contextlib.suppress(TypeError, ValueError):
                    total_cost += float(getattr(usage, "cost_usd", 0.0) or 0.0)
        for sub in getattr(capture, "subprocess_records", ()) or ():
            n_llm_calls += 1
            usage_dict = sub.get("usage") if isinstance(sub, dict) else None
            if isinstance(usage_dict, dict):
                with contextlib.suppress(TypeError, ValueError):
                    total_cost += float(usage_dict.get("cost_usd", 0.0) or 0.0)

        # Tool-side telemetry: read whatever the test stashed under
        # _TOOL_TELEMETRY_KEY (populated by the test when it wants
        # per-test tool-trace metrics — empty otherwise).
        tool_summary = request_tool_telemetry_holder.get(test_nodeid, {}) or {}
        tool_names = tool_summary.get("tool_names", [])
        n_tool_calls = int(tool_summary.get("n_tool_calls", 0) or 0)
        n_tool_errors = int(tool_summary.get("n_tool_errors", 0) or 0)

        record: dict[str, Any] = {
            "test_nodeid": test_nodeid,
            "outcome": outcome,
            "elapsed_s": round(elapsed_s, 4),
            "cost_usd": round(total_cost, 6),
            "n_llm_calls": n_llm_calls,
            "n_tool_calls": n_tool_calls,
            "n_tool_errors": n_tool_errors,
            "tool_names": list(tool_names),
        }

        if judge_state:
            judge_block = {
                "label": judge_state.get("judge_label"),
                "confidence": judge_state.get("judge_confidence"),
                "cost_usd": judge_state.get("judge_cost_usd"),
                "latency_ms": judge_state.get("judge_latency_ms"),
                "fell_back": judge_state.get("judge_fell_back"),
                "reasoning": judge_state.get("judge_reasoning"),
                "test_id": judge_state.get("test_id"),
            }
            # Roll judge spend into the headline cost so cost-over-time
            # plots reflect total spend (agent + judge), and surface
            # the breakdown via the judge.cost_usd sub-field.
            judge_cost = judge_state.get("judge_cost_usd") or 0.0
            with contextlib.suppress(TypeError, ValueError):
                record["cost_usd"] = round(total_cost + float(judge_cost), 6)
            record["judge"] = judge_block

        out_dir.mkdir(parents=True, exist_ok=True)
        summary_path = out_dir / _SUMMARY_FILENAME
        with summary_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, separators=(",", ":")) + "\n")
            fh.flush()
            with contextlib.suppress(OSError, ValueError):
                os.fsync(fh.fileno())
    except Exception:
        # Telemetry must never break a test.
        pass


# Module-level dict carrying the latest tool-telemetry stashed by a
# test (via ``record_tool_telemetry``). Keyed by nodeid so the autouse
# fixture can pick it up in its finally block. Cleared after the
# summary line is written.
request_tool_telemetry_holder: dict[str, dict[str, Any]] = {}


def record_tool_telemetry(
    test_nodeid: str,
    *,
    tool_names: list[str],
    n_tool_errors: int,
) -> None:
    """Stash per-test tool telemetry for the conftest summary writer.

    Tests that wire the judge layer also call this with the agent
    response's tool names + error count, so the SUMMARY.jsonl line
    carries n_tool_calls / n_tool_errors / tool_names alongside the
    judge verdict.
    """
    request_tool_telemetry_holder[test_nodeid] = {
        "tool_names": list(tool_names),
        "n_tool_calls": len(tool_names),
        "n_tool_errors": int(n_tool_errors),
    }


@pytest_asyncio.fixture(autouse=True)
async def _record_live_calls(request, judge_state) -> AsyncIterator[None]:
    """Autouse fixture — record LLM calls for ``@pytest.mark.live`` tests.

    No-ops for non-live tests so the fixture imposes no cost on the
    unit / non-live tier.

    Honors ``KAOS_TESTS_NO_RECORD=1`` for opt-out.

    Also emits a per-test SUMMARY.jsonl line (one record per @live
    test) under ``runs/<date>/SUMMARY.jsonl`` for diff-able
    cost-over-time tracking, per the corpus-stress follow-up plan
    P1.3.
    """
    is_live = request.node.get_closest_marker("live") is not None
    if not is_live or os.environ.get("KAOS_TESTS_NO_RECORD") == "1":
        yield
        return

    extra = {
        "markers": [m.name for m in request.node.iter_markers()],
        "anthropic_key_present": "ANTHROPIC_API_KEY" in os.environ,
        "openai_key_present": "OPENAI_API_KEY" in os.environ,
    }

    today = _today_dir()
    t_start = time.perf_counter()
    async with record_live_test(
        request.node.nodeid,
        out_dir=today,
        extra_metadata=extra,
    ) as capture:
        yield
        # After the test body has run, pytest's makereport hook has
        # populated the stash. Forward the real outcome into the
        # recorder's capture object so the header reflects it.
        outcome, longrepr = _read_outcome(request.node)
        capture.outcome_override = outcome  # consumed by record_live_test
        if outcome == "failed" and longrepr:
            capture.error_override = longrepr

    # Per-test SUMMARY.jsonl line. Written AFTER the record_live_test
    # context exits so capture.invocations / subprocess_records are
    # the post-test totals.
    elapsed = time.perf_counter() - t_start
    final_outcome, _ = _read_outcome(request.node)
    _write_summary_line(
        out_dir=today,
        test_nodeid=request.node.nodeid,
        outcome=final_outcome,
        elapsed_s=elapsed,
        capture=capture,
        judge_state=judge_state,
    )
    # Pop the tool-telemetry entry so a later test in the same
    # process can't accidentally inherit it.
    request_tool_telemetry_holder.pop(request.node.nodeid, None)
