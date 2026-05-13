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

import datetime
import os
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
import pytest_asyncio

from tests.integration._recorder import record_live_test

# Stash key for the test-phase outcome. ``StashKey`` is pytest's
# typed scratchpad on ``Item.stash``.
_OUTCOME_KEY = pytest.StashKey[dict[str, str]]()


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


@pytest_asyncio.fixture(autouse=True)
async def _record_live_calls(request) -> AsyncIterator[None]:
    """Autouse fixture — record LLM calls for ``@pytest.mark.live`` tests.

    No-ops for non-live tests so the fixture imposes no cost on the
    unit / non-live tier.

    Honors ``KAOS_TESTS_NO_RECORD=1`` for opt-out.
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

    async with record_live_test(
        request.node.nodeid,
        out_dir=_today_dir(),
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
