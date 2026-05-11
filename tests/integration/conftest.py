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
"""

from __future__ import annotations

import datetime
import os
from collections.abc import AsyncIterator
from pathlib import Path

import pytest_asyncio

from tests.integration._recorder import record_live_test


def _today_dir() -> Path:
    """Today's run subdirectory under tests/integration/runs/."""
    here = Path(__file__).parent
    today = datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%d")
    return here / "runs" / today


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
    ):
        yield
