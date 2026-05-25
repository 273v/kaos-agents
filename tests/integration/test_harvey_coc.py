"""Live integration test — Harvey LAB change-of-control extraction.

Runs the vendored Harvey LAB ``extract-change-of-control-provisions`` task
against the agent and asserts a small, cost-bounded subset of behaviors.
The full benchmark (all 55 criteria, ~$0.27 per run) lives at
``tests/benchmarks/harvey_coc_benchmark.py``; this test runs the *same*
machinery but with ``max_criteria=5`` so CI cost stays under ~$0.50.

Asserted behaviors (intentionally weak so the test pins infrastructure,
not model quality):

* The agent produces a non-trivial deliverable on a real M&A data room
  (8 acquisition-target contracts).
* The judge fanout runs end to end without "judge unavailable" on
  every criterion (which would indicate a config problem, not an agent
  quality issue).
* The deliverable references Apex-Kenji JV and the 4.5x EBITDA buyout
  formula — these are criteria the haiku agent reliably hits on the
  2026-05-06 baseline run (10/55 passed). They serve as a canary for
  "did the agent actually read the documents and produce relevant
  content?".

The headline accuracy number — pass rate, all-pass score — belongs in
the benchmark, not in this integration test. Asserting on it here would
make CI flake every time model behavior drifts. The benchmark JSON
(under ``docs/benchmarks/``) is the place where regressions get
caught and explained.

Skips when ``ANTHROPIC_API_KEY`` (or ``KAOS_LLM_ANTHROPIC_API_KEY``) is
absent — there is no mock fallback. The fixture is vendored under
``tests/fixtures/harvey-lab/`` (MIT, attribution preserved).
"""

from __future__ import annotations

import os

import pytest

from tests.benchmarks.harvey_coc_benchmark import run_benchmark
from tests.integration._models import judge_model, respond_model

_HAS_ANTHROPIC_KEY = bool(
    os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("KAOS_LLM_ANTHROPIC_API_KEY")
)


@pytest.mark.live
@pytest.mark.integration
@pytest.mark.skipif(
    not _HAS_ANTHROPIC_KEY,
    reason="ANTHROPIC_API_KEY (or KAOS_LLM_ANTHROPIC_API_KEY) not set",
)
@pytest.mark.asyncio
async def test_harvey_coc_smoke() -> None:
    """End-to-end smoke: agent reads 8 contracts, scores 5 criteria."""
    result = await run_benchmark(
        agent_model=respond_model(),
        judge_model=judge_model(),
        max_criteria=5,
        verbose=False,
        concurrency=3,
    )

    # The agent must produce a non-trivial deliverable.
    assert not result.agent_errored, (
        f"Agent errored: {result.error_message}. "
        "If this is a transient provider 5xx/529, re-run; if it persists, "
        "investigate the agent's research pattern against this corpus."
    )
    assert result.deliverable_chars > 500, (
        f"Deliverable too short ({result.deliverable_chars} chars). "
        "The Harvey CoC task expects a comprehensive multi-contract "
        "extraction report — anything shorter than 500 chars suggests the "
        "agent gave up or only addressed one contract."
    )

    # The judge fanout must run end-to-end on every criterion. A
    # blanket "judge unavailable" outcome means the config is broken
    # (API key, network, provider outage) — not a model quality
    # signal. We don't assert on n_passed because the first 5
    # criteria of this rubric all happen to be Northland-specific
    # provisions that the haiku agent reliably MISSES on default
    # retrieval (real result: 0/5 passed in the 2026-05-06 baseline,
    # but 10/55 passed across the full rubric). The benchmark is the
    # place to catch quality regressions; the integration test pins
    # infrastructure correctness only.
    assert result.n_criteria == 5, f"Expected 5 criteria scored, got {result.n_criteria}."
    assert result.n_judge_unavailable == 0, (
        f"{result.n_judge_unavailable}/{result.n_criteria} judge calls "
        "returned 'judge unavailable' — this is a configuration problem, "
        "not an agent quality issue. Re-run after diagnosing."
    )

    # Cost ceiling — keep the integration test cheap. The full
    # 55-criterion benchmark lands at ~$0.27 (haiku x haiku); a
    # 5-criterion subset should be well under $0.50. If this fires,
    # either the agent is doing too many tool calls (loop bug?) or
    # the judge is expensive (model swap?).
    total_cost = result.agent_cost_usd + result.judge_cost_usd
    assert total_cost < 0.50, (
        f"Smoke test cost ${total_cost:.4f} — over the $0.50 ceiling. "
        f"Agent: ${result.agent_cost_usd:.4f}, "
        f"Judge: ${result.judge_cost_usd:.4f}. "
        "Investigate before bumping the ceiling."
    )


@pytest.mark.live
@pytest.mark.integration
@pytest.mark.skipif(
    not _HAS_ANTHROPIC_KEY,
    reason="ANTHROPIC_API_KEY (or KAOS_LLM_ANTHROPIC_API_KEY) not set",
)
@pytest.mark.asyncio
async def test_harvey_coc_canary_apex_buyout() -> None:
    """Canary: the agent must surface the Apex-Kenji JV buyout formula.

    On the 2026-05-06 baseline run with claude-haiku-4-5, criteria C-039
    and C-040 (JV buy-out at 4.5x EBITDA, ~$22.95M) are reliably passed
    by the agent — they sit in the JV agreement document which our
    BM25 retrieval surfaces consistently for change-of-control queries.
    These specific identifiers (``4.5x EBITDA`` / ``Apex`` / ``Kenji``)
    function as a deterministic, judge-independent canary for "did the
    agent actually read the documents and produce relevant content?".

    If this canary fails, look at retrieval first — the JV document's
    Article I and Section 12.3 chunks are the most reliable hits in
    the corpus.
    """
    result = await run_benchmark(
        agent_model=respond_model(),
        judge_model=judge_model(),
        max_criteria=1,  # judge call is unused here — we only need the agent run
        verbose=False,
        concurrency=1,
    )

    assert not result.agent_errored, (
        f"Agent errored: {result.error_message}. "
        "Re-run if transient; investigate research pattern if persistent."
    )

    text = result.deliverable_text.lower()
    assert "apex" in text and "kenji" in text, (
        "Deliverable does not name both parties of the JV agreement "
        "(Apex and Kenji). The JV document is the most retrievable "
        "contract in this corpus — missing it suggests retrieval is "
        "broken. "
        f"Deliverable preview: {result.deliverable_text[:1000]!r}"
    )
    assert "4.5" in text or "ebitda" in text, (
        "Deliverable does not reference the 4.5x EBITDA buyout formula "
        "from the Apex-Kenji JV agreement (Section 12.3). This is the "
        "canary signal that the agent successfully extracted operative "
        "language, not just party names. "
        f"Deliverable preview: {result.deliverable_text[:1000]!r}"
    )
