"""Tier 2: single tool call — kaos-source-fr-search.

Goal: verify the tool bridge surfaces structured fields (regression
test for the dict(params) FR fix and the structuredContent bridge
fix), the span tree stitches turn -> tool_call cleanly, and ReAct
makes exactly one tool call when prompted carefully.

Cost target: ~$0.05 (sonnet on a small ReAct loop).
"""

from __future__ import annotations

import pytest

from tests.integration.ladder.conftest import (
    assert_within_budget,
    collect_events,
    model_for_tier,
    text_from_summary,
    tool_call_starts,
    turn_spans,
)

pytestmark = pytest.mark.live

TIER = 2
BUDGET_USD = 0.10


@pytest.mark.asyncio
async def test_single_tool_call_fr_search(ladder_runner_factory, turn_session_id):
    """One Federal Register search call returns full structured fields."""
    runner, _ = ladder_runner_factory(
        agent_kwargs={
            "instructions": (
                "You are a research assistant. Use the kaos-source-fr-search tool "
                "exactly once with the user's query parameters, then report the "
                "title from results[0]."
            ),
            "model": model_for_tier(TIER),
            "tools": ("kaos-source-fr-search",),
        },
        register_source=True,
    )
    msg = (
        "Use kaos-source-fr-search exactly once with: term=PFAS, "
        "agency=environmental-protection-agency, doc_type=NOTICE, "
        "per_page=1, order=newest. From results[0], report THREE fields "
        "verbatim: title, document_number, and publication_date."
    )
    events = await collect_events(runner.run(msg, turn_session_id))

    assert_within_budget(events, budget_usd=BUDGET_USD, label="tier-2")

    # Span shape: turn pair + at least one tool_call/start
    spans = turn_spans(events)
    assert len(spans) >= 2, f"need >=2 turn spans, got {len(spans)}"
    tool_starts = tool_call_starts(events)
    assert len(tool_starts) >= 1, (
        "expected >=1 tool_call/start span (the FR search). "
        "Zero spans means the v2 tool-bridging fix regressed."
    )

    # Span-tree stitching: every tool_call/start parents under a turn
    for tc in tool_starts:
        assert tc.parent_span_id, (
            f"tool_call span has parent_span_id=None - the collect_events / "
            f"span-stack fix regressed: {tc.name}"
        )

    # Answer regression: the FR API returns real titles + document
    # numbers in YYYY-NNNNN format + ISO dates. The agent's answer
    # must contain a real document_number AND a publication_date —
    # values it CAN'T produce without seeing the tool's
    # structuredContent. The earlier "answer contains PFAS" check
    # was theater because PFAS is in the prompt; the model could echo
    # it without ever calling the tool. document_number and date are
    # only knowable from the tool's structured response — a hard
    # regression signal for the tool-bridge structuredContent fix.
    import re

    text = text_from_summary(events)
    assert len(text) >= 50, f"answer too short — bridge dropping data? got: {text!r}"
    assert re.search(r"\b(20\d{2})-\d{4,5}\b", text), (
        f"answer must contain a Federal Register document number "
        f"(YYYY-NNNNN pattern). Got: {text[:300]!r}. "
        f"This value comes only from the tool's structuredContent — "
        f"if absent, the bridge is dropping structured data again."
    )
    assert re.search(r"\b20\d{2}-\d{2}-\d{2}\b", text), (
        f"answer must contain a publication_date in YYYY-MM-DD form "
        f"from the tool result. Got: {text[:300]!r}"
    )
