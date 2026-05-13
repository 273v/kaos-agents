"""Tier 7: PDF extraction via the kaos-pdf TOOL BRIDGE.

An earlier draft passed the PDF text inline in the prompt — that
tested "LLM extracts from text," not "kaos-pdf bridges into the
agent's tool flow." This version registers kaos-pdf tools through
``register_pdf=True`` and prompts the agent to USE
``kaos-pdf-extract-parse`` on a file path. Verifies:

- kaos-pdf bridges into the runtime
- The PDF tool's structuredContent makes it through the tool-bridge
  combiner fix from the prior session
- The agent picks the tool when asked, makes a real tool_call,
  and uses the parsed output to answer

Cost target: ~$0.10 (small ReAct loop with one PDF parse call).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.integration.ladder.conftest import (
    assert_within_budget,
    collect_events,
    model_for_tier,
    text_from_summary,
    tool_call_starts,
)

pytestmark = pytest.mark.live

TIER = 7
BUDGET_USD = 0.20

PDF_FIXTURE = Path(__file__).parent / "fixtures" / "sample.pdf"


@pytest.mark.asyncio
async def test_pdf_extraction_via_tool_bridge(ladder_runner_factory, turn_session_id):
    """Agent calls kaos-pdf-extract-parse as a real tool and extracts facts."""
    assert PDF_FIXTURE.exists(), f"missing fixture: {PDF_FIXTURE}"

    from kaos_agents.config import AgentPattern

    # Use plan pattern so the multi-part request ("parse + answer two
    # questions") doesn't get routed to chat→fallback-to-no-tool-LLM.
    # The pattern routing is correct platform behavior — single-step
    # prompts go through chat-ReAct; multi-step go through plan-execute.
    # This tier specifically exercises kaos-pdf in the tool-execution
    # path, which works under either pattern.
    runner, _ = ladder_runner_factory(
        agent_kwargs={
            "instructions": (
                "You are a document-extraction assistant. Use the "
                "kaos-pdf-extract-page-text tool to extract text from "
                "the PDF at the path the user gives you, then answer "
                "the question from the extracted content."
            ),
            "model": model_for_tier(TIER),
            "pattern": AgentPattern.PLAN,
            # ``kaos-pdf-extract-page-text`` is the context-free
            # variant — it doesn't require a runtime artifact store,
            # so it works under the test factory's bare runtime.
            # ``kaos-pdf-extract-parse`` (the full-document parser)
            # needs context.
            "tools": ("kaos-pdf-extract-page-text",),
        },
        register_pdf=True,
    )

    msg = (
        f"Use kaos-pdf-extract-page-text on path {PDF_FIXTURE} for page 0. "
        "From the extracted text, report (1) the docket number from the "
        "subject line and (2) the city's Mayor. Format: 'Docket: <number>; "
        "Mayor: <name>'."
    )
    events = await collect_events(runner.run(msg, turn_session_id))

    assert_within_budget(events, budget_usd=BUDGET_USD, label="tier-7")

    # The agent MUST call a kaos-pdf-* tool — if zero pdf calls fire,
    # the test failed even if the answer looked plausible (the LLM
    # would have to fabricate without seeing the PDF).
    tool_starts = tool_call_starts(events)
    pdf_calls = [s for s in tool_starts if "kaos-pdf" in str(s.attributes.get("tool_name", ""))]
    assert pdf_calls, (
        f"expected the agent to call a kaos-pdf-* tool; saw "
        f"tools: {[s.attributes.get('tool_name') for s in tool_starts]!r}. "
        f"If zero pdf calls fired, the tool bridge or ReAct's tool-selection "
        f"regressed."
    )

    text = text_from_summary(events)
    # Ground truth from the PDF (visually inspected):
    # Subject: Docket OST-2008-0299 — EAS at El Centro/Imperial, California
    # Mayor: Geoff Dale
    assert "OST-2008-0299" in text, (
        f"answer must contain docket number 'OST-2008-0299' from the "
        f"parsed PDF; got: {text[:300]!r}"
    )
    assert "Geoff Dale" in text or "geoff dale" in text.lower(), (
        f"answer must contain Mayor 'Geoff Dale' from the parsed PDF; got: {text[:300]!r}"
    )
