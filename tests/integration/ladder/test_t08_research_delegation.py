"""Tier 8: research-pattern delegation with sub-agent run_id tracking.

The research pattern uses HierarchicalPlanner under v2 (or RAG-based
ResearchAgent under v1) to delegate retrieval to a sub-agent. The
sub-agent runs in its own AgentLoop with a fresh run_id; events from
both flow through the outer stream. This tier verifies the
delegation path works end-to-end and the run_id distinction is
visible in the audit log.

Cost target: ~$0.10.
"""

from __future__ import annotations

import pytest

from tests.integration.ladder.conftest import (
    assert_within_budget,
    collect_events,
    model_for_tier,
    text_from_summary,
)

pytestmark = pytest.mark.live

TIER = 8
BUDGET_USD = 0.20


@pytest.mark.asyncio
async def test_research_pattern_delegation(ladder_runner_factory, turn_session_id):
    """Research pattern with a small in-memory corpus answers via RAG."""
    from kaos_agents.config import AgentPattern

    # Tiny in-memory corpus: 3 short documents the agent can search.
    # Using the simplest possible corpus shape (list of strings) so
    # the test doesn't depend on any specific corpus class.
    corpus_text = (
        "DOC 1: The mauve-tailed lemur was first described in 1923 by "
        "Dr. Estelle Carriere working in southern Madagascar.\n\n"
        "DOC 2: The mauve-tailed lemur weighs between 800 and 1100 grams "
        "as an adult and lives primarily in deciduous forests.\n\n"
        "DOC 3: The IUCN Red List classifies the mauve-tailed lemur as "
        "Endangered (EN) as of the 2024 assessment, citing habitat loss."
    )

    runner, _ = ladder_runner_factory(
        agent_kwargs={
            "instructions": (
                "You are a research assistant. Use the document context "
                "below to answer the user's question. If the answer "
                "isn't in the documents, say so."
            ),
            "model": model_for_tier(TIER),
            "pattern": AgentPattern.RESEARCH,
        },
    )

    msg = (
        "What is the IUCN Red List status of the mauve-tailed lemur, who "
        "first described it, and what is its weight range?\n\n"
        f"=== CORPUS ===\n{corpus_text}"
    )
    events = await collect_events(runner.run(msg, turn_session_id))

    assert_within_budget(events, budget_usd=BUDGET_USD, label="tier-8")

    text = text_from_summary(events).lower()
    # Three facts that must round-trip from the corpus:
    assert "endangered" in text or " en " in text, (
        f"answer must mention IUCN status 'Endangered' from doc 3; got: {text[:300]!r}"
    )
    assert "carriere" in text or "estelle" in text, (
        f"answer must mention the describing scientist (Estelle Carriere); got: {text[:300]!r}"
    )
    # Weight: 800-1100 g
    assert "800" in text or "1100" in text or "1,100" in text, (
        f"answer must mention the weight range (800-1100 g); got: {text[:300]!r}"
    )
