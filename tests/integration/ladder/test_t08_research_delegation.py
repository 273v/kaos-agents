"""Tier 8: research pattern with a real ContentDocumentCorpus.

An earlier draft passed the corpus INLINE in the prompt — that
tested "LLM answers from in-prompt context," not "research pattern
+ RAG retrieval over a corpus." This version builds a
``ContentDocumentCorpus`` from 3 short docs, passes it to the
Runner via the ``corpus`` kwarg, and lets the research pattern
retrieve+ground the answer.

Verifies:

- Corpus plumbing from Runner -> ResearchAgent -> RAG
- The grounded-answer path produces something useful even when the
  user's question requires synthesizing across multiple corpus docs
- The 3 facts buried in 3 separate docs all surface in the answer
  (validates retrieval AND synthesis, not just retrieval of one doc)

Cost target: ~$0.10 (one Anthropic call via the research+RAG path).
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


def _build_corpus() -> object:
    """A tiny 3-doc corpus about a made-up animal so the model can't
    fabricate from training data — every fact must come from the corpus."""
    from kaos_content.corpus import ContentDocumentCorpus
    from kaos_content.parsers.plain import parse_plain_text

    docs_text = (
        # Doc 0 — discovery + describer
        "The mauve-tailed lemur (Lemur purpurus, fictional) was first "
        "described in 1923 by Dr. Estelle Carriere working in southern "
        "Madagascar. She published the description in the Bulletin of "
        "Madagascar Zoology, volume 4.",
        # Doc 1 — physical characteristics
        "Adult mauve-tailed lemurs weigh between 800 and 1100 grams and "
        "measure 35-42 cm head-to-tail. They live primarily in deciduous "
        "forests of the south-east region and forage at dusk.",
        # Doc 2 — conservation status
        "The IUCN Red List classifies the mauve-tailed lemur as "
        "Endangered (EN) as of the 2024 assessment, citing habitat loss "
        "from agricultural expansion as the primary threat.",
    )
    docs = [parse_plain_text(t) for t in docs_text]
    uris = [f"doc-{i}-mauve-lemur" for i in range(len(docs))]
    return ContentDocumentCorpus(docs, doc_uris=uris)


@pytest.mark.asyncio
async def test_research_with_real_corpus(ladder_runner_factory, turn_session_id):
    """ResearchAgent + ContentDocumentCorpus answers from 3 separate docs."""
    from kaos_agents.config import AgentPattern

    corpus = _build_corpus()

    runner, _ = ladder_runner_factory(
        agent_kwargs={
            "instructions": (
                "You are a research assistant. Answer the user's question "
                "using only the documents in the provided corpus. If the "
                "answer requires combining facts from multiple documents, "
                "do so."
            ),
            "model": model_for_tier(TIER),
            "pattern": AgentPattern.RESEARCH,
        },
        corpus=corpus,
    )
    # Question that requires answering from THREE different docs —
    # no single doc has all the facts, so retrieval-only-of-one-doc
    # fails the synthesis check.
    msg = (
        "Who first described the mauve-tailed lemur, what does it weigh, "
        "and what is its IUCN Red List status?"
    )
    events = await collect_events(runner.run(msg, turn_session_id))

    assert_within_budget(events, budget_usd=BUDGET_USD, label="tier-8")

    text = text_from_summary(events).lower()

    # All 3 facts must round-trip via retrieval+synthesis.
    # 1. Describer (from doc 0): "Estelle Carriere"
    assert "carriere" in text or "estelle" in text, (
        f"answer must mention Estelle Carriere (doc 0); got: {text[:300]!r}"
    )
    # 2. Weight (from doc 1): 800-1100 grams
    has_weight = any(n in text for n in ("800", "1100", "1,100"))
    assert has_weight, f"answer must mention weight range; got: {text[:300]!r}"
    # 3. IUCN status (from doc 2): "Endangered"
    assert "endangered" in text or " en " in text, (
        f"answer must mention IUCN status Endangered (doc 2); got: {text[:300]!r}"
    )
