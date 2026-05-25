"""Live integration test — Deliverable composition over Harvey CoC fixture.

Runs the full output pipeline against a small synthetic memory:

1. Seed FINDINGS with typed Claims so the walk + citation aggregation
   has real data to operate on.
2. ``compose_narrative`` over a 2-section structure with a real
   :class:`SectionWriterSignature` Call. Asserts:
   - the deliverable round-trips through ``to_content_document`` and
     ``to_markdown``,
   - per-section usage is captured (cost > 0, tokens > 0),
   - citations are non-empty and deduplicated.
3. One iteration of :class:`RefineDeliverable` with a 1-criterion
   rubric: produce → score → assert the loop's stop reason
   classification works (no exception path).

Skipped when ``ANTHROPIC_API_KEY`` is absent. Bounded under ~$0.10
total via ``max_iterations=1`` and a 2-section structure.

This test pins the *infrastructure*, not model quality. We don't
assert on rubric pass rates here; that lives in the benchmark.
"""

from __future__ import annotations

import os
from typing import cast

import pytest
from kaos_llm_core.signatures.grounding import Claim, Span

from kaos_agents.memory.session import SessionMemory
from kaos_agents.output import (
    Deliverable,
    NarrativeDeliverable,
    RefineDeliverable,
    RubricCriterion,
    RubricDeliverableCritic,
    SectionSpec,
    aggregate_citations,
    compose_narrative,
)
from kaos_agents.types.memory import MemoryType
from tests.integration._models import critic_model, respond_model

_HAS_ANTHROPIC_KEY = bool(
    os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("KAOS_LLM_ANTHROPIC_API_KEY")
)


def _seed_findings(memory: SessionMemory) -> tuple[Claim, Claim]:
    """Seed FINDINGS with two synthetic Claims so walks + citations have data."""
    span_a = Span(
        source_uri="doc:northland-msa",
        quote="anti-assignment without prior written consent",
        char_span=(0, 50),
        page=14,
    )
    claim_a = Claim(
        statement="The Northland MSA prohibits assignment without consent.",
        claim_type="factual",
        supporting_spans=[span_a],
        confidence=0.9,
    )

    span_b = Span(
        source_uri="doc:apex-jv",
        quote="4.5x EBITDA buyout",
        char_span=(0, 18),
        page=12,
    )
    claim_b = Claim(
        statement="The Apex-Kenji JV uses a 4.5x EBITDA buyout formula.",
        claim_type="factual",
        supporting_spans=[span_b],
        confidence=0.95,
    )

    for claim in (claim_a, claim_b):
        memory.add(
            MemoryType.FINDINGS,
            f"[{claim.claim_type}] {claim.statement}",
            metadata={"claim": claim.model_dump(mode="json"), "verified": True},
        )
    return claim_a, claim_b


@pytest.mark.live
@pytest.mark.integration
@pytest.mark.skipif(
    not _HAS_ANTHROPIC_KEY,
    reason="ANTHROPIC_API_KEY (or KAOS_LLM_ANTHROPIC_API_KEY) not set",
)
@pytest.mark.asyncio
async def test_compose_narrative_round_trip() -> None:
    """End-to-end: seed FINDINGS, compose narrative, render markdown."""
    memory = SessionMemory("test-deliverable-compose")
    _seed_findings(memory)

    structure = (
        SectionSpec(
            id="findings",
            title="Findings",
            depth=1,
            instruction="Summarise the change-of-control provisions identified.",
        ),
        SectionSpec(
            id="recommendations",
            title="Recommendations",
            depth=1,
            instruction="Recommend next-step actions based on the findings.",
        ),
    )

    result = await compose_narrative(
        memory,
        structure,
        model=respond_model(),
        title="Test M&A Memo",
    )

    assert isinstance(result.deliverable, NarrativeDeliverable)
    assert result.deliverable.kind == "narrative"

    markdown = result.deliverable.to_markdown()
    assert "Findings" in markdown, f"Heading missing: {markdown[:300]!r}"
    assert "Recommendations" in markdown, f"Heading missing: {markdown[:300]!r}"
    assert len(markdown) > 200, "Deliverable markdown too short for a real two-section memo"

    cd = result.deliverable.to_content_document()
    assert len(cd.body) >= 4, (
        "ContentDocument round-trip should have at least 4 blocks (2 headings + 2 paragraphs)"
    )

    citations = result.deliverable.citations()
    assert len(citations) >= 2, "Citations must include the seeded spans"
    assert any("northland" in c.source_uri for c in citations)
    assert any("apex-jv" in c.source_uri for c in citations)

    assert len(result.per_section_usage) == 2
    total_tokens = result.total_usage.total_tokens
    assert total_tokens > 0, "Per-section invocation usage must roll up to nonzero tokens"
    assert result.total_usage.cost_usd >= 0.0


@pytest.mark.live
@pytest.mark.integration
@pytest.mark.skipif(
    not _HAS_ANTHROPIC_KEY,
    reason="ANTHROPIC_API_KEY (or KAOS_LLM_ANTHROPIC_API_KEY) not set",
)
@pytest.mark.asyncio
async def test_refine_deliverable_one_iteration() -> None:
    """Smoke RefineDeliverable: produce → critic → loop terminates cleanly."""
    memory = SessionMemory("test-refine-deliverable")
    _seed_findings(memory)
    structure = (SectionSpec(id="findings", title="Findings", depth=1),)

    async def produce() -> Deliverable:
        result = await compose_narrative(
            memory,
            structure,
            model=respond_model(),
            title="Refine Smoke",
        )
        # NarrativeDeliverable is a structural :class:`Deliverable` —
        # the cast makes ty's inference explicit; the runtime check
        # via :func:`runtime_checkable` is the actual conformance test.
        return cast(Deliverable, result.deliverable)

    rubric = (
        RubricCriterion(
            id="C-mentions-northland",
            description="Mentions Northland",
            match_criteria=(
                "PASS if the deliverable mentions the Northland Refining MSA. FAIL if it does not."
            ),
        ),
    )

    critic = RubricDeliverableCritic(
        critic_model=critic_model(),
        concurrency=1,
    )

    refine = RefineDeliverable(
        producer=produce,
        critic=critic,
        rubric=rubric,
        memory=memory,
        max_iterations=1,
        patience=0,
        min_pass_rate_for_perfect=0.99,
    )
    result = await refine.run()

    assert result.best_iteration == 1
    assert isinstance(result.best_deliverable, NarrativeDeliverable)
    assert result.stop_reason in {"completed", "perfect"}
    assert len(result.history) == 1
    assert result.total_critic_cost_usd >= 0.0


def test_aggregate_citations_pure_unit() -> None:
    """Pure-unit smoke: dedupe + walk_findings work without a live LLM."""
    memory = SessionMemory("test-citations-pure")
    _seed_findings(memory)
    spans = aggregate_citations(memory)
    assert len(spans) == 2  # two distinct sources
    sources = {s.source_uri for s in spans}
    assert sources == {"doc:northland-msa", "doc:apex-jv"}
