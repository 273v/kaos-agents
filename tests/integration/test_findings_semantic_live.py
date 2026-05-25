"""Sprint-2 #6 — Live integration test for semantic-selector mode.

Reproduces the PA6 silent-failure: K7 token selector enumerates
zero useful candidates when the user's question uses different
vocabulary than the document. The original failure scenario was a
PPTX board deck with a slide titled "Cyber Risk" whose body talked
about "multi-factor authentication and quarterly penetration
testing" but never contained the literal word "cyber". Asked
"What's the cyber risk mitigation?", the K7 token selector with
``selector_arg="cyber"`` returned no candidates and the synthesis
truthfully reported "no mitigation found" — a recall failure that
looked like an LLM failure.

The two scenarios this file covers (one cheap live LLM call per
scenario, total spend < $0.05):

1. ``select_by="semantic"`` recovers the planted mitigation by
   asking a cheap rewrite LLM ("anthropic:claude-haiku-4-5") for
   the vocabulary that would appear in a document containing the
   answer, then running the union of token matches across those
   terms.
2. ``select_by="token"`` with ``selector_arg="cyber"`` on the
   same document surfaces the structured low-recall warning in
   ``structuredContent["warnings"]`` — the failure mode the user
   was tripping over in PA6.

Skipped without ``ANTHROPIC_API_KEY``. Cost budget: $0.10 total
for the file. The semantic scenario is one rewrite call
(~$0.001) + one filter chunk (~$0.005) + one synthesis call
(~$0.01). The token scenario in scenario 2 doesn't even need
synthesis — the warning fires from Phase 1.
"""

from __future__ import annotations

import os

import pytest
from kaos_content.model.document import ContentDocument
from kaos_content.shortcuts import paragraph

from tests.integration._models import critic_model

requires_anthropic = pytest.mark.skipif(
    "ANTHROPIC_API_KEY" not in os.environ,
    reason=(
        "ANTHROPIC_API_KEY missing — Sprint-2 #6 semantic-selector "
        "live test requires an Anthropic key for the rewrite call."
    ),
)


# The planted mitigation phrase — same as the PA6 reproduction.
# Critically, the body never contains the literal word "cyber".
# A token selector with selector_arg="cyber" must enumerate ZERO
# candidates over this doc; semantic mode must recover at least
# the mitigation sentence.
PLANTED_MITIGATION = "multi-factor authentication and quarterly penetration testing"


def _build_cyber_doc() -> ContentDocument:
    """Build a small ContentDocument modeling the PA6 failure case.

    No literal "cyber" in the body. The mitigation is in plain
    English; the question will be abstract ("cyber risk
    mitigation"). The token selector misses; the semantic
    rewrite should bridge.
    """
    return ContentDocument(
        body=(
            paragraph("Quarterly board review of operational risks."),
            paragraph("Revenue grew 14% YoY to $312 million in Q3."),
            paragraph(
                "Top operational risk this quarter is credential stuffing "
                "against partner SSO portals."
            ),
            paragraph(
                f"Board-approved mitigation is {PLANTED_MITIGATION} across all "
                "admin systems by end of Q1 2026."
            ),
            paragraph(
                "Tabletop incident exercise completed in October; remediation "
                "gaps tracked to closure."
            ),
            paragraph("Insurance renewal completed at 6% premium increase."),
            paragraph("Vendor concentration risk flagged on two top-five suppliers."),
        ),
    )


@pytest.fixture
def runtime():
    from kaos_core.registry.container import KaosRuntime

    # In-memory + global isolation — the session_id below is
    # hard-coded so the disk VFS would false-green on re-run.
    return KaosRuntime.test_mode()


@pytest.fixture
def context(runtime):
    from kaos_core.base.context import KaosContext

    return KaosContext.create(session_id="findings-semantic-live", runtime=runtime)


async def _store_doc(runtime, context, doc: ContentDocument) -> str:
    from kaos_content.artifacts import store_document

    manifest = await store_document(doc, runtime, context, name="cyber-deck")
    return manifest.artifact_id


@pytest.mark.live
@requires_anthropic
class TestSemanticSelectorLive:
    """Sprint-2 #6 — semantic mode recovers PA6 vocabulary gap.

    Two-scenario test: scenario 1 proves the semantic path
    recovers what the token path misses; scenario 2 proves the
    token-path low-recall warning fires on exactly that case.
    Both together close the PA6 transparency gap — the agent now
    either recovers (semantic) or honestly tells the caller why
    it might be missing things (token + warning).
    """

    async def test_semantic_mode_recovers_planted_mitigation(self, runtime, context) -> None:
        """select_by='semantic' on the PA6 failure doc must surface
        the planted mitigation sentence in the synthesis answer."""
        from kaos_agents.tools.findings import AgentFindingsTool

        artifact_id = await _store_doc(runtime, context, _build_cyber_doc())

        tool = AgentFindingsTool()
        result = await tool.execute(
            {
                "artifact_id": artifact_id,
                "question": "What is the cyber risk mitigation plan in this deck?",
                "select_by": "semantic",
                "filter_model": critic_model(),
                "synthesis_model": critic_model(),
                "semantic_rewrite_model": critic_model(),
                "chunk_size": 20,
                "num_parallel": 3,
                "relevance_threshold": 0.4,
            },
            context,
        )

        assert not result.isError, f"Tool errored: {result.text}"
        payload = result.structuredContent
        assert payload is not None, "Tool returned no structured payload"

        # Sprint-2 #6 wire shape — semantic_terms always present.
        assert "semantic_terms" in payload
        assert "warnings" in payload
        assert "semantic_rewrite_cost_usd" in payload
        # The rewrite returned a non-empty term list.
        assert len(payload["semantic_terms"]) >= 1, (
            "Semantic-rewrite returned no terms — sanitizer or rewrite LLM regression"
        )
        # The rewrite cost is non-zero (real LLM call happened).
        assert payload["semantic_rewrite_cost_usd"] > 0, (
            "Rewrite cost zero — usage wiring regression"
        )

        # The rewrite should have produced at least one term that
        # actually appears in the doc — that's the whole point.
        # We don't pin specific terms (the LLM is free to vary
        # the expansion); we just verify the union pinned at least
        # one candidate.
        assert payload["total_enumerated"] >= 1, (
            f"Semantic-rewrite produced terms but the union match "
            f"found nothing. Terms: {payload['semantic_terms']!r}"
        )

        # Phase 2 kept at least one survivor — the planted
        # mitigation sentence is exactly the kind of thing the
        # filter should keep.
        assert payload["total_filtered"] >= 1, (
            "Phase 2 filter dropped every candidate; filter prompt "
            "regression or relevance_threshold too strict"
        )

        # The synthesized answer must surface the mitigation
        # phrase — proof the right sentence reached synthesis,
        # not a hallucination.
        answer_lc = payload["answer"].lower()
        assert "multi-factor" in answer_lc or "penetration testing" in answer_lc, (
            f"Synthesized answer doesn't surface the planted mitigation. "
            f"Semantic terms: {payload['semantic_terms']!r}. "
            f"Answer: {payload['answer']!r}"
        )

        # Cost envelope — rewrite (~$0.001) + filter (~$0.005) +
        # synthesis (~$0.01) under $0.05 ceiling.
        assert 0 < payload["total_cost_usd"] < 0.05, (
            f"total_cost_usd={payload['total_cost_usd']} outside sanity gate ($0.05)"
        )

        # The rewrite produced a non-empty term list (we already
        # asserted) — print it for the agent report.
        print(
            f"\n[semantic] rewrite terms: {payload['semantic_terms']!r}\n"
            f"           total cost ${payload['total_cost_usd']:.4f}"
        )

    async def test_token_mode_surfaces_low_recall_warning(self, runtime, context) -> None:
        """select_by='token' selector_arg='cyber' on the PA6 doc
        must surface the low-recall warning in structuredContent.

        The doc body never contains "cyber" verbatim — token
        selector enumerates 0 candidates. Question is > 6 words.
        Both gates fire → warning attaches."""
        from kaos_agents.tools.findings import AgentFindingsTool

        artifact_id = await _store_doc(runtime, context, _build_cyber_doc())

        tool = AgentFindingsTool()
        result = await tool.execute(
            {
                "artifact_id": artifact_id,
                "question": "What is the cyber risk mitigation plan in this deck?",
                "select_by": "token",
                "selector_arg": "cyber",
                "filter_model": critic_model(),
                "synthesis_model": critic_model(),
                "chunk_size": 20,
            },
            context,
        )

        assert not result.isError, f"Tool errored: {result.text}"
        payload = result.structuredContent
        assert payload is not None

        # Phase 1 enumerated zero candidates (the literal "cyber"
        # is not in the body of the doc).
        assert payload["total_enumerated"] == 0, (
            f"Token selector expected to miss; enumerated="
            f"{payload['total_enumerated']}. PA6 reproduction "
            "broken — the doc body must not contain 'cyber'."
        )

        # Refusal contract still fires.
        assert payload["refusal_reason"] == "no_candidates_enumerated", (
            f"Expected refusal=no_candidates_enumerated, got {payload['refusal_reason']!r}"
        )

        # The Sprint-2 #6 warning is present.
        warnings = payload["warnings"]
        assert isinstance(warnings, list)
        assert len(warnings) >= 1, (
            "Low-recall warning did not fire on the PA6 failure "
            "case — Sprint-2 #6 transparency contract broken"
        )
        warn = warnings[0]
        assert warn["kind"] == "low_recall_token_selector"
        assert "semantic" in warn["message"]
        assert "cyber" in warn["message"]
        assert warn["details"]["candidate_count"] == 0
        assert warn["details"]["selector_arg"] == "cyber"
