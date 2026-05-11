"""Sprint-2 #5 — Live consistency / quality gate for FindingsAgent.

The trust-skeptic Probe 1 (commit ``47b5f7d``,
``docs/design/skeptic-trust-findings.md``) ran 5 passes of K7 over
the same NDA + same question and measured surviving-text Jaccard of
0.84-0.92 across 10 pairwise comparisons, with 1-2 obligations
dropped between runs. Two associates running the same query would
notice the missing items during cite-check.

After Sprint-2 #5 lands, the same probe must measure surviving
finding_id-set Jaccard >= 0.95. The 5-run test below is the
acceptance gate.

Total live spend on this file: ~5 * $0.01 (Haiku) ~ $0.05 per run
of the suite. The 2-run union test adds another ~$0.02. Cap is
conservatively $0.30 — if cost spikes much above that, something
regressed (provider switched away from Haiku, chunk_size blew up,
etc.).
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from kaos_agents.patterns.findings import (
    FindingsAgent,
    every_sentence_selector,
    sentences_with_token_selector,
)

NDA_DIR = Path.home() / "projects" / "273v" / "kelvin-app" / "samples" / "docx"
ACME_NDA = NDA_DIR / "MNDA - Acme.docx"

requires_anthropic = pytest.mark.skipif(
    "ANTHROPIC_API_KEY" not in os.environ,
    reason="ANTHROPIC_API_KEY missing",
)

requires_acme_nda = pytest.mark.skipif(
    not ACME_NDA.exists(),
    reason=f"Acme NDA fixture missing at {ACME_NDA}",
)


# Filter on Haiku, synthesis also on Haiku for the consistency test —
# the trust-skeptic ran with Sonnet for synthesis but the quality
# claim under test ("two associates see the same surviving set") is a
# Phase-2 property. Using Haiku for both keeps the live spend bounded
# (~$0.01/run instead of ~$0.03/run) while still proving the same
# determinism gate that costs $0.15+ on Sonnet.
INNER_MODEL = "anthropic:claude-haiku-4-5"
SYNTH_MODEL = "anthropic:claude-haiku-4-5"


def _view_for(docx_path: Path):
    from kaos_content.views import DocumentView
    from kaos_nlp_core._defaults import get_default_punkt_tokenizer
    from kaos_office import parse_docx

    doc = parse_docx(str(docx_path))
    return DocumentView(doc, sentence_segmenter=get_default_punkt_tokenizer())


def _pairwise_jaccard(sets: list[set[str]]) -> tuple[float, float, float]:
    """Min/median/max Jaccard across all ordered pairs of ``sets``."""
    if len(sets) < 2:
        return 1.0, 1.0, 1.0
    scores: list[float] = []
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            a = sets[i]
            b = sets[j]
            union = a | b
            scores.append(len(a & b) / len(union) if union else 1.0)
    scores.sort()
    return scores[0], scores[len(scores) // 2], scores[-1]


@pytest.mark.live
@requires_anthropic
@requires_acme_nda
class TestFindingsConsistencyLive:
    """Five-run consistency gate — the quality bar from
    Sprint-2 #5 / trust-skeptic Probe 1."""

    async def test_five_run_finding_id_jaccard_above_threshold(self) -> None:
        """Run K7 5 times on the same NDA, same question, ``temperature=0``.

        Acceptance: 5-run Jaccard over surviving finding_id sets >= 0.95.

        Note the difference from the trust-skeptic baseline: that
        report measured Jaccard on *surviving text strings* (since
        the old ``uuid4()`` ids rotated per run). With deterministic
        finding_ids we can now compare on id-sets directly — the
        same sentence in two runs has the same id by construction,
        so any Jaccard gap reflects real disagreement on which
        sentences survive Phase 2, not id-rotation noise.
        """
        view = _view_for(ACME_NDA)
        question = "What are the obligations of the receiving party?"

        survivor_id_sets: list[set[str]] = []
        total_cost = 0.0
        for run_idx in range(5):
            agent = FindingsAgent(
                selector=every_sentence_selector,
                filter_model=INNER_MODEL,
                synthesis_model=SYNTH_MODEL,
                chunk_size=20,
                num_parallel=3,
                relevance_threshold=0.4,
                temperature=0.0,
            )
            result = await agent.run(question, view)
            survivor_id_sets.append({f.candidate.finding_id for f in result.findings})
            total_cost += result.total_cost_usd
            assert result.total_enumerated >= 50, (
                f"Run {run_idx}: Phase 1 enumerated only "
                f"{result.total_enumerated} candidates — selector regression?"
            )
            assert result.total_filtered >= 1, (
                f"Run {run_idx}: Phase 2 dropped everything — filter too strict?"
            )

        # All 5 runs must produce the same id-space (deterministic
        # selector), so the union is a sanity bound on disagreement.
        min_jacc, median_jacc, max_jacc = _pairwise_jaccard(survivor_id_sets)

        # Stamp the actual measurement onto the test output for
        # reproducibility / commit-message attribution.
        print(
            f"\nFindings consistency (5 runs, temperature=0):"
            f"\n  pairwise Jaccard min={min_jacc:.3f} "
            f"median={median_jacc:.3f} max={max_jacc:.3f}"
            f"\n  total live spend = ${total_cost:.4f}"
        )

        assert min_jacc >= 0.95, (
            f"Sprint-2 #5 quality gate failed: minimum pairwise "
            f"Jaccard = {min_jacc:.3f} < 0.95. Two associates running "
            f"the same query would still see materially different "
            f"surviving sets. Survivor counts: "
            f"{[len(s) for s in survivor_id_sets]!r}."
        )

        # Cost sanity: ~$0.01/run * 5 ~ $0.05; allow 6x headroom.
        assert total_cost < 0.30, (
            f"5-run total cost ${total_cost:.4f} exceeded the $0.30 "
            f"sanity gate. Did the model default change?"
        )

    async def test_runs_union_mode_captures_indemnification(self) -> None:
        """Trust-skeptic Probe 1 saw indemnification covered in 4/5
        single-pass runs (dropped from synthesis once). The
        ``runs=2`` union mode unions surviving findings across both
        passes, which should capture the indemnification sentence
        every time on this NDA.

        The Acme MNDA has at least one sentence containing the
        substring ``indemnif`` (verified against the live trust-
        probe corpus — see ``docs/design/skeptic-trust-findings.md``
        survivor lists). With ``runs=2``, the union has to contain
        at least one survivor whose text matches ``indemnif``.
        """
        view = _view_for(ACME_NDA)
        agent = FindingsAgent(
            selector=sentences_with_token_selector("indemnif"),
            filter_model=INNER_MODEL,
            synthesis_model=SYNTH_MODEL,
            chunk_size=20,
            num_parallel=3,
            relevance_threshold=0.4,
            temperature=0.0,
            runs=2,
        )
        result = await agent.run("What does this agreement say about indemnification?", view)

        # Phase 1 (token selector) is deterministic; either the NDA
        # mentions indemnification or it doesn't. If it doesn't,
        # there's no quality claim to test — flag and skip cleanly.
        if result.total_enumerated == 0:
            pytest.skip(
                "Acme NDA has no 'indemnif' substring — re-fixture "
                "or pick a different NDA for this gate."
            )

        # Union mode -> filter_calls = 2 * (chunks for the selector).
        # We don't pin a specific number (depends on doc) but we
        # require >= 2 since runs=2.
        assert result.filter_calls >= 2, (
            f"runs=2 should issue >= 2 filter calls; saw {result.filter_calls}."
        )

        # The Phase-1 selector only returns sentences whose lowercase
        # text contains 'indemnif'; the union must therefore also
        # only contain such sentences.
        survivor_texts = [f.candidate.text.lower() for f in result.findings]
        assert any("indemnif" in t for t in survivor_texts), (
            f"Union mode dropped every indemnification sentence — "
            f"the indemnification clause was missed by both runs. "
            f"survivor count = {len(survivor_texts)}; "
            f"enumerated = {result.total_enumerated}."
        )

        # Cost sanity: 2 filter passes + 1 synthesis ≈ $0.02-0.03.
        assert result.total_cost_usd < 0.20, (
            f"runs=2 total cost ${result.total_cost_usd:.4f} exceeded the $0.20 sanity gate."
        )
