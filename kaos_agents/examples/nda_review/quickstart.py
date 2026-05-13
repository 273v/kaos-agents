"""60-second NDA-batch review demo — kaos-agents v0.1.0a1.

Reviews 5 mutual NDAs from a deal-room and surfaces:
  - Governing-law deviations from the house standard
  - Term-length variance across the batch
  - Confidentiality-period drift across the batch
  - The EMNA signature-block template artifact
  - Other deviations a careful senior associate would flag

What this demonstrates that a chat-bot cannot:
  - FindingsAgent (recall-first) over a multi-doc DOCX corpus —
    every sentence in every NDA is enumerated, then filtered, then
    synthesised into a typed answer
  - Typed findings with provenance: each surviving sentence carries
    a deterministic finding_id, document filename, block_ref, and
    page number — a partner can verify the cite
  - Refusal contract: when no candidate survives the filter, the
    agent returns FindingsRefusal instead of hallucinating
  - max_cost_usd as a contract: caps the per-document review and
    reports budget_exceeded truthfully when the cap binds
  - Audit trail: every LLM call routed through kaos-llm-core is
    captured (schema-v4 JSONL, redaction-by-default) under the
    configured recorder dir

Run (after `pip install 'kaos-agents[llm,office]'`)::

    python -m kaos_agents.examples.nda_review.quickstart

Cost: ~$0.05-0.10 end-to-end with the default Haiku 4.5 filter +
Sonnet 4.6 synthesis. Anthropic Haiku 4.5 alone is enough for the
filter step (the only quality-critical call is synthesis); switch
``SYNTHESIS_MODEL`` below to ``claude-haiku-4-5`` to drop to ~$0.02.

Requires ``ANTHROPIC_API_KEY`` in the environment.
"""

from __future__ import annotations

import asyncio
import os
from importlib.resources import files as _resource_files

from kaos_agents.patterns.findings import (
    FindingsAgent,
    FindingsResult,
    every_sentence_selector,
)

# The fixtures ship as package data so the demo works whether the
# user is in a monorepo checkout or a `pip install kaos-agents`. See
# ``kaos_agents/examples/nda_review/ndas/README.md`` for provenance.
NDAS_DIR = _resource_files("kaos_agents.examples.nda_review").joinpath("ndas")

# Anthropic Haiku 4.5 is the verified-green default per the v0.1.0a1
# provider matrix (README "Known limitations"). For production legal
# review where missing a clause is unacceptable, swap to Sonnet 4.6
# for synthesis and consider runs=2 (union mode) for audit-grade
# consistency.
FILTER_MODEL = "anthropic:claude-haiku-4-5"
SYNTHESIS_MODEL = "anthropic:claude-sonnet-4-6"

# Cap the per-document review at $0.50. The FindingsAgent contract
# is strict at wave-level — if the cap binds, the agent stops
# dispatching new filter chunks and surfaces ``budget_exceeded=True``
# instead of overshooting silently.
MAX_COST_PER_DOC_USD = 0.50

# The senior-counsel question a partner would actually ask before
# signing a batch of NDAs from the deal-room. Phrased to cover the
# four deviation classes the ground-truth memo documents
# (`tests/integration/ladder/fixtures/nda_ground_truth.py`).
REVIEW_QUESTION = (
    "Review this NDA for deviations from a standard mutual NDA. "
    "Identify (1) governing law and venue, (2) term length, (3) "
    "confidentiality survival period, (4) presence and scope of "
    "any non-solicitation clause, and (5) any template artifacts "
    "or drafting anomalies a careful reviewer would flag (e.g. a "
    "signature block naming a different party than the recitals). "
    "Cite the relevant text inline with the finding_id from each "
    "surviving sentence."
)


def _load_view(docx_path: str):
    """Parse a DOCX into a kaos-content DocumentView.

    The lazy import keeps the example importable even when
    ``kaos_office`` is not installed — the install hint surfaces
    cleanly from ``run()`` rather than at module load.
    """
    from kaos_content.views import DocumentView
    from kaos_nlp_core._defaults import get_default_punkt_tokenizer
    from kaos_office import parse_docx

    doc = parse_docx(docx_path)
    return DocumentView(doc, sentence_segmenter=get_default_punkt_tokenizer())


async def review_one(name: str, docx_path: str) -> tuple[str, FindingsResult]:
    """Review a single NDA and return ``(filename, FindingsResult)``."""
    view = _load_view(docx_path)
    agent = FindingsAgent(
        selector=every_sentence_selector,
        filter_model=FILTER_MODEL,
        synthesis_model=SYNTHESIS_MODEL,
        chunk_size=20,
        num_parallel=3,
        relevance_threshold=0.4,
        max_cost_usd=MAX_COST_PER_DOC_USD,
    )
    result = await agent.run(REVIEW_QUESTION, view)
    return name, result


async def run() -> dict[str, FindingsResult]:
    """Review all 5 NDAs and return a ``{filename: FindingsResult}`` map.

    Exposed as a public coroutine so the live regression harness
    (``tests/integration/test_quickstart_nda_review_live.py``) can
    drive the example without re-encoding its logic.
    """
    if "ANTHROPIC_API_KEY" not in os.environ:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. Export it (or pass via your "
            "shell config) before running the quickstart. The verified "
            "v0.1.0a1 provider for FindingsAgent is Anthropic Haiku 4.5; "
            "see README 'Known limitations' for the full provider matrix."
        )

    nda_paths = sorted(p for p in NDAS_DIR.iterdir() if p.name.endswith(".docx"))
    if len(nda_paths) != 5:
        raise RuntimeError(
            f"Expected 5 NDA fixtures under {NDAS_DIR}, found {len(nda_paths)}. "
            "The fixtures ship as package data — reinstall kaos-agents."
        )

    print(f"Loaded 5 NDAs: {', '.join(p.name for p in nda_paths)}")

    # Run the 5 reviews concurrently — five independent FindingsAgent
    # pipelines, one per NDA. The agent is stateless across calls.
    results = await asyncio.gather(*(review_one(p.name, str(p)) for p in nda_paths))
    return dict(results)


def print_report(results: dict[str, FindingsResult]) -> None:
    """Pretty-print the batch review."""
    total_cost = 0.0
    total_findings = 0
    print()
    print("=" * 72)
    print("NDA BATCH REVIEW")
    print("=" * 72)
    for name, result in results.items():
        # result is a FindingsResult; the attributes below are the
        # public typed contract on
        # kaos_agents.patterns.findings.FindingsResult.
        cost = result.total_cost_usd
        total_cost += cost
        total_findings += len(result.findings)

        print()
        print(f"--- {name} ---")
        refusal = result.refusal
        if refusal is not None:
            # Honest refusal — the agent looked at every sentence and
            # determined the document does not answer the question.
            print(f"  REFUSAL: {refusal.reason}")
            print(f"  {refusal.message}")
            continue

        print(f"  Cost: ${cost:.4f}  budget_exceeded={result.budget_exceeded}")
        print(
            f"  Enumerated {result.total_enumerated} sentences, "
            f"{result.total_filtered} survived the filter."
        )
        print()
        print("  Synthesis:")
        # Indent the answer so it reads as a block.
        for line in result.answer.splitlines():
            print(f"    {line}")
        print()
        # Top 5 surviving findings with provenance — the cites a
        # partner can verify.
        top_findings = sorted(result.findings, key=lambda f: f.relevance, reverse=True)[:5]
        if top_findings:
            print("  Top surviving findings (with provenance):")
            for f in top_findings:
                page = f.candidate.page
                fid = f.candidate.finding_id
                snippet = f.candidate.text.strip().replace("\n", " ")
                if len(snippet) > 110:
                    snippet = snippet[:107] + "..."
                page_str = f"p.{page}" if page is not None else "p.?"
                print(f"    [{fid}]({page_str}) score={f.relevance:.2f}  {snippet}")

    print()
    print("=" * 72)
    print(f"TOTAL: {total_findings} findings across 5 NDAs, ${total_cost:.4f} spent.")
    print("=" * 72)
    print()
    print("Next steps:")
    print("  - See kaos_agents/examples/nda_review/ndas/README.md for fixture provenance")
    print("  - Read CLAUDE.md for the 8-step turn-loop design and the 15-event taxonomy")
    print(
        "  - For production legal review: bump synthesis to Sonnet 4.6 "
        "with runs=2 (audit-grade consistency)"
    )


def main() -> int:
    """Sync entry point — exists so this module is runnable via
    ``python -m kaos_agents.examples.nda_review.quickstart``.
    """
    results = asyncio.run(run())
    print_report(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
