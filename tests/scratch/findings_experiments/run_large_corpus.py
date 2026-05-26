"""CS-B family experiment 2: 50-doc S07 stress on FindingsAgent.

Experiment 1 (``run.py``) proved FindingsAgent recovers the planted
needles in tight single-doc fixtures. This experiment asks the
scaling question: when the corpus is 53 docs (50 distractors + 3
needles) flattened into one ``DocumentView``, does Phase 1
enumeration still surface the right candidates, and does Phase 2
filter still cull the noise efficiently?

The real corpus-stress S07 generates 50 distractors via
``synth_corpus`` and sprinkles 3 needle docs through different file
formats. For this experiment we skip the file-format mux (FindingsAgent
operates on ``ContentDocument``, not file bytes) and instead concentrate
on the **scale** axis: how does enumeration cost + filter cost grow,
and does the synthesis still pick the right citations?

Three variants:

* **S07_50doc_concat** — 53 docs concatenated into one
  ContentDocument, every_sentence_selector across the whole thing.
* **S07_50doc_token** — same corpus, but a
  ``sentences_with_token_selector("project")`` to test whether a
  cheaper Phase 1 still recovers all 3 needles. Catches the case
  where every_sentence_selector is over-eager.
* **S07_50doc_pertoken** — three separate FindingsAgent runs, one
  per project name. Tests "narrow first, then synthesize" vs the
  monolithic approach.

What we measure beyond the basics from run.py: cost growth ratio
vs the single-doc baseline, latency, whether any needle is dropped
when distractors are present.

Run::

    cd /home/mjbommar/projects/273v/kaos-agents
    KAOS_TEST_RESPOND_MODEL=anthropic:claude-sonnet-4-6 \\
    KAOS_TEST_CRITIC_MODEL=anthropic:claude-sonnet-4-6 \\
        uv run python tests/scratch/findings_experiments/run_large_corpus.py
"""

from __future__ import annotations

import asyncio
import json
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from kaos_content.model.blocks import Paragraph
from kaos_content.model.document import ContentDocument
from kaos_content.model.inlines import Text
from kaos_content.views.document_view import DocumentView
from kaos_nlp_core._defaults import get_default_punkt_tokenizer

from kaos_agents.patterns.findings import (
    FindingsAgent,
    FindingsResult,
    every_sentence_selector,
    sentences_with_token_selector,
)
from tests.integration._models import critic_model, respond_model

FILTER_MODEL = critic_model()
SYNTH_MODEL = respond_model()


# ─── Distractor generator ───────────────────────────────────────────
#
# Cheap, deterministic, topically plausible. Five paragraphs each
# (matches the real synth_corpus default), all related to either
# finance or engineering so the cluster signal is preserved. No
# planted needles in distractors by construction.

_FINANCE_BITS = (
    "Q3 revenue grew {pct}% year-over-year to ${amount}M, driven by "
    "strong performance in the enterprise segment.",
    "Gross margin held steady at {gm}% despite ongoing supply-chain "
    "pressures and a {fx} basis-point FX headwind.",
    "Operating expenses came in at ${opex}M, slightly below the "
    "midpoint of guidance.",
    "Cash and equivalents at quarter-end totaled ${cash}M, providing "
    "{months} months of runway at current burn.",
    "We maintained net retention above {nrr}% across our top-50 "
    "customer cohort, consistent with prior quarters.",
)
_ENGINEERING_BITS = (
    "The latency p99 for the inference path dropped to {ms}ms after "
    "the batch-coalescing change shipped in v{ver}.",
    "We rolled out the new schema migration over {days} days with "
    "zero customer-visible incidents.",
    "GPU utilization on the training fleet now averages {util}% across "
    "all clusters, up from {prev}% in the prior period.",
    "The retry budget for the eligibility-check service was tightened "
    "to {budget}% to reduce thundering-herd effects.",
    "We deprecated {n} unused API endpoints this cycle as part of the "
    "surface-area reduction initiative.",
)


def _distractor_paragraphs(rng: random.Random, cluster: str, n: int = 5) -> tuple[str, ...]:
    """Return n plausible-looking sentences for the given cluster."""
    bits = _FINANCE_BITS if cluster == "finance" else _ENGINEERING_BITS
    paragraphs: list[str] = []
    for _ in range(n):
        template = rng.choice(bits)
        formatted = template.format(
            pct=rng.randint(3, 27),
            amount=rng.randint(50, 850),
            gm=rng.randint(58, 76),
            fx=rng.randint(20, 180),
            opex=rng.randint(40, 220),
            cash=rng.randint(200, 1400),
            months=rng.randint(18, 48),
            nrr=rng.randint(108, 134),
            ms=rng.randint(35, 220),
            ver=rng.randint(2, 14),
            days=rng.randint(3, 21),
            util=rng.randint(58, 92),
            prev=rng.randint(40, 78),
            budget=rng.randint(2, 12),
            n=rng.randint(4, 23),
        )
        paragraphs.append(formatted)
    return tuple(paragraphs)


# ─── Needle docs ────────────────────────────────────────────────────

NEEDLE_DOCS: tuple[tuple[str, str], ...] = (
    (
        "Project Alpha — Funding",
        "Project Alpha closed its Series B funding round at $42M, "
        "led by Greylock with participation from Sequoia.",
    ),
    (
        "Project Bravo — Launch",
        "Project Bravo launch date is set for 2026-11-01, contingent "
        "on the final security review.",
    ),
    (
        "Project Charlie — Customers",
        "Project Charlie has 17 enrolled customers as of this writing, "
        "up from 12 last quarter.",
    ),
)
NEEDLES = ("$42M", "2026-11-01", "17")


def _needle_paragraphs(rng: random.Random, header: str, needle_sentence: str) -> tuple[str, ...]:
    """Wrap each needle sentence in finance-flavored surroundings.

    Mirrors the real S07 layout: each needle doc has its own header +
    a few related-but-not-the-fact sentences + the planted needle.
    """
    return (
        header,
        rng.choice(_FINANCE_BITS).format(
            pct=rng.randint(3, 27), amount=rng.randint(50, 850),
            gm=rng.randint(58, 76), fx=rng.randint(20, 180),
            opex=rng.randint(40, 220), cash=rng.randint(200, 1400),
            months=rng.randint(18, 48), nrr=rng.randint(108, 134),
        ),
        needle_sentence,
        "Additional context is available in the appendix to the "
        "operating-committee deck.",
    )


# ─── Corpus assembly ────────────────────────────────────────────────


def build_large_corpus(seed: int = 707, n_distractors: int = 50) -> ContentDocument:
    """Build one ContentDocument from 50 distractor docs + 3 needle docs.

    Needles are sprinkled at positions 10, 25, 40 (mirroring the real
    S07 layout). Each doc contributes a "document header" paragraph
    + body paragraphs so the corpus reads as a multi-doc concatenation.
    """
    rng = random.Random(seed)
    blocks: list[Paragraph] = []

    # Pre-generate distractor docs.
    distractor_docs: list[tuple[str, tuple[str, ...]]] = []
    for i in range(n_distractors):
        cluster = "finance" if i % 2 == 0 else "engineering"
        header = f"Document {i + 1:04d} ({cluster})"
        body = _distractor_paragraphs(rng, cluster)
        distractor_docs.append((header, body))

    # Pre-generate needle docs.
    needle_docs: list[tuple[str, tuple[str, ...]]] = []
    for header, needle_sentence in NEEDLE_DOCS:
        body = _needle_paragraphs(rng, header, needle_sentence)
        needle_docs.append((header, body))

    # Sprinkle layout: 10 distractors, needle 0, 15 distractors,
    # needle 1, 15 distractors, needle 2, remaining distractors.
    interleaved: list[tuple[str, tuple[str, ...]]] = []
    interleaved.extend(distractor_docs[:10])
    interleaved.append(needle_docs[0])
    interleaved.extend(distractor_docs[10:25])
    interleaved.append(needle_docs[1])
    interleaved.extend(distractor_docs[25:40])
    interleaved.append(needle_docs[2])
    interleaved.extend(distractor_docs[40:])

    # Flatten into one ContentDocument. Each doc becomes a header
    # paragraph + body paragraphs.
    for header, body in interleaved:
        blocks.append(Paragraph(children=(Text(value=f"=== {header} ==="),)))
        for paragraph_text in body:
            blocks.append(Paragraph(children=(Text(value=paragraph_text),)))

    return ContentDocument(body=tuple(blocks))


# ─── Experiment variants ────────────────────────────────────────────


@dataclass(frozen=True)
class Variant:
    name: str
    description: str
    """Distinguishes how Phase 1 is set up."""


VARIANTS = (
    Variant(
        name="every_sentence",
        description=(
            "every_sentence_selector across the entire 53-doc concat. "
            "Maximum recall, highest filter cost."
        ),
    ),
    Variant(
        name="token_project",
        description=(
            "sentences_with_token_selector('project') — Phase 1 narrows "
            "to sentences mentioning 'project'. Lower recall, lower cost."
        ),
    ),
)


_QUESTION = (
    "The corpus contains documents about three projects: Alpha, "
    "Bravo, and Charlie. For each, report the single key fact "
    "stated in the corpus. Be specific."
)


# ─── Driver ────────────────────────────────────────────────────────


def _check_needles(answer: str) -> tuple[bool, list[str]]:
    missing = [n for n in NEEDLES if n not in answer]
    return (not missing, missing)


async def run_variant(variant: Variant, view: DocumentView) -> dict:
    print(f"\n{'=' * 70}")
    print(f"  S07_50doc_{variant.name}")
    print(f"  {variant.description}")
    print(f"{'=' * 70}")

    selector = (
        every_sentence_selector
        if variant.name == "every_sentence"
        else sentences_with_token_selector("project")
    )
    agent = FindingsAgent(
        selector=selector,
        filter_model=FILTER_MODEL,
        synthesis_model=SYNTH_MODEL,
        chunk_size=20,
        num_parallel=4,
        relevance_threshold=0.4,
    )

    t0 = time.monotonic()
    result: FindingsResult = await agent.run(_QUESTION, view)
    elapsed = time.monotonic() - t0

    all_present, missing = _check_needles(result.answer)
    summary = {
        "scenario": f"S07_50doc_{variant.name}",
        "description": variant.description,
        "needle_present": all_present,
        "missing_needles": missing,
        "answer_head": result.answer[:600],
        "answer_length": len(result.answer),
        "total_enumerated": result.total_enumerated,
        "total_filtered": result.total_filtered,
        "filter_calls": result.filter_calls,
        "filter_cost_usd": round(result.filter_cost_usd, 4),
        "synthesis_cost_usd": round(result.synthesis_cost_usd, 4),
        "total_cost_usd": round(
            result.filter_cost_usd + result.synthesis_cost_usd, 4
        ),
        "elapsed_seconds": round(elapsed, 2),
        "refused": result.refusal is not None,
        "models": {"filter": FILTER_MODEL, "synthesis": SYNTH_MODEL},
    }
    print(
        f"  ✓ needle_present={summary['needle_present']} "
        f"survivors={summary['total_filtered']}/{summary['total_enumerated']} "
        f"filter_calls={summary['filter_calls']} "
        f"cost=${summary['total_cost_usd']:.4f} "
        f"time={summary['elapsed_seconds']:.1f}s"
    )
    if not summary["needle_present"]:
        print(f"  ✗ MISSING: {summary['missing_needles']}")
    print(f"  answer: {summary['answer_head']!r}")
    return summary


async def main() -> int:
    print(f"\nS07 50-doc stress — {len(VARIANTS)} variants")
    print(f"Filter model:     {FILTER_MODEL}")
    print(f"Synthesis model:  {SYNTH_MODEL}\n")

    print("Building corpus...")
    t0 = time.monotonic()
    doc = build_large_corpus(seed=707, n_distractors=50)
    print(
        f"  built ContentDocument with {len(doc.body)} blocks "
        f"in {time.monotonic() - t0:.2f}s"
    )

    view = DocumentView(doc, sentence_segmenter=get_default_punkt_tokenizer())
    all_sentences = list(view.sentences)
    print(f"  segmented into {len(all_sentences)} sentences\n")

    results = []
    for variant in VARIANTS:
        try:
            summary = await run_variant(variant, view)
        except Exception as exc:
            print(f"  ✗ EXCEPTION: {type(exc).__name__}: {exc}")
            summary = {
                "scenario": f"S07_50doc_{variant.name}",
                "error": f"{type(exc).__name__}: {exc}",
            }
        results.append(summary)

    total_cost = sum(r.get("total_cost_usd", 0.0) for r in results)
    n_pass = sum(1 for r in results if r.get("needle_present"))
    print(f"\n{'=' * 70}")
    print(
        f"  SUMMARY: {n_pass}/{len(results)} all needles present, "
        f"total cost ${total_cost:.4f}"
    )
    print(f"{'=' * 70}")

    out_path = Path(__file__).parent / "results.jsonl"
    with out_path.open("a") as fh:
        for r in results:
            fh.write(json.dumps(r) + "\n")
    print(f"  → appended to {out_path}\n")
    return 0 if n_pass == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
