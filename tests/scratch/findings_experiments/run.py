"""CS-B family experiment: does FindingsAgent fix the failures?

Three failure flavors from the corpus-stress suite, all reduced to
minimal single-file fixtures so the experiment runs in ~2 min and
costs under $0.20. No VFS, no tool plumbing — just a hand-built
``ContentDocument`` + ``DocumentView`` + ``FindingsAgent.run()``.

What we measure per scenario:

* **needle_present**: did the synthesized answer text contain the
  planted needle verbatim? (Same standard as the corpus-stress suite.)
* **findings_count**: how many candidates survived Phase 2 filter?
* **filter_cost_usd / synthesis_cost_usd**: spend breakdown.
* **refusal**: did FindingsAgent emit a structured refusal? When the
  pipeline can't ground an answer, the contract is "say so" — that
  beats both fabrication (CS-B2) and silent give-up (CS-B3).

The ChatAgent comparison side is **not** rebuilt here — we already
have empirical data on the failing baseline from the corpus-stress
suite. The hypothesis being tested is "does FindingsAgent close the
gap when called directly on the right content?" If yes, the
remediation is dispatch ("when do we route to FindingsAgent?"). If
no, the remediation is contract ("FindingsAgent itself needs to be
tighter").

Run::

    cd /home/mjbommar/projects/273v/kaos-agents
    KAOS_TEST_RESPOND_MODEL=anthropic:claude-sonnet-4-6 \\
    KAOS_TEST_CRITIC_MODEL=anthropic:claude-sonnet-4-6 \\
        uv run python tests/scratch/findings_experiments/run.py
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

# Make tests.integration._models importable (test-tier helper).
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
)
from tests.integration._models import critic_model, respond_model

# ─── Model selection ────────────────────────────────────────────────

FILTER_MODEL = critic_model()  # Phase 2 — relevance scoring
SYNTH_MODEL = respond_model()  # Phase 3 — answer composition


# ─── Scenario fixtures ──────────────────────────────────────────────


@dataclass(frozen=True)
class Scenario:
    name: str
    description: str
    paragraphs: tuple[str, ...]
    question: str
    needle: str  # The exact string the answer must contain
    """The planted fact (verbatim) that the synthesis must surface."""


# CS-B3 / S05 — give-up cliff. Three-doc shape collapsed to one doc;
# the JSON line is one paragraph among unrelated others. Tests
# whether FindingsAgent can locate the value when the question
# is unambiguous.
S05 = Scenario(
    name="S05_give_up_cliff",
    description="JSON config_token in a mixed-format doc (CS-B3).",
    paragraphs=(
        "Release 2.4.1 ships with bug fixes and performance improvements.",
        "Run the installer with --quiet for unattended mode. "
        "Logs land in /var/log/kaos/.",
        'The configuration JSON contains: "environment": "production", '
        '"config_token": "KAOS-S05-JSON-OK", "feature_flags": {"corpus_v2": true}.',
    ),
    question=(
        "What is the value of 'config_token' in the JSON configuration? "
        "Answer with the exact token string."
    ),
    needle="KAOS-S05-JSON-OK",
)

# CS-B2 / S03-surrogate — hallucination. Real S03 needs OCR; we
# substitute a text doc where the planted needle competes with
# plausible-but-wrong distractors elsewhere in the corpus. The
# failure mode we're probing: does the LLM invent a different
# codename ("NIGHTFALL") or surface the actual one ("7K-FALCON-2026")?
S03_SURROGATE = Scenario(
    name="S03_surrogate_hallucination",
    description="Project Oryx codename surrounded by plausible distractors (CS-B2).",
    paragraphs=(
        "INTERNAL MEMO — DO NOT DISTRIBUTE",
        "Last quarter we evaluated three codename candidates for "
        "Project Oryx. NIGHTFALL was used as a working title during "
        "early scoping but was discarded after the legal review.",
        "MOONRAKER was proposed by the product team but rejected "
        "for trademark reasons.",
        "PHOENIX was held in reserve for a future project.",
        "After review, the Project Oryx codename was finalized as "
        "7K-FALCON-2026. Distribution: leadership only.",
        "All earlier working titles should be retired from external "
        "communications effective immediately.",
    ),
    question=(
        "What is the final Project Oryx codename? Quote it verbatim."
    ),
    needle="7K-FALCON-2026",
)

# CS-B / S07-simplified — multi-needle synthesis. Three facts in
# one document. Tests whether FindingsAgent's single synthesis
# pass can assemble the multi-fact answer or whether it drops one.
S07_MULTI = Scenario(
    name="S07_multi_needle_synthesis",
    description="Three project facts in one doc.",
    paragraphs=(
        "PROJECT BRIEF — Q4 2026",
        "Project Alpha closed its Series B funding round at $42M, "
        "led by Greylock with participation from Sequoia.",
        "Mid-year reporting shows steady growth across all business units.",
        "Project Bravo launch date is set for 2026-11-01, contingent "
        "on the final security review.",
        "The engineering team has flagged three integration risks but "
        "none are P0.",
        "Project Charlie has 17 enrolled customers as of this writing, "
        "up from 12 last quarter.",
        "Churn remains under 4% and NPS continues to trend positive.",
    ),
    question=(
        "For each of Projects Alpha, Bravo, and Charlie, report the "
        "single key fact stated in this brief. Be specific."
    ),
    # We check three needles for S07 — see _check_needles below.
    needle="$42M | 2026-11-01 | 17",
)

ALL_SCENARIOS = (S05, S03_SURROGATE, S07_MULTI)


# ─── Helpers ────────────────────────────────────────────────────────


def build_view(scenario: Scenario) -> DocumentView:
    """One ``ContentDocument``, one ``Paragraph`` per scenario string.

    Each Paragraph holds a single ``Text`` run. We don't need
    structure beyond that for the experiment — the
    ``every_sentence_selector`` will further segment via the punkt
    tokenizer.
    """
    doc = ContentDocument(
        body=tuple(
            Paragraph(children=(Text(value=p),)) for p in scenario.paragraphs
        )
    )
    return DocumentView(doc, sentence_segmenter=get_default_punkt_tokenizer())


def _check_needles(scenario: Scenario, answer: str) -> tuple[bool, list[str]]:
    """Return ``(all_present, missing_needles)``.

    Multi-needle scenarios pack `|` between their planted facts;
    single-needle scenarios just have one string.
    """
    needles = [n.strip() for n in scenario.needle.split("|")]
    missing = [n for n in needles if n not in answer]
    return (not missing, missing)


def _format_result(scenario: Scenario, result: FindingsResult, elapsed: float) -> dict:
    """Snapshot the result into a JSON-safe summary record."""
    all_present, missing = _check_needles(scenario, result.answer)
    return {
        "scenario": scenario.name,
        "description": scenario.description,
        "question": scenario.question,
        "needle": scenario.needle,
        "needle_present": all_present,
        "missing_needles": missing,
        "answer_head": result.answer[:400],
        "answer_length": len(result.answer),
        "total_enumerated": result.total_enumerated,
        "total_filtered": result.total_filtered,
        "filter_cost_usd": round(result.filter_cost_usd, 4),
        "synthesis_cost_usd": round(result.synthesis_cost_usd, 4),
        "total_cost_usd": round(
            result.filter_cost_usd + result.synthesis_cost_usd, 4
        ),
        "elapsed_seconds": round(elapsed, 2),
        "refused": result.refusal is not None,
        "refusal_reason": (
            result.refusal.reason if result.refusal else None
        ),
        "models": {
            "filter": FILTER_MODEL,
            "synthesis": SYNTH_MODEL,
        },
    }


# ─── Experiment driver ──────────────────────────────────────────────


async def run_scenario(scenario: Scenario) -> dict:
    """Execute one FindingsAgent run against a scenario fixture."""
    print(f"\n{'=' * 70}")
    print(f"  {scenario.name}")
    print(f"  {scenario.description}")
    print(f"{'=' * 70}")
    print(f"  question: {scenario.question}")
    print(f"  needle:   {scenario.needle}")
    print(f"  models:   filter={FILTER_MODEL} synth={SYNTH_MODEL}")

    view = build_view(scenario)
    agent = FindingsAgent(
        selector=every_sentence_selector,
        filter_model=FILTER_MODEL,
        synthesis_model=SYNTH_MODEL,
        chunk_size=20,
        num_parallel=3,
        relevance_threshold=0.4,
    )

    t0 = time.monotonic()
    result = await agent.run(scenario.question, view)
    elapsed = time.monotonic() - t0

    summary = _format_result(scenario, result, elapsed)
    print(
        f"  ✓ needle_present={summary['needle_present']} "
        f"survivors={summary['total_filtered']}/{summary['total_enumerated']} "
        f"cost=${summary['total_cost_usd']:.4f} "
        f"time={summary['elapsed_seconds']:.1f}s"
    )
    if not summary["needle_present"]:
        print(f"  ✗ MISSING: {summary['missing_needles']}")
    print(f"  answer: {summary['answer_head']!r}")
    return summary


async def main() -> int:
    print(f"\nCS-B family experiment — {len(ALL_SCENARIOS)} scenarios")
    print(f"Filter model:     {FILTER_MODEL}")
    print(f"Synthesis model:  {SYNTH_MODEL}\n")

    results = []
    for scenario in ALL_SCENARIOS:
        try:
            summary = await run_scenario(scenario)
        except Exception as exc:
            print(f"  ✗ EXCEPTION: {type(exc).__name__}: {exc}")
            summary = {
                "scenario": scenario.name,
                "error": f"{type(exc).__name__}: {exc}",
            }
        results.append(summary)

    # Aggregate
    total_cost = sum(r.get("total_cost_usd", 0.0) for r in results)
    n_pass = sum(1 for r in results if r.get("needle_present"))
    print(f"\n{'=' * 70}")
    print(f"  SUMMARY: {n_pass}/{len(results)} needles present, "
          f"total cost ${total_cost:.4f}")
    print(f"{'=' * 70}")

    # Persist for later comparison.
    out_path = Path(__file__).parent / "results.jsonl"
    with out_path.open("a") as fh:
        for r in results:
            fh.write(json.dumps(r) + "\n")
    print(f"  → wrote {out_path}\n")

    return 0 if n_pass == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
