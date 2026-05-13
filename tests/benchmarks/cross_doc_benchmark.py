"""Cross-document synthesis benchmark — the actual law-firm workflow.

Sister benchmark to ``multiformat_e2e.py`` and ``hard_refusal_benchmark.py``.
Loads the 3-document EDGAR cross-doc corpus
(``kaos-llm-core/tests/fixtures/cross-doc-corpus/``) and runs questions
that REQUIRE the agent to reason across multiple documents:

- enumeration: "How many employment agreements are in this corpus?"
- comparison: "Compare the effective dates. Which is earlier?"
- entity rollup: "List all parties across all agreements."
- type disambiguation: "Which agreement involves real property?"
- role disambiguation: "Which agreement describes a CLCO position?"
- single-doc lookup with corpus pressure: "What is the Galera salary?"
- hard refusal across docs: "Compare non-compete durations" (clauses
  not present in the 80-line excerpts).
- hard refusal across docs: "Find conflicts between noncompete
  provisions" (same — operative text not in excerpt).

This is the gap multiformat_e2e.py and hard_refusal_benchmark.py both
miss: every question there is single-doc. Real legal workflows are
multi-document — comparing contracts, finding conflicts, listing
parties across a deal room.

Usage::

    # LLM judge — recommended.
    uv run python tests/benchmarks/cross_doc_benchmark.py --judge llm

    # Save results.
    uv run python tests/benchmarks/cross_doc_benchmark.py --judge llm --json results.json

The benchmark reuses ``multiformat_e2e.py``'s machinery via monkey-
patching the corpus path + question loader. No copy-pasted logic.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

from tests.benchmarks.multiformat_e2e import (
    _DEFAULT_JUDGE_MODEL,
    run_benchmark,
)

# Cross-doc corpus lives in kaos-llm-core/tests/fixtures/cross-doc-corpus.
# Paths are relative to the kaos-modules monorepo root (3 dirs up from
# this file: kaos-agents/tests/benchmarks/ → kaos-modules/).
_CROSS_DOC_DIR = (
    Path(__file__).resolve().parent.parent.parent.parent
    / "kaos-llm-core/tests/fixtures/cross-doc-corpus"
)
_CROSS_DOC_QUESTIONS = _CROSS_DOC_DIR / "cross-doc-questions.jsonl"


def _load_cross_doc_questions() -> list[dict]:
    if not _CROSS_DOC_QUESTIONS.exists():
        msg = (
            f"Cross-doc questions fixture not found at "
            f"{_CROSS_DOC_QUESTIONS}. The file ships in "
            f"kaos-llm-core/tests/fixtures/cross-doc-corpus/."
        )
        raise FileNotFoundError(msg)
    questions = []
    with _CROSS_DOC_QUESTIONS.open() as f:
        for line in f:
            line = line.strip()
            if line:
                questions.append(json.loads(line))
    return questions


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    parser.add_argument("--model", default=None)
    parser.add_argument("--json", default=None)
    parser.add_argument(
        "--judge",
        default="fuzzy",
        help=("Scoring mode: 'fuzzy', 'llm' (claude-haiku-4-5), 'llm:<model>', or 'none'."),
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    judge_mode = "fuzzy"
    judge_model = _DEFAULT_JUDGE_MODEL
    if args.judge == "none":
        judge_mode = "none"
    elif args.judge == "fuzzy":
        judge_mode = "fuzzy"
    elif args.judge == "llm":
        judge_mode = "llm"
    elif args.judge.startswith("llm:"):
        judge_mode = "llm"
        judge_model = args.judge[len("llm:") :]
    else:
        parser.error(f"unknown --judge value: {args.judge!r}")

    if not _CROSS_DOC_DIR.exists():
        sys.stderr.write(f"Cross-doc corpus not found: {_CROSS_DOC_DIR}\n")
        sys.exit(1)

    # Monkey-patch the corpus dir + question loader so multiformat_e2e's
    # run_benchmark uses the cross-doc fixtures instead of the standard
    # multiformat ones. Cleaner than reimplementing run_benchmark.
    from tests.benchmarks import multiformat_e2e

    multiformat_e2e._CORPUS_DIR = _CROSS_DOC_DIR  # type: ignore[attr-defined]
    multiformat_e2e._load_questions = _load_cross_doc_questions  # ty: ignore[invalid-assignment]

    result = asyncio.run(
        run_benchmark(
            model=args.model,
            verbose=args.verbose,
            judge_mode=judge_mode,
            judge_model=judge_model,
        )
    )
    result.judge_mode = judge_mode
    result.judge_model = judge_model if judge_mode == "llm" else ""

    sys.stdout.write(f"\n{'=' * 60}\n")
    sys.stdout.write("CROSS-DOCUMENT SYNTHESIS BENCHMARK\n")
    sys.stdout.write(f"{'=' * 60}\n")
    sys.stdout.write(f"Documents:        {result.n_documents} ({_CROSS_DOC_DIR.name})\n")
    sys.stdout.write(f"Questions:        {result.n_questions}\n")
    sys.stdout.write(f"Correct answers:  {result.n_correct_answers}\n")
    sys.stdout.write(f"Correct refusals: {result.n_correct_refusals}\n")
    sys.stdout.write(f"Wrong answers:    {result.n_wrong_answers}\n")
    sys.stdout.write(f"Wrong refusals:   {result.n_wrong_refusals}\n")
    sys.stdout.write(f"Errors:           {result.n_errors}\n")
    sys.stdout.write(f"Accuracy:         {result.accuracy:.1%}\n")
    sys.stdout.write(f"Avg latency:      {result.avg_latency_s:.1f}s\n")
    sys.stdout.write(f"Judge:            {result.judge_mode}")
    if result.judge_model:
        sys.stdout.write(f" ({result.judge_model})")
    sys.stdout.write("\n")
    if result.judge_total_cost_usd > 0:
        sys.stdout.write(f"Judge cost:       ${result.judge_total_cost_usd:.4f}\n")
    sys.stdout.flush()

    json_path = args.json
    if json_path is None:
        today = time.strftime("%Y-%m-%d")
        benchmarks_dir = Path(__file__).resolve().parent.parent / "docs" / "benchmarks"
        json_path = str(benchmarks_dir / f"cross-doc-{today}.json")
    out_path = Path(json_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    output = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "corpus_dir": str(_CROSS_DOC_DIR),
        "questions_file": str(_CROSS_DOC_QUESTIONS),
        "n_documents": result.n_documents,
        "summary": {
            "n_questions": result.n_questions,
            "n_correct_answers": result.n_correct_answers,
            "n_correct_refusals": result.n_correct_refusals,
            "n_wrong_answers": result.n_wrong_answers,
            "n_wrong_refusals": result.n_wrong_refusals,
            "n_errors": result.n_errors,
            "accuracy": result.accuracy,
            "avg_latency_s": result.avg_latency_s,
            "judge_mode": result.judge_mode,
            "judge_model": result.judge_model,
            "judge_total_cost_usd": result.judge_total_cost_usd,
        },
        "questions": [asdict(q) for q in result.questions],
    }
    out_path.write_text(json.dumps(output, indent=2))
    sys.stdout.write(f"\nSaved to {json_path}\n")


if __name__ == "__main__":
    main()
