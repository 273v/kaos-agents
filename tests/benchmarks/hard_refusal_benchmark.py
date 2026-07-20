"""Hard-refusal benchmark — adversarial 'doc is here, answer isn't' refusals.

Sister benchmark to ``multiformat_e2e.py`` that uses
``kaos-llm-core/tests/fixtures/multiformat-corpus/hard-refusal-questions.jsonl``
instead of the standard 12-question fixture. Every question is
``answerable=false`` and follows the **mf08 pattern**: the relevant
document IS in the corpus but the specific answer is NOT.

Examples (paraphrased):
- "What is the penalty for using SHALL incorrectly?" — RFC 2119 is
  in the corpus; it defines SHALL but does not specify a penalty.
- "What algorithms are approved for password hashing?" — NIST 800-63B
  is in the corpus; it covers length and composition but not algorithms.
- "What is the statute of limitations for a 10b-5 action?" — the rule
  text is in the corpus; the statute of limitations is in 28 USC 1658(b),
  which is NOT in the corpus.

This is the failure mode that distinguishes a research agent from a
hallucination machine. Easy refusals (Voyager 2 / GDPR / Delaware
filing fee — completely off-topic) are layups; hard refusals require
the agent to recognize "I have the relevant document but it doesn't
answer this specific question." A law firm cannot deploy an agent
that cannot tell the difference.

Usage::

    # Default — fuzzy refusal matching (quick smoke).
    uv run python tests/benchmarks/hard_refusal_benchmark.py

    # LLM judge (recommended — more reliable refusal detection).
    uv run python tests/benchmarks/hard_refusal_benchmark.py --judge llm

    # Save results.
    uv run python tests/benchmarks/hard_refusal_benchmark.py --judge llm --json results.json

The benchmark loads the same multiformat-corpus directory as
``multiformat_e2e.py`` so the agent has access to the same documents.
Only the question file changes.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

# Reuse all of multiformat_e2e's machinery — load_questions is the only
# thing we override.
from tests.benchmarks.multiformat_e2e import (
    _CORPUS_DIR,
    _DEFAULT_JUDGE_MODEL,
    run_benchmark,
)

_HARD_QUESTIONS_PATH = _CORPUS_DIR / "hard-refusal-questions.jsonl"


def _load_hard_questions() -> list[dict]:
    questions = []
    if not _HARD_QUESTIONS_PATH.exists():
        msg = (
            f"Hard-refusal questions fixture not found at "
            f"{_HARD_QUESTIONS_PATH}. The file ships in "
            f"kaos-llm-core/tests/fixtures/multiformat-corpus/. "
            f"Make sure kaos-llm-core is checked out."
        )
        raise FileNotFoundError(msg)
    with _HARD_QUESTIONS_PATH.open() as f:
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
        help=(
            "Scoring mode: 'fuzzy' (default), 'llm' (claude-haiku-4-5), "
            "'llm:<model>' (explicit model spec), or 'none' (skip)."
        ),
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

    if not _CORPUS_DIR.exists():
        sys.stderr.write(f"Corpus not found: {_CORPUS_DIR}\n")
        sys.exit(1)
    if not _HARD_QUESTIONS_PATH.exists():
        sys.stderr.write(f"Hard-refusal fixture not found: {_HARD_QUESTIONS_PATH}\n")
        sys.exit(1)

    # Monkey-patch the question loader on the imported module so
    # run_benchmark picks up our hard-refusal questions instead of the
    # standard multiformat ones. Cleaner than copy-pasting run_benchmark.
    from tests.benchmarks import multiformat_e2e

    multiformat_e2e._load_questions = _load_hard_questions

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
    sys.stdout.write("HARD-REFUSAL BENCHMARK\n")
    sys.stdout.write(f"{'=' * 60}\n")
    sys.stdout.write(f"Documents:        {result.n_documents} ({_CORPUS_DIR.name})\n")
    sys.stdout.write(f"Questions:        {result.n_questions}\n")
    sys.stdout.write(f"Correct refusals: {result.n_correct_refusals}\n")
    sys.stdout.write(f"Wrong answers:    {result.n_wrong_answers}\n")
    sys.stdout.write(f"Errors:           {result.n_errors}\n")
    refusal_recall = result.n_correct_refusals / result.n_questions if result.n_questions else 0.0
    sys.stdout.write(f"Refusal recall:   {refusal_recall:.1%}\n")
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
        json_path = str(benchmarks_dir / f"hard-refusal-{today}.json")
    out_path = Path(json_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    output = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "corpus_dir": str(_CORPUS_DIR),
        "questions_file": str(_HARD_QUESTIONS_PATH),
        "n_documents": result.n_documents,
        "summary": {
            "n_questions": result.n_questions,
            "n_correct_refusals": result.n_correct_refusals,
            "n_wrong_answers": result.n_wrong_answers,
            "n_errors": result.n_errors,
            "refusal_recall": refusal_recall,
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
