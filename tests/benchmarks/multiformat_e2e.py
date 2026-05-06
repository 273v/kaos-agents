"""End-to-end multi-format document Q&A benchmark.

Tests the full pipeline: load mixed-format files → BM25 retrieval → RAG →
cited answer. Uses the WS-3.7 multiformat corpus (10 docs, 5 formats,
12 questions with ground truth).

This is the acceptance test for "can I dump a bunch of files into a folder
and get answers?"

Usage::

    # Run all 12 questions with the historical fuzzy-hint scorer
    uv run python tests/benchmarks/multiformat_e2e.py

    # Use the LLM-as-judge scorer (default model claude-haiku-4-5)
    uv run python tests/benchmarks/multiformat_e2e.py --judge llm

    # LLM judge with explicit model
    uv run python tests/benchmarks/multiformat_e2e.py --judge llm:anthropic:claude-haiku-4-5

    # Disable scoring entirely (just record agent outputs)
    uv run python tests/benchmarks/multiformat_e2e.py --judge none

    # Save results
    uv run python tests/benchmarks/multiformat_e2e.py --json results.json

    # Verbose (show agent events)
    uv run python tests/benchmarks/multiformat_e2e.py -v
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

_CORPUS_DIR = Path(__file__).resolve().parent.parent.parent.parent / (
    "kaos-llm-core/tests/fixtures/multiformat-corpus"
)
_QUESTIONS_PATH = _CORPUS_DIR / "multiformat-questions.jsonl"

_FUZZY_MATCH_THRESHOLD = 0.5  # BM25 score threshold (hint as query, answer as doc)
_WORD_OVERLAP_THRESHOLD = 0.5  # Fallback: 50% of hint content words in answer

_DEFAULT_JUDGE_MODEL = "anthropic:claude-haiku-4-5"

# Phrases that count as a refusal in the fuzzy scorer.
_REFUSAL_PHRASES = (
    "insufficient",
    "not find",
    "not contain",
    "no information",
    "not available",
    "cannot answer",
    "don't have",
    "do not have",
    "unable to answer",
    "could not find",
)


def _fuzzy_hint_match(answer: str, hint: str) -> bool:
    """Check if the answer contains the expected hint content.

    Uses kaos-nlp-core BM25 scoring when available (treating the hint as a
    query and the answer as a document). Falls back to word overlap.
    More robust than exact substring — tolerates paraphrasing, reordering,
    and synonyms.
    """
    if not hint or not answer:
        return False
    # Exact match first (fast path)
    if hint.lower() in answer.lower():
        return True

    # Try BM25-based scoring via kaos-nlp-core
    try:
        from kaos_nlp_core.search import Searcher

        records = [{"id": 0, "text": answer}]
        searcher = Searcher.from_documents(records)
        hits = searcher.search(hint, top_k=1)
        if hits and hits[0].score > _FUZZY_MATCH_THRESHOLD:
            return True
    except ImportError:
        pass

    # Fallback: word overlap (content words only)
    hint_words = {
        w
        for w in hint.lower().split()
        if len(w) > 2
        and w
        not in {
            "the",
            "and",
            "for",
            "are",
            "was",
            "with",
            "that",
            "this",
            "from",
            "have",
            "has",
            "been",
            "not",
            "but",
            "its",
            "each",
        }
    }
    if not hint_words:
        return hint.lower() in answer.lower()
    answer_lower = answer.lower()
    found = sum(1 for w in hint_words if w in answer_lower)
    return found / len(hint_words) >= _WORD_OVERLAP_THRESHOLD


@dataclass(frozen=True, slots=True)
class QuestionResult:
    """Result for one question.

    ``answer_correct`` is the headline verdict — it tracks whichever
    judge mode is selected (fuzzy by default, LLM when ``--judge llm``).
    ``fuzzy_correct`` and ``llm_correct`` always carry both verdicts when
    available so downstream analysis can compare them; the unused one is
    ``None``.
    """

    question_id: str
    question: str
    answerable: bool
    agent_answered: bool
    agent_refused: bool
    agent_errored: bool
    answer_correct: bool
    expected_doc_found: bool
    citations_count: int
    answer_length: int
    latency_s: float
    answer_text: str = ""
    error_message: str = ""
    # Per-judge verdicts. None means that judge was not run.
    fuzzy_correct: bool | None = None
    llm_correct: bool | None = None
    llm_confidence: float | None = None
    llm_reasoning: str = ""
    llm_model: str = ""
    llm_cost_usd: float = 0.0


@dataclass(slots=True)  # Mutable: accumulated
class BenchmarkResult:
    """Aggregate benchmark results."""

    corpus_dir: str = ""
    n_documents: int = 0
    n_questions: int = 0
    n_correct_answers: int = 0
    n_correct_refusals: int = 0
    n_wrong_answers: int = 0
    n_wrong_refusals: int = 0
    n_errors: int = 0
    accuracy: float = 0.0
    avg_latency_s: float = 0.0
    judge_mode: str = "fuzzy"
    judge_model: str = ""
    judge_total_cost_usd: float = 0.0
    questions: list[QuestionResult] = field(default_factory=list)


def _load_questions() -> list[dict]:
    """Load the multiformat questions JSONL."""
    questions = []
    with _QUESTIONS_PATH.open() as f:
        for line in f:
            line = line.strip()
            if line:
                questions.append(json.loads(line))
    return questions


def _is_refusal_text(answer: str) -> bool:
    """Heuristic refusal detector. Used by the fuzzy scorer when no
    explicit ``EvidenceInsufficient`` event fires (some refusal paths
    only emit text saying "I don't have enough evidence").

    Lives at module level so the LLM judge can compare its verdict
    against the same fuzzy refusal call the harness uses.
    """
    a = answer.lower()
    return any(p in a for p in _REFUSAL_PHRASES)


async def run_benchmark(
    *,
    model: str | None = None,
    verbose: bool = False,
    judge_mode: str = "fuzzy",
    judge_model: str = _DEFAULT_JUDGE_MODEL,
) -> BenchmarkResult:
    """Run the full multi-format E2E benchmark."""
    from kaos_core.registry.container import KaosRuntime
    from kaos_core.vfs.core import VirtualFileSystem

    from kaos_agents.cli_chat import _load_files_into_memory
    from kaos_agents.config import Agent
    from kaos_agents.events import (
        CitationFound,
        EvidenceInsufficient,
        RunError,
        TextDelta,
    )
    from kaos_agents.memory.session import SessionMemory
    from kaos_agents.memory.store import SessionStore
    from kaos_agents.memory.types import MemoryType
    from kaos_agents.runner import Runner
    from kaos_agents.settings import DEFAULT_MODEL
    from kaos_agents.tools import register_agent_tools

    # Load corpus
    sys.stdout.write(f"Loading corpus from {_CORPUS_DIR}...\n")
    sys.stdout.flush()

    corpus_files = sorted(
        f
        for f in _CORPUS_DIR.iterdir()
        if f.is_file()
        and f.suffix.lower() != ".jsonl"
        and f.name not in ("README.md", "extraction-golden.jsonl")
    )

    session_id = "multiformat-e2e"
    memory = SessionMemory(session_id)
    n_loaded = _load_files_into_memory(corpus_files, memory, verbose=verbose)

    n_docs = memory.section_item_count(MemoryType.DOCUMENTS)
    sys.stdout.write(f"  {n_loaded} files loaded ({n_docs} documents in memory)\n")
    sys.stdout.flush()

    # Persist to VFS
    vfs = VirtualFileSystem()
    store = SessionStore(vfs)
    await store.save(memory)

    # Build agent
    runtime = KaosRuntime.default()
    register_agent_tools(runtime)

    agent = Agent.create(
        instructions=(
            f"You are a research assistant with access to {n_docs} documents. "
            "Answer questions by citing specific documents. "
            "If you cannot find sufficient evidence, say so explicitly."
        ),
        model=model or DEFAULT_MODEL,
        pattern="research",
    )
    runner = Runner(agent, runtime=runtime, vfs=vfs)

    # Load questions
    questions = _load_questions()
    sys.stdout.write(f"Running {len(questions)} questions...\n\n")
    sys.stdout.flush()

    result = BenchmarkResult(
        corpus_dir=str(_CORPUS_DIR),
        n_documents=n_docs,
        n_questions=len(questions),
    )

    for i, q in enumerate(questions):
        qid = q["id"]
        question = q["question"]
        answerable = q.get("answerable", True)
        expected_doc = q.get("expected_doc_id", "")
        expected_hint = q.get("expected_answer_hint", "").lower()

        t0 = time.perf_counter()
        text_parts: list[str] = []
        citations: list[str] = []
        refused = False
        errored = False
        error_msg = ""

        try:
            async for event in runner.run(question, session_id):
                if isinstance(event, CitationFound):
                    citations.append(getattr(event, "source_uri", ""))
                elif isinstance(event, EvidenceInsufficient):
                    refused = True
                elif isinstance(event, TextDelta):
                    text_parts.append(event.content)
                elif isinstance(event, RunError):
                    errored = True
                    error_msg = event.message[:200]
        except Exception as exc:
            errored = True
            error_msg = f"{type(exc).__name__}: {exc}"[:200]

        latency = time.perf_counter() - t0
        answer_text = "".join(text_parts).lower()
        answered = bool(answer_text) and not refused and not errored

        # Fuzzy correctness check (always run — kept as a baseline
        # signal even when judge_mode="llm" so we can compare).
        if answerable:
            fuzzy_correct = answered and _fuzzy_hint_match(answer_text, expected_hint)
            expected_doc_found = (
                any(expected_doc.split("/")[-1].split(".")[0] in c for c in citations)
                if expected_doc and citations
                else False
            )
        else:
            fuzzy_correct = refused or _is_refusal_text(answer_text)
            expected_doc_found = True  # N/A for unanswerable

        # Optional LLM-as-judge verdict. Falls back to fuzzy on any
        # judge failure (the llm_judge module has its own graceful
        # failure mode contract).
        llm_correct: bool | None = None
        llm_confidence: float | None = None
        llm_reasoning_text = ""
        llm_model_used = ""
        llm_call_cost = 0.0
        if judge_mode == "llm":
            try:
                from kaos_agents.benchmarks.llm_judge import llm_judge

                verdict = await llm_judge(
                    question=question,
                    expected_hint=expected_hint or None,
                    expected_answerable=answerable,
                    agent_answer=answer_text,
                    agent_refused=refused or _is_refusal_text(answer_text),
                    model=judge_model,
                )
                # llm_judge returns dict (via JudgeVerdict.to_dict()).
                # When the judge is unavailable (provider 529, missing API
                # key, etc.) the dict has reasoning prefixed with
                # "judge unavailable:" — treat that as None so the harness
                # falls back to fuzzy. Otherwise we'd score the agent's
                # answer based on a transient infra failure.
                reasoning = str(verdict.get("reasoning", "") or "")
                judge_failed = reasoning.startswith("judge unavailable:")
                if judge_failed:
                    llm_correct = None
                    llm_confidence = None
                else:
                    llm_correct = bool(verdict.get("correct", False))
                    llm_confidence = float(verdict.get("confidence", 0.0) or 0.0)
                llm_reasoning_text = reasoning
                llm_model_used = str(verdict.get("judge_model", "") or "")
                llm_call_cost = float(verdict.get("judge_cost_usd", 0.0) or 0.0)
                result.judge_total_cost_usd += llm_call_cost
            except Exception as exc:
                llm_reasoning_text = f"judge unavailable: {exc!r}"

        # Promote whichever verdict the caller requested as the
        # headline ``answer_correct``. If the LLM judge was requested
        # but failed (None), fall back to fuzzy so we don't silently
        # discard the run.
        if judge_mode == "llm" and llm_correct is not None:
            answer_correct = llm_correct
        else:
            answer_correct = fuzzy_correct

        qr = QuestionResult(
            question_id=qid,
            question=question[:80],
            answerable=answerable,
            agent_answered=answered,
            agent_refused=refused,
            agent_errored=errored,
            answer_correct=answer_correct,
            expected_doc_found=expected_doc_found,
            citations_count=len(citations),
            answer_length=len(answer_text),
            latency_s=latency,
            answer_text=answer_text[:1000],  # truncate for compact JSON
            error_message=error_msg,
            fuzzy_correct=fuzzy_correct,
            llm_correct=llm_correct,
            llm_confidence=llm_confidence,
            llm_reasoning=llm_reasoning_text,
            llm_model=llm_model_used,
            llm_cost_usd=llm_call_cost,
        )
        result.questions.append(qr)

        # Categorize
        if answer_correct:
            if answerable:
                result.n_correct_answers += 1
            else:
                result.n_correct_refusals += 1
        elif errored:
            result.n_errors += 1
        elif answerable and not answered:
            result.n_wrong_refusals += 1
        else:
            result.n_wrong_answers += 1

        status = "CORRECT" if answer_correct else ("ERROR" if errored else "WRONG")
        marker = "A" if answerable else "R"
        sys.stdout.write(
            f"  [{i + 1}/{len(questions)}] [{marker}] {status} "
            f"cites={len(citations)} {latency:.1f}s: {question[:60]}\n"
        )
        sys.stdout.flush()

    # Compute aggregates
    total_correct = result.n_correct_answers + result.n_correct_refusals
    result.accuracy = total_correct / result.n_questions if result.n_questions else 0
    result.avg_latency_s = (
        sum(q.latency_s for q in result.questions) / len(result.questions)
        if result.questions
        else 0
    )

    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Multi-format E2E document Q&A benchmark")
    parser.add_argument("--model", default=None, help="LLM model (default: from settings)")
    parser.add_argument("--json", type=str, default=None)
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument(
        "--judge",
        default="fuzzy",
        help=(
            "Scoring mode: 'fuzzy' (default — historical hint matcher), "
            "'llm' (claude-haiku-4-5), 'llm:<model>' (explicit model spec), "
            "or 'none' (skip judging — record agent outputs only)."
        ),
    )
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
        parser.error(f"unknown --judge value: {args.judge!r} (expected fuzzy/llm/llm:<model>/none)")

    if not _CORPUS_DIR.exists():
        sys.stderr.write(f"Corpus not found: {_CORPUS_DIR}\n")
        sys.exit(1)

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

    # Print summary
    sys.stdout.write(f"\n{'=' * 60}\n")
    sys.stdout.write("MULTI-FORMAT E2E BENCHMARK\n")
    sys.stdout.write(f"{'=' * 60}\n")
    sys.stdout.write(f"Documents:        {result.n_documents} ({_CORPUS_DIR.name})\n")
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
    # Fuzzy vs LLM agreement when both are present (judge_mode='llm').
    if judge_mode == "llm" and result.questions:
        agree = sum(
            1
            for q in result.questions
            if q.fuzzy_correct is not None
            and q.llm_correct is not None
            and q.fuzzy_correct == q.llm_correct
        )
        total = sum(
            1 for q in result.questions if q.fuzzy_correct is not None and q.llm_correct is not None
        )
        if total:
            sys.stdout.write(f"Fuzzy↔LLM agree:  {agree}/{total} ({agree / total:.0%})\n")
    sys.stdout.flush()

    # Save results
    json_path = args.json
    if json_path is None:
        today = time.strftime("%Y-%m-%d")
        # __file__ is tests/benchmarks/multiformat_e2e.py — we want
        # the package's docs/benchmarks/, not tests/docs/benchmarks/.
        benchmarks_dir = Path(__file__).resolve().parent.parent.parent / "docs" / "benchmarks"
        json_path = str(benchmarks_dir / f"multiformat-e2e-{today}.json")

    out_path = Path(json_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    output = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "corpus_dir": str(_CORPUS_DIR),
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
