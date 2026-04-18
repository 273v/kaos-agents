"""End-to-end multi-format document Q&A benchmark.

Tests the full pipeline: load mixed-format files → BM25 retrieval → RAG →
cited answer. Uses the WS-3.7 multiformat corpus (10 docs, 5 formats,
12 questions with ground truth).

This is the acceptance test for "can I dump a bunch of files into a folder
and get answers?"

Usage::

    # Run all 12 questions
    uv run python tests/benchmarks/multiformat_e2e.py

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


@dataclass(frozen=True, slots=True)
class QuestionResult:
    """Result for one question."""

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
    error_message: str = ""


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


async def run_benchmark(
    *,
    model: str | None = None,
    verbose: bool = False,
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
        f for f in _CORPUS_DIR.iterdir()
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

        # Check correctness
        if answerable:
            # Did the agent answer with the expected content?
            answer_correct = answered and expected_hint in answer_text
            expected_doc_found = any(
                expected_doc.split("/")[-1].split(".")[0] in c
                for c in citations
            ) if expected_doc and citations else False
        else:
            # Did the agent correctly refuse?
            answer_correct = refused or (
                "insufficient" in answer_text
                or "not find" in answer_text
                or "not contain" in answer_text
                or "no information" in answer_text
                or "not available" in answer_text
                or "cannot answer" in answer_text
                or "don't have" in answer_text
            )
            expected_doc_found = True  # N/A for unanswerable

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
            error_message=error_msg,
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
        if result.questions else 0
    )

    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Multi-format E2E document Q&A benchmark")
    parser.add_argument("--model", default=None, help="LLM model (default: from settings)")
    parser.add_argument("--json", type=str, default=None)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    if not _CORPUS_DIR.exists():
        sys.stderr.write(f"Corpus not found: {_CORPUS_DIR}\n")
        sys.exit(1)

    result = asyncio.run(run_benchmark(model=args.model, verbose=args.verbose))

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
    sys.stdout.flush()

    # Save results
    json_path = args.json
    if json_path is None:
        today = time.strftime("%Y-%m-%d")
        benchmarks_dir = Path(__file__).resolve().parent.parent / "docs" / "benchmarks"
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
        },
        "questions": [asdict(q) for q in result.questions],
    }
    out_path.write_text(json.dumps(output, indent=2))
    sys.stdout.write(f"\nSaved to {json_path}\n")


if __name__ == "__main__":
    main()
