"""Scale E2E benchmark — 100+ docs across all formats.

Loads ALL test fixture files from kaos-pdf, kaos-office, kaos-web, and
kaos-llm-core into a single session. Runs the same 12 multiformat
questions to test retrieval at scale (can the agent find needles in
100+ docs?).

Usage::

    uv run python tests/benchmarks/scale_e2e.py
    uv run python tests/benchmarks/scale_e2e.py --json results.json
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
    BenchmarkResult,
    QuestionResult,
    _load_questions,
)

_FIXTURE_ROOTS = [
    "kaos-pdf/tests/fixtures",
    "kaos-office/tests/fixtures",
    "kaos-web/tests/fixtures",
    "kaos-llm-core/tests/fixtures/multiformat-corpus",
    "kaos-llm-core/tests/fixtures/grounding-corpus",
]


def _find_all_fixtures(repo_root: Path) -> list[Path]:
    """Find all parseable fixture files across the repo."""
    from kaos_agents.cli_chat import _SUPPORTED_EXTENSIONS

    files: list[Path] = []
    for rel in _FIXTURE_ROOTS:
        root = repo_root / rel
        if not root.exists():
            continue
        for f in root.rglob("*"):
            if (
                f.is_file()
                and f.suffix.lower() in _SUPPORTED_EXTENSIONS
                and "bad" not in f.name.lower()
            ):
                files.append(f)
    return sorted(files)


async def run_scale_benchmark(
    *,
    model: str | None = None,
    verbose: bool = False,
) -> BenchmarkResult:
    """Load 100+ docs, run 12 questions."""
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

    repo_root = Path(__file__).resolve().parent.parent.parent.parent
    files = _find_all_fixtures(repo_root)
    sys.stdout.write(f"Found {len(files)} fixture files\n")
    sys.stdout.flush()

    session_id = "scale-e2e"
    memory = SessionMemory(session_id)

    t0 = time.perf_counter()
    n_loaded = _load_files_into_memory(files, memory, verbose=verbose)
    t1 = time.perf_counter()

    n_docs = memory.section_item_count(MemoryType.DOCUMENTS)
    total_chars = sum(len(item.content) for item in memory.get(MemoryType.DOCUMENTS))
    sys.stdout.write(
        f"Loaded {n_loaded}/{len(files)} in {t1 - t0:.1f}s — "
        f"{n_docs} docs, {total_chars:,} chars ({total_chars // max(n_docs, 1):,} avg)\n"
    )
    sys.stdout.flush()

    # Persist
    vfs = VirtualFileSystem()
    store = SessionStore(vfs)
    await store.save(memory)

    # Build agent
    runtime = KaosRuntime.default()
    register_agent_tools(runtime)

    agent = Agent.create(
        instructions=(
            f"You are a research assistant with access to {n_docs} documents "
            f"({total_chars:,} total chars). "
            "Answer questions by citing specific documents. "
            "If you cannot find sufficient evidence, say so explicitly."
        ),
        model=model or DEFAULT_MODEL,
        pattern="research",
    )
    runner = Runner(agent, runtime=runtime, vfs=vfs)

    questions = _load_questions()
    sys.stdout.write(f"\nRunning {len(questions)} questions against {n_docs} docs...\n\n")
    sys.stdout.flush()

    result = BenchmarkResult(
        corpus_dir="all-fixtures",
        n_documents=n_docs,
        n_questions=len(questions),
    )

    for i, q in enumerate(questions):
        qid = q["id"]
        question = q["question"]
        answerable = q.get("answerable", True)
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

        if answerable:
            answer_correct = answered and expected_hint in answer_text
        else:
            answer_correct = refused or any(
                phrase in answer_text
                for phrase in (
                    "insufficient", "not find", "not contain",
                    "no information", "not available", "cannot answer", "don't have",
                )
            )

        qr = QuestionResult(
            question_id=qid,
            question=question[:80],
            answerable=answerable,
            agent_answered=answered,
            agent_refused=refused,
            agent_errored=errored,
            answer_correct=answer_correct,
            expected_doc_found=bool(citations) if answerable else True,
            citations_count=len(citations),
            answer_length=len(answer_text),
            latency_s=latency,
            error_message=error_msg,
        )
        result.questions.append(qr)

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

    total_correct = result.n_correct_answers + result.n_correct_refusals
    result.accuracy = total_correct / result.n_questions if result.n_questions else 0
    result.avg_latency_s = (
        sum(q.latency_s for q in result.questions) / len(result.questions)
        if result.questions else 0
    )
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Scale E2E benchmark (100+ docs)")
    parser.add_argument("--model", default=None)
    parser.add_argument("--json", type=str, default=None)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    result = asyncio.run(run_scale_benchmark(model=args.model, verbose=args.verbose))

    sys.stdout.write(f"\n{'=' * 60}\n")
    sys.stdout.write(f"SCALE E2E BENCHMARK ({result.n_documents} docs)\n")
    sys.stdout.write(f"{'=' * 60}\n")
    sys.stdout.write(f"Correct answers:  {result.n_correct_answers}/7\n")
    sys.stdout.write(f"Correct refusals: {result.n_correct_refusals}/5\n")
    sys.stdout.write(f"Wrong:            {result.n_wrong_answers + result.n_wrong_refusals}\n")
    sys.stdout.write(f"Errors:           {result.n_errors}\n")
    sys.stdout.write(f"Accuracy:         {result.accuracy:.1%}\n")
    sys.stdout.write(f"Avg latency:      {result.avg_latency_s:.1f}s\n")
    sys.stdout.flush()

    json_path = args.json
    if json_path is None:
        today = time.strftime("%Y-%m-%d")
        benchmarks_dir = Path(__file__).resolve().parent.parent / "docs" / "benchmarks"
        json_path = str(benchmarks_dir / f"scale-e2e-{today}.json")

    out_path = Path(json_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    output = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "n_documents": result.n_documents,
        "summary": {
            "accuracy": result.accuracy,
            "correct_answers": result.n_correct_answers,
            "correct_refusals": result.n_correct_refusals,
            "wrong": result.n_wrong_answers + result.n_wrong_refusals,
            "errors": result.n_errors,
            "avg_latency_s": result.avg_latency_s,
        },
        "questions": [asdict(q) for q in result.questions],
    }
    out_path.write_text(json.dumps(output, indent=2))
    sys.stdout.write(f"\nSaved to {json_path}\n")


if __name__ == "__main__":
    main()
