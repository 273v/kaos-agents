"""Generic Harvey LAB benchmark runner — works for any vendored task.

Sister of ``harvey_coc_benchmark.py``, but task-agnostic. The CoC
benchmark wires in a domain-specific structured-extraction pipeline
(``change-of-control`` schema). This runner is for the *free-form*
pipeline only — the agent reads all documents, produces a single
deliverable, and is judged against the task's rubric.

Use this to test whether the free-form Sonnet pipeline (which is what
generalizes across practice areas) produces competent deliverables on
tasks that aren't M&A contract review.

Usage::

    # litigation: motion-to-dismiss issue memo
    uv run python tests/benchmarks/harvey_lab_benchmark.py \\
      --task tests/fixtures/harvey-lab/analyze-counterparty-motion-to-dismiss \\
      --model anthropic:claude-sonnet-4-6 \\
      --judge-model anthropic:claude-haiku-4-5 \\
      --json docs/benchmarks/harvey-mtd-sonnet-2026-05-06.json

The ``--task`` directory must contain ``task.json`` and a
``documents/`` subdirectory matching the upstream Harvey LAB layout.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

_DEFAULT_JUDGE_MODEL = "anthropic:claude-haiku-4-5"
_DEFAULT_AGENT_MODEL = "anthropic:claude-haiku-4-5"
_JUDGE_CONCURRENCY = 5


@dataclass(frozen=True, slots=True)
class CriterionResult:
    """Per-criterion verdict + provenance."""

    criterion_id: str
    title: str
    passed: bool
    confidence: float
    reasoning: str
    judge_model: str
    judge_cost_usd: float


@dataclass(slots=True)
class BenchmarkResult:
    """Aggregate result of a Harvey LAB run."""

    fixture_dir: str = ""
    task_title: str = ""
    n_documents: int = 0
    n_criteria: int = 0
    n_passed: int = 0
    n_failed: int = 0
    n_judge_unavailable: int = 0
    pooled_pass_rate: float = 0.0
    all_pass_score: float = 0.0
    agent_model: str = ""
    judge_model: str = ""
    agent_latency_s: float = 0.0
    judge_latency_s: float = 0.0
    agent_cost_usd: float = 0.0
    judge_cost_usd: float = 0.0
    deliverable_chars: int = 0
    agent_errored: bool = False
    error_message: str = ""
    deliverable_text: str = ""
    criteria: list[CriterionResult] = field(default_factory=list)


def _load_task(task_dir: Path) -> dict:
    task_json = task_dir / "task.json"
    if not task_json.exists():
        msg = f"task.json not found at {task_json}."
        raise FileNotFoundError(msg)
    return json.loads(task_json.read_text())


def _list_documents(task_dir: Path) -> list[Path]:
    docs_dir = task_dir / "documents"
    if not docs_dir.exists():
        msg = f"documents/ directory not found at {docs_dir}."
        raise FileNotFoundError(msg)
    return sorted(p for p in docs_dir.iterdir() if p.is_file())


async def _run_freeform_agent(
    *,
    task_title: str,
    instructions: str,
    documents: list[Path],
    docs_dir: Path,
    model: str,
    verbose: bool,
    pattern: str = "research",
) -> tuple[str, float, float, bool, str]:
    """Free-form research-pattern agent run.

    Loads every document into ``MemoryType.DOCUMENTS`` and runs a single
    research-pattern agent turn against the task instructions. Returns
    the deliverable text + telemetry. Same code path the CoC benchmark
    uses for its free-form mode, generalized for any task.
    """
    from kaos_core.registry.container import KaosRuntime
    from kaos_core.vfs.core import VirtualFileSystem

    from kaos_agents.cli.chat import _load_files_into_memory
    from kaos_agents.config import Agent
    from kaos_agents.events import RunError, TextDelta, TurnSummary
    from kaos_agents.memory.session import SessionMemory
    from kaos_agents.memory.store import SessionStore
    from kaos_agents.runtime.runner import Runner
    from kaos_agents.tools import register_agent_tools
    from kaos_agents.types.memory import MemoryType

    sys.stdout.write(f"Loading {len(documents)} documents from {docs_dir}...\n")
    sys.stdout.flush()

    session_id = f"harvey-lab-{abs(hash(task_title)) % 10_000}"
    memory = SessionMemory(session_id)
    n_loaded = _load_files_into_memory(documents, memory, verbose=verbose)
    n_chunks = memory.section_item_count(MemoryType.DOCUMENTS)
    sys.stdout.write(f"  {n_loaded} files loaded ({n_chunks} chunks in memory)\n")
    sys.stdout.flush()

    vfs = VirtualFileSystem()
    store = SessionStore(vfs)
    await store.save(memory)

    runtime = KaosRuntime.default()
    register_agent_tools(runtime)

    agent = Agent.create(
        instructions=(
            f"You are a senior associate working on the following task: "
            f"{task_title}. You have access to {n_chunks} document chunks "
            f"across {len(documents)} source files. "
            "Produce the deliverable described in the task instructions. "
            "Cite specific section numbers, dates, parties, and quoted "
            "language from the source documents. Apply the relevant "
            "controlling law and identify procedural posture explicitly. "
            "If the source documents do not support a finding, say so — "
            "do not invent facts. The deliverable will be evaluated against "
            "a structured rubric of pass/fail criteria; specificity and "
            "explicit citations matter."
        ),
        model=model,
        pattern=pattern,
    )
    runner = Runner(agent, runtime=runtime, vfs=vfs)

    sys.stdout.write("Running agent turn...\n")
    sys.stdout.flush()

    text_parts: list[str] = []
    final_text: str = ""
    errored = False
    error_msg = ""
    cost_usd = 0.0
    t0 = time.perf_counter()

    try:
        async for event in runner.run(instructions, session_id):
            if isinstance(event, TextDelta):
                text_parts.append(event.content)
                if verbose:
                    sys.stdout.write(event.content)
                    sys.stdout.flush()
            elif isinstance(event, TurnSummary):
                cost_usd += float(getattr(event, "cost_usd", 0.0) or 0.0)
                # TurnSummary.text is the canonical final response after
                # memory persistence; streamed TextDelta sometimes
                # truncates on research-pattern Sonnet. Take the longer
                # of the two so a partial stream never silently wins.
                final_text = str(getattr(event, "text", "") or "")
            elif isinstance(event, RunError):
                errored = True
                error_msg = event.message[:500]
    except Exception as exc:
        errored = True
        error_msg = f"{type(exc).__name__}: {exc}"[:500]

    latency = time.perf_counter() - t0
    streamed = "".join(text_parts)
    deliverable = final_text if len(final_text) >= len(streamed) else streamed
    sys.stdout.write(
        f"\nAgent run complete: {latency:.1f}s, {len(deliverable):,} chars, ${cost_usd:.4f}\n"
    )
    sys.stdout.flush()
    return deliverable, latency, cost_usd, errored, error_msg


async def _judge_one(
    *,
    criterion: dict,
    deliverable: str,
    judge_model: str,
    sem: asyncio.Semaphore,
) -> CriterionResult:
    """Score one criterion under a concurrency-limited semaphore."""
    from kaos_agents.benchmarks.rubric_judge import rubric_judge

    async with sem:
        verdict = await rubric_judge(
            criterion_id=criterion["id"],
            criterion_title=criterion["title"],
            match_criteria=criterion["match_criteria"],
            deliverable=deliverable,
            model=judge_model,
        )
    return CriterionResult(
        criterion_id=str(verdict.get("criterion_id", criterion["id"])),
        title=criterion["title"],
        passed=bool(verdict.get("passed", False)),
        confidence=float(verdict.get("confidence", 0.0) or 0.0),
        reasoning=str(verdict.get("reasoning", "") or ""),
        judge_model=str(verdict.get("judge_model", "") or ""),
        judge_cost_usd=float(verdict.get("judge_cost_usd", 0.0) or 0.0),
    )


async def _judge_all(
    *,
    criteria: list[dict],
    deliverable: str,
    judge_model: str,
    concurrency: int,
) -> list[CriterionResult]:
    """Score every criterion concurrently with bounded parallelism."""
    sem = asyncio.Semaphore(max(1, concurrency))
    tasks = [
        asyncio.create_task(
            _judge_one(
                criterion=c,
                deliverable=deliverable,
                judge_model=judge_model,
                sem=sem,
            )
        )
        for c in criteria
    ]
    results: list[CriterionResult] = []
    for completed, coro in enumerate(asyncio.as_completed(tasks), start=1):
        r = await coro
        marker = "PASS" if r.passed else "FAIL"
        if "judge unavailable:" in r.reasoning:
            marker = "JUDGE_UNAVAIL"
        sys.stdout.write(
            f"  [{completed}/{len(criteria)}] {marker:>13} {r.criterion_id}: {r.title[:70]}\n"
        )
        sys.stdout.flush()
        results.append(r)
    results.sort(key=lambda r: r.criterion_id)
    return results


async def run_benchmark(
    task_dir: Path,
    *,
    agent_model: str = _DEFAULT_AGENT_MODEL,
    judge_model: str = _DEFAULT_JUDGE_MODEL,
    max_criteria: int | None = None,
    verbose: bool = False,
    concurrency: int = _JUDGE_CONCURRENCY,
    pattern: str = "research",
) -> BenchmarkResult:
    """Run a Harvey LAB free-form benchmark end to end."""
    task = _load_task(task_dir)
    documents = _list_documents(task_dir)
    criteria = task["criteria"]
    if max_criteria is not None and max_criteria > 0:
        criteria = criteria[:max_criteria]

    sys.stdout.write(f"\n{'=' * 60}\n")
    sys.stdout.write(f"HARVEY LAB BENCHMARK — {task['title']}\n")
    sys.stdout.write(f"{'=' * 60}\n")
    sys.stdout.write(f"Fixture: {task_dir}\n")
    sys.stdout.write(f"Documents: {len(documents)}\n")
    sys.stdout.write(f"Criteria: {len(criteria)} (of {len(task['criteria'])} total)\n")
    sys.stdout.write(f"Agent model: {agent_model}\n")
    sys.stdout.write(f"Judge model: {judge_model}\n\n")
    sys.stdout.flush()

    deliverable, agent_latency, agent_cost, errored, error_msg = await _run_freeform_agent(
        task_title=task["title"],
        instructions=task["instructions"],
        documents=documents,
        docs_dir=task_dir / "documents",
        model=agent_model,
        verbose=verbose,
        pattern=pattern,
    )

    result = BenchmarkResult(
        fixture_dir=str(task_dir),
        task_title=task["title"],
        n_documents=len(documents),
        n_criteria=len(criteria),
        agent_model=agent_model,
        judge_model=judge_model,
        agent_latency_s=agent_latency,
        agent_cost_usd=agent_cost,
        deliverable_chars=len(deliverable),
        agent_errored=errored,
        error_message=error_msg,
        deliverable_text=deliverable,
    )

    if errored or not deliverable.strip():
        sys.stdout.write("\nAgent run produced no usable deliverable — skipping rubric judging.\n")
        sys.stdout.flush()
        return result

    sys.stdout.write(f"\nJudging {len(criteria)} criteria...\n")
    sys.stdout.flush()
    t0 = time.perf_counter()
    criterion_results = await _judge_all(
        criteria=criteria,
        deliverable=deliverable,
        judge_model=judge_model,
        concurrency=concurrency,
    )
    judge_latency = time.perf_counter() - t0

    n_passed = sum(1 for c in criterion_results if c.passed)
    n_failed = sum(
        1 for c in criterion_results if not c.passed and "judge unavailable:" not in c.reasoning
    )
    n_judge_unavailable = sum(1 for c in criterion_results if "judge unavailable:" in c.reasoning)
    judge_cost = sum(c.judge_cost_usd for c in criterion_results)

    result.criteria = criterion_results
    result.n_passed = n_passed
    result.n_failed = n_failed
    result.n_judge_unavailable = n_judge_unavailable
    result.pooled_pass_rate = n_passed / len(criterion_results) if criterion_results else 0.0
    result.all_pass_score = (
        1.0 if n_passed == len(criterion_results) and n_judge_unavailable == 0 else 0.0
    )
    result.judge_latency_s = judge_latency
    result.judge_cost_usd = judge_cost
    return result


def _print_summary(result: BenchmarkResult) -> None:
    sys.stdout.write(f"\n{'=' * 60}\n")
    sys.stdout.write("BENCHMARK SUMMARY\n")
    sys.stdout.write(f"{'=' * 60}\n")
    sys.stdout.write(f"Task:                {result.task_title}\n")
    sys.stdout.write(f"Documents:           {result.n_documents}\n")
    sys.stdout.write(f"Criteria evaluated:  {result.n_criteria}\n")
    sys.stdout.write(f"  Passed:            {result.n_passed}\n")
    sys.stdout.write(f"  Failed:            {result.n_failed}\n")
    sys.stdout.write(f"  Judge unavailable: {result.n_judge_unavailable}\n")
    sys.stdout.write(f"Pooled pass rate:    {result.pooled_pass_rate:.1%}\n")
    sys.stdout.write(f"All-pass score:      {result.all_pass_score:.1f}\n")
    sys.stdout.write(f"Agent latency:       {result.agent_latency_s:.1f}s\n")
    sys.stdout.write(f"Judge latency:       {result.judge_latency_s:.1f}s\n")
    sys.stdout.write(f"Agent cost:          ${result.agent_cost_usd:.4f}\n")
    sys.stdout.write(f"Judge cost:          ${result.judge_cost_usd:.4f}\n")
    sys.stdout.write(f"Total cost:          ${result.agent_cost_usd + result.judge_cost_usd:.4f}\n")
    sys.stdout.write(f"Deliverable size:    {result.deliverable_chars:,} chars\n")
    if result.agent_errored:
        sys.stdout.write(f"\nAGENT ERROR: {result.error_message}\n")
    sys.stdout.flush()


def _save_json(result: BenchmarkResult, json_path: str) -> None:
    out_path = Path(json_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    output = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "fixture_dir": result.fixture_dir,
        "task_title": result.task_title,
        "n_documents": result.n_documents,
        "summary": {
            "n_criteria": result.n_criteria,
            "n_passed": result.n_passed,
            "n_failed": result.n_failed,
            "n_judge_unavailable": result.n_judge_unavailable,
            "pooled_pass_rate": result.pooled_pass_rate,
            "all_pass_score": result.all_pass_score,
            "agent_model": result.agent_model,
            "judge_model": result.judge_model,
            "agent_latency_s": result.agent_latency_s,
            "judge_latency_s": result.judge_latency_s,
            "agent_cost_usd": result.agent_cost_usd,
            "judge_cost_usd": result.judge_cost_usd,
            "deliverable_chars": result.deliverable_chars,
            "agent_errored": result.agent_errored,
            "error_message": result.error_message,
        },
        "deliverable": result.deliverable_text,
        "criteria": [asdict(c) for c in result.criteria],
    }
    out_path.write_text(json.dumps(output, indent=2))
    sys.stdout.write(f"\nSaved to {json_path}\n")
    sys.stdout.flush()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generic Harvey LAB benchmark runner.")
    parser.add_argument(
        "--task",
        required=True,
        help=(
            "Path to a Harvey LAB task fixture directory containing "
            "``task.json`` and a ``documents/`` subdirectory."
        ),
    )
    parser.add_argument(
        "--model",
        default=_DEFAULT_AGENT_MODEL,
        help=f"Agent LLM model (default: {_DEFAULT_AGENT_MODEL}).",
    )
    parser.add_argument(
        "--judge-model",
        default=_DEFAULT_JUDGE_MODEL,
        help=f"Judge LLM model (default: {_DEFAULT_JUDGE_MODEL}).",
    )
    parser.add_argument(
        "--max-criteria",
        type=int,
        default=None,
        help="Score only the first N criteria. Default: all.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=_JUDGE_CONCURRENCY,
        help=f"Max concurrent judge calls (default: {_JUDGE_CONCURRENCY}).",
    )
    parser.add_argument(
        "--pattern",
        default="research",
        choices=["chat", "research", "plan-execute"],
        help=(
            "Agent pattern. 'research' uses RAG (short citation-grounded "
            "answers); 'chat' uses simple-respond + ReAct (long-form prose). "
            "Pick 'chat' for tasks where the deliverable is a long memo."
        ),
    )
    parser.add_argument("--json", default=None, help="Output JSON path.")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    task_dir = Path(args.task).resolve()
    if not task_dir.exists():
        sys.stderr.write(f"Task directory not found: {task_dir}\n")
        sys.exit(1)

    if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("KAOS_LLM_ANTHROPIC_API_KEY")):
        sys.stderr.write("ANTHROPIC_API_KEY (or KAOS_LLM_ANTHROPIC_API_KEY) is required.\n")
        sys.exit(2)

    result = asyncio.run(
        run_benchmark(
            task_dir,
            agent_model=args.model,
            judge_model=args.judge_model,
            max_criteria=args.max_criteria,
            verbose=args.verbose,
            concurrency=args.concurrency,
            pattern=args.pattern,
        )
    )

    _print_summary(result)

    json_path = args.json
    if json_path is None:
        today = time.strftime("%Y-%m-%d")
        slug = task_dir.name
        # __file__ is tests/benchmarks/harvey_lab_benchmark.py — we want
        # the package's docs/benchmarks/, not tests/docs/benchmarks/.
        benchmarks_dir = Path(__file__).resolve().parent.parent.parent / "docs" / "benchmarks"
        json_path = str(benchmarks_dir / f"harvey-{slug}-{today}.json")
    _save_json(result, json_path)


if __name__ == "__main__":
    main()
