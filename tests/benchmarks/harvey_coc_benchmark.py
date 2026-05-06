"""Harvey LAB change-of-control benchmark — port of one M&A task.

Ports
[`tasks/corporate-ma/extract-change-of-control-provisions`](https://github.com/harveyai/harvey-labs/tree/main/tasks/corporate-ma/extract-change-of-control-provisions)
from the [Harvey LAB](https://github.com/harveyai/harvey-labs) (MIT-licensed)
benchmark into our integration-test infrastructure.

The shape is fundamentally different from ``multiformat_e2e.py``:

* **One agent run produces one deliverable.** The agent reads 8 acquisition-
  target contracts and writes a single comprehensive change-of-control
  extraction report.
* **Many rubric criteria evaluate that deliverable.** The vendored
  ``task.json`` ships 55 PASS/FAIL criteria — each judged independently by
  :func:`kaos_agents.benchmarks.rubric_judge.rubric_judge`.
* **All-pass scoring.** Following Harvey's published methodology, the
  headline score is 1.0 iff *every* criterion passes, else 0.0. The pooled
  criterion pass rate (passes / total) is the secondary signal that lets
  partial credit show up in dashboards.

Usage::

    # Full run with LLM judge (default) — ~$0.10-$0.15 end to end.
    uv run python tests/benchmarks/harvey_coc_benchmark.py

    # Bound to the first 10 criteria for cheaper iteration.
    uv run python tests/benchmarks/harvey_coc_benchmark.py --max-criteria 10

    # Override the agent model.
    uv run python tests/benchmarks/harvey_coc_benchmark.py --model anthropic:claude-sonnet-4-6

    # Override the judge model.
    uv run python tests/benchmarks/harvey_coc_benchmark.py --judge-model anthropic:claude-haiku-4-5

    # Save to a specific file (default: docs/benchmarks/harvey-coc-<date>.json).
    uv run python tests/benchmarks/harvey_coc_benchmark.py --json /tmp/harvey-coc.json

    # Verbose: stream agent events as they arrive.
    uv run python tests/benchmarks/harvey_coc_benchmark.py -v

The benchmark reuses ``cli_chat._load_files_into_memory`` so the agent
sees exactly what the ``kaos-agent chat`` CLI would see if a user ran
``kaos-agent chat --files documents/*.docx``. That keeps the benchmark
honest — it tests the actual product surface, not a private codepath.
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

_FIXTURE_DIR = (
    Path(__file__).resolve().parent.parent
    / "fixtures/harvey-lab/extract-change-of-control-provisions"
)
_TASK_JSON = _FIXTURE_DIR / "task.json"
_DOCS_DIR = _FIXTURE_DIR / "documents"

_DEFAULT_JUDGE_MODEL = "anthropic:claude-haiku-4-5"
_DEFAULT_AGENT_MODEL = "anthropic:claude-haiku-4-5"

# Bounded concurrency on the judge fanout — providers rate-limit
# parallel json-mode calls aggressively, and 5 is a safe default that
# still finishes 55 criteria in well under a minute.
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
    """Aggregate benchmark results."""

    fixture_dir: str = ""
    task_title: str = ""
    n_documents: int = 0
    n_criteria: int = 0
    n_passed: int = 0
    n_failed: int = 0
    n_judge_unavailable: int = 0
    pooled_pass_rate: float = 0.0
    all_pass_score: float = 0.0  # 1.0 iff every criterion passed
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


def _load_task() -> dict:
    if not _TASK_JSON.exists():
        msg = (
            f"Harvey LAB CoC fixture not found at {_TASK_JSON}. "
            "Run scripts/refresh-harvey-fixtures.sh or vendor manually "
            "from harveyai/harvey-labs."
        )
        raise FileNotFoundError(msg)
    return json.loads(_TASK_JSON.read_text())


def _list_documents() -> list[Path]:
    if not _DOCS_DIR.exists():
        msg = f"Harvey LAB documents dir not found at {_DOCS_DIR}."
        raise FileNotFoundError(msg)
    return sorted(p for p in _DOCS_DIR.iterdir() if p.is_file())


async def _run_agent(
    *,
    instructions: str,
    documents: list[Path],
    model: str,
    verbose: bool,
) -> tuple[str, float, float, bool, str]:
    """Run a single agent turn and return (deliverable, latency, cost,
    errored, error_message).
    """
    from kaos_core.registry.container import KaosRuntime
    from kaos_core.vfs.core import VirtualFileSystem

    from kaos_agents.cli_chat import _load_files_into_memory
    from kaos_agents.config import Agent
    from kaos_agents.events import RunError, TextDelta, TurnComplete
    from kaos_agents.memory.session import SessionMemory
    from kaos_agents.memory.store import SessionStore
    from kaos_agents.memory.types import MemoryType
    from kaos_agents.runner import Runner
    from kaos_agents.tools import register_agent_tools

    sys.stdout.write(f"Loading {len(documents)} documents from {_DOCS_DIR}...\n")
    sys.stdout.flush()

    session_id = "harvey-coc-benchmark"
    memory = SessionMemory(session_id)
    n_loaded = _load_files_into_memory(documents, memory, verbose=verbose)
    n_docs = memory.section_item_count(MemoryType.DOCUMENTS)
    sys.stdout.write(f"  {n_loaded} files loaded ({n_docs} chunks in memory)\n")
    sys.stdout.flush()

    vfs = VirtualFileSystem()
    store = SessionStore(vfs)
    await store.save(memory)

    runtime = KaosRuntime.default()
    register_agent_tools(runtime)

    agent = Agent.create(
        instructions=(
            "You are a senior M&A associate reviewing acquisition target "
            "contracts. You have access to "
            f"{n_docs} document chunks across {len(documents)} contracts. "
            "When the user gives you a review task, produce a comprehensive "
            "extraction report that addresses every contract in the data "
            "room. For each provision you identify, cite the specific "
            "section number and quote the operative language. Flag risks "
            "explicitly. If a contract is missing a relevant provision, "
            "say so — do not invent provisions that are not in the source "
            "text."
        ),
        model=model,
        pattern="research",
    )
    runner = Runner(agent, runtime=runtime, vfs=vfs)

    sys.stdout.write("Running agent turn...\n")
    sys.stdout.flush()

    text_parts: list[str] = []
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
            elif isinstance(event, TurnComplete):
                cost_usd += float(getattr(event, "cost_usd", 0.0) or 0.0)
            elif isinstance(event, RunError):
                errored = True
                error_msg = event.message[:500]
    except Exception as exc:
        errored = True
        error_msg = f"{type(exc).__name__}: {exc}"[:500]

    latency = time.perf_counter() - t0
    deliverable = "".join(text_parts)
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
    """Score a single criterion under a concurrency-limited semaphore."""
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
    concurrency: int = _JUDGE_CONCURRENCY,
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
    # Sort by criterion id so JSON output is stable across runs.
    results.sort(key=lambda r: r.criterion_id)
    return results


async def run_benchmark(
    *,
    agent_model: str = _DEFAULT_AGENT_MODEL,
    judge_model: str = _DEFAULT_JUDGE_MODEL,
    max_criteria: int | None = None,
    verbose: bool = False,
    concurrency: int = _JUDGE_CONCURRENCY,
) -> BenchmarkResult:
    """Run the Harvey LAB CoC benchmark end to end."""
    task = _load_task()
    documents = _list_documents()
    criteria = task["criteria"]
    if max_criteria is not None and max_criteria > 0:
        criteria = criteria[:max_criteria]

    sys.stdout.write(f"\n{'=' * 60}\n")
    sys.stdout.write("HARVEY LAB — CHANGE-OF-CONTROL EXTRACTION BENCHMARK\n")
    sys.stdout.write(f"{'=' * 60}\n")
    sys.stdout.write(f"Task: {task['title']}\n")
    sys.stdout.write(f"Documents: {len(documents)}\n")
    sys.stdout.write(f"Criteria: {len(criteria)} (of {len(task['criteria'])} total)\n")
    sys.stdout.write(f"Agent model: {agent_model}\n")
    sys.stdout.write(f"Judge model: {judge_model}\n\n")
    sys.stdout.flush()

    deliverable, agent_latency, agent_cost, errored, error_msg = await _run_agent(
        instructions=task["instructions"],
        documents=documents,
        model=agent_model,
        verbose=verbose,
    )

    result = BenchmarkResult(
        fixture_dir=str(_FIXTURE_DIR),
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
    parser = argparse.ArgumentParser(
        description="Harvey LAB change-of-control extraction benchmark."
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
        help="Score only the first N criteria. Default: all 55.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=_JUDGE_CONCURRENCY,
        help=f"Max concurrent judge calls (default: {_JUDGE_CONCURRENCY}).",
    )
    parser.add_argument("--json", default=None, help="Output JSON path.")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    if not _FIXTURE_DIR.exists():
        sys.stderr.write(f"Fixture not found: {_FIXTURE_DIR}\n")
        sys.exit(1)

    if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("KAOS_LLM_ANTHROPIC_API_KEY")):
        sys.stderr.write(
            "ANTHROPIC_API_KEY (or KAOS_LLM_ANTHROPIC_API_KEY) is required "
            "to run the agent + judge.\n"
        )
        sys.exit(2)

    result = asyncio.run(
        run_benchmark(
            agent_model=args.model,
            judge_model=args.judge_model,
            max_criteria=args.max_criteria,
            verbose=args.verbose,
            concurrency=args.concurrency,
        )
    )

    _print_summary(result)

    json_path = args.json
    if json_path is None:
        today = time.strftime("%Y-%m-%d")
        benchmarks_dir = Path(__file__).resolve().parent.parent / "docs" / "benchmarks"
        json_path = str(benchmarks_dir / f"harvey-coc-{today}.json")
    _save_json(result, json_path)


if __name__ == "__main__":
    main()
