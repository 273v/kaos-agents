"""Eval harness — runs pathologies x models, captures live traces.

Invocation contract:

    pytest tests/integration/eval -m live \
        --models openai:gpt-5.4-mini,anthropic:claude-sonnet-4-6 \
        --max-cost-per-case 0.50

or directly:

    python -m tests.integration.eval.harness \
        --models openai:gpt-5.4-mini,anthropic:claude-sonnet-4-6

The harness shells out to ``kaos-agent chat --message ... --log
<jsonl> --model M --pattern P --max-cost C --with-all`` and parses
the resulting JSONL into the event list the `Pathology.expected_signals`
predicates consume. We use the CLI rather than the Python API because
(a) the CLI is the load-bearing public entry point and (b) shell-out
gives us process isolation per case (no shared SessionMemory across
pathologies, no agent-state leakage).

Cost: each pathology x model = 1 shell invocation. Default model
pair (gpt-5.4-mini + claude-sonnet-4-6) at ~$0.05-$0.50 per case
x 4 default pathologies x 2 models = ~$0.40-$4.00 per full run.
The `--max-cost-per-case` arg passes through to ``kaos-agent chat``
which has its own circuit breaker.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import os
import re
import sys
import tempfile
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from tests.integration.eval.pathologies import (
    PATHOLOGY_PACK,
    Pathology,
)

# Two canonical frontier models, one per provider, that the project
# pins as the lower-bound coverage matrix (see
# `feedback_kaos_oss_legal_research_bar`).
DEFAULT_MODELS: tuple[str, ...] = (
    "openai:gpt-5.4-mini",
    "anthropic:claude-sonnet-4-6",
)


@dataclass(frozen=True, slots=True)
class SignalOutcome:
    name: str
    severity: Literal["required", "preferred"]
    passed: bool
    explanation: str = ""


@dataclass(frozen=True, slots=True)
class EvalResult:
    """One pathology x one model = one EvalResult."""

    pathology_code: str
    pathology_name: str
    model: str
    duration_s: float
    cost_usd: float
    tool_spans_called: tuple[str, ...]
    intent_classified: str | None
    assistant_text_snippet: str
    signal_outcomes: tuple[SignalOutcome, ...]
    stdout_tail: str
    error: str | None = None

    @property
    def required_passed(self) -> bool:
        return all(s.passed for s in self.signal_outcomes if s.severity == "required")

    @property
    def preferred_passed_count(self) -> int:
        return sum(1 for s in self.signal_outcomes if s.severity == "preferred" and s.passed)

    @property
    def preferred_total(self) -> int:
        return sum(1 for s in self.signal_outcomes if s.severity == "preferred")


@dataclass(frozen=True, slots=True)
class EvalReport:
    started_at: float
    finished_at: float
    results: tuple[EvalResult, ...]

    @property
    def required_pass_rate(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.required_passed for r in self.results) / len(self.results)

    @property
    def total_cost_usd(self) -> float:
        return sum(r.cost_usd for r in self.results)

    def render_markdown(self) -> str:
        """Render a matrix Markdown table for the PR description."""
        if not self.results:
            return "# kaos-agents-bench (empty run)\n"
        models = sorted({r.model for r in self.results})
        pathology_codes = sorted({r.pathology_code for r in self.results})
        lines: list[str] = []
        lines.append("# kaos-agents-bench live evaluation")
        lines.append("")
        lines.append(
            f"_{len(self.results)} cases · {len(models)} model(s) · "
            f"required pass {self.required_pass_rate:.0%} · "
            f"cost ${self.total_cost_usd:.4f}_"
        )
        lines.append("")
        header = "| Pathology | " + " | ".join(models) + " |"
        sep = "|" + "---|" * (len(models) + 1)
        lines.append(header)
        lines.append(sep)
        by_key: dict[tuple[str, str], EvalResult] = {
            (r.pathology_code, r.model): r for r in self.results
        }
        for code in pathology_codes:
            cells: list[str] = []
            for model in models:
                r = by_key.get((code, model))
                if r is None:
                    cells.append("—")
                    continue
                req = "✅" if r.required_passed else "❌"
                pref = (
                    f" ({r.preferred_passed_count}/{r.preferred_total} pref)"
                    if r.preferred_total
                    else ""
                )
                cells.append(f"{req}{pref}")
            lines.append(f"| {code} | " + " | ".join(cells) + " |")
        lines.append("")
        lines.append("## Per-case detail")
        for r in self.results:
            lines.append("")
            lines.append(
                f"### `{r.pathology_code}` x `{r.model}` — {'✅' if r.required_passed else '❌'}"
            )
            lines.append(f"- duration: {r.duration_s:.1f}s · cost: ${r.cost_usd:.4f}")
            lines.append(f"- intent classified: `{r.intent_classified or 'n/a'}`")
            lines.append(
                f"- tool spans: {len(r.tool_spans_called)}"
                + (
                    f" — `{', '.join(r.tool_spans_called[:6])}`"
                    + (" …" if len(r.tool_spans_called) > 6 else "")
                    if r.tool_spans_called
                    else ""
                )
            )
            if r.error:
                lines.append(f"- **error**: `{r.error}`")
            for s in r.signal_outcomes:
                mark = "✅" if s.passed else "❌"
                tag = "" if s.severity == "required" else " _(preferred)_"
                lines.append(f"  - {mark} {s.name}{tag}")
        return "\n".join(lines) + "\n"

    def to_json(self) -> str:
        """Serialise to JSON for archival / CI artefacts."""
        return json.dumps(
            {
                "started_at": self.started_at,
                "finished_at": self.finished_at,
                "duration_s": self.finished_at - self.started_at,
                "required_pass_rate": self.required_pass_rate,
                "total_cost_usd": self.total_cost_usd,
                "results": [
                    {
                        "pathology_code": r.pathology_code,
                        "pathology_name": r.pathology_name,
                        "model": r.model,
                        "duration_s": r.duration_s,
                        "cost_usd": r.cost_usd,
                        "tool_spans_called": list(r.tool_spans_called),
                        "intent_classified": r.intent_classified,
                        "assistant_text_snippet": r.assistant_text_snippet,
                        "required_passed": r.required_passed,
                        "signal_outcomes": [
                            {
                                "name": s.name,
                                "severity": s.severity,
                                "passed": s.passed,
                                "explanation": s.explanation,
                            }
                            for s in r.signal_outcomes
                        ],
                        "error": r.error,
                    }
                    for r in self.results
                ],
            },
            indent=2,
        )


def _load_event_trace(jsonl_path: Path) -> list[dict[str, Any]]:
    """Read a kaos-agent ``--log`` JSONL into a list of decoded events.

    The CLI's `--log` writes one event per line. Each event is the
    serialised KaosEvent. We extract a flat per-event record and
    normalise the few fields the predicate library reads.
    """
    out: list[dict[str, Any]] = []
    for raw in jsonl_path.read_text().splitlines():
        raw = raw.strip()
        if not raw:
            continue
        try:
            line = json.loads(raw)
        except json.JSONDecodeError:
            continue
        # Two shapes observed: bare event object OR {"event": {...}}
        ev = line.get("event") if isinstance(line, dict) and "event" in line else line
        if not isinstance(ev, dict):
            continue
        out.append(ev)
    return out


def _extract_run_summary(events: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Pull tool-span names, intent, text-delta snippet, cost from trace."""
    tool_spans: list[str] = []
    intent: str | None = None
    text_parts: list[str] = []
    cost_usd = 0.0
    for ev in events:
        t = ev.get("type")
        if t == "span":
            name = ev.get("name") or ""
            if name.startswith("tool."):
                tool_spans.append(name)
        elif t == "intent_classified":
            intent = ev.get("intent") or intent
        elif t == "text_delta":
            c = ev.get("content")
            if isinstance(c, str):
                text_parts.append(c)
        elif t in ("usage_observed", "turn_summary"):
            c = ev.get("cost_usd")
            if isinstance(c, int | float) and c > cost_usd:
                cost_usd = float(c)
    text = "".join(text_parts)
    return {
        "tool_spans": tuple(tool_spans),
        "intent": intent,
        "text_snippet": (text[:240] + "…") if len(text) > 240 else text,
        "cost_usd": cost_usd,
    }


_COST_RE = re.compile(r"\$([0-9]+\.[0-9]{2,4})")


def _extract_cli_cost(stdout: str) -> float | None:
    """Last resort: read cost from kaos-agent's stdout banner.

    The CLI prints e.g. ``[done] 3 tool(s), 65949 tokens, $0.0200``.
    """
    matches = _COST_RE.findall(stdout)
    if not matches:
        return None
    try:
        return float(matches[-1])
    except ValueError:
        return None


async def run_pathology(
    pathology: Pathology,
    model: str,
    *,
    session_prefix: str = "eval",
    timeout_s: float = 240.0,
    verbose: bool = False,
) -> EvalResult:
    """Run one (pathology, model) pair via ``kaos-agent chat`` subprocess.

    Returns an EvalResult; never raises.
    """
    t_start = time.monotonic()
    session_id = f"{session_prefix}-{pathology.code.lower()}-{int(time.time() * 1000)}"
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".jsonl", delete=False, prefix=f"{session_id}-"
    ) as fp:
        log_path = Path(fp.name)
    try:
        cmd = [
            "uv",
            "run",
            "kaos-agent",
            "chat",
            "--message",
            pathology.prompt,
            "--session",
            session_id,
            "--model",
            model,
            "--pattern",
            pathology.pattern,
            "--with-all",
            "--max-cost",
            f"{pathology.max_cost_usd}",
            "--log",
            str(log_path),
        ]
        if verbose:
            cmd.append("--verbose")

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            cwd=str(_kaos_agents_root()),
        )
        try:
            stdout_bytes, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout_s)
        except TimeoutError:
            proc.kill()
            await proc.wait()
            return EvalResult(
                pathology_code=pathology.code,
                pathology_name=pathology.name,
                model=model,
                duration_s=time.monotonic() - t_start,
                cost_usd=0.0,
                tool_spans_called=(),
                intent_classified=None,
                assistant_text_snippet="",
                signal_outcomes=tuple(
                    SignalOutcome(
                        name=s.name,
                        severity=s.severity,
                        passed=False,
                        explanation="timeout",
                    )
                    for s in pathology.expected_signals
                ),
                stdout_tail="",
                error=f"timeout after {timeout_s}s",
            )
        stdout = stdout_bytes.decode("utf-8", errors="replace")

        if not log_path.exists() or log_path.stat().st_size == 0:
            return EvalResult(
                pathology_code=pathology.code,
                pathology_name=pathology.name,
                model=model,
                duration_s=time.monotonic() - t_start,
                cost_usd=_extract_cli_cost(stdout) or 0.0,
                tool_spans_called=(),
                intent_classified=None,
                assistant_text_snippet="",
                signal_outcomes=tuple(
                    SignalOutcome(
                        name=s.name,
                        severity=s.severity,
                        passed=False,
                        explanation="no event log produced",
                    )
                    for s in pathology.expected_signals
                ),
                stdout_tail=stdout[-2000:],
                error=f"no log; exit={proc.returncode}",
            )

        events = _load_event_trace(log_path)
        summary = _extract_run_summary(events)

        signal_outcomes: list[SignalOutcome] = []
        for sig in pathology.expected_signals:
            try:
                passed = bool(sig.check(events))
            except Exception as exc:
                signal_outcomes.append(
                    SignalOutcome(
                        name=sig.name,
                        severity=sig.severity,
                        passed=False,
                        explanation=f"predicate error: {exc!r}",
                    )
                )
                continue
            signal_outcomes.append(
                SignalOutcome(
                    name=sig.name,
                    severity=sig.severity,
                    passed=passed,
                    explanation=sig.explanation,
                )
            )

        cost = summary["cost_usd"] or (_extract_cli_cost(stdout) or 0.0)
        return EvalResult(
            pathology_code=pathology.code,
            pathology_name=pathology.name,
            model=model,
            duration_s=time.monotonic() - t_start,
            cost_usd=cost,
            tool_spans_called=summary["tool_spans"],
            intent_classified=summary["intent"],
            assistant_text_snippet=summary["text_snippet"],
            signal_outcomes=tuple(signal_outcomes),
            stdout_tail=stdout[-2000:],
            error=None,
        )
    finally:
        with contextlib.suppress(OSError):
            log_path.unlink(missing_ok=True)


def _kaos_agents_root() -> Path:
    """Return the kaos-agents repo root.

    Used as the subprocess cwd so ``uv run`` resolves the correct
    virtualenv even when the eval harness is invoked from outside.
    """
    here = Path(__file__).resolve()
    # tests/integration/eval/harness.py  →  parents[3] is repo root
    return here.parents[3]


async def run_pathology_matrix(
    *,
    pathologies: Sequence[Pathology] = PATHOLOGY_PACK,
    models: Sequence[str] = DEFAULT_MODELS,
    concurrency: int = 4,
    verbose: bool = False,
) -> EvalReport:
    """Run every (pathology, model) pair with bounded concurrency."""
    sem = asyncio.Semaphore(concurrency)
    t_start = time.time()

    async def _gated(p: Pathology, m: str) -> EvalResult:
        async with sem:
            return await run_pathology(p, m, verbose=verbose)

    tasks = [_gated(p, m) for p in pathologies for m in models]
    results = await asyncio.gather(*tasks)
    return EvalReport(
        started_at=t_start,
        finished_at=time.time(),
        results=tuple(results),
    )


def _parse_models(arg: str) -> tuple[str, ...]:
    return tuple(m.strip() for m in arg.split(",") if m.strip())


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="kaos-agents-bench",
        description="Run the agentic-pathology eval harness.",
    )
    parser.add_argument(
        "--models",
        type=str,
        default=",".join(DEFAULT_MODELS),
        help="Comma-separated provider:model strings.",
    )
    parser.add_argument(
        "--pathology",
        type=str,
        default="",
        help="If set, only run pathologies whose code is in this comma-list.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=4,
        help="Max concurrent CLI subprocesses.",
    )
    parser.add_argument(
        "--out-md",
        type=str,
        default="",
        help="Write the Markdown report to this path (default: stdout).",
    )
    parser.add_argument(
        "--out-json",
        type=str,
        default="",
        help="Write the JSON report to this path.",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    models = _parse_models(args.models)
    if args.pathology:
        wanted = {c.strip() for c in args.pathology.split(",") if c.strip()}
        pack = tuple(p for p in PATHOLOGY_PACK if p.code in wanted)
        if not pack:
            print(
                f"no pathologies match {wanted!r}; have {[p.code for p in PATHOLOGY_PACK]}",
                file=sys.stderr,
            )
            return 2
    else:
        pack = PATHOLOGY_PACK

    if not _have_api_keys(models):
        print(
            "warning: at least one selected model lacks an API key in env",
            file=sys.stderr,
        )

    report = asyncio.run(
        run_pathology_matrix(
            pathologies=pack,
            models=models,
            concurrency=args.concurrency,
            verbose=args.verbose,
        )
    )

    md = report.render_markdown()
    if args.out_md:
        Path(args.out_md).write_text(md)
        print(f"wrote {args.out_md}", file=sys.stderr)
    else:
        print(md)
    if args.out_json:
        Path(args.out_json).write_text(report.to_json())
        print(f"wrote {args.out_json}", file=sys.stderr)

    # Exit code: 0 iff every required signal passed on every model.
    return 0 if report.required_pass_rate >= 1.0 else 1


def _have_api_keys(models: Sequence[str]) -> bool:
    needed: set[str] = set()
    for m in models:
        if m.startswith("openai:"):
            needed.add("OPENAI_API_KEY")
        elif m.startswith("anthropic:"):
            needed.add("ANTHROPIC_API_KEY")
        elif m.startswith("google:"):
            needed.add("GOOGLE_API_KEY")
        elif m.startswith("xai:"):
            needed.add("XAI_API_KEY")
    return all(os.environ.get(k) for k in needed)


if __name__ == "__main__":
    sys.exit(_main())
