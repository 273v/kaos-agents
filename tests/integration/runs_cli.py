#!/usr/bin/env python3
"""kaos-agents-runs — viewer / diff over captured live-test telemetry.

The autouse recorder fixture in ``tests/integration/conftest.py``
writes one JSONL per ``@pytest.mark.live`` test under
``tests/integration/runs/<YYYY-MM-DD>/`` plus a summary line to
``tests/integration/runs/INDEX.jsonl``. This CLI gives an opinionated
read over those artifacts so you don't have to compose ``jq`` +
``diff`` by hand.

Subcommands::

    list       List captured runs with cost / outcome / timestamps.
    show       Pretty-print one captured run (header + every call).
    diff       Side-by-side compare two runs of the same test.
    summary    Per-day / per-commit cost rollups.

All subcommands accept ``--runs-dir`` (default: this file's parent
``runs/`` sibling), so they work both from the repo root and from
``tests/integration/``.

Examples::

    # Today's runs, by cost desc
    python tests/integration/runs_cli.py list --sort cost

    # All runs of the strict-rubric reflexion test
    python tests/integration/runs_cli.py list --grep strict_rubric

    # Show one run pretty-printed
    python tests/integration/runs_cli.py show <test_name_substr>

    # Diff the two most-recent runs of the same test
    python tests/integration/runs_cli.py diff <test_name_substr>

    # Per-commit cost rollup
    python tests/integration/runs_cli.py summary --by commit
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Index + per-file loaders
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class IndexEntry:
    """One row from INDEX.jsonl. Authoritative source of truth for listing."""

    test_nodeid: str
    outcome: str
    elapsed_s: float
    total_cost_usd: float
    call_count: int
    end_ts_utc: str
    file: str
    git_short_sha: str


def _default_runs_dir() -> Path:
    return Path(__file__).parent / "runs"


def load_index(runs_dir: Path) -> list[IndexEntry]:
    """Read INDEX.jsonl into a list of typed entries. Empty when missing."""
    path = runs_dir / "INDEX.jsonl"
    if not path.exists():
        return []
    entries: list[IndexEntry] = []
    with path.open("r", encoding="utf-8") as fh:
        for raw in fh:
            raw = raw.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
                entries.append(
                    IndexEntry(
                        test_nodeid=obj.get("test_nodeid", ""),
                        outcome=obj.get("outcome", ""),
                        elapsed_s=float(obj.get("elapsed_s", 0.0)),
                        total_cost_usd=float(obj.get("total_cost_usd", 0.0)),
                        call_count=int(obj.get("call_count", 0)),
                        end_ts_utc=obj.get("end_ts_utc", ""),
                        file=obj.get("file", ""),
                        git_short_sha=obj.get("git_short_sha", ""),
                    )
                )
            except (ValueError, KeyError, TypeError):
                continue
    return entries


def load_run(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Read one captured-run JSONL: returns ``(header, [calls...])``."""
    header: dict[str, Any] = {}
    calls: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for raw in fh:
            raw = raw.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except ValueError:
                continue
            kind = obj.get("kind", "")
            if kind == "header" and not header:
                header = obj
            elif kind == "invocation":
                calls.append(obj)
    return header, calls


# ---------------------------------------------------------------------------
# Filtering / matching
# ---------------------------------------------------------------------------


def filter_entries(
    entries: list[IndexEntry],
    *,
    grep: str | None = None,
    outcome: str | None = None,
    commit: str | None = None,
    date: str | None = None,
) -> list[IndexEntry]:
    """Apply free-form filters to a list of entries."""
    out = entries
    if grep:
        out = [e for e in out if grep in e.test_nodeid or grep in e.file]
    if outcome:
        out = [e for e in out if e.outcome == outcome]
    if commit:
        out = [e for e in out if e.git_short_sha.startswith(commit)]
    if date:
        out = [e for e in out if e.end_ts_utc.startswith(date) or date in e.file]
    return out


def sort_entries(entries: list[IndexEntry], *, by: str) -> list[IndexEntry]:
    """Sort entries by one of: time | cost | calls | name."""
    keys = {
        "time": lambda e: e.end_ts_utc,
        "cost": lambda e: e.total_cost_usd,
        "calls": lambda e: e.call_count,
        "name": lambda e: e.test_nodeid,
    }
    if by not in keys:
        raise SystemExit(f"--sort: unknown key {by!r}; choose from {sorted(keys)}")
    return sorted(entries, key=keys[by], reverse=(by != "name"))


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------


def fmt_short_test_name(nodeid: str, *, width: int = 60) -> str:
    """pytest nodeid is verbose; show just the test method + parametrize."""
    if "::" in nodeid:
        nodeid = "::".join(nodeid.rsplit("::", 2)[-2:])
    if len(nodeid) > width:
        nodeid = "…" + nodeid[-(width - 1) :]
    return nodeid


def render_list(entries: list[IndexEntry]) -> None:
    """One-line-per-entry table to stdout."""
    if not entries:
        print("no entries match the filters", file=sys.stderr)
        return
    print(f"{'outcome':<8} {'cost':>9} {'calls':>5} {'elapsed':>8}  {'commit':<9}  test")
    print(f"{'-' * 8} {'-' * 9} {'-' * 5} {'-' * 8}  {'-' * 9}  {'-' * 60}")
    for e in entries:
        print(
            f"{e.outcome:<8} ${e.total_cost_usd:>8.4f} {e.call_count:>5d} "
            f"{e.elapsed_s:>7.1f}s  {e.git_short_sha:<9}  "
            f"{fmt_short_test_name(e.test_nodeid)}"
        )


def render_show(header: dict[str, Any], calls: list[dict[str, Any]]) -> None:
    """Pretty-print one captured run."""
    print("=" * 78)
    print(f"  test     : {header.get('test_nodeid', '?')}")
    print(f"  outcome  : {header.get('outcome', '?')}")
    print(f"  elapsed  : {header.get('elapsed_s', 0):.2f}s")
    print(f"  cost     : ${header.get('total_cost_usd', 0):.4f}")
    print(f"  calls    : {header.get('call_count', 0)}")
    print(
        f"  tokens   : in={header.get('total_input_tokens', 0)} "
        f"out={header.get('total_output_tokens', 0)}"
    )
    git = header.get("git", {})
    if git:
        print(
            f"  git      : {git.get('short_sha', '?')} on {git.get('branch', '?')}"
            f" (dirty={git.get('dirty', '?')})"
        )
    print(f"  started  : {header.get('start_ts_utc', '?')}")
    if header.get("error"):
        print(f"  error    : {header['error']}")
    print("=" * 78)
    for c in calls:
        seq = c.get("call_seq", "?")
        model = c.get("model", "?")
        trace = c.get("trace") or {}
        tokens = trace.get("total_tokens", 0)
        cost = trace.get("cost_usd", 0.0)
        latency = trace.get("latency_ms", 0.0)
        retries = trace.get("retries", 0)
        call_name = trace.get("call_name", "?")
        print(f"\n--- call #{seq}  {call_name}  model={model}")
        print(f"    tokens={tokens}  cost=${cost:.5f}  latency={latency:.0f}ms  retries={retries}")
        inputs = trace.get("inputs", {})
        for k, v in inputs.items():
            s = str(v)
            s = s[:300] + ("…" if len(s) > 300 else "")
            print(f"    in.{k}: {s}")
        output = c.get("output")
        if output is not None:
            s = json.dumps(output) if isinstance(output, dict) else str(output)
            s = s[:500] + ("…" if len(s) > 500 else "")
            print(f"    output : {s}")
        if c.get("error"):
            print(f"    error  : {c['error']}")


def render_diff(
    header_a: dict[str, Any],
    calls_a: list[dict[str, Any]],
    header_b: dict[str, Any],
    calls_b: list[dict[str, Any]],
) -> None:
    """Side-by-side comparison of two captured runs of the same test."""
    print("=" * 78)
    print(
        f"  A: {header_a.get('start_ts_utc', '?')}  "
        f"git={header_a.get('git', {}).get('short_sha', '?')}"
    )
    print(
        f"  B: {header_b.get('start_ts_utc', '?')}  "
        f"git={header_b.get('git', {}).get('short_sha', '?')}"
    )
    print("=" * 78)

    def row(label: str, a: Any, b: Any) -> None:
        a_s = str(a)
        b_s = str(b)
        marker = "  " if a_s == b_s else "≠ "
        print(f"{marker}{label:<22} {a_s:<25}  →  {b_s}")

    row("outcome", header_a.get("outcome"), header_b.get("outcome"))
    row(
        "cost_usd",
        f"${header_a.get('total_cost_usd', 0):.4f}",
        f"${header_b.get('total_cost_usd', 0):.4f}",
    )
    row("call_count", header_a.get("call_count"), header_b.get("call_count"))
    row("elapsed_s", f"{header_a.get('elapsed_s', 0):.2f}", f"{header_b.get('elapsed_s', 0):.2f}")
    row("input_tokens", header_a.get("total_input_tokens"), header_b.get("total_input_tokens"))
    row("output_tokens", header_a.get("total_output_tokens"), header_b.get("total_output_tokens"))

    print("\n--- per-call delta ---")
    n = max(len(calls_a), len(calls_b))
    for i in range(n):
        a = calls_a[i] if i < len(calls_a) else None
        b = calls_b[i] if i < len(calls_b) else None
        if a is None and b is not None:
            b_out = b.get("output") or {}
            b_score = b_out.get("score", "?") if isinstance(b_out, dict) else "?"
            print(f"call#{i + 1}  (A: missing)  B: {b.get('model')} score={b_score}")
            continue
        if b is None and a is not None:
            a_out = a.get("output") or {}
            a_score = a_out.get("score", "?") if isinstance(a_out, dict) else "?"
            print(f"call#{i + 1}  A: {a.get('model')} score={a_score}  (B: missing)")
            continue
        if a is None or b is None:
            continue
        a_out = a.get("output") or {}
        b_out = b.get("output") or {}
        a_marker = a_out if isinstance(a_out, dict) else str(a_out)[:80]
        b_marker = b_out if isinstance(b_out, dict) else str(b_out)[:80]
        # Highlight common interesting fields
        for field in ("score", "specialist_name", "confidence"):
            if isinstance(a_marker, dict) and field in a_marker:
                a_val = a_marker.get(field)
                b_val = b_marker.get(field) if isinstance(b_marker, dict) else None
                mk = "≠ " if a_val != b_val else "  "
                print(f"  call#{i + 1}  {mk}{field:<18} {a_val!r}  →  {b_val!r}")
                break
        else:
            # Generic fallback — compare the output blobs as strings
            a_s = json.dumps(a_marker)[:80] if isinstance(a_marker, dict) else str(a_marker)[:80]
            b_s = json.dumps(b_marker)[:80] if isinstance(b_marker, dict) else str(b_marker)[:80]
            mk = "≠ " if a_s != b_s else "  "
            print(f"  call#{i + 1}  {mk}output           {a_s}  →  {b_s}")


def render_summary(entries: list[IndexEntry], *, by: str) -> None:
    """Per-day or per-commit cost rollups."""
    if by not in ("commit", "day", "test"):
        raise SystemExit(f"summary --by: choose commit | day | test, got {by!r}")
    grouped: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "count": 0,
            "cost": 0.0,
            "passed": 0,
            "failed": 0,
            "calls": 0,
        }
    )
    for e in entries:
        if by == "commit":
            key = e.git_short_sha or "(no-git)"
        elif by == "day":
            key = (e.end_ts_utc or "")[:10] or "(no-date)"
        else:
            key = e.test_nodeid
        g = grouped[key]
        g["count"] += 1
        g["cost"] += e.total_cost_usd
        g["passed"] += 1 if e.outcome == "passed" else 0
        g["failed"] += 1 if e.outcome == "failed" else 0
        g["calls"] += e.call_count

    label = {"commit": "commit", "day": "date", "test": "test"}[by]
    print(f"{label:<60} {'n':>5} {'pass':>5} {'fail':>5} {'cost':>9} {'calls':>6}")
    print(f"{'-' * 60} {'-' * 5} {'-' * 5} {'-' * 5} {'-' * 9} {'-' * 6}")
    for key in sorted(grouped):
        g = grouped[key]
        label_s = key if len(key) <= 60 else "…" + key[-59:]
        print(
            f"{label_s:<60} {g['count']:>5d} {g['passed']:>5d} {g['failed']:>5d} "
            f"${g['cost']:>8.4f} {g['calls']:>6d}"
        )


# ---------------------------------------------------------------------------
# Subcommand drivers
# ---------------------------------------------------------------------------


def cmd_list(args: argparse.Namespace) -> int:
    runs_dir = Path(args.runs_dir).resolve()
    entries = load_index(runs_dir)
    entries = filter_entries(
        entries,
        grep=args.grep,
        outcome=args.outcome,
        commit=args.commit,
        date=args.date,
    )
    entries = sort_entries(entries, by=args.sort)
    if args.limit:
        entries = entries[: args.limit]
    if args.json:
        for e in entries:
            print(
                json.dumps(
                    {
                        "test_nodeid": e.test_nodeid,
                        "outcome": e.outcome,
                        "total_cost_usd": e.total_cost_usd,
                        "call_count": e.call_count,
                        "elapsed_s": e.elapsed_s,
                        "end_ts_utc": e.end_ts_utc,
                        "git_short_sha": e.git_short_sha,
                        "file": e.file,
                    }
                )
            )
    else:
        render_list(entries)
    return 0


def _resolve_runs_for_substr(entries: list[IndexEntry], substr: str) -> list[IndexEntry]:
    """Find INDEX entries whose nodeid contains ``substr``."""
    matches = [e for e in entries if substr in e.test_nodeid]
    return matches


def cmd_show(args: argparse.Namespace) -> int:
    runs_dir = Path(args.runs_dir).resolve()
    entries = load_index(runs_dir)
    matches = _resolve_runs_for_substr(entries, args.test)
    if not matches:
        print(f"no run matches {args.test!r}", file=sys.stderr)
        return 2
    # Pick most recent by default; ``--all`` shows all
    matches = sorted(matches, key=lambda e: e.end_ts_utc, reverse=True)
    if not args.all:
        matches = matches[:1]
    for e in matches:
        # ``e.file`` is relative to tests/integration/ — resolve to absolute.
        path = (runs_dir.parent / e.file).resolve()
        if not path.exists():
            print(f"  (missing file: {path})", file=sys.stderr)
            continue
        header, calls = load_run(path)
        render_show(header, calls)
    return 0


def cmd_diff(args: argparse.Namespace) -> int:
    runs_dir = Path(args.runs_dir).resolve()
    entries = load_index(runs_dir)
    matches = _resolve_runs_for_substr(entries, args.test)
    if len(matches) < 2:
        print(
            f"need at least 2 runs to diff; got {len(matches)} matching {args.test!r}",
            file=sys.stderr,
        )
        return 2
    matches = sorted(matches, key=lambda e: e.end_ts_utc)
    a = (
        matches[-2]
        if not args.a
        else next((e for e in matches if args.a in e.end_ts_utc), matches[-2])
    )
    b = (
        matches[-1]
        if not args.b
        else next((e for e in matches if args.b in e.end_ts_utc), matches[-1])
    )
    path_a = (runs_dir.parent / a.file).resolve()
    path_b = (runs_dir.parent / b.file).resolve()
    if not path_a.exists() or not path_b.exists():
        print(f"missing file: {path_a if not path_a.exists() else path_b}", file=sys.stderr)
        return 2
    header_a, calls_a = load_run(path_a)
    header_b, calls_b = load_run(path_b)
    render_diff(header_a, calls_a, header_b, calls_b)
    return 0


def cmd_summary(args: argparse.Namespace) -> int:
    runs_dir = Path(args.runs_dir).resolve()
    entries = load_index(runs_dir)
    entries = filter_entries(
        entries,
        grep=args.grep,
        outcome=args.outcome,
        commit=args.commit,
        date=args.date,
    )
    render_summary(entries, by=args.by)
    return 0


# ---------------------------------------------------------------------------
# Argparse
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="runs_cli", description=__doc__)
    parser.add_argument(
        "--runs-dir",
        default=str(_default_runs_dir()),
        help="Path to the captured-runs directory (default: ./runs/)",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_list = sub.add_parser("list", help="List captured runs.")
    p_list.add_argument("--grep", help="Substring filter on test_nodeid / file.")
    p_list.add_argument("--outcome", choices=["passed", "failed"], help="Filter by outcome.")
    p_list.add_argument("--commit", help="Filter by short SHA prefix.")
    p_list.add_argument("--date", help="Filter by date (YYYY-MM-DD).")
    p_list.add_argument(
        "--sort",
        choices=["time", "cost", "calls", "name"],
        default="time",
    )
    p_list.add_argument("--limit", type=int, default=0, help="Cap rows shown.")
    p_list.add_argument("--json", action="store_true", help="Emit JSON lines.")
    p_list.set_defaults(func=cmd_list)

    p_show = sub.add_parser("show", help="Pretty-print one run.")
    p_show.add_argument("test", help="Substring of the test nodeid.")
    p_show.add_argument("--all", action="store_true", help="Show every match, not just the latest.")
    p_show.set_defaults(func=cmd_show)

    p_diff = sub.add_parser("diff", help="Compare two runs of the same test.")
    p_diff.add_argument("test", help="Substring of the test nodeid.")
    p_diff.add_argument("--a", help="Pick run A by end_ts substring (default: 2nd-most-recent).")
    p_diff.add_argument("--b", help="Pick run B by end_ts substring (default: most-recent).")
    p_diff.set_defaults(func=cmd_diff)

    p_sum = sub.add_parser("summary", help="Cost / outcome rollups.")
    p_sum.add_argument("--by", choices=["commit", "day", "test"], default="day")
    p_sum.add_argument("--grep", help="Substring filter.")
    p_sum.add_argument("--outcome", choices=["passed", "failed"])
    p_sum.add_argument("--commit", help="Filter by short SHA prefix.")
    p_sum.add_argument("--date", help="Filter by date (YYYY-MM-DD).")
    p_sum.set_defaults(func=cmd_summary)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
