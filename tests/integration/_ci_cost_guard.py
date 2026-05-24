"""CI-side aggregate cost guard.

Sums ``cost_usd`` across every per-test ``SUMMARY.jsonl`` record and
exits non-zero if the total exceeds ``--budget``. Run from the
``aggregate cost guard`` step of ``integration-judge.yml`` (corpus-
stress-suite follow-up P1.2).

The per-test ``cost_usd`` field in SUMMARY.jsonl already includes the
judge spend (see ``tests/integration/conftest.py::_write_summary_line``
where ``record["cost_usd"]`` is rolled up as
``total_cost + judge_cost``). We still surface the judge breakdown
from ``judge.cost_usd`` separately so the per-test diagnostic table
shows where the spend went.

Hard fail: exits non-zero when over budget — never just warns. The
CI workflow relies on this exit code to fail the job.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--summary-glob", required=True)
    p.add_argument("--budget", type=float, required=True)
    args = p.parse_args()

    # ``Path.glob`` only takes a relative pattern, so we anchor on
    # cwd and let the caller-supplied glob (e.g.
    # ``tests/integration/runs/*/SUMMARY.jsonl``) walk from there.
    paths = sorted(Path().glob(args.summary_glob))

    total = 0.0
    rows: list[tuple[str, float]] = []
    files_seen = 0
    for path in paths:
        files_seen += 1
        try:
            with path.open(encoding="utf-8") as f:
                for raw in f:
                    line = raw.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    # SUMMARY.jsonl rolls judge spend into cost_usd
                    # already (see conftest._write_summary_line) — do
                    # not double-count by adding judge.cost_usd back in.
                    cost = float(rec.get("cost_usd", 0.0) or 0.0)
                    total += cost
                    rows.append((rec.get("test_nodeid", "?"), cost))
        except OSError as exc:
            print(f"WARN: could not read {path}: {exc}", file=sys.stderr)
            continue

    print(
        f"Aggregate cost: ${total:.4f} across {len(rows)} tests "
        f"in {files_seen} SUMMARY.jsonl file(s) "
        f"(budget ${args.budget:.2f})",
        file=sys.stderr,
    )
    rows.sort(key=lambda r: -r[1])
    print("Top 10 most expensive tests:", file=sys.stderr)
    for nodeid, cost in rows[:10]:
        print(f"  ${cost:.4f}  {nodeid}", file=sys.stderr)

    if total > args.budget:
        print(
            f"COST BUDGET EXCEEDED: ${total:.4f} > ${args.budget:.2f}",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
