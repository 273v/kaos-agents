"""CI-side PR comment writer for the integration-judge lane.

Reads every ``SUMMARY.jsonl`` produced by the corpus-stress + web-
tools suites (one line per ``@pytest.mark.live`` test, written by
``tests/integration/conftest.py``) and posts a markdown table to the
PR via ``gh pr comment``.

Empty-input safety: if the glob matches no files (e.g. the suite
crashed before writing any line), the script posts a one-line note
saying so instead of throwing. Same goes for malformed records.

Invocation (see ``.github/workflows/integration-judge.yml``):

    python tests/integration/_ci_pr_comment.py \\
        --summary-glob 'tests/integration/runs/*/SUMMARY.jsonl' \\
        --pr 123 --budget 5.00
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


def _load_records(summary_glob: str) -> list[dict[str, Any]]:
    """Slurp every JSONL record across all SUMMARY.jsonl files matching glob."""
    out: list[dict[str, Any]] = []
    for path in sorted(Path().glob(summary_glob)):
        try:
            with path.open(encoding="utf-8") as f:
                for raw in f:
                    line = raw.strip()
                    if not line:
                        continue
                    try:
                        out.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        except OSError:
            continue
    return out


def _short_test_id(nodeid: str, judge: dict[str, Any] | None) -> str:
    """Derive a short label like ``S01`` / ``W11`` from the judge ``test_id``
    if present, otherwise from the nodeid tail."""
    if judge and judge.get("test_id"):
        return str(judge["test_id"])
    # Last :: component is the test func name; strip the leading "test_".
    tail = nodeid.rsplit("::", 1)[-1]
    if tail.startswith("test_"):
        tail = tail[len("test_") :]
    return tail


def _outcome_glyph(outcome: str) -> str:
    return {
        "passed": "pass",
        "failed": "FAIL",
        "skipped": "skip",
    }.get(outcome, outcome or "?")


def _judge_cell(judge: dict[str, Any] | None) -> str:
    if not judge:
        return "—"
    label = judge.get("label") or "?"
    conf = judge.get("confidence")
    if conf is not None:
        try:
            return f"{label} ({float(conf):.2f})"
        except (TypeError, ValueError):
            return str(label)
    return str(label)


def _tools_cell(rec: dict[str, Any]) -> str:
    n = int(rec.get("n_tool_calls", 0) or 0)
    n_err = int(rec.get("n_tool_errors", 0) or 0)
    if n == 0:
        return "0"
    if n_err:
        return f"{n} ({n_err} err)"
    return str(n)


def _render_markdown(
    records: list[dict[str, Any]],
    *,
    pr_number: int,
    budget: float | None,
) -> str:
    if not records:
        return (
            f"## Integration Judge Results — PR #{pr_number}\n\n"
            "No per-test SUMMARY.jsonl records were produced. "
            "The integration suites likely crashed before any test "
            "completed — see the workflow logs."
        )

    lines: list[str] = []
    lines.append(f"## Integration Judge Results — PR #{pr_number}\n")
    lines.append("| Test | Outcome | Judge | Cost | Tools |")
    lines.append("|------|---------|-------|------|-------|")

    total_cost = 0.0
    total_judge_cost = 0.0
    n_pass = n_fail = n_skip = 0
    for rec in records:
        nodeid = rec.get("test_nodeid", "?")
        judge = rec.get("judge") if isinstance(rec.get("judge"), dict) else None
        cost = float(rec.get("cost_usd", 0.0) or 0.0)
        total_cost += cost
        if judge:
            jc = judge.get("cost_usd")
            with contextlib.suppress(TypeError, ValueError):
                total_judge_cost += float(jc or 0.0)

        outcome = rec.get("outcome", "?")
        if outcome == "passed":
            n_pass += 1
        elif outcome == "failed":
            n_fail += 1
        elif outcome == "skipped":
            n_skip += 1

        test_id = _short_test_id(nodeid, judge)
        lines.append(
            f"| {test_id} | {_outcome_glyph(outcome)} | "
            f"{_judge_cell(judge)} | ${cost:.4f} | {_tools_cell(rec)} |"
        )

    lines.append("")
    summary = (
        f"**Total: ${total_cost:.4f} across {len(records)} tests** "
        f"(judge: ${total_judge_cost:.4f}; "
        f"pass: {n_pass} / fail: {n_fail} / skip: {n_skip})"
    )
    if budget is not None:
        over = total_cost > budget
        summary += f". Budget: ${budget:.2f}"
        if over:
            summary += " — **EXCEEDED**"
    lines.append(summary)

    return "\n".join(lines)


def _post_comment(body: str, *, pr_number: int) -> int:
    """Post the comment via ``gh pr comment``. Best-effort.

    Returns 0 on success or when ``gh`` is unavailable (we don't want
    a missing CLI to fail the workflow — the table is already in the
    job logs via stdout).
    """
    print(body)  # always echo so the table is visible in workflow logs

    if not shutil.which("gh"):
        print("WARN: gh CLI not available; skipping PR comment", file=sys.stderr)
        return 0

    if not os.environ.get("GH_TOKEN") and not os.environ.get("GITHUB_TOKEN"):
        print("WARN: no GH_TOKEN/GITHUB_TOKEN; skipping PR comment", file=sys.stderr)
        return 0

    # Write body to a tempfile so we don't run into shell-quoting woes.
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".md", delete=False) as tf:
        tf.write(body)
        body_path = Path(tf.name)

    try:
        result = subprocess.run(
            ["gh", "pr", "comment", str(pr_number), "--body-file", str(body_path)],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            print(
                f"WARN: gh pr comment failed (rc={result.returncode}): {result.stderr.strip()}",
                file=sys.stderr,
            )
        return 0  # never fail the workflow on a comment-posting hiccup
    finally:
        with contextlib.suppress(OSError):
            body_path.unlink()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--summary-glob", required=True)
    p.add_argument("--pr", type=int, required=True, help="PR number")
    p.add_argument("--budget", type=float, default=None, help="Total cost budget in USD")
    args = p.parse_args()

    records = _load_records(args.summary_glob)
    body = _render_markdown(records, pr_number=args.pr, budget=args.budget)
    return _post_comment(body, pr_number=args.pr)


if __name__ == "__main__":
    sys.exit(main())
