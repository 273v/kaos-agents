"""Live test telemetry recorder.

Captures every LLM call made during a test by patching
``kaos_llm_core.programs.call.Call._execute`` for the test's
duration. Each :class:`~kaos_llm_core.programs._invocation.Invocation`
returned (whether the test passed or raised) is serialized to a
JSONL file under ``tests/integration/runs/<date>/<nodeid>.jsonl``.

The header line of each file is a JSON object with test identity +
environment (git SHA, branch, timestamps, outcome). The remaining
lines are per-call records carrying the full
:class:`~kaos_llm_core.observability.traces.ExecutionTrace` —
inputs, outputs, model, tokens, cost, latency, retries, error.

Why this exists: the live tests for G6 (reflexion), G7 (router),
the pattern + parity suites — they all exercise real LLM behaviour
and then drop every byte of evidence at process exit. For software
serving regulated industries (legal, financial), that's an audit-
trail gap. SOC 2 CC7.2 / FINRA 4511 / HIPAA §164.312(b) all want a
durable record of automated decisions. This recorder gives us that
record per-test, keyed to a git SHA so you can diff behavioral
changes between commits.

The record is *additive*: tests still pass/fail the same way; the
recorder runs as a passive observer. No test logic changes are
needed when the recorder is enabled.

Usage (auto-installed by ``conftest.py``):

    @pytest.mark.live
    async def test_something():
        # The autouse fixture wraps this test in record_live_test().
        # Every Call.invoke that happens during the test is captured.
        ...

Manual usage:

    async with record_live_test("test_id", out_dir=Path("...")):
        await some_async_work()
"""

from __future__ import annotations

import datetime
import json
import subprocess
import time
import traceback
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Git context
# ---------------------------------------------------------------------------


def _git(*args: str, cwd: Path | None = None) -> str:
    """Run a git command; return stdout or "" on failure (best-effort)."""
    try:
        out = subprocess.run(
            ["git", *args],
            check=False,
            capture_output=True,
            text=True,
            cwd=cwd,
        )
        return out.stdout.strip()
    except (FileNotFoundError, OSError):
        return ""


def git_context(repo_root: Path) -> dict[str, str]:
    """Capture the git state at the moment the test starts."""
    return {
        "sha": _git("rev-parse", "HEAD", cwd=repo_root),
        "short_sha": _git("rev-parse", "--short", "HEAD", cwd=repo_root),
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD", cwd=repo_root),
        "dirty": "yes" if _git("status", "--porcelain", cwd=repo_root) else "no",
    }


# ---------------------------------------------------------------------------
# Invocation serialization
# ---------------------------------------------------------------------------


def _safe_dump(obj: Any) -> Any:
    """Coerce ``obj`` into something JSON can serialize."""
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, (datetime.datetime, datetime.date)):
        return obj.isoformat()
    if isinstance(obj, (list, tuple, set, frozenset)):
        return [_safe_dump(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _safe_dump(v) for k, v in obj.items()}
    # Pydantic v2 — use mode="json" so datetime, Enum, etc. coerce.
    if hasattr(obj, "model_dump"):
        try:
            return obj.model_dump(mode="json")
        except Exception:
            try:
                return _safe_dump(obj.model_dump())
            except Exception:
                pass
    # Dataclasses
    if hasattr(obj, "__dataclass_fields__"):
        try:
            return {f: _safe_dump(getattr(obj, f)) for f in obj.__dataclass_fields__}
        except Exception:
            pass
    # Enum
    if hasattr(obj, "value") and hasattr(obj, "name") and hasattr(type(obj), "__members__"):
        return obj.value
    # Last resort
    return repr(obj)


def serialize_invocation(invocation: Any, *, call_seq: int) -> dict[str, Any]:
    """Convert an :class:`Invocation` to a JSON-ready dict.

    Pulls the full ExecutionTrace (inputs / outputs / model / tokens /
    cost / latency / retries / error). Errors during serialization
    are swallowed and recorded as ``serialize_error`` so the recorder
    never breaks a test.
    """
    try:
        trace = getattr(invocation, "trace", None)
        usage = getattr(invocation, "usage", None)
        return {
            "kind": "invocation",
            "call_seq": call_seq,
            "invocation_id": getattr(invocation, "id", None),
            "model": getattr(invocation, "model", None),
            "output": _safe_dump(getattr(invocation, "output", None)),
            "error": (
                repr(invocation.error) if getattr(invocation, "error", None) is not None else None
            ),
            "trace": _safe_dump(trace) if trace is not None else None,
            "usage": _safe_dump(usage) if usage is not None else None,
        }
    except Exception as exc:
        return {
            "kind": "invocation",
            "call_seq": call_seq,
            "serialize_error": repr(exc),
        }


# ---------------------------------------------------------------------------
# Patch + capture
# ---------------------------------------------------------------------------


class _Capture:
    """Holds the recorded Invocations for one test."""

    __slots__ = ("invocations", "lock_count")

    def __init__(self) -> None:
        self.invocations: list[Any] = []
        # Re-entrant counter — nested Calls (Program-of-Programs) still
        # all share one capture but we only patch once per outer scope.
        self.lock_count: int = 0


@asynccontextmanager
async def record_live_test(
    test_nodeid: str,
    *,
    out_dir: Path,
    extra_metadata: dict[str, Any] | None = None,
) -> AsyncIterator[_Capture]:
    """Async context manager that captures every Call._execute during the block.

    On exit, writes a JSONL file at ``out_dir/<sanitized-nodeid>.jsonl``
    with one header line + one line per recorded Invocation.

    The capture is best-effort: a serialization error on one
    Invocation does not stop the others from being recorded, and a
    test failure / exception still writes whatever was captured up
    to that point. The exception is re-raised after the file is
    written.

    Args:
        test_nodeid: pytest nodeid (e.g. ``"tests/integration/...::test_x"``).
            Used as the filename stem after sanitization.
        out_dir: Directory to write the JSONL into. Created on demand.
        extra_metadata: Optional dict of test-level metadata merged
            into the header (e.g., test markers, parametrize IDs).

    Yields:
        The :class:`_Capture` object — useful for asserting on call
        counts inside a test, though tests typically just check the
        side-effect file post-hoc.
    """
    from kaos_llm_core.programs.call import Call  # lazy — keeps deferred import order

    capture = _Capture()
    original_execute = Call._execute  # bound method ref

    async def patched_execute(self_call, inputs: dict[str, Any]) -> Any:  # type: ignore[no-untyped-def]
        try:
            invocation = await original_execute(self_call, inputs)
        except BaseException as exc:
            # Recover the partial Invocation if the Call attached one
            # to the exception (Call.__call__ contract guarantees this
            # on ValidationRetryExhaustedError / CallError).
            partial = getattr(exc, "invocation", None)
            if partial is not None:
                capture.invocations.append(partial)
            raise
        capture.invocations.append(invocation)
        return invocation

    # Patch only once even with nested record_live_test calls.
    if capture.lock_count == 0:
        Call._execute = patched_execute  # ty: ignore[invalid-assignment]
    capture.lock_count += 1

    start_ts = datetime.datetime.now(datetime.UTC).isoformat()
    start_perf = time.perf_counter()
    repo_root = _find_repo_root(out_dir)

    outcome = "passed"
    error_repr: str | None = None
    error_tb: str | None = None
    try:
        yield capture
    except BaseException as exc:
        outcome = "failed"
        error_repr = repr(exc)
        error_tb = traceback.format_exc()
        raise
    finally:
        capture.lock_count -= 1
        if capture.lock_count == 0:
            Call._execute = original_execute

        elapsed_s = time.perf_counter() - start_perf
        end_ts = datetime.datetime.now(datetime.UTC).isoformat()

        # Aggregate cost across captured invocations.
        total_cost = 0.0
        total_input_tokens = 0
        total_output_tokens = 0
        for inv in capture.invocations:
            try:
                usage = getattr(inv, "usage", None)
                if usage is not None:
                    total_cost += float(getattr(usage, "cost_usd", 0.0) or 0.0)
                    total_input_tokens += int(getattr(usage, "input_tokens", 0) or 0)
                    total_output_tokens += int(getattr(usage, "output_tokens", 0) or 0)
            except Exception:
                continue

        out_dir.mkdir(parents=True, exist_ok=True)
        fname = _sanitize_nodeid(test_nodeid) + ".jsonl"
        out_path = out_dir / fname

        header = {
            "kind": "header",
            "test_nodeid": test_nodeid,
            "start_ts_utc": start_ts,
            "end_ts_utc": end_ts,
            "elapsed_s": round(elapsed_s, 4),
            "outcome": outcome,
            "error": error_repr,
            "traceback": error_tb,
            "call_count": len(capture.invocations),
            "total_cost_usd": round(total_cost, 6),
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
            "git": git_context(repo_root) if repo_root else {},
            "python_version": _python_version(),
            "schema_version": 1,
            **(extra_metadata or {}),
        }

        try:
            with out_path.open("w", encoding="utf-8") as fh:
                fh.write(json.dumps(header, separators=(",", ":")) + "\n")
                for i, inv in enumerate(capture.invocations, start=1):
                    record = serialize_invocation(inv, call_seq=i)
                    fh.write(json.dumps(record, separators=(",", ":")) + "\n")
        except Exception as exc:
            # The recorder must never break tests. If we can't write
            # the JSONL, write a tiny error stub and move on.
            try:
                with out_path.open("w", encoding="utf-8") as fh:
                    fh.write(
                        json.dumps(
                            {
                                "kind": "header",
                                "test_nodeid": test_nodeid,
                                "recorder_error": repr(exc),
                            }
                        )
                        + "\n"
                    )
            except Exception:
                pass

        # Append to the rolling index (one line per recorded run).
        try:
            index_path = out_dir.parent / "INDEX.jsonl"
            with index_path.open("a", encoding="utf-8") as fh:
                fh.write(
                    json.dumps(
                        {
                            "test_nodeid": test_nodeid,
                            "outcome": outcome,
                            "elapsed_s": round(elapsed_s, 4),
                            "total_cost_usd": round(total_cost, 6),
                            "call_count": len(capture.invocations),
                            "end_ts_utc": end_ts,
                            "file": str(out_path.relative_to(out_dir.parent.parent)),
                            "git_short_sha": header["git"].get("short_sha", "")
                            if header["git"]
                            else "",
                        },
                        separators=(",", ":"),
                    )
                    + "\n"
                )
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sanitize_nodeid(nodeid: str) -> str:
    """Make a pytest nodeid filesystem-safe."""
    bad = '/\\:?*"<> '
    out = "".join("_" if c in bad else c for c in nodeid)
    # Trim long names; 200 chars is generous for any sane filesystem.
    return out[-200:] if len(out) > 200 else out


def _find_repo_root(start: Path) -> Path | None:
    """Walk up from ``start`` looking for a ``.git`` directory."""
    cur = start.resolve()
    while cur != cur.parent:
        if (cur / ".git").exists():
            return cur
        cur = cur.parent
    return None


def _python_version() -> str:
    import sys

    return f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"


__all__ = [
    "git_context",
    "record_live_test",
    "serialize_invocation",
]
