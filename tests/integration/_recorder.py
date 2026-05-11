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

import asyncio
import datetime
import json
import os
import subprocess
import tempfile
import time
import traceback
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

# KC6 — env var read by kaos-llm-core's env_recorder when a subprocess
# imports it. The parent test sets this var (in the subprocess env only,
# never in its own os.environ) so cross-process LLM calls land in the
# test's run directory and can be stitched into the main JSONL on exit.
_SUBPROCESS_ENV_VAR = "KAOS_LLM_CORE_RECORDER_DIR"

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
    """Holds the recorded Invocations for one test.

    Attributes:
        invocations: Every Invocation that ran inside the context manager.
        lock_count: Re-entrant counter so nested ``record_live_test``
            calls share one patch site.
        outcome_override: When set (typically by the conftest fixture
            after reading pytest's makereport stash), takes precedence
            over the context-manager-level pass/fail detection. Needed
            because pytest's autouse-fixture ``yield`` doesn't see
            test-body assertion failures — pytest catches them first.
        error_override: Long-form failure repr from pytest's report.
            Only consulted when ``outcome_override == "failed"``.
        subprocess_dir: Temporary directory passed to subprocesses via
            ``KAOS_LLM_CORE_RECORDER_DIR`` (KC6). Subprocess
            kaos-llm-core imports auto-install the env recorder and
            append per-PID JSONL records here. The records are
            stitched into the main JSONL on exit.
        subprocess_records: Stitched records from the subprocess_dir,
            populated at exit by ``_collect_subprocess_records()``.
    """

    __slots__ = (
        "error_override",
        "invocations",
        "lock_count",
        "outcome_override",
        "subprocess_dir",
        "subprocess_records",
    )

    def __init__(self) -> None:
        self.invocations: list[Any] = []
        self.lock_count: int = 0
        self.outcome_override: str | None = None
        self.error_override: str | None = None
        self.subprocess_dir: Path | None = None
        self.subprocess_records: list[dict[str, Any]] = []


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

    # KC6: set up a per-test subprocess capture dir + Popen patch so
    # LLM calls inside spawned children land in our JSONL stitch.
    capture.subprocess_dir = Path(tempfile.mkdtemp(prefix="kaos-recorder-sub-"))
    _subprocess_patches = _install_subprocess_env_patches(capture.subprocess_dir)

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

        # KC6: roll back the subprocess.Popen / asyncio patches and
        # stitch any subprocess-written JSONL records into the
        # capture. Done before the file write so the main JSONL
        # carries both in-process and subprocess records.
        _restore_subprocess_env_patches(_subprocess_patches)
        try:
            capture.subprocess_records = _collect_subprocess_records(capture.subprocess_dir)
        except Exception:
            capture.subprocess_records = []
        _cleanup_subprocess_dir(capture.subprocess_dir)

        # The caller (typically the pytest conftest fixture) may have
        # stamped a definitive outcome on the capture after ``yield``
        # returned — pytest catches test-body assertion errors itself
        # so the BaseException branch above doesn't fire for them.
        # Honor the override when present.
        if capture.outcome_override is not None:
            outcome = capture.outcome_override
        if capture.error_override and error_repr is None:
            error_repr = capture.error_override

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

        # KC6: roll subprocess records into the same totals so the
        # header reflects total spend across both surfaces.
        for sub_record in capture.subprocess_records:
            usage_dict = sub_record.get("usage") or {}
            if not isinstance(usage_dict, dict):
                continue
            try:
                total_cost += float(usage_dict.get("cost_usd", 0.0) or 0.0)
                total_input_tokens += int(usage_dict.get("input_tokens", 0) or 0)
                total_output_tokens += int(usage_dict.get("output_tokens", 0) or 0)
            except (TypeError, ValueError):
                continue

        out_dir.mkdir(parents=True, exist_ok=True)
        fname = _sanitize_nodeid(test_nodeid) + ".jsonl"
        out_path = out_dir / fname

        total_subprocess_calls = len(capture.subprocess_records)
        total_calls = len(capture.invocations) + total_subprocess_calls

        header = {
            "kind": "header",
            "test_nodeid": test_nodeid,
            "start_ts_utc": start_ts,
            "end_ts_utc": end_ts,
            "elapsed_s": round(elapsed_s, 4),
            "outcome": outcome,
            "error": error_repr,
            "traceback": error_tb,
            "call_count": total_calls,
            "in_process_call_count": len(capture.invocations),
            "subprocess_call_count": total_subprocess_calls,
            "total_cost_usd": round(total_cost, 6),
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
            "git": git_context(repo_root) if repo_root else {},
            "python_version": _python_version(),
            "schema_version": 2,
            **(extra_metadata or {}),
        }

        try:
            with out_path.open("w", encoding="utf-8") as fh:
                fh.write(json.dumps(header, separators=(",", ":")) + "\n")
                for i, inv in enumerate(capture.invocations, start=1):
                    record = serialize_invocation(inv, call_seq=i)
                    fh.write(json.dumps(record, separators=(",", ":")) + "\n")
                # KC6: stitched subprocess records, tagged so consumers
                # can distinguish their origin from in-process records.
                for j, sub_record in enumerate(
                    capture.subprocess_records, start=len(capture.invocations) + 1
                ):
                    stitched = {**sub_record, "call_seq": j, "source": "subprocess"}
                    fh.write(json.dumps(stitched, separators=(",", ":")) + "\n")
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
                            "call_count": total_calls,
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


# ---------------------------------------------------------------------------
# KC6 — subprocess capture helpers
# ---------------------------------------------------------------------------


def _inject_env(env_arg: Any, recorder_dir: Path) -> dict[str, str]:
    """Build the env dict that subprocess.Popen will use.

    Adds ``KAOS_LLM_CORE_RECORDER_DIR`` to whatever the caller passed.
    When ``env_arg`` is None, copies os.environ so we don't strip the
    subprocess of every other var.
    """
    # Caller may have passed os.environ or a custom dict; copy to
    # avoid mutating the original. None → start from os.environ so
    # the subprocess keeps every other var the parent has.
    base = dict(os.environ) if env_arg is None else dict(env_arg)
    base[_SUBPROCESS_ENV_VAR] = str(recorder_dir)
    return base


def _install_subprocess_env_patches(recorder_dir: Path) -> dict[str, Any]:
    """Patch subprocess.Popen + asyncio subprocess helpers to inject the env var.

    Returns the originals dict so ``_restore_subprocess_env_patches``
    can roll them back. Idempotency / nesting is handled by the
    caller's lock_count check — this function unconditionally wraps.
    """
    originals: dict[str, Any] = {}

    original_popen_init = subprocess.Popen.__init__
    originals["popen_init"] = original_popen_init

    def patched_popen_init(self: Any, *args: Any, **kwargs: Any) -> None:
        kwargs["env"] = _inject_env(kwargs.get("env"), recorder_dir)
        return original_popen_init(self, *args, **kwargs)

    subprocess.Popen.__init__ = patched_popen_init  # ty: ignore[invalid-assignment]

    original_create_exec = asyncio.create_subprocess_exec
    originals["asyncio_exec"] = original_create_exec

    async def patched_create_exec(*args: Any, **kwargs: Any) -> Any:
        kwargs["env"] = _inject_env(kwargs.get("env"), recorder_dir)
        return await original_create_exec(*args, **kwargs)

    asyncio.create_subprocess_exec = patched_create_exec  # ty: ignore[invalid-assignment]

    original_create_shell = asyncio.create_subprocess_shell
    originals["asyncio_shell"] = original_create_shell

    async def patched_create_shell(*args: Any, **kwargs: Any) -> Any:
        kwargs["env"] = _inject_env(kwargs.get("env"), recorder_dir)
        return await original_create_shell(*args, **kwargs)

    asyncio.create_subprocess_shell = patched_create_shell  # ty: ignore[invalid-assignment]

    return originals


def _restore_subprocess_env_patches(originals: dict[str, Any]) -> None:
    """Roll back the patches installed by ``_install_subprocess_env_patches``."""
    if "popen_init" in originals:
        subprocess.Popen.__init__ = originals["popen_init"]
    if "asyncio_exec" in originals:
        asyncio.create_subprocess_exec = originals["asyncio_exec"]
    if "asyncio_shell" in originals:
        asyncio.create_subprocess_shell = originals["asyncio_shell"]


def _collect_subprocess_records(recorder_dir: Path | None) -> list[dict[str, Any]]:
    """Read every ``subprocess-<pid>-<hex>.jsonl`` under ``recorder_dir``.

    Returns the parsed records as a flat list, sorted by PID + line
    order so deterministic across runs. Missing dir or unreadable
    files contribute zero records.
    """
    if recorder_dir is None or not recorder_dir.exists():
        return []
    records: list[dict[str, Any]] = []
    for jsonl_path in sorted(recorder_dir.glob("subprocess-*.jsonl")):
        try:
            with jsonl_path.open("r", encoding="utf-8") as fh:
                for raw in fh:
                    stripped = raw.strip()
                    if not stripped:
                        continue
                    try:
                        records.append(json.loads(stripped))
                    except json.JSONDecodeError:
                        # Corrupt line — skip but keep the rest.
                        continue
        except OSError:
            continue
    return records


def _cleanup_subprocess_dir(recorder_dir: Path | None) -> None:
    """Remove the subprocess recorder tempdir. Best-effort."""
    if recorder_dir is None:
        return
    try:
        import shutil

        shutil.rmtree(recorder_dir, ignore_errors=True)
    except Exception:
        pass


__all__ = [
    "git_context",
    "record_live_test",
    "serialize_invocation",
]
