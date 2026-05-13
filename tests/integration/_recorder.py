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

Crash-safe contract (schema_version=4):

* The JSONL is **streamed**: the header is written + flushed +
  ``fsync()``ed immediately when the recorder starts. Each
  completed Invocation is appended + flushed + ``fsync()``ed as it
  finishes, **before** the call returns to the test.
* The trailer line (``kind="trailer"``) carries the final outcome,
  elapsed time, total cost, and stitched subprocess records. It is
  written at exit. It is **optional** — readers MUST tolerate a
  JSONL that ends after N invocation lines with no trailer.
* A reader walking a JSONL captured under SIGTERM may also see the
  last line truncated mid-write (Linux's ``fsync`` does not give a
  hard guarantee under hard-kill / pod-eviction / OOM-kill). The
  reader MUST swallow ``json.JSONDecodeError`` on the final line
  and use the lines it has.
* The header field ``streaming=true`` documents this for consumers
  so they can branch on it cleanly.

Transparency redaction (schema_version=4 — KC16-4):

The audit found that the schema-v3 recorder persisted full document
bodies (50KB+ ``message`` fields) verbatim into JSONL captures, and
those captures were being committed to git. For a regulated-industry
adopter every CI-processed document becomes a secondary data plane.
KC16-4 changes the on-disk shape WITHOUT changing what the recorder
captures conceptually:

* Every string value longer than ``redaction_threshold_chars``
  (default 2048; configurable via ``KAOS_RECORDER_REDACTION_THRESHOLD``)
  is replaced with ``"<first 200 chars> ... [TRUNCATED N chars;
  sha256=<16 hex>]"``. The sha256 prefix lets you dedup captures of
  the same document; 16 hex chars (~64 bits) is enough entropy for
  that but not enough to reconstruct the document.
* Metadata fields (``model``, ``invocation_id``, token counts, cost,
  latency, error class) are untouched. Redaction only kicks in on
  string values above the threshold.
* The header carries ``redaction_enabled``, ``redaction_threshold_chars``;
  the trailer carries ``redacted_count`` (total fields truncated).
* **Opt-out for local development**: set
  ``KAOS_RECORDER_FULL_TEXT=1`` in the env. The header records
  ``redaction_enabled=false`` so downstream consumers can see
  whether a capture was redacted. Do NOT do this in CI for any
  capture that may be committed.
* Backward compat: ``runs_cli.load_run`` still reads schema-3 files;
  the only header/trailer field changes are additive.

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
import contextlib
import datetime
import hashlib
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

# KC16-4 — redaction tunables. The threshold balances "preserve enough
# context to be useful for debugging" against "don't commit the entire
# document to git". 2KB ≈ 500 words ≈ "the LLM call shape was X" without
# also exporting the document body. The hash entropy (16 hex chars =
# 64 bits) is enough to dedup repeated captures of the same document
# but not enough to reconstruct it.
_REDACTION_DEFAULT_THRESHOLD_CHARS = 2048
_REDACTION_PREFIX_CHARS = 200
_REDACTION_HASH_HEX_LEN = 16
_FULL_TEXT_ENV_VAR = "KAOS_RECORDER_FULL_TEXT"
_THRESHOLD_ENV_VAR = "KAOS_RECORDER_REDACTION_THRESHOLD"


def _redaction_config() -> tuple[bool, int]:
    """Read redaction config from env. Returns (enabled, threshold_chars).

    ``KAOS_RECORDER_FULL_TEXT=1`` disables redaction entirely.
    ``KAOS_RECORDER_REDACTION_THRESHOLD`` overrides the threshold.
    Invalid threshold values fall back to the default rather than
    erroring (telemetry must not break a test).
    """
    full_text = os.environ.get(_FULL_TEXT_ENV_VAR, "").strip().lower()
    enabled = full_text not in {"1", "true", "yes", "on"}
    raw = os.environ.get(_THRESHOLD_ENV_VAR, "").strip()
    threshold = _REDACTION_DEFAULT_THRESHOLD_CHARS
    if raw:
        try:
            parsed = int(raw)
            if parsed > 0:
                threshold = parsed
        except ValueError:
            pass
    return enabled, threshold


def _truncate_string(value: str, threshold: int, counter: list[int]) -> str:
    """Redact a single string. Mutates ``counter[0]`` in place.

    Format: ``<first 200 chars> ... [TRUNCATED N chars; sha256=<16 hex>]``.
    The sha256 covers the *full* original string (UTF-8 bytes) so two
    captures of the same document collapse to the same hash. The
    counter is a one-element list so callers can read it after a
    recursive walk without threading return values everywhere.
    """
    if len(value) <= threshold:
        return value
    full_len = len(value)
    digest = hashlib.sha256(value.encode("utf-8", errors="replace")).hexdigest()
    prefix = value[:_REDACTION_PREFIX_CHARS]
    counter[0] += 1
    return f"{prefix} ... [TRUNCATED {full_len} chars; sha256={digest[:_REDACTION_HASH_HEX_LEN]}]"


def _redact(value: Any, threshold: int, counter: list[int]) -> Any:
    """Walk a JSON-ready value and redact long strings in place.

    Only operates on the structure ``_safe_dump`` actually produces:
    ``str``, ``int``, ``float``, ``bool``, ``None``, ``list``, ``dict``.
    Anything else is returned unchanged (it has already been coerced
    to one of those by ``_safe_dump``).
    """
    if isinstance(value, str):
        return _truncate_string(value, threshold, counter)
    if isinstance(value, list):
        return [_redact(item, threshold, counter) for item in value]
    if isinstance(value, dict):
        return {k: _redact(v, threshold, counter) for k, v in value.items()}
    return value


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


def serialize_invocation(
    invocation: Any,
    *,
    call_seq: int,
    redaction_enabled: bool = True,
    redaction_threshold_chars: int = _REDACTION_DEFAULT_THRESHOLD_CHARS,
    redacted_counter: list[int] | None = None,
) -> dict[str, Any]:
    """Convert an :class:`Invocation` to a JSON-ready dict.

    Pulls the full ExecutionTrace (inputs / outputs / model / tokens /
    cost / latency / retries / error). Errors during serialization
    are swallowed and recorded as ``serialize_error`` so the recorder
    never breaks a test.

    KC16-4: when ``redaction_enabled`` is True, every string value
    longer than ``redaction_threshold_chars`` is replaced with a
    truncation marker. The ``redacted_counter`` (a one-element list)
    is incremented in place for each truncated value so the trailer
    can report the total. Metadata fields (model, invocation_id,
    error class, token counts) are not subject to redaction — they
    are short by construction.
    """
    if redacted_counter is None:
        redacted_counter = [0]
    try:
        trace = getattr(invocation, "trace", None)
        usage = getattr(invocation, "usage", None)
        # Metadata fields stay as-is — they're short and structural,
        # not document content.
        record: dict[str, Any] = {
            "kind": "invocation",
            "call_seq": call_seq,
            "invocation_id": getattr(invocation, "id", None),
            "model": getattr(invocation, "model", None),
            "error": (
                repr(invocation.error) if getattr(invocation, "error", None) is not None else None
            ),
        }
        # Content-bearing fields go through _safe_dump first (to coerce
        # to JSON-ready primitives) then through _redact (to truncate
        # long strings). When redaction is off, _redact is a no-op
        # at the string layer.
        output_dump = _safe_dump(getattr(invocation, "output", None))
        trace_dump = _safe_dump(trace) if trace is not None else None
        usage_dump = _safe_dump(usage) if usage is not None else None
        if redaction_enabled:
            output_dump = _redact(output_dump, redaction_threshold_chars, redacted_counter)
            trace_dump = _redact(trace_dump, redaction_threshold_chars, redacted_counter)
            # usage is small typed numbers — skip the walk.
        record["output"] = output_dump
        record["trace"] = trace_dump
        record["usage"] = usage_dump
        return record
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
        out_path: Resolved JSONL output path. Created at
            ``__aenter__`` time so per-invocation streaming has a
            target.
        file_handle: Open text-mode file handle for incremental
            writes. ``None`` once the trailer has been written and
            the handle closed.
        write_lock: ``asyncio.Lock`` serializing writes from
            concurrent Calls running inside the same test. The
            patched ``_execute`` may be invoked from many awaitables;
            without this, two writes can interleave and produce a
            mangled JSONL line.
        call_seq_counter: Monotonic call sequence assigned at write
            time. Pre-streaming the recorder assigned seq at exit;
            now we hand each Invocation a seq the moment it
            completes so the on-disk order matches reality.
    """

    __slots__ = (
        "call_seq_counter",
        "error_override",
        "file_handle",
        "invocations",
        "lock_count",
        "out_path",
        "outcome_override",
        "redacted_counter",
        "redaction_enabled",
        "redaction_threshold_chars",
        "subprocess_dir",
        "subprocess_records",
        "write_lock",
    )

    def __init__(self) -> None:
        self.invocations: list[Any] = []
        self.lock_count: int = 0
        self.outcome_override: str | None = None
        self.error_override: str | None = None
        self.subprocess_dir: Path | None = None
        self.subprocess_records: list[dict[str, Any]] = []
        self.out_path: Path | None = None
        self.file_handle: Any = None
        self.write_lock: asyncio.Lock = asyncio.Lock()
        self.call_seq_counter: int = 0
        # KC16-4: redaction state — populated at __aenter__ from env vars.
        # One-element list so the counter can be mutated from inside
        # ``_redact`` without threading return values.
        enabled, threshold = _redaction_config()
        self.redaction_enabled: bool = enabled
        self.redaction_threshold_chars: int = threshold
        self.redacted_counter: list[int] = [0]


def _write_jsonl_line_crashsafe(fh: Any, record: dict[str, Any]) -> None:
    """Append one JSONL line and flush both Python + OS buffers.

    Schema-v3 streaming contract — every per-invocation line is
    written + ``flush()``ed + ``os.fsync()``ed before control
    returns. This bounds the audit-trail loss window to "the line
    currently being written" rather than "every line since
    process start".

    On Linux a successful ``fsync()`` is the strongest portable
    durability guarantee Python gives us, but it is **not** a hard
    real-time guarantee:

    * ``TextIOWrapper`` keeps a write buffer above the OS layer;
      we must call ``fh.flush()`` first so the Python-level buffer
      drains into the underlying ``BufferedWriter`` / file descriptor.
    * ``fileno()`` is what ``os.fsync`` wants — not the Python file
      object. Passing the wrong thing silently is a common gotcha.
    * Some filesystems (tmpfs, certain network mounts) treat fsync
      as a no-op or only sync data without the directory entry; for
      the recorder's purposes this is acceptable because the
      directory is created + written to up front, and the same
      caveat applies to every other durability strategy in the
      KAOS stack.

    Silent on failure by intent: telemetry must never break a
    test. If we can't write to the JSONL, the test outcome is
    still authoritative.
    """
    try:
        fh.write(json.dumps(record, separators=(",", ":")) + "\n")
        fh.flush()
        # OSError: fsync not supported on this fd (e.g. some tmpfs).
        # ValueError: fileno() is closed (race during shutdown).
        # In both cases we've already flushed the Python buffer,
        # which is the best we can do.
        with contextlib.suppress(OSError, ValueError):
            os.fsync(fh.fileno())
    except Exception:
        # Recorder must never break a test. Drop the write silently.
        pass


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

    # ------------------------------------------------------------------
    # __aenter__ work: open the JSONL, write the header, start streaming.
    # ------------------------------------------------------------------
    start_ts = datetime.datetime.now(datetime.UTC).isoformat()
    start_perf = time.perf_counter()
    repo_root = _find_repo_root(out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)
    fname = _sanitize_nodeid(test_nodeid) + ".jsonl"
    out_path = out_dir / fname
    capture.out_path = out_path

    streaming_header = {
        "kind": "header",
        "test_nodeid": test_nodeid,
        "start_ts_utc": start_ts,
        "git": git_context(repo_root) if repo_root else {},
        "python_version": _python_version(),
        # Schema 4 (KC16-4) adds transparency-lens redaction fields.
        # Backward compatible: schema-3 readers ignore unknown header
        # keys, and ``runs_cli.load_run`` was already permissive.
        "schema_version": 4,
        "streaming": True,
        # Audit-trail discipline: declare the durability contract so a
        # reader can branch on it without having to know the recorder
        # version. See module docstring for the full contract.
        "trailer_optional": True,
        "partial_last_line_tolerated": True,
        # KC16-4: redaction policy advertised in the header so a
        # downstream consumer can branch on whether a capture is
        # truncated (CI default) or full-text (local dev opt-out).
        "redaction_enabled": capture.redaction_enabled,
        "redaction_threshold_chars": capture.redaction_threshold_chars,
        **(extra_metadata or {}),
    }

    try:
        capture.file_handle = out_path.open("w", encoding="utf-8")
        _write_jsonl_line_crashsafe(capture.file_handle, streaming_header)
    except Exception:
        # If we can't even open the file, fall back to a None handle;
        # the rest of the recorder will silently no-op writes. Tests
        # remain authoritative.
        capture.file_handle = None

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
                # Stream the partial so the audit trail survives the
                # outer test failure / SIGTERM / OOM-kill window.
                await _stream_invocation(capture, partial)
            raise
        capture.invocations.append(invocation)
        await _stream_invocation(capture, invocation)
        return invocation

    # Patch only once even with nested record_live_test calls.
    if capture.lock_count == 0:
        Call._execute = patched_execute  # ty: ignore[invalid-assignment]
    capture.lock_count += 1

    # KC6: set up a per-test subprocess capture dir + Popen patch so
    # LLM calls inside spawned children land in our JSONL stitch.
    capture.subprocess_dir = Path(tempfile.mkdtemp(prefix="kaos-recorder-sub-"))
    _subprocess_patches = _install_subprocess_env_patches(capture.subprocess_dir)

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
        # capture. Done before the trailer write so the trailer
        # carries both in-process and subprocess totals.
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
        # trailer reflects total spend across both surfaces.
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

        total_subprocess_calls = len(capture.subprocess_records)
        total_calls = len(capture.invocations) + total_subprocess_calls

        # Schema-v4 trailer: the post-hoc "what actually happened"
        # line. Optional by contract — readers MUST tolerate runs
        # without it (process killed before exit). KC16-4 adds the
        # ``redacted_count`` total so an audit reader can verify
        # the redaction pass actually fired.
        trailer = {
            "kind": "trailer",
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
            "redacted_count": capture.redacted_counter[0],
            "schema_version": 4,
        }

        fh = capture.file_handle
        if fh is not None:
            # Append the stitched subprocess records as their own
            # streamed lines BEFORE the trailer — keeps the per-line
            # JSONL discipline (one record per line, trailer last).
            for sub_record in capture.subprocess_records:
                capture.call_seq_counter += 1
                stitched = {
                    **sub_record,
                    "call_seq": capture.call_seq_counter,
                    "source": "subprocess",
                }
                _write_jsonl_line_crashsafe(fh, stitched)

            _write_jsonl_line_crashsafe(fh, trailer)

            with contextlib.suppress(Exception):
                fh.close()
            capture.file_handle = None

        # Append to the rolling index (one line per recorded run).
        try:
            index_path = out_dir.parent / "INDEX.jsonl"
            git_info = streaming_header.get("git", {}) or {}
            git_short_sha = git_info.get("short_sha", "") if isinstance(git_info, dict) else ""
            with index_path.open("a", encoding="utf-8") as ifh:
                ifh.write(
                    json.dumps(
                        {
                            "test_nodeid": test_nodeid,
                            "outcome": outcome,
                            "elapsed_s": round(elapsed_s, 4),
                            "total_cost_usd": round(total_cost, 6),
                            "call_count": total_calls,
                            "end_ts_utc": end_ts,
                            "file": str(out_path.relative_to(out_dir.parent.parent)),
                            "git_short_sha": git_short_sha,
                        },
                        separators=(",", ":"),
                    )
                    + "\n"
                )
                ifh.flush()
                with contextlib.suppress(OSError, ValueError):
                    os.fsync(ifh.fileno())
        except Exception:
            pass


async def _stream_invocation(capture: _Capture, invocation: Any) -> None:
    """Serialize + write one Invocation under the capture's write lock.

    Called from the patched ``Call._execute`` for every completed
    Invocation (successful or partial). Holds the write lock so
    concurrent Calls running inside the same test don't interleave
    JSONL lines. Silent on failure — telemetry must never break
    the test.
    """
    fh = capture.file_handle
    if fh is None:
        return
    async with capture.write_lock:
        capture.call_seq_counter += 1
        try:
            record = serialize_invocation(
                invocation,
                call_seq=capture.call_seq_counter,
                redaction_enabled=capture.redaction_enabled,
                redaction_threshold_chars=capture.redaction_threshold_chars,
                redacted_counter=capture.redacted_counter,
            )
        except Exception as exc:
            record = {
                "kind": "invocation",
                "call_seq": capture.call_seq_counter,
                "serialize_error": repr(exc),
            }
        _write_jsonl_line_crashsafe(fh, record)


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

    Returns the parsed invocation records as a flat list, sorted by
    PID + line order so deterministic across runs. Missing dir or
    unreadable files contribute zero records.

    Filters to ``kind == "invocation"`` so the schema_version=3 header
    lines (written by ``env_recorder.install_from_env``) are not
    stitched as if they were invocations. A truncated final line is
    swallowed, matching the schema-v3 crash-safety contract.
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
                        obj = json.loads(stripped)
                    except json.JSONDecodeError:
                        # Corrupt / truncated line — skip but keep the rest.
                        continue
                    # Schema-v3: skip the header line; only invocations
                    # are stitched into the parent JSONL.
                    if obj.get("kind") == "invocation":
                        records.append(obj)
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
