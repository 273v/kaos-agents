"""Durability tests for the kaos-agents live-test recorder (sprint-3 #8).

Sprint-3 #8 (transparency lens / audit-trail survives non-clean
exits): the recorder used to write ALL JSONL at ``__aexit__``. Under
SIGTERM / pod-eviction / OOM-kill mid-test, **zero JSONL was
produced**.

These tests pin the new streaming contract:

1. The header line is written + ``fsync``ed at ``__aenter__`` —
   before any LLM call has run.
2. Each completed Invocation is written + ``fsync``ed before
   control returns to the test body.
3. A real ``multiprocessing.Process`` SIGTERM'd between calls
   leaves a JSONL with header + N invocation lines (where N is the
   number of completed calls). The trailer is optional and may be
   missing — readers MUST tolerate that.
4. The line content is well-formed JSON (every recorded line, not
   just the first).

We use ``multiprocessing.Process`` (not threading) because the
defining characteristic of the regression hazard is **real SIGTERM**.
Threads can't be SIGTERM'd individually.
"""

from __future__ import annotations

import asyncio
import json
import multiprocessing
import os
import signal
import sys
import time
from pathlib import Path
from typing import Any

import pytest

# The subprocess workers do their own kaos_llm_core import inside
# ``_run_streaming_child`` to make sure the test module is importable
# even when kaos_llm_core isn't installed in the test environment.


# ---------------------------------------------------------------------------
# Subprocess workers (top-level so multiprocessing can pickle them).
# ---------------------------------------------------------------------------


def _run_streaming_child(
    out_dir_str: str,
    sentinel_after_call_path: str,
    nodeid: str,
    *,
    sleep_after_call_seconds: float = 30.0,
) -> None:
    """Subprocess target: open ``record_live_test``, run ONE patched call, idle.

    Steps:
    1. Open ``record_live_test()`` against ``out_dir_str``.
    2. Patch ``Call._execute`` to return a fake Invocation with a
       deterministic id + model + usage.
    3. Run the patched call once.
    4. Touch ``sentinel_after_call_path`` so the parent knows the
       JSONL line has been written + fsync'd.
    5. Sleep ``sleep_after_call_seconds`` so the parent can SIGTERM us.

    On SIGTERM, the child dies without entering ``__aexit__``. The
    streaming contract guarantees the header + the one invocation
    line are still on disk.
    """
    # Make sure the child resolves the in-tree tests package the same
    # way pytest does.
    pkg_root = Path(__file__).resolve().parents[2]
    if str(pkg_root) not in sys.path:
        sys.path.insert(0, str(pkg_root))

    from kaos_llm_core.programs._invocation import Invocation, TokenUsage
    from kaos_llm_core.programs.call import Call

    from tests.integration._recorder import record_live_test

    async def fake_execute(self_call: Any, inputs: dict[str, Any]) -> Any:
        # Return a deterministic Invocation. The recorder's
        # serialize_invocation handles the rest.
        return Invocation(
            id="inv-durability-test",
            client=None,
            model="function-durability-test",
            context=None,
            extras={},
            output={"answer": "streamed"},
            trace=None,
            usage=TokenUsage(input_tokens=3, output_tokens=2, total_tokens=5, cost_usd=0.0),
            error=None,
        )

    # Install the fake BEFORE record_live_test so the recorder
    # captures our fake as its ``original_execute``. The recorder
    # wraps whatever ``Call._execute`` is at __aenter__ time.
    Call._execute = fake_execute

    async def body() -> None:
        async with record_live_test(nodeid, out_dir=Path(out_dir_str)):
            # Run one call through the recorder's wrapper, which
            # in turn calls our captured fake_execute.
            inv = await Call._execute(None, {})  # ty: ignore[invalid-argument-type]
            assert inv is not None
            # Tell the parent the line has been written + fsync'd.
            Path(sentinel_after_call_path).write_text("ok", encoding="utf-8")
            # Idle long enough for the parent to SIGTERM us. If we
            # ever escape this sleep without SIGTERM, the test will
            # still observe the streaming behavior (header + inv
            # line written before the sleep), but the trailer will
            # also land — assertions allow both.
            await asyncio.sleep(sleep_after_call_seconds)

    asyncio.run(body())


def _run_zero_calls_child(out_dir_str: str, sentinel_path: str, nodeid: str) -> None:
    """Subprocess target: open the recorder, touch a sentinel, idle.

    Used to verify that the streaming header lands on disk even
    before any LLM call runs.
    """
    pkg_root = Path(__file__).resolve().parents[2]
    if str(pkg_root) not in sys.path:
        sys.path.insert(0, str(pkg_root))

    from tests.integration._recorder import record_live_test

    async def body() -> None:
        async with record_live_test(nodeid, out_dir=Path(out_dir_str)):
            Path(sentinel_path).write_text("ok", encoding="utf-8")
            await asyncio.sleep(30.0)

    asyncio.run(body())


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _wait_for_path(path: Path, timeout_s: float = 15.0) -> bool:
    """Poll for the existence of ``path``. Returns True if it appears."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if path.exists():
            return True
        time.sleep(0.05)
    return False


def _spawn_with_method(method: str, target: Any, args: tuple[Any, ...]) -> multiprocessing.Process:
    """Spawn a process under the requested start method."""
    ctx = multiprocessing.get_context(method)
    # ``BaseContext.Process`` is the canonical multiprocessing entry
    # point; ty's stub doesn't model it via __getattr__, so widen the
    # access manually.
    proc: multiprocessing.Process = ctx.Process(target=target, args=args)  # ty: ignore[unresolved-attribute]
    proc.start()
    return proc


def _require_pid(proc: multiprocessing.Process) -> int:
    """``Process.pid`` is ``int | None`` until ``start()`` returns.

    All callers here have already ``start()``ed; this helper makes
    that invariant explicit for the type checker and fails loud if
    the assumption is ever violated.
    """
    pid = proc.pid
    assert pid is not None, "process has not been started yet"
    return pid


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestStreamingHeaderOnEnter:
    """The header line must hit disk at ``__aenter__``, not at exit."""

    def test_header_present_before_any_call(self, tmp_path: Path) -> None:
        """Even with zero calls, SIGTERM after enter leaves a readable header."""
        out_dir = tmp_path / "runs"
        out_dir.mkdir()
        sentinel = tmp_path / "after-enter.sentinel"

        proc = _spawn_with_method(
            "spawn",
            _run_zero_calls_child,
            (str(out_dir), str(sentinel), "test_zero_calls"),
        )
        try:
            appeared = _wait_for_path(sentinel)
            assert appeared, "child failed to reach the recorder body"

            # The header should already be on disk at this point.
            jsonl_files = list(out_dir.glob("test_zero_calls*.jsonl"))
            assert len(jsonl_files) == 1, f"expected 1 jsonl, got {jsonl_files}"
            content = jsonl_files[0].read_text(encoding="utf-8")
            lines = [ln for ln in content.split("\n") if ln.strip()]
            assert len(lines) == 1, f"expected header-only, got {len(lines)} lines"
            header = json.loads(lines[0])
            assert header["kind"] == "header"
            assert header["streaming"] is True
            # Schema bumped to 4 in KC16-4 alongside redaction.
            assert header["schema_version"] == 4
            assert header["test_nodeid"] == "test_zero_calls"
            assert header["trailer_optional"] is True
            assert header["partial_last_line_tolerated"] is True
        finally:
            # SIGTERM the idling child. If the child has already
            # exited (e.g. the sleep finished, which would itself
            # be a test failure), kill is a no-op.
            if proc.is_alive():
                os.kill(_require_pid(proc), signal.SIGTERM)
            proc.join(timeout=10.0)
            if proc.is_alive():
                proc.kill()
                proc.join(timeout=5.0)


class TestStreamingInvocationOnCompletion:
    """A completed Invocation must hit disk before the next line of test body runs."""

    def test_invocation_persists_across_sigterm(self, tmp_path: Path) -> None:
        """Spawn child → wait for invocation written → SIGTERM → assert JSONL.

        The acceptance gate for sprint-3 #8: a real SIGTERM after
        the first Call completes leaves a JSONL with the header +
        the one invocation line on disk. Before this change the
        JSONL was empty because everything was deferred to
        ``__aexit__``.
        """
        out_dir = tmp_path / "runs"
        out_dir.mkdir()
        sentinel = tmp_path / "after-call.sentinel"

        proc = _spawn_with_method(
            "spawn",
            _run_streaming_child,
            (str(out_dir), str(sentinel), "test_invocation_sigterm"),
        )
        try:
            appeared = _wait_for_path(sentinel)
            assert appeared, "child failed to write the post-call sentinel"

            # Give the child's fsync() a brief moment to settle on
            # tmpfs — sentinel goes through the same write path so
            # the JSONL should already be durable, but a 50ms cushion
            # absorbs FS-level scheduling jitter.
            time.sleep(0.05)

            os.kill(_require_pid(proc), signal.SIGTERM)
            proc.join(timeout=10.0)
            if proc.is_alive():
                proc.kill()
                proc.join(timeout=5.0)
                pytest.fail("child did not exit on SIGTERM")

            jsonl_files = list(out_dir.glob("test_invocation_sigterm*.jsonl"))
            assert len(jsonl_files) == 1, f"expected 1 jsonl, got {jsonl_files}"
            raw = jsonl_files[0].read_text(encoding="utf-8")
            lines = [ln for ln in raw.split("\n") if ln.strip()]

            # Acceptance: header + at least 1 invocation line. The
            # trailer SHOULD be missing because SIGTERM hit during
            # the sleep, not during __aexit__.
            assert len(lines) >= 2, f"expected header + 1+ invocation, got {len(lines)} lines"

            header = json.loads(lines[0])
            assert header["kind"] == "header"
            assert header["streaming"] is True
            # Schema bumped to 4 in KC16-4 alongside redaction.
            assert header["schema_version"] == 4
            assert header["test_nodeid"] == "test_invocation_sigterm"

            # All subsequent lines should be well-formed JSON.
            for i, line in enumerate(lines[1:], start=1):
                parsed = json.loads(line)
                assert "kind" in parsed, f"line {i} missing kind: {line!r}"

            # At least one invocation should have made it.
            invocations = [
                json.loads(ln) for ln in lines[1:] if json.loads(ln).get("kind") == "invocation"
            ]
            assert len(invocations) >= 1, "expected >=1 invocation line, got 0"
            inv = invocations[0]
            assert inv["model"] == "function-durability-test"
            assert inv["invocation_id"] == "inv-durability-test"

            # The trailer is OPTIONAL by the schema-v3 contract. We
            # don't assert its absence — pytest/CI scheduling could
            # in theory let the child reach __aexit__ before SIGTERM
            # is delivered — but we DO assert that a missing trailer
            # is fine.
            trailer_lines = [ln for ln in lines if json.loads(ln).get("kind") == "trailer"]
            # No assertion on len(trailer_lines): the streaming
            # contract permits either 0 or 1.
            assert len(trailer_lines) <= 1
        finally:
            if proc.is_alive():
                os.kill(_require_pid(proc), signal.SIGKILL)
                proc.join(timeout=5.0)


class TestPartialLineTolerance:
    """The runs_cli loader must handle truncated final lines."""

    def test_runs_cli_loads_partial_jsonl(self, tmp_path: Path) -> None:
        """A JSONL with header + 1 full invocation + 1 truncated line still loads."""
        from tests.integration.runs_cli import load_run

        jsonl_path = tmp_path / "partial.jsonl"
        header = {
            "kind": "header",
            "test_nodeid": "x::y",
            "schema_version": 3,
            "streaming": True,
            "start_ts_utc": "2026-05-11T00:00:00+00:00",
            "git": {"short_sha": "abc1234"},
        }
        inv = {"kind": "invocation", "call_seq": 1, "model": "m1"}
        # Truncated mid-write: missing closing brace + newline.
        truncated_tail = '{"kind":"invocation","call_seq":2,"model":"m2"'
        jsonl_path.write_text(
            json.dumps(header) + "\n" + json.dumps(inv) + "\n" + truncated_tail,
            encoding="utf-8",
        )

        merged_header, calls = load_run(jsonl_path)

        # Header is present and parseable.
        assert merged_header["test_nodeid"] == "x::y"
        assert merged_header["schema_version"] == 3
        # The one well-formed invocation survived; the truncated
        # tail was silently dropped per schema-v3 tolerance.
        assert len(calls) == 1
        assert calls[0]["model"] == "m1"

    def test_runs_cli_loads_no_trailer(self, tmp_path: Path) -> None:
        """A JSONL with header + N invocations but NO trailer still loads."""
        from tests.integration.runs_cli import load_run

        jsonl_path = tmp_path / "no_trailer.jsonl"
        header = {"kind": "header", "test_nodeid": "x", "schema_version": 3}
        invs = [{"kind": "invocation", "call_seq": i, "model": f"m{i}"} for i in (1, 2, 3)]
        jsonl_path.write_text(
            "\n".join([json.dumps(header)] + [json.dumps(i) for i in invs]) + "\n",
            encoding="utf-8",
        )

        merged_header, calls = load_run(jsonl_path)
        # No trailer → outcome is absent. Consumers branch on this.
        assert merged_header.get("outcome") is None
        assert len(calls) == 3
        assert [c["model"] for c in calls] == ["m1", "m2", "m3"]

    def test_runs_cli_overlays_trailer_onto_header(self, tmp_path: Path) -> None:
        """When a trailer IS present, its fields overlay the header."""
        from tests.integration.runs_cli import load_run

        jsonl_path = tmp_path / "full.jsonl"
        header = {
            "kind": "header",
            "test_nodeid": "x",
            "schema_version": 3,
            "git": {"short_sha": "abc1234"},
        }
        inv = {"kind": "invocation", "call_seq": 1, "model": "m1"}
        trailer = {
            "kind": "trailer",
            "outcome": "passed",
            "total_cost_usd": 0.0123,
            "call_count": 1,
            "elapsed_s": 1.5,
        }
        jsonl_path.write_text(
            "\n".join([json.dumps(header), json.dumps(inv), json.dumps(trailer)]) + "\n",
            encoding="utf-8",
        )

        merged_header, calls = load_run(jsonl_path)
        assert merged_header["outcome"] == "passed"
        assert merged_header["total_cost_usd"] == pytest.approx(0.0123)
        assert merged_header["call_count"] == 1
        # The header fields survive too.
        assert merged_header["git"]["short_sha"] == "abc1234"
        assert len(calls) == 1
