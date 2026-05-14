"""Redaction tests for the kaos-agents live-test recorder (KC16-4).

KC16-4 (HIGH · transparency lens) — Probe 4 + Probe 5 of
``docs/design/kc16-audit-findings.md`` flagged that pre-KC16-4 the
recorder persisted full document bodies (50KB+ ``message`` fields)
verbatim into JSONL and those JSONLs landed in git as a secondary
data plane. KC16-4 redacts every string > 2048 chars to
``<first 200 chars> ... [TRUNCATED N chars; sha256=<16 hex>]``.

These tests pin the redaction contract:

1. A captured Invocation with a 50,000-char input field is truncated
   in the on-disk JSONL: prefix preserved, hash present, full length
   preserved in the marker, original bytes gone.
2. ``KAOS_RECORDER_FULL_TEXT=1`` disables redaction — the full
   string is preserved.
3. Header advertises ``schema_version=4`` + ``redaction_enabled`` +
   ``redaction_threshold_chars``; trailer carries ``redacted_count``.
4. ``runs_cli.load_run`` reads the new schema-4 file cleanly, AND
   continues to read pre-KC16-4 schema-3 files for backward compat.
5. Metadata fields (model, invocation_id, error class, token counts)
   are never redacted — they're short by construction and structural.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from tests.integration._recorder import (
    _REDACTION_DEFAULT_THRESHOLD_CHARS,
    _REDACTION_HASH_HEX_LEN,
    _REDACTION_PREFIX_CHARS,
    record_live_test,
    serialize_invocation,
)

# ---------------------------------------------------------------------------
# Test fixtures — synthesize a fake Invocation that looks like what the
# real kaos_llm_core.programs._invocation.Invocation produces, including
# the model_dump() shape for the trace.
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class _FakeTokenUsage:
    """Mirrors kaos_llm_core.programs._invocation.TokenUsage shape."""

    input_tokens: int = 100
    output_tokens: int = 50
    total_tokens: int = 150
    cost_usd: float = 0.001


class _FakeTrace:
    """Minimal model_dump-able stand-in for ExecutionTrace.

    The real trace is a Pydantic model with ``model_dump(mode="json")``;
    we mimic it so the redaction walk hits inputs/outputs the same way
    it does in production.
    """

    def __init__(self, message: str, output: str) -> None:
        self._message = message
        self._output = output

    def model_dump(self, mode: str = "python") -> dict[str, Any]:
        return {
            "trace_id": "trace-abc",
            "call_name": "_FakeSig",
            "signature": "_FakeSig",
            "inputs": {"message": self._message, "system": "short"},
            "outputs": {"answer": self._output},
            "model": "anthropic:claude-haiku-4-5",
            "codec": "JSONCodec",
            "input_tokens": 100,
            "output_tokens": 50,
            "total_tokens": 150,
            "cost_usd": 0.001,
            "latency_ms": 250.0,
            "retries": 0,
            "examples_used": 0,
            "children": [],
            "error": None,
            "timestamp": "2026-05-11T00:00:00Z",
        }


class _FakeInvocation:
    """Minimal stand-in for Invocation."""

    def __init__(self, message: str, output: str) -> None:
        self.id = "inv-test-12345"
        self.model = "anthropic:claude-haiku-4-5"
        self.output = {"answer": output}
        self.error = None
        self.trace = _FakeTrace(message, output)
        self.usage = _FakeTokenUsage()


# ---------------------------------------------------------------------------
# Direct serialize_invocation tests — no file I/O, no asyncio.
# ---------------------------------------------------------------------------


class TestSerializeInvocationRedaction:
    """Direct unit tests on serialize_invocation()."""

    def test_long_input_string_is_truncated(self) -> None:
        """A 50,000-char input message becomes a truncation marker."""
        big = "x" * 50_000
        small = "ok"
        inv = _FakeInvocation(message=big, output=small)
        counter: list[int] = [0]

        record = serialize_invocation(
            inv,
            call_seq=1,
            redaction_enabled=True,
            redaction_threshold_chars=2048,
            redacted_counter=counter,
        )

        # The trace inputs.message survives as a redacted string.
        message_field = record["trace"]["inputs"]["message"]
        assert isinstance(message_field, str), (
            f"expected redacted string, got {type(message_field).__name__}"
        )
        assert message_field.startswith("x" * _REDACTION_PREFIX_CHARS), (
            "first 200 chars should be preserved"
        )
        assert "[TRUNCATED 50000 chars; sha256=" in message_field, (
            f"truncation marker missing: {message_field[:300]!r}"
        )
        # sha256 prefix has the right length + hex shape.
        sha_part = message_field.rsplit("sha256=", 1)[1].rstrip("]")
        assert len(sha_part) == _REDACTION_HASH_HEX_LEN
        assert all(c in "0123456789abcdef" for c in sha_part)
        # And the hash matches the input.
        expected_hash = hashlib.sha256(big.encode("utf-8")).hexdigest()[:_REDACTION_HASH_HEX_LEN]
        assert sha_part == expected_hash

        # Counter incremented.
        assert counter[0] >= 1

    def test_short_string_is_not_truncated(self) -> None:
        """Strings at or below the threshold pass through verbatim."""
        msg = "x" * 100  # well under 2048
        inv = _FakeInvocation(message=msg, output="ok")
        counter: list[int] = [0]

        record = serialize_invocation(
            inv,
            call_seq=1,
            redaction_enabled=True,
            redaction_threshold_chars=2048,
            redacted_counter=counter,
        )

        assert record["trace"]["inputs"]["message"] == msg
        assert "TRUNCATED" not in record["trace"]["inputs"]["message"]

    def test_metadata_fields_never_redacted(self) -> None:
        """model / invocation_id / error / usage stay untouched."""
        inv = _FakeInvocation(message="x" * 50_000, output="x" * 50_000)
        counter: list[int] = [0]

        record = serialize_invocation(
            inv,
            call_seq=1,
            redaction_enabled=True,
            redaction_threshold_chars=2048,
            redacted_counter=counter,
        )

        assert record["invocation_id"] == "inv-test-12345"
        assert record["model"] == "anthropic:claude-haiku-4-5"
        assert record["error"] is None
        # usage stays a small numeric dict — no string body to redact.
        assert record["usage"]["total_tokens"] == 150
        assert record["usage"]["cost_usd"] == 0.001

    def test_disabled_redaction_preserves_full_text(self) -> None:
        """With redaction_enabled=False, the full string is preserved."""
        big = "x" * 50_000
        inv = _FakeInvocation(message=big, output="ok")
        counter: list[int] = [0]

        record = serialize_invocation(
            inv,
            call_seq=1,
            redaction_enabled=False,
            redaction_threshold_chars=2048,
            redacted_counter=counter,
        )

        assert record["trace"]["inputs"]["message"] == big
        assert counter[0] == 0

    def test_default_threshold_is_2048(self) -> None:
        """A 2048-char string is preserved; 2049 is truncated.

        Boundary test for the off-by-one we'd otherwise debug later.
        """
        at_boundary = "x" * _REDACTION_DEFAULT_THRESHOLD_CHARS
        over_boundary = "x" * (_REDACTION_DEFAULT_THRESHOLD_CHARS + 1)

        inv_at = _FakeInvocation(message=at_boundary, output="ok")
        inv_over = _FakeInvocation(message=over_boundary, output="ok")
        counter: list[int] = [0]

        record_at = serialize_invocation(
            inv_at,
            call_seq=1,
            redaction_enabled=True,
            redaction_threshold_chars=_REDACTION_DEFAULT_THRESHOLD_CHARS,
            redacted_counter=counter,
        )
        record_over = serialize_invocation(
            inv_over,
            call_seq=2,
            redaction_enabled=True,
            redaction_threshold_chars=_REDACTION_DEFAULT_THRESHOLD_CHARS,
            redacted_counter=counter,
        )

        assert record_at["trace"]["inputs"]["message"] == at_boundary
        assert "TRUNCATED" in record_over["trace"]["inputs"]["message"]

    def test_counter_increments_per_field_redacted(self) -> None:
        """Counter goes up once per truncated string in the record."""
        big = "x" * 50_000
        # Both message AND output exceed the threshold → expect 2 redactions
        # (output dict's "answer" + trace inputs.message + trace outputs.answer
        # → so 3 total in this fake).
        inv = _FakeInvocation(message=big, output=big)
        counter: list[int] = [0]

        serialize_invocation(
            inv,
            call_seq=1,
            redaction_enabled=True,
            redaction_threshold_chars=2048,
            redacted_counter=counter,
        )

        # output.answer, trace.inputs.message, trace.outputs.answer
        assert counter[0] == 3


# ---------------------------------------------------------------------------
# End-to-end test through record_live_test() — file is written, schema-4
# header lands, trailer carries redacted_count.
# ---------------------------------------------------------------------------


def _patch_call_execute_to_yield(invocation: Any) -> Any:
    """Patch Call._execute to return one canned Invocation."""
    from kaos_llm_core.programs.call import Call

    original = Call._execute

    async def fake_execute(self_call: Any, inputs: dict[str, Any]) -> Any:
        return invocation

    Call._execute = fake_execute  # ty: ignore[invalid-assignment]
    return original


def _restore_call_execute(original: Any) -> None:
    from kaos_llm_core.programs.call import Call

    Call._execute = original


class TestRecordLiveTestRedaction:
    """End-to-end test through record_live_test()."""

    @pytest.mark.asyncio
    async def test_schema4_header_and_trailer_with_redaction(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Header advertises schema-4 + redaction; trailer carries count."""
        # Make sure env doesn't disable redaction.
        monkeypatch.delenv("KAOS_RECORDER_FULL_TEXT", raising=False)
        monkeypatch.delenv("KAOS_RECORDER_REDACTION_THRESHOLD", raising=False)

        big = "x" * 50_000
        inv = _FakeInvocation(message=big, output="answer")
        original = _patch_call_execute_to_yield(inv)
        try:
            from kaos_llm_core.programs.call import Call

            async with record_live_test("test_redaction", out_dir=tmp_path):
                await Call._execute(None, {})  # ty: ignore[invalid-argument-type]
        finally:
            _restore_call_execute(original)

        jsonl_files = list(tmp_path.glob("test_redaction*.jsonl"))
        assert len(jsonl_files) == 1, f"expected 1 jsonl, got {jsonl_files}"
        lines = [ln for ln in jsonl_files[0].read_text(encoding="utf-8").split("\n") if ln.strip()]
        assert len(lines) >= 3, f"expected header + inv + trailer, got {len(lines)}"

        header = json.loads(lines[0])
        assert header["kind"] == "header"
        assert header["schema_version"] == 4
        assert header["redaction_enabled"] is True
        assert header["redaction_threshold_chars"] == 2048

        # Find the invocation line (between header and trailer).
        inv_lines = [json.loads(ln) for ln in lines if '"kind":"invocation"' in ln]
        assert len(inv_lines) == 1
        inv_rec = inv_lines[0]
        msg = inv_rec["trace"]["inputs"]["message"]
        assert "TRUNCATED 50000 chars; sha256=" in msg
        assert msg.startswith("x" * _REDACTION_PREFIX_CHARS)

        trailer = json.loads(lines[-1])
        assert trailer["kind"] == "trailer"
        assert trailer["schema_version"] == 4
        # We redacted >= 1 field (trace.inputs.message at minimum).
        assert trailer["redacted_count"] >= 1

        # The 50000-char string should NOT appear anywhere in the
        # serialized JSONL — that's the whole point of the redaction.
        raw = jsonl_files[0].read_text(encoding="utf-8")
        # The marker is ~250 chars; the body would be 50,000. Total
        # file size must be much smaller than 50K + envelope.
        # (Sanity check: file < 10 KB even with header + trailer.)
        assert len(raw) < 10_000, f"redacted file unexpectedly large: {len(raw)} bytes"
        # And the literal 50000-x block does not appear.
        assert "x" * 1000 not in raw, "redaction failed: long run of x's still present"

    @pytest.mark.asyncio
    async def test_full_text_env_var_disables_redaction(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """KAOS_RECORDER_FULL_TEXT=1 disables redaction; full text persists."""
        monkeypatch.setenv("KAOS_RECORDER_FULL_TEXT", "1")
        monkeypatch.delenv("KAOS_RECORDER_REDACTION_THRESHOLD", raising=False)

        big = "x" * 50_000
        inv = _FakeInvocation(message=big, output="answer")
        original = _patch_call_execute_to_yield(inv)
        try:
            from kaos_llm_core.programs.call import Call

            async with record_live_test("test_full_text", out_dir=tmp_path):
                await Call._execute(None, {})  # ty: ignore[invalid-argument-type]
        finally:
            _restore_call_execute(original)

        jsonl_files = list(tmp_path.glob("test_full_text*.jsonl"))
        assert len(jsonl_files) == 1
        raw = jsonl_files[0].read_text(encoding="utf-8")
        lines = [ln for ln in raw.split("\n") if ln.strip()]

        header = json.loads(lines[0])
        assert header["redaction_enabled"] is False
        # The full 50_000-char string survives.
        assert "x" * 1000 in raw, (
            "full-text opt-out should preserve the long string, but it was redacted"
        )
        # Find the invocation line.
        inv_lines = [json.loads(ln) for ln in lines if '"kind":"invocation"' in ln]
        assert len(inv_lines) == 1
        assert inv_lines[0]["trace"]["inputs"]["message"] == big

        trailer = json.loads(lines[-1])
        # With redaction disabled, count is 0.
        assert trailer["redacted_count"] == 0

    @pytest.mark.asyncio
    async def test_custom_threshold_env_var(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """KAOS_RECORDER_REDACTION_THRESHOLD overrides the default."""
        monkeypatch.delenv("KAOS_RECORDER_FULL_TEXT", raising=False)
        monkeypatch.setenv("KAOS_RECORDER_REDACTION_THRESHOLD", "100")

        # 200 chars — well over 100 threshold, well under default 2048.
        msg = "y" * 200
        inv = _FakeInvocation(message=msg, output="ok")
        original = _patch_call_execute_to_yield(inv)
        try:
            from kaos_llm_core.programs.call import Call

            async with record_live_test("test_custom_threshold", out_dir=tmp_path):
                await Call._execute(None, {})  # ty: ignore[invalid-argument-type]
        finally:
            _restore_call_execute(original)

        jsonl_files = list(tmp_path.glob("test_custom_threshold*.jsonl"))
        raw = jsonl_files[0].read_text(encoding="utf-8")
        lines = [ln for ln in raw.split("\n") if ln.strip()]
        header = json.loads(lines[0])
        assert header["redaction_threshold_chars"] == 100
        # The 200-char message should be truncated.
        inv_lines = [json.loads(ln) for ln in lines if '"kind":"invocation"' in ln]
        captured = inv_lines[0]["trace"]["inputs"]["message"]
        assert "TRUNCATED 200 chars; sha256=" in captured


# ---------------------------------------------------------------------------
# PA18 — per-attempt rotation: consecutive runs of the same test must
# produce distinct files, not overwrite each other.
# ---------------------------------------------------------------------------


class TestRecorderPerAttemptRotation:
    """The recorder must NEVER overwrite a previous attempt's capture.

    Originally the filename was ``<sanitized-nodeid>.jsonl`` and the
    file handle was opened in ``"w"`` (truncate) mode — so a flaky
    test's first attempt's capture was silently lost when the next
    attempt opened the same path. PA18 surfaced this when looking for
    the original Sonnet-4-6 Jaccard 0.621 outlier — the captures had
    been overwritten by subsequent re-runs.

    The fix is a UTC start-timestamp suffix on the filename.
    """

    @pytest.mark.asyncio
    async def test_consecutive_attempts_produce_distinct_files(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Two back-to-back ``record_live_test`` blocks for the same
        nodeid must leave two JSONL files on disk, not one."""
        monkeypatch.delenv("KAOS_RECORDER_FULL_TEXT", raising=False)
        inv = _FakeInvocation(message="first", output="ok-1")
        original = _patch_call_execute_to_yield(inv)
        try:
            from kaos_llm_core.programs.call import Call

            async with record_live_test("test_pa18_rotation", out_dir=tmp_path):
                await Call._execute(None, {})  # ty: ignore[invalid-argument-type]
        finally:
            _restore_call_execute(original)

        # Force a 1-second gap so the second attempt's UTC timestamp
        # differs in the seconds field (filename resolution is 1s).
        time.sleep(1.1)

        inv2 = _FakeInvocation(message="second", output="ok-2")
        original = _patch_call_execute_to_yield(inv2)
        try:
            from kaos_llm_core.programs.call import Call

            async with record_live_test("test_pa18_rotation", out_dir=tmp_path):
                await Call._execute(None, {})  # ty: ignore[invalid-argument-type]
        finally:
            _restore_call_execute(original)

        jsonl_files = sorted(tmp_path.glob("test_pa18_rotation*.jsonl"))
        assert len(jsonl_files) == 2, (
            f"expected 2 distinct capture files (one per attempt), "
            f"got {len(jsonl_files)}: {jsonl_files} — PA18 regression: "
            f"the second attempt overwrote the first."
        )

        # Filenames must include the timestamp suffix.
        for path in jsonl_files:
            assert "__20" in path.name, (
                f"capture filename {path.name!r} missing the "
                f"``__YYYYMMDDTHHMMSSZ`` UTC timestamp suffix that PA18 "
                f"introduced for per-attempt rotation."
            )

        # Each file's content reflects its own attempt's invocation —
        # they're not just two copies of the second attempt.
        bodies = [path.read_text(encoding="utf-8") for path in jsonl_files]
        assert "ok-1" in bodies[0] and "ok-1" not in bodies[1], (
            "first attempt's output bled into the second file"
        )
        assert "ok-2" in bodies[1] and "ok-2" not in bodies[0], (
            "second attempt's output bled into the first file"
        )


# ---------------------------------------------------------------------------
# Backward compat — runs_cli still parses schema-3 (pre-KC16-4) files.
# ---------------------------------------------------------------------------


class TestRunsCliBackwardCompat:
    """runs_cli.load_run reads both schema-3 and schema-4 files."""

    def test_load_run_handles_schema_4(self, tmp_path: Path) -> None:
        """A schema-4 file with redaction fields loads cleanly."""
        from tests.integration.runs_cli import load_run

        jsonl = tmp_path / "schema4.jsonl"
        header = {
            "kind": "header",
            "test_nodeid": "schema4::test",
            "schema_version": 4,
            "streaming": True,
            "redaction_enabled": True,
            "redaction_threshold_chars": 2048,
            "git": {"short_sha": "abc1234"},
        }
        inv = {
            "kind": "invocation",
            "call_seq": 1,
            "model": "m1",
            "trace": {"inputs": {"message": "short"}},
        }
        trailer = {
            "kind": "trailer",
            "schema_version": 4,
            "outcome": "passed",
            "total_cost_usd": 0.001,
            "call_count": 1,
            "elapsed_s": 0.5,
            "redacted_count": 3,
        }
        jsonl.write_text(
            "\n".join([json.dumps(header), json.dumps(inv), json.dumps(trailer)]) + "\n",
            encoding="utf-8",
        )

        merged_header, calls = load_run(jsonl)
        assert merged_header["schema_version"] == 4
        assert merged_header["outcome"] == "passed"
        assert merged_header["redacted_count"] == 3
        assert merged_header["redaction_enabled"] is True
        assert len(calls) == 1

    def test_load_run_still_handles_schema_3(self, tmp_path: Path) -> None:
        """Pre-KC16-4 schema-3 files keep loading."""
        from tests.integration.runs_cli import load_run

        jsonl = tmp_path / "schema3.jsonl"
        header = {
            "kind": "header",
            "test_nodeid": "schema3::test",
            "schema_version": 3,
            "streaming": True,
            "git": {"short_sha": "old1234"},
        }
        inv = {"kind": "invocation", "call_seq": 1, "model": "legacy"}
        trailer = {
            "kind": "trailer",
            "schema_version": 3,
            "outcome": "passed",
            "total_cost_usd": 0.0,
            "call_count": 1,
            "elapsed_s": 0.1,
        }
        jsonl.write_text(
            "\n".join([json.dumps(header), json.dumps(inv), json.dumps(trailer)]) + "\n",
            encoding="utf-8",
        )

        merged_header, calls = load_run(jsonl)
        assert merged_header["schema_version"] == 3
        assert merged_header["outcome"] == "passed"
        # New fields absent in schema-3 — readers must tolerate.
        assert "redacted_count" not in merged_header
        assert len(calls) == 1
