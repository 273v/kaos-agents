"""KC6 tests for the cross-process subprocess capture path.

Verifies the kaos-agents recorder's subprocess-capture machinery:
1. ``_inject_env`` adds the env var without clobbering the caller's env.
2. ``_install_subprocess_env_patches`` rewrites subprocess.Popen so
   spawned children inherit ``KAOS_LLM_CORE_RECORDER_DIR``.
3. ``_collect_subprocess_records`` reads a multi-PID JSONL set into
   one flat list.
4. End-to-end: an actual ``python -c`` subprocess that imports
   ``kaos_llm_core`` + runs one Call writes a JSONL record that the
   recorder stitches into the main test JSONL.
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from tests.integration._recorder import (
    _SUBPROCESS_ENV_VAR,
    _collect_subprocess_records,
    _inject_env,
    _install_subprocess_env_patches,
    _restore_subprocess_env_patches,
    record_live_test,
)


class TestInjectEnv:
    def test_none_starts_from_os_environ(self, tmp_path: Path) -> None:
        env = _inject_env(None, tmp_path)
        # Picks up at least PATH from the parent.
        assert "PATH" in env
        assert env[_SUBPROCESS_ENV_VAR] == str(tmp_path)

    def test_custom_dict_is_copied(self, tmp_path: Path) -> None:
        original = {"FOO": "bar"}
        env = _inject_env(original, tmp_path)
        assert env["FOO"] == "bar"
        assert env[_SUBPROCESS_ENV_VAR] == str(tmp_path)
        # Original is NOT mutated — caller's dict stays clean.
        assert _SUBPROCESS_ENV_VAR not in original

    def test_override_replaces_existing_var(self, tmp_path: Path) -> None:
        original = {_SUBPROCESS_ENV_VAR: "/old/path"}
        env = _inject_env(original, tmp_path)
        assert env[_SUBPROCESS_ENV_VAR] == str(tmp_path)


class TestPopenPatch:
    def test_patch_injects_env_var(self, tmp_path: Path) -> None:
        """A child process spawned during the patch sees the env var."""
        originals = _install_subprocess_env_patches(tmp_path)
        try:
            # Use a child that just prints its env var value. We don't
            # need kaos_llm_core for this — only that the env var
            # reaches the child.
            result = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    f"import os; print(os.environ.get('{_SUBPROCESS_ENV_VAR}', 'UNSET'))",
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            assert result.stdout.strip() == str(tmp_path)
        finally:
            _restore_subprocess_env_patches(originals)

    def test_restore_rolls_back(self, tmp_path: Path) -> None:
        """After restore, subsequent subprocess calls have no recorder env."""
        originals = _install_subprocess_env_patches(tmp_path)
        _restore_subprocess_env_patches(originals)
        # Clear the var from the parent env so the child can't see it
        # through Popen's default env=None inheritance.
        parent_had = os.environ.pop(_SUBPROCESS_ENV_VAR, None)
        try:
            result = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    f"import os; print(os.environ.get('{_SUBPROCESS_ENV_VAR}', 'UNSET'))",
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            assert result.stdout.strip() == "UNSET"
        finally:
            if parent_had is not None:
                os.environ[_SUBPROCESS_ENV_VAR] = parent_had

    @pytest.mark.asyncio
    async def test_patch_injects_into_asyncio_subprocess(self, tmp_path: Path) -> None:
        """asyncio.create_subprocess_exec also receives the env var."""
        originals = _install_subprocess_env_patches(tmp_path)
        try:
            script = (
                "import os, sys; "
                f"sys.stdout.write(os.environ.get('{_SUBPROCESS_ENV_VAR}', 'UNSET'))"
            )
            proc = await asyncio.create_subprocess_exec(
                sys.executable,
                "-c",
                script,
                stdout=asyncio.subprocess.PIPE,
            )
            stdout, _ = await proc.communicate()
            assert stdout.decode().strip() == str(tmp_path)
        finally:
            _restore_subprocess_env_patches(originals)


class TestCollectSubprocessRecords:
    def test_returns_empty_when_dir_missing(self, tmp_path: Path) -> None:
        missing = tmp_path / "does-not-exist"
        assert _collect_subprocess_records(missing) == []

    def test_returns_empty_when_dir_none(self) -> None:
        assert _collect_subprocess_records(None) == []

    def test_returns_empty_when_no_jsonl(self, tmp_path: Path) -> None:
        (tmp_path / "stray.txt").write_text("not jsonl")
        assert _collect_subprocess_records(tmp_path) == []

    def test_reads_multiple_pids(self, tmp_path: Path) -> None:
        """Two subprocess JSONLs are stitched into one flat list."""
        (tmp_path / "subprocess-1234-abcd.jsonl").write_text(
            json.dumps({"kind": "invocation", "pid": 1234, "model": "m1"}) + "\n"
        )
        (tmp_path / "subprocess-5678-efef.jsonl").write_text(
            json.dumps({"kind": "invocation", "pid": 5678, "model": "m2"})
            + "\n"
            + json.dumps({"kind": "invocation", "pid": 5678, "model": "m3"})
            + "\n"
        )
        records = _collect_subprocess_records(tmp_path)
        assert len(records) == 3
        # Sorted by filename → 1234 group first, then 5678 group.
        assert records[0]["pid"] == 1234
        assert records[1]["pid"] == 5678
        assert records[2]["pid"] == 5678

    def test_skips_corrupt_lines(self, tmp_path: Path) -> None:
        """A bad JSON line in the middle doesn't stop the rest from being read."""
        path = tmp_path / "subprocess-1-abcd.jsonl"
        path.write_text(
            json.dumps({"kind": "invocation", "model": "ok1"}) + "\n"
            "this is not json\n" + json.dumps({"kind": "invocation", "model": "ok2"}) + "\n"
        )
        records = _collect_subprocess_records(tmp_path)
        assert len(records) == 2
        assert records[0]["model"] == "ok1"
        assert records[1]["model"] == "ok2"


class TestEndToEndSubprocessCapture:
    """Spawn a real Python subprocess that imports kaos_llm_core and
    runs one Call. The recorder must stitch its JSONL into the main
    test JSONL on exit. This is the KC6 acceptance gate.
    """

    @pytest.mark.asyncio
    async def test_subprocess_call_lands_in_main_jsonl(self, tmp_path: Path) -> None:
        # Subprocess child: imports kaos_llm_core, sets up a
        # FunctionClient, runs one Call. Expects the env-recorder to
        # write a JSONL record to KAOS_LLM_CORE_RECORDER_DIR.
        child_script = """
import asyncio
from typing import Any
from kaos_llm_client import ModelProfile, ProviderResponse, UsageInfo
from kaos_llm_client.providers.function import FunctionClient
from kaos_llm_client.types import ContentPart
import kaos_llm_core.programs.call as call_mod
from kaos_llm_core import InputField, OutputField, Signature
from kaos_llm_core.programs.call import Call

def handler(messages: list[dict[str, Any]], profile: ModelProfile) -> ProviderResponse:
    return ProviderResponse.model_construct(
        provider="function",
        model="function-test-subproc",
        raw={},
        parts=[ContentPart.model_construct(type="text", text='{"answer": "from-subprocess"}')],
        usage=UsageInfo.model_construct(input_tokens=11, output_tokens=7, total_tokens=18),
        stop_reason="end_turn",
        status_code=200,
        response_headers={},
    )

client = FunctionClient(model="function-test-subproc", function=handler)
call_mod.create_client = lambda *a, **kw: client

class _Sig(Signature):
    \"\"\"Test.\"\"\"
    text: str = InputField(description="in")
    answer: str = OutputField(description="out")

call = Call(_Sig, model="function-test-subproc")
result = asyncio.run(call(text="hi"))
print(result.answer)
"""

        out_dir = tmp_path / "runs"
        out_dir.mkdir()

        async with record_live_test(
            test_nodeid="kc6_subprocess_e2e",
            out_dir=out_dir,
        ) as capture:
            # The recorder has patched subprocess.Popen by now —
            # children inherit the env var pointing at
            # capture.subprocess_dir.
            assert capture.subprocess_dir is not None
            result = subprocess.run(
                [sys.executable, "-c", child_script],
                capture_output=True,
                text=True,
                check=True,
            )
            assert "from-subprocess" in result.stdout

        # The subprocess JSONL should have been stitched into the
        # main JSONL. Read the main JSONL and assert.
        main_jsonl = out_dir / "kc6_subprocess_e2e.jsonl"
        assert main_jsonl.exists()
        lines = [json.loads(line) for line in main_jsonl.read_text().splitlines()]

        # Schema-v3 streaming: first line is the at-start header,
        # last line is the at-exit trailer with the final counts.
        header = lines[0]
        trailer = lines[-1]
        assert header["kind"] == "header"
        assert header["streaming"] is True
        assert header["schema_version"] == 3
        assert trailer["kind"] == "trailer"
        assert trailer["subprocess_call_count"] >= 1, (
            f"expected >=1 subprocess record, got trailer={trailer}"
        )

        subproc_lines = [line for line in lines[1:-1] if line.get("source") == "subprocess"]
        assert len(subproc_lines) >= 1
        # The subprocess record should carry the model + usage from
        # the FunctionClient handler.
        record = subproc_lines[0]
        assert record["model"] == "function-test-subproc"
        assert record["usage"]["total_tokens"] == 18

        # The recorder's tempdir should have been cleaned up.
        assert not capture.subprocess_dir.exists()
