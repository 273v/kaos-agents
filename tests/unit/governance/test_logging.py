"""Tests for kaos_agents.governance.logging — JSONLAuditLogger."""

from __future__ import annotations

from typing import Any

import pytest

from kaos_agents.events import IntentClassified, RunError, TextDelta
from kaos_agents.governance.logging import JSONLAuditLogger, _default_path_resolver


class _StubVFS:
    """In-memory VFS double matching the kaos_core.vfs.core.VirtualFileSystem
    interface we exercise: async ``read``/``write`` with optional context_id."""

    def __init__(self) -> None:
        self.store: dict[str, bytes] = {}
        self.read_calls: list[tuple[str, str | None]] = []
        self.write_calls: list[tuple[str, bytes, str | None]] = []
        self.fail_writes = False
        self.fail_reads_existing_only = False

    async def read(self, path: str, context_id: str | None = None) -> bytes:
        self.read_calls.append((path, context_id))
        if path not in self.store:
            raise FileNotFoundError(path)
        return self.store[path]

    async def write(self, path: str, data: bytes, context_id: str | None = None) -> int:
        self.write_calls.append((path, data, context_id))
        if self.fail_writes:
            raise RuntimeError("write failure")
        self.store[path] = data
        return len(data)


def _intent_event(seq: int = 0, *, session_id: str = "sess-1") -> IntentClassified:
    return IntentClassified(
        timestamp=1.0 + seq,
        sequence=seq,
        session_id=session_id,
        run_id="run-1",
        intent="tool_use",
        confidence=0.9,
        reasoning="example",
    )


def _text_delta_event(seq: int = 1, *, session_id: str = "sess-1") -> TextDelta:
    return TextDelta(
        timestamp=2.0 + seq,
        sequence=seq,
        session_id=session_id,
        run_id="run-1",
        content="hello",
    )


def test_default_path_resolver_basic() -> None:
    assert _default_path_resolver("abc") == "audit/session-abc.jsonl"


def test_default_path_resolver_unscoped_when_blank() -> None:
    assert _default_path_resolver("") == "audit/session-unscoped.jsonl"


def test_default_path_resolver_replaces_slashes() -> None:
    assert _default_path_resolver("a/b") == "audit/session-a_b.jsonl"


@pytest.mark.asyncio
async def test_logger_with_no_vfs_is_noop() -> None:
    """vfs=None: hook accepts events without raising or persisting."""
    hook = JSONLAuditLogger(vfs=None)
    await hook.on_event(_intent_event())  # must not raise


@pytest.mark.asyncio
async def test_logger_writes_one_line_per_event() -> None:
    vfs = _StubVFS()
    hook = JSONLAuditLogger(vfs=vfs)
    await hook.on_event(_intent_event())
    assert len(vfs.write_calls) == 1
    path, data, _ = vfs.write_calls[0]
    assert path == "audit/session-sess-1.jsonl"
    text = data.decode("utf-8")
    assert text.endswith("\n")
    assert text.count("\n") == 1
    assert '"intent_classified"' in text
    assert '"sess-1"' in text


@pytest.mark.asyncio
async def test_logger_uses_custom_path_resolver() -> None:
    vfs = _StubVFS()

    def resolver(session_id: str) -> str:
        return f"custom/{session_id}.log"

    hook = JSONLAuditLogger(vfs=vfs, path_for_session=resolver)
    await hook.on_event(_intent_event(session_id="abc"))
    assert vfs.write_calls[0][0] == "custom/abc.log"


@pytest.mark.asyncio
async def test_logger_include_event_kinds_filter() -> None:
    vfs = _StubVFS()
    hook = JSONLAuditLogger(vfs=vfs, include_event_kinds=("IntentClassified",))
    await hook.on_event(_intent_event())
    await hook.on_event(_text_delta_event())  # filtered out
    assert len(vfs.write_calls) == 1
    text = vfs.write_calls[0][1].decode("utf-8")
    assert "intent_classified" in text
    assert "text_delta" not in text


@pytest.mark.asyncio
async def test_logger_swallows_vfs_write_failure() -> None:
    vfs = _StubVFS()
    vfs.fail_writes = True
    hook = JSONLAuditLogger(vfs=vfs)
    # Must not raise — observability never crashes the run.
    await hook.on_event(_intent_event())
    assert vfs.write_calls and vfs.write_calls[0][1]  # attempt was made
    assert vfs.store == {}  # but nothing committed (because the stub raised)


@pytest.mark.asyncio
async def test_logger_accumulates_multiple_events() -> None:
    vfs = _StubVFS()
    hook = JSONLAuditLogger(vfs=vfs)
    for seq in range(3):
        await hook.on_event(_intent_event(seq=seq))
    final = vfs.store["audit/session-sess-1.jsonl"].decode("utf-8")
    # Three lines, all terminated with newlines.
    lines = [line for line in final.split("\n") if line]
    assert len(lines) == 3
    assert all('"intent_classified"' in line for line in lines)


@pytest.mark.asyncio
async def test_logger_unscoped_when_session_id_blank() -> None:
    """Events that lack a session_id land in the ``unscoped`` log."""
    vfs = _StubVFS()
    hook = JSONLAuditLogger(vfs=vfs)
    # session_id is required by the pydantic model — use empty string.
    event = IntentClassified(
        timestamp=1.0,
        sequence=0,
        session_id="",
        run_id="run-1",
        intent="respond",
        confidence=0.5,
        reasoning="",
    )
    await hook.on_event(event)
    assert vfs.write_calls[0][0] == "audit/session-unscoped.jsonl"


@pytest.mark.asyncio
async def test_logger_handles_run_error_event() -> None:
    """Errors are events too — they round-trip via the normal serde path."""
    vfs = _StubVFS()
    hook = JSONLAuditLogger(vfs=vfs)
    event = RunError(
        timestamp=3.0,
        sequence=2,
        session_id="sess-1",
        run_id="run-1",
        error_type="RuntimeError",
        message="boom",
    )
    await hook.on_event(event)
    line = vfs.write_calls[0][1].decode("utf-8")
    assert '"run_error"' in line
    assert '"RuntimeError"' in line


@pytest.mark.asyncio
async def test_logger_metadata_is_well_formed() -> None:
    """Hook metadata names follow KAOS naming convention."""
    md = JSONLAuditLogger.metadata()
    assert md.name == "kaos-agents-jsonl-audit-logger"
    assert md.listens_to == ()


@pytest.mark.asyncio
async def test_logger_threads_context_id_to_vfs() -> None:
    """``context_id`` is forwarded to both read and write."""
    vfs = _StubVFS()
    hook = JSONLAuditLogger(vfs=vfs, context_id="ctx-42")
    await hook.on_event(_intent_event())
    # First read attempt (file doesn't exist yet) and the write both use the ctx.
    assert any(call[1] == "ctx-42" for call in vfs.read_calls)
    assert vfs.write_calls[0][2] == "ctx-42"


@pytest.mark.asyncio
async def test_logger_appends_newline_when_existing_lacks_one() -> None:
    """Read-modify-write: existing content without trailing newline gets one."""
    vfs = _StubVFS()
    vfs.store["audit/session-sess-1.jsonl"] = b'{"older":"event"}'  # no \n
    hook = JSONLAuditLogger(vfs=vfs)
    await hook.on_event(_intent_event())
    final = vfs.store["audit/session-sess-1.jsonl"].decode("utf-8")
    lines = final.split("\n")
    # First line preserved, second is the new event, third is empty (trailing \n).
    assert lines[0] == '{"older":"event"}'
    assert '"intent_classified"' in lines[1]
    assert lines[2] == ""


def test_logger_default_construction_no_args() -> None:
    """Bare constructor produces a usable no-op hook."""
    hook = JSONLAuditLogger()
    assert isinstance(hook, JSONLAuditLogger)


def test_logger_pathresolver_type_is_callable() -> None:
    from kaos_agents.governance.logging import PathResolver

    resolver: PathResolver = lambda sid: f"x/{sid}"  # noqa: E731
    assert resolver("y") == "x/y"


@pytest.mark.parametrize("ev_kinds", [None, ("RunError",), ("IntentClassified",)])
def test_logger_constructor_accepts_filter_variants(ev_kinds: Any) -> None:
    JSONLAuditLogger(vfs=None, include_event_kinds=ev_kinds)
