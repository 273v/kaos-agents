"""Unit tests for :class:`MatterIsolationHook` (plan §Issue 2).

The hook is the agent-loop-side companion to the SPA-side BAA gate
(``app/services/baa_gate.py`` in kaos-ui). When a Runner is bound
to a specific ``matter_id``, the hook inspects every tool-call
event's args for cross-matter references and refuses the call.

Acceptance row from plan §Issue 2:

    Runner constructed with matter_id=A, tool calls into matter=B
    raise MatterIsolationError.
"""

from __future__ import annotations

import pytest

from kaos_agents.events.spans import Span, SpanPhase, SpanSubject
from kaos_agents.hooks.base import HookAction
from kaos_agents.memory.isolation import (
    MatterIsolationError,
    MatterIsolationHook,
    _scan_value_for_matter_ids,
)

_TS = 1747915200.0  # 2026-05-22T12:00:00Z as POSIX float
_SID = "01KSTEST"
_RID = "turn-0000-aaa"


def _start_span(args: object, tool_name: str = "test-tool") -> Span:
    """Build a representative ``Span(TOOL_CALL, START)`` event."""
    return Span(
        timestamp=_TS,
        sequence=0,
        session_id=_SID,
        run_id=_RID,
        subject=SpanSubject.TOOL_CALL,
        phase=SpanPhase.START,
        span_id="span_abc123",
        attributes={"tool_name": tool_name, "args": args},
    )


# ── _scan_value_for_matter_ids ───────────────────────────────────────


@pytest.mark.unit
def test_scan_finds_matter_uri_form() -> None:
    """``matter:<id>`` canonical URI is detected."""
    assert _scan_value_for_matter_ids("matter:ABC-2026-0001") == {"ABC-2026-0001"}


@pytest.mark.unit
def test_scan_finds_matters_path_form() -> None:
    """``matters/<id>/...`` VFS path form is detected."""
    assert _scan_value_for_matter_ids("matters/XYZ-9999/contracts/nda.pdf") == {"XYZ-9999"}


@pytest.mark.unit
def test_scan_finds_matter_id_query_fragment() -> None:
    """``matter_id=<id>`` query-fragment form is detected."""
    assert _scan_value_for_matter_ids(
        "https://api.example.com/data?matter_id=ABC-2026-0001&foo=bar"
    ) == {"ABC-2026-0001"}


@pytest.mark.unit
def test_scan_finds_matter_id_in_dict_kwarg() -> None:
    """Literal ``matter_id`` key in a kwarg dict is detected."""
    assert _scan_value_for_matter_ids({"matter_id": "ABC-2026-0001", "query": "test"}) == {
        "ABC-2026-0001"
    }


@pytest.mark.unit
def test_scan_recurses_into_nested_dicts_and_lists() -> None:
    """Nested structures are walked one level deep."""
    args = {
        "files": [
            "matters/CASE-A/file1.pdf",
            "matters/CASE-B/file2.pdf",
        ],
        "options": {"matter_id": "CASE-C"},
    }
    found = _scan_value_for_matter_ids(args)
    assert found == {"CASE-A", "CASE-B", "CASE-C"}


@pytest.mark.unit
def test_scan_returns_empty_for_unrelated_strings() -> None:
    """Free-text args without matter references yield empty set."""
    assert _scan_value_for_matter_ids("What's the weather today?") == set()
    assert _scan_value_for_matter_ids({"q": "hello world"}) == set()
    assert _scan_value_for_matter_ids(["a", "b", "c"]) == set()


# ── Hook behavior ────────────────────────────────────────────────────


@pytest.mark.unit
@pytest.mark.asyncio
async def test_hook_is_noop_when_matter_id_is_none() -> None:
    """Legacy sessions (no matter_id) must bypass the gate so
    pre-existing behaviour stays unchanged. Operators opt in by
    setting matter_id on the session."""
    hook = MatterIsolationHook(matter_id=None)
    event = _start_span({"matter_id": "OTHER-MATTER", "query": "test"})
    action = await hook.on_tool_call_start(event)
    assert action is HookAction.CONTINUE


@pytest.mark.unit
@pytest.mark.asyncio
async def test_hook_continues_on_same_matter() -> None:
    """A tool call whose args reference the bound matter is allowed."""
    hook = MatterIsolationHook(matter_id="ABC-2026-0001")
    event = _start_span({"file": "matters/ABC-2026-0001/nda.pdf", "query": "find term"})
    action = await hook.on_tool_call_start(event)
    assert action is HookAction.CONTINUE


@pytest.mark.unit
@pytest.mark.asyncio
async def test_hook_raises_on_cross_matter_args() -> None:
    """Plan §Issue 2 acceptance: Runner bound to A, tool call into B
    raises MatterIsolationError."""
    hook = MatterIsolationHook(matter_id="MATTER-A")
    event = _start_span(
        {"file": "matters/MATTER-B/contracts/nda.pdf"},
        tool_name="kaos-pdf-extract-parse",
    )
    with pytest.raises(MatterIsolationError) as exc_info:
        await hook.on_tool_call_start(event)
    msg = str(exc_info.value)
    assert "MATTER-A" in msg
    assert "MATTER-B" in msg
    assert "kaos-pdf-extract-parse" in msg
    assert "Fix" in msg
    assert "Alternative" in msg


@pytest.mark.unit
@pytest.mark.asyncio
async def test_hook_raises_on_matter_uri_form() -> None:
    """Canonical ``matter:<id>`` URI form is caught too."""
    hook = MatterIsolationHook(matter_id="MATTER-A")
    event = _start_span("matter:MATTER-B/files/foo")
    with pytest.raises(MatterIsolationError):
        await hook.on_tool_call_start(event)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_hook_continues_when_args_have_no_matter_reference() -> None:
    """Free-text args (search queries, prompts) don't trip the gate
    even when matter_id is bound — they're not addressing the KB."""
    hook = MatterIsolationHook(matter_id="MATTER-A")
    event = _start_span({"query": "what is the governing law clause?"})
    action = await hook.on_tool_call_start(event)
    assert action is HookAction.CONTINUE


@pytest.mark.unit
@pytest.mark.asyncio
async def test_hook_continues_when_event_is_not_tool_call() -> None:
    """The hook fires only on TOOL_CALL spans; TURN / STEP / LLM_CALL
    spans pass through (they're handled by other hooks)."""
    hook = MatterIsolationHook(matter_id="MATTER-A")
    event = Span(
        timestamp=_TS,
        sequence=0,
        session_id=_SID,
        run_id=_RID,
        subject=SpanSubject.TURN,  # not TOOL_CALL
        phase=SpanPhase.START,
        span_id="span_turn_1",
        attributes={"args": {"matter_id": "MATTER-B"}},
    )
    action = await hook.on_tool_call_start(event)
    assert action is HookAction.CONTINUE


@pytest.mark.unit
@pytest.mark.asyncio
async def test_hook_continues_when_args_key_absent() -> None:
    """An in-test stub Span without ``args`` is treated as no-op
    rather than raising defensively."""
    hook = MatterIsolationHook(matter_id="MATTER-A")
    event = Span(
        timestamp=_TS,
        sequence=0,
        session_id=_SID,
        run_id=_RID,
        subject=SpanSubject.TOOL_CALL,
        phase=SpanPhase.START,
        span_id="span_x",
        attributes={"tool_name": "x"},
    )
    action = await hook.on_tool_call_start(event)
    assert action is HookAction.CONTINUE


@pytest.mark.unit
def test_hook_exposes_bound_matter_id() -> None:
    """The ``matter_id`` property lets the Runner / audit layer
    query what the hook is bound to."""
    hook = MatterIsolationHook(matter_id="ABC-2026-0001")
    assert hook.matter_id == "ABC-2026-0001"
    none_hook = MatterIsolationHook(matter_id=None)
    assert none_hook.matter_id is None
