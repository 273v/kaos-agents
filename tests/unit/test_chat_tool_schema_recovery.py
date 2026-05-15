"""Unit tests for the tool-schema rejection parser + drop helpers.

Pins FIX-16: when ONE bridged tool ships an invalid JSON Schema, the
chat pattern drops that single tool from the catalog and retries
ReAct instead of failing the whole turn. The parser shape was
designed against a real openai:gpt-5.5 400 response captured in the
single-user-chat backend log during the FIX-14 reproducer.
"""

from __future__ import annotations

from dataclasses import dataclass

from kaos_agents.patterns._tool_schema import (
    drop_tool_at_index,
    drop_tool_by_name,
    extract_invalid_tool_index,
    extract_invalid_tool_name,
    is_tool_schema_rejection,
    tool_name_of,
)

# Real exception text captured from kaos-pdf 0.1.0a2 + openai gpt-5.5
# during the FIX-14 reproducer. Preserved verbatim (modulo Python dict
# repr quoting) so a future change to provider error rendering will
# surface in this test file rather than only in production.
_OPENAI_400_LIVE = (
    "openai returned 400: Invalid schema for function 'kaos-pdf-extract-parse': "
    "In context=('properties', 'pages'), array schema missing items. "
    "({'provider': 'openai', 'model': 'gpt-5.5', 'status_code': 400, "
    "'raw_error': {'error': {'message': \"Invalid schema for function "
    "'kaos-pdf-extract-parse': In context=('properties', 'pages'), array "
    "schema missing items.\", 'type': 'invalid_request_error', "
    "'param': 'tools[9].function.parameters', "
    "'code': 'invalid_function_parameters'}}, "
    "'fix': 'Check request parameters. Ensure required fields are present.', "
    "'retry_after': None})"
)

_OPENAI_400_NEXT_TOOL = (
    "openai returned 400: Invalid schema for function 'kaos-office-parse-xlsx': "
    "In context=('properties', 'sheets'), array schema missing items. "
    "({'param': 'tools[26].function.parameters', "
    "'code': 'invalid_function_parameters'})"
)


def test_is_tool_schema_rejection_matches_live_payload() -> None:
    assert is_tool_schema_rejection(_OPENAI_400_LIVE) is True
    assert is_tool_schema_rejection(_OPENAI_400_NEXT_TOOL) is True


def test_is_tool_schema_rejection_rejects_unrelated_errors() -> None:
    assert is_tool_schema_rejection("rate limited 429") is False
    assert is_tool_schema_rejection("openai timeout after 60s") is False
    assert is_tool_schema_rejection("") is False


def test_extract_invalid_tool_name_pulls_function_name() -> None:
    assert extract_invalid_tool_name(_OPENAI_400_LIVE) == "kaos-pdf-extract-parse"
    assert extract_invalid_tool_name(_OPENAI_400_NEXT_TOOL) == "kaos-office-parse-xlsx"
    assert extract_invalid_tool_name("no tool name here") is None


def test_extract_invalid_tool_index_pulls_tools_n() -> None:
    assert extract_invalid_tool_index(_OPENAI_400_LIVE) == 9
    assert extract_invalid_tool_index(_OPENAI_400_NEXT_TOOL) == 26
    assert extract_invalid_tool_index("missing index") is None


@dataclass
class _FakeMeta:
    name: str


@dataclass
class _FakeTool:
    metadata: _FakeMeta


def _mktool(name: str) -> _FakeTool:
    return _FakeTool(metadata=_FakeMeta(name=name))


def test_tool_name_of_walks_metadata_attribute() -> None:
    assert tool_name_of(_mktool("kaos-pdf-extract-parse")) == "kaos-pdf-extract-parse"


def test_tool_name_of_walks_callable_metadata() -> None:
    class CallableMeta:
        def __call__(self) -> _FakeMeta:
            return _FakeMeta(name="kaos-callable")

    @dataclass
    class CallableMetaTool:
        metadata: CallableMeta

    tool = CallableMetaTool(metadata=CallableMeta())
    assert tool_name_of(tool) == "kaos-callable"


def test_tool_name_of_falls_back_to_bare_name() -> None:
    @dataclass
    class BareNameTool:
        name: str

    assert tool_name_of(BareNameTool(name="bare")) == "bare"
    # When neither shape applies, return None — caller must handle.
    assert tool_name_of(object()) is None


def test_drop_tool_by_name_removes_exact_match() -> None:
    tools = [_mktool("a"), _mktool("b"), _mktool("c")]
    result = drop_tool_by_name(tools, "b")
    assert [tool_name_of(t) for t in result] == ["a", "c"]
    # Original list preserved
    assert [tool_name_of(t) for t in tools] == ["a", "b", "c"]


def test_drop_tool_by_name_no_match_returns_full_list() -> None:
    tools = [_mktool("a"), _mktool("b")]
    result = drop_tool_by_name(tools, "missing")
    assert [tool_name_of(t) for t in result] == ["a", "b"]


def test_drop_tool_at_index_returns_dropped_name() -> None:
    tools = [_mktool("a"), _mktool("b"), _mktool("c")]
    new_list, dropped = drop_tool_at_index(tools, 1)
    assert dropped == "b"
    assert [tool_name_of(t) for t in new_list] == ["a", "c"]


def test_drop_tool_at_index_out_of_range_is_noop() -> None:
    tools = [_mktool("a"), _mktool("b")]
    new_list, dropped = drop_tool_at_index(tools, 5)
    assert dropped is None
    assert [tool_name_of(t) for t in new_list] == ["a", "b"]


def test_drop_tool_at_index_negative_is_noop() -> None:
    """Defensive: a -1 index from a buggy extractor should not wrap to the end."""
    tools = [_mktool("a"), _mktool("b")]
    new_list, dropped = drop_tool_at_index(tools, -1)
    assert dropped is None
    assert [tool_name_of(t) for t in new_list] == ["a", "b"]
