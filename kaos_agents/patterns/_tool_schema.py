"""Tool-schema rejection helpers — extract the offending tool from a
provider-error string so :mod:`kaos_agents.patterns.chat` can drop the
bad tool and retry instead of failing the whole turn.

OpenAI (and structurally any strict JSON-Schema validator at the
provider boundary) returns HTTP 400 with a body like::

    {'error': {'message': "Invalid schema for function 'TOOL': ...",
               'type': 'invalid_request_error',
               'param': 'tools[N].function.parameters',
               'code': 'invalid_function_parameters'}}

The error gets surfaced up through ``kaos_llm_client`` /
``kaos_llm_core`` as a Python exception whose ``str(exc)`` contains the
JSON-rendered payload. Parsing that string is the cheapest portable
extraction — provider clients differ in their exception types but all
preserve the body in ``str(exc)`` (verified against the live log from
FIX-14's openai:gpt-5.5 reproducer).

Two pieces of identity:

- ``extract_invalid_tool_name`` returns the function name, the more
  reliable signal because index-based dropping is sensitive to list
  ordering inside ReAct.
- ``extract_invalid_tool_index`` returns the 0-based index from
  ``tools[N]``, the fallback for providers whose message phrasing
  doesn't include the function name.

Both return ``None`` when the input doesn't look like a tool-schema
rejection — the caller MUST treat ``None`` as "not a recoverable
schema error, fall through to the generic fallback".
"""

from __future__ import annotations

import re
from typing import Any

# Patterns are deliberately tolerant of surrounding noise (provider
# wrappers tend to add prefixes like "openai returned 400: ..." plus a
# trailing dict-repr). Both anchors are unique enough to extract
# without false positives.
_INVALID_FUNCTION_NAME_RE = re.compile(
    r"Invalid schema for function ['\"]([^'\"]+)['\"]",
)
_INVALID_FUNCTION_INDEX_RE = re.compile(
    r"tools\[(\d+)\]\.function",
)
_INVALID_FUNCTION_CODE_RE = re.compile(
    r"invalid_function_parameters",
)


def is_tool_schema_rejection(exc_message: str) -> bool:
    """True when the exception text matches a tool-schema rejection.

    The check is conservative — we look for the OpenAI-style
    ``invalid_function_parameters`` code OR the
    ``Invalid schema for function`` phrasing OR the
    ``tools[N].function`` param pointer. Any one is enough to suggest
    the error is recoverable by dropping a tool.
    """
    return bool(
        _INVALID_FUNCTION_CODE_RE.search(exc_message)
        or _INVALID_FUNCTION_NAME_RE.search(exc_message)
        or _INVALID_FUNCTION_INDEX_RE.search(exc_message)
    )


def extract_invalid_tool_name(exc_message: str) -> str | None:
    """Return the function name from an "Invalid schema for function ..."
    error, or ``None`` when the message lacks that anchor.

    Prefer this over the index-based extractor: tool lists are
    rebuilt per turn, so an index is only meaningful for the failed
    call's exact ordering.
    """
    match = _INVALID_FUNCTION_NAME_RE.search(exc_message)
    return match.group(1) if match else None


def extract_invalid_tool_index(exc_message: str) -> int | None:
    """Return the 0-based tool index from ``param: 'tools[N].function...'``.

    Fallback used when the name regex misses (some providers truncate
    the message). The caller should map the index back to the tool name
    via the tool list it just passed in.
    """
    match = _INVALID_FUNCTION_INDEX_RE.search(exc_message)
    return int(match.group(1)) if match else None


def tool_name_of(tool: Any) -> str | None:
    """Best-effort tool-name extraction shared with ``filter_tools``.

    Mirrors :func:`kaos_agents.context.tool_filter._tool_name` so the
    schema-rejection retry uses the same identity the SessionToolSet
    filter does. Returns ``None`` when the tool exposes no recoverable
    name field.
    """
    meta = getattr(tool, "metadata", None)
    if meta is not None:
        if callable(meta):
            meta = meta()
        name = getattr(meta, "name", None)
        if name is not None:
            return str(name)
    name = getattr(tool, "name", None)
    return str(name) if name is not None else None


def drop_tool_by_name(tools: list[Any], name: str) -> list[Any]:
    """Return a new list with every tool matching ``name`` removed."""
    return [t for t in tools if tool_name_of(t) != name]


def drop_tool_at_index(tools: list[Any], index: int) -> tuple[list[Any], str | None]:
    """Return (new_list, dropped_name) — defensive when index is out of range."""
    if not 0 <= index < len(tools):
        return list(tools), None
    dropped = tool_name_of(tools[index])
    new_list = [t for i, t in enumerate(tools) if i != index]
    return new_list, dropped


__all__ = [
    "drop_tool_at_index",
    "drop_tool_by_name",
    "extract_invalid_tool_index",
    "extract_invalid_tool_name",
    "is_tool_schema_rejection",
    "tool_name_of",
]
