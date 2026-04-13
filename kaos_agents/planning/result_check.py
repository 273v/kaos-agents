"""Shared result checking utilities.

Single source of truth for error detection in tool/LLM results.
Used by act.py, evaluate.py, compose.py, and plan_execute.py.
"""

from __future__ import annotations

# Error prefixes that indicate a step failure.
# Tool bridge (tool_bridge.py) produces "ERROR:" prefix.
# JSON error responses use '{"error":' prefix.
_ERROR_PREFIXES = ("ERROR:", '{"error":')


def is_error_result(text: str) -> bool:
    """Check if a result string indicates an error.

    This is the single authoritative check. All planning code must use
    this instead of inline startswith() calls.
    """
    return any(text.startswith(prefix) for prefix in _ERROR_PREFIXES)


def is_empty_result(text: str) -> bool:
    """Check if a result string is empty or whitespace-only."""
    return not text.strip()
