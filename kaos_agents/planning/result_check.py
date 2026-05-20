"""Shared result checking utilities.

Single source of truth for error detection in tool/LLM results.
Used by act.py, evaluate.py, compose.py, plan_execute.py, and the
CircuitBreaker (via :func:`is_uninformative_result`).
"""

from __future__ import annotations

import re

# Error prefixes that indicate a step failure.
# Tool bridge (tool_bridge.py) produces "ERROR:" prefix.
# JSON error responses use '{"error":' prefix.
_ERROR_PREFIXES = ("ERROR:", '{"error":')


# Default patterns that mark a *successful* tool call as carrying no
# usable signal (zero results / empty array / explicit "no matches").
#
# These are GENERIC across tool families — no tool-name-specific
# phrases. Each pattern is a compiled regex applied with `.search()`
# against the tool's textual result. A pattern firing means the agent
# saw no actionable data and any retry with a similar prompt is very
# unlikely to produce different output. The CircuitBreaker counts
# consecutive firings as failures so the loop terminates rather than
# spinning (session DEB had 12 consecutive zero-result web searches
# in a row with no signal to break the loop).
_DEFAULT_UNINFORMATIVE_PATTERNS: tuple[re.Pattern[str], ...] = (
    # "No results", "no result found", "No Results Found For ..."
    re.compile(r"\bno\s+results?\b", re.IGNORECASE),
    # "No matches", "no match found"
    re.compile(r"\bno\s+match(?:es)?\b", re.IGNORECASE),
    # "No hits"
    re.compile(r"\bno\s+hits?\b", re.IGNORECASE),
    # "0 results" / "0 hits" / "0 matches" — anchored on word boundary
    # before 0 so "1230 results" never matches.
    re.compile(
        r"\b0\s+(?:results?|hits?|matches?|items?|rows?|records?|documents?|files?|entries)\b",
        re.IGNORECASE,
    ),
    # JSON-style empty list field: `"results": []`, `"hits": []`,
    # `"matches": []`, etc. Allows whitespace around the colon and
    # inside the brackets.
    re.compile(
        r"\"(?:results?|hits?|matches?|items?|rows?|records?|documents?|files?|entries|data)\"\s*:\s*\[\s*\]"
    ),
    # JSON-style explicit zero count: `"total_matches": 0`, `"count": 0`,
    # `"n_results": 0`, etc.
    re.compile(
        r"\"(?:total_matches|count|total|n_results?|num_results?|hit_count|match_count|result_count)\"\s*:\s*0\b"
    ),
    # Bare empty array as the entire body.
    re.compile(r"^\s*\[\s*\]\s*$"),
)


def is_error_result(text: str) -> bool:
    """Check if a result string indicates an error.

    This is the single authoritative check. All planning code must use
    this instead of inline startswith() calls.
    """
    return any(text.startswith(prefix) for prefix in _ERROR_PREFIXES)


def is_empty_result(text: str) -> bool:
    """Check if a result string is empty or whitespace-only."""
    return not text.strip()


def is_uninformative_result(
    text: str,
    *,
    extra_patterns: tuple[re.Pattern[str], ...] = (),
) -> bool:
    """Return True iff a textual tool result carries no usable signal.

    Distinct from :func:`is_error_result`: a tool can succeed
    (``is_error=False``) yet return nothing actionable — zero search
    hits, an empty array, an explicit "no matches" string. When that
    happens repeatedly the agent has no new signal to drive its next
    reasoning step and will either spin (session
    ``01KS2DEBYT341F1F16B3BRQRV0`` had 12 consecutive zero-result web
    searches in a row) or fabricate. Counting consecutive uninformative
    returns as :class:`~kaos_agents.action.circuit.CircuitBreaker`
    failures terminates the loop instead of letting it run forever.

    Predicate is generic — it does NOT inspect tool names. It matches a
    set of conservative phrasings that real-world kaos-* tools emit on
    empty returns:

    * empty / whitespace-only text (via :func:`is_empty_result`)
    * "no results" / "no matches" / "no hits" phrases
    * explicit "0 results" / "0 hits" / "0 matches" counts
    * JSON-style empty list fields
      (``"results": []``, ``"hits": []``, ``"matches": []``,
      ``"items": []``, etc.)
    * JSON-style explicit zero count
      (``"total_matches": 0``, ``"count": 0``, etc.)
    * The bare empty array ``[]`` as the entire body

    Does NOT match:

    * Any text that :func:`is_error_result` returns True for. The error
      path is owned by the existing predicate — do not double-count.
    * Any text that does not contain one of the explicit markers above.
      Informativeness is best-effort and the predicate errs on the side
      of NOT firing. Tools whose empty-state phrasing is non-standard
      can opt in via ``extra_patterns``.

    Args:
        text: The textual result body — typically the
            ``"result_summary"`` attribute of a
            :class:`~kaos_agents.events.Span` with
            ``subject=TOOL_CALL`` and ``phase=COMPLETE``.
        extra_patterns: Caller-supplied compiled regexes to OR in on
            top of the defaults — useful when a tool family uses
            non-standard empty-state phrasing. Default is the empty
            tuple.

    Returns:
        True iff the result carries no usable signal.
    """
    if is_error_result(text):
        return False  # owned by is_error_result; do not double-count.
    if is_empty_result(text):
        return True
    return any(pat.search(text) for pat in (*_DEFAULT_UNINFORMATIVE_PATTERNS, *extra_patterns))
