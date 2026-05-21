"""Tool-call argument PII scrubber (broad-reliability roadmap §B0.7).

Adversarial scenario the F-03 audit flagged:

    User uploads ``client_letter.docx`` containing SSNs and case
    numbers. User asks "search the web for similar cases." The agent
    calls ``kaos-web-search(query="John Q. Public 123-45-6789 fraud
    case")``. The SSN goes into SerpAPI / Brave / Exa query args.
    Logs at the search provider now have client PII.

This module is the pre-execution gate that mutates tool-call kwargs
to mask PII *before* the underlying ``KaosTool.execute()`` runs.
``actions/tool_bridge.py`` calls :func:`scrub_tool_args` from its
executor (after the permission gate, before the asyncio.timeout-
wrapped invoke).

Patterns covered (v1, USA-centric — the legal-AI use case):

- **SSN** — ``NNN-NN-NNNN`` or ``NNNNNNNNN`` (9-digit standalone).
- **EIN** — ``NN-NNNNNNN`` (employer identification number).
- **Credit card** — Luhn-validated 13-19 digit runs. We don't false-
  positive on long numerics like phone numbers or invoice IDs by
  checking the Luhn checksum.
- **US phone (DEA-flag-only)** — phones are often legitimately passed
  to search, so we DO NOT scrub them by default. The pattern lives
  here for future opt-in.

Sentinels:

- :data:`SCRUB_MASK` is the replacement string used for matched runs
  (``"***SCRUBBED***"``). Callers writing audit trails can grep for
  this token to see what fired.
- :data:`SCRUB_FIELD_PREFIX` flags WHOLE-FIELD masks (used when a
  scalar value matches end-to-end) so downstream consumers can tell
  the difference between "an SSN was found inside a larger query"
  and "the entire arg WAS a credit-card number."

The scrubber is conservative by design:

- Only inspects ``str`` values inside the kwargs tree. Numeric IDs
  (``int`` / ``float``) pass through untouched.
- Recurses into nested ``dict`` / ``list`` / ``tuple`` containers but
  leaves the container shape unchanged.
- Returns a NEW kwargs object — never mutates the caller's input.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Final

SCRUB_MASK: Final[str] = "***SCRUBBED***"
SCRUB_FIELD_PREFIX: Final[str] = "***SCRUBBED:"

# ── Regex patterns ──────────────────────────────────────────────────
#
# SSN: NNN-NN-NNNN preferred. Bare 9-digit runs are common false-
# positives (zip-9, order IDs), so we accept them only when not
# adjacent to other digits (``\b`` word boundaries) AND when surrounded
# by non-digit context. The hyphenated form is the load-bearing case.
_SSN_HYPHEN_RE: Final = re.compile(r"\b(\d{3}-\d{2}-\d{4})\b")
_SSN_BARE_RE: Final = re.compile(r"(?<!\d)(\d{9})(?!\d)")

# EIN: NN-NNNNNNN. The hyphen is a strong signal — IRS-issued
# identifiers always render with the dash.
_EIN_RE: Final = re.compile(r"\b(\d{2}-\d{7})\b")

# Credit card: 13-19 digit run with optional spaces or dashes between
# every 4-digit group. Luhn-validated to suppress false positives.
_CC_RE: Final = re.compile(r"\b(\d{4}[ -]?\d{4}[ -]?\d{4}[ -]?\d{1,7})\b")


def _luhn_check(digits: str) -> bool:
    """Return True if ``digits`` (digits-only, 13-19 chars) passes Luhn."""
    if not 13 <= len(digits) <= 19 or not digits.isdigit():
        return False
    total = 0
    parity = len(digits) % 2
    for i, ch in enumerate(digits):
        n = int(ch)
        if i % 2 == parity:
            n *= 2
            if n > 9:
                n -= 9
        total += n
    return total % 10 == 0


def _looks_like_ssn(s: str) -> bool:
    """A whole-string SSN match — drives ``SCRUB_FIELD_PREFIX``."""
    return bool(_SSN_HYPHEN_RE.fullmatch(s.strip()))


# ── Public dataclass ────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class ScrubResult:
    """Outcome of a scrubber pass on one kwargs payload.

    ``kwargs`` is the new (possibly transformed) kwargs dict.
    ``matches`` is the list of pattern names that fired at least
    once (e.g. ``["ssn", "credit_card"]``) for telemetry / audit
    annotations. Empty when nothing was scrubbed.
    """

    kwargs: dict[str, Any]
    matches: tuple[str, ...]

    @property
    def scrubbed(self) -> bool:
        """Convenience: did any pattern fire?"""
        return bool(self.matches)


# ── Core scrubber ────────────────────────────────────────────────────


def _scrub_string(value: str, matches: set[str]) -> str:
    """Apply all regex patterns to one string, mutating ``matches``."""
    out = value

    # 1. Credit cards (Luhn-validated so we don't false-positive on
    #    order IDs / phone-with-extension).
    def _cc_sub(m: re.Match[str]) -> str:
        raw = m.group(1)
        digits = re.sub(r"[ -]", "", raw)
        if _luhn_check(digits):
            matches.add("credit_card")
            return SCRUB_MASK
        return raw

    out = _CC_RE.sub(_cc_sub, out)

    # 2. SSN with hyphens (high-precision).
    if _SSN_HYPHEN_RE.search(out):
        matches.add("ssn")
        out = _SSN_HYPHEN_RE.sub(SCRUB_MASK, out)

    # 3. Bare 9-digit SSN. Only fires when surrounded by non-digit
    #    context (already enforced by the lookarounds). Conservative
    #    fallback for "Jane Doe 123456789 fraud case" tokens.
    if _SSN_BARE_RE.search(out):
        matches.add("ssn")
        out = _SSN_BARE_RE.sub(SCRUB_MASK, out)

    # 4. EIN.
    if _EIN_RE.search(out):
        matches.add("ein")
        out = _EIN_RE.sub(SCRUB_MASK, out)

    return out


def _scrub_value(value: Any, matches: set[str]) -> Any:
    """Recursively scrub a value of arbitrary nesting."""
    if isinstance(value, str):
        # Whole-string SSN → emit a labeled placeholder so audit
        # trails can tell the difference between "an SSN appeared
        # inside a larger query" and "the entire arg IS the SSN".
        if _looks_like_ssn(value):
            matches.add("ssn")
            return f"{SCRUB_FIELD_PREFIX}ssn***"
        return _scrub_string(value, matches)
    if isinstance(value, dict):
        return {k: _scrub_value(v, matches) for k, v in value.items()}
    if isinstance(value, list):
        return [_scrub_value(v, matches) for v in value]
    if isinstance(value, tuple):
        return tuple(_scrub_value(v, matches) for v in value)
    # Numbers, bools, None, custom objects — pass through unchanged.
    # The scrubber is a string-level defense; integer SSNs would
    # require a separate policy (and are extremely rare in tool args).
    return value


def scrub_tool_args(kwargs: dict[str, Any]) -> ScrubResult:
    """Scrub PII from a tool-call kwargs dict.

    Returns a :class:`ScrubResult` carrying the new kwargs (deep-
    copy of input with PII replaced) and the set of pattern names
    that fired. The caller is expected to:

    1. Use ``result.kwargs`` when invoking the underlying tool.
    2. If ``result.scrubbed``, annotate the tool call's
       ``args_preview`` / log line so the audit trail records that
       PII was filtered.

    The scrubber is intentionally conservative — patterns are USA-
    centric (SSN / EIN / Luhn-valid credit cards) and field shapes
    that aren't strings pass through untouched. False-negative bias
    is preferred over false-positive bias (no one wants their
    legitimate query mangled).
    """
    matches: set[str] = set()
    new_kwargs = _scrub_value(kwargs, matches)
    return ScrubResult(kwargs=new_kwargs, matches=tuple(sorted(matches)))


__all__ = [
    "SCRUB_FIELD_PREFIX",
    "SCRUB_MASK",
    "ScrubResult",
    "scrub_tool_args",
]
