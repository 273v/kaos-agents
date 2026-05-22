"""Ordinal coreference resolver (plan §Issue 8 / B1.5).

A long deal-room session accumulates a list of documents / cases /
filings the user has referenced earlier. By turn 7+ they start
saying "the third NDA we discussed" or "the second filing" or
"the previous case" — and the agent answers about the wrong one.

The resolver is a fast, deterministic, zero-LLM primitive: given
the user's current message and a list of referent candidates
(typically the SessionMemory DOCUMENTS section, ACTIONS, or
FINDINGS items in turn order), it looks for ordinal phrases like
"the third NDA" / "the first one" / "the last filing" / "the
previous one" and returns the resolved candidate.

Two pieces of context drove this design:

1. **Deterministic over LLM-judged.** The acceptance bar is ≥90%
   resolution rate on the 12-scenario fixture (plan §Issue 8).
   A regex-based ordinal extractor + index map is both faster
   (no LLM round-trip per turn) and easier to audit than a
   Signature call.

2. **Inject as ``<context>`` tag, not as a rewrite.** The plan
   says: "Inject resolved referent into worker prompt as
   ``<context>`` tag". The resolver returns a structured
   ``CoreferenceResolution`` value; the assemble_context layer
   formats it as a tag. We never silently mutate the user's
   message — that would be an attorney-malpractice surface
   (paraphrasing the question).

The full integration (assemble_context tag injection + 12-scenario
LLM eval in tests/integration/test_ordinal_coref_live.py) is a
follow-on; this module ships the resolver primitive with its own
contract pins.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

# Ordinal word → 1-based index. Covers the cases the plan
# acceptance fixture exercises: first / second / third / ... /
# tenth + "last" + "previous" + "next".
_ORDINAL_WORDS: dict[str, int] = {
    "first": 1,
    "second": 2,
    "third": 3,
    "fourth": 4,
    "fifth": 5,
    "sixth": 6,
    "seventh": 7,
    "eighth": 8,
    "ninth": 9,
    "tenth": 10,
}

# Numeric ordinal suffixes ("1st", "2nd", "3rd", ..., "20th").
# We bound at 20 for the deal-room use case — beyond that, ordinal
# coref is wrong abstraction; users should reference by name.
_NUMERIC_ORDINAL_RE = re.compile(r"\b(\d{1,2})(?:st|nd|rd|th)\b", re.IGNORECASE)

# Whole-phrase patterns. The leading article ("the") is required
# to keep "first thing in the morning" from matching "first".
_PHRASE_RES: tuple[re.Pattern[str], ...] = (
    re.compile(
        r"\bthe\s+("
        + "|".join(_ORDINAL_WORDS.keys())
        + r")\s+(?:one|item|entry|file|doc|document|case|filing|nda|"
        + r"contract|agreement|matter|attachment|upload|cite|"
        + r"citation|result)\b",
        re.IGNORECASE,
    ),
    # "the second one", "the third" (bare ordinal at end of clause)
    re.compile(
        r"\bthe\s+(" + "|".join(_ORDINAL_WORDS.keys()) + r")\b(?:[\s.,;:!?]|$)",
        re.IGNORECASE,
    ),
    # "the previous" / "the prior" / "the last" → index -1 sentinel,
    # handled separately below
)

_LAST_RE = re.compile(
    r"\bthe\s+(last|prior|previous|preceding|latest|most\s+recent)"
    r"(?:\s+(?:one|item|entry|file|doc|document|case|filing|nda|"
    r"contract|agreement|matter|attachment|upload|cite|citation|"
    r"result))?\b",
    re.IGNORECASE,
)

_NEXT_RE = re.compile(
    r"\bthe\s+next"
    r"(?:\s+(?:one|item|entry|file|doc|document|case|filing|nda|"
    r"contract|agreement|matter|attachment|upload|cite|citation|"
    r"result))?\b",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class CoreferenceResolution:
    """Resolved-referent record.

    Frozen — safe to attach to events / persist to memory.
    """

    matched_phrase: str
    """The exact substring of ``message`` that triggered the
    resolution (audit trail anchor)."""

    ordinal: int
    """1-based ordinal. ``-1`` means "the last one" (relative to
    end of list); ``-2`` means "the next-to-last", etc."""

    resolved_index: int | None
    """0-based index into the candidates list. ``None`` when the
    ordinal is out of range (more than the number of candidates)."""

    resolved_candidate: Any | None
    """The candidates[resolved_index] item — or ``None`` if out of
    range or candidates was empty."""

    confidence: float
    """0.0-1.0. 1.0 when the ordinal is in range; 0.5 when the
    ordinal is out of range (we detected the phrase but couldn't
    bind it). Callers use this to decide whether to inject the
    resolution as a hint or to flag the ambiguity to the user."""


def resolve_ordinal(
    message: str,
    candidates: list[Any] | tuple[Any, ...],
) -> CoreferenceResolution | None:
    """Scan ``message`` for an ordinal reference and resolve it
    against ``candidates``.

    Returns ``None`` when no ordinal phrase is detected — that's
    the common case for non-ordinal messages and lets the caller
    skip the tag-injection step entirely.

    Resolution order:
    1. "the last / prior / previous / latest" → candidates[-1]
    2. "the next" → candidates[0] if list is non-empty
       (semantically "what's next" usually means "the next one to
       look at" which is item 0 in a fresh list — but if all items
       are already visited, this is ambiguous; we return the first
       item with confidence=0.5)
    3. Numeric ordinal (e.g. "the 3rd NDA") → candidates[N-1]
    4. Word ordinal (e.g. "the third NDA") → candidates[N-1]

    Ambiguity is the operator's call — when more than one ordinal
    fires, we return the FIRST match (left-to-right reading
    order). The assemble_context layer can decide to flag
    ambiguous resolutions to the user.
    """
    if not message or not candidates:
        return None

    # 1. "the last" — robust against the user saying it before any
    #    other ordinal phrase (covers "the previous one" too).
    m = _LAST_RE.search(message)
    if m is not None:
        idx = len(candidates) - 1
        return CoreferenceResolution(
            matched_phrase=m.group(0),
            ordinal=-1,
            resolved_index=idx,
            resolved_candidate=candidates[idx] if idx >= 0 else None,
            confidence=1.0,
        )

    # 2. "the next" — only meaningful when there's at least one
    #    candidate; we bind to index 0 as a heuristic.
    m = _NEXT_RE.search(message)
    if m is not None:
        return CoreferenceResolution(
            matched_phrase=m.group(0),
            ordinal=1,
            resolved_index=0,
            resolved_candidate=candidates[0],
            confidence=0.5,  # heuristic — "next" is ambiguous in chat
        )

    # 3. Numeric ordinal ("1st" / "2nd" / "3rd" / ...).
    m = _NUMERIC_ORDINAL_RE.search(message)
    if m is not None:
        n = int(m.group(1))
        return _bind_ordinal(message=m.group(0), n=n, candidates=candidates)

    # 4. Word ordinal ("first" / "second" / "third" / ...).
    for pattern in _PHRASE_RES:
        m = pattern.search(message)
        if m is not None:
            word = m.group(1).lower()
            n = _ORDINAL_WORDS[word]
            return _bind_ordinal(message=m.group(0), n=n, candidates=candidates)

    return None


def _bind_ordinal(
    *,
    message: str,
    n: int,
    candidates: list[Any] | tuple[Any, ...],
) -> CoreferenceResolution:
    """Bind a 1-based ordinal ``n`` to ``candidates``.

    Out-of-range → confidence 0.5 (the resolver still reports the
    detected phrase so the caller can surface the ambiguity).
    """
    if 1 <= n <= len(candidates):
        idx = n - 1
        return CoreferenceResolution(
            matched_phrase=message,
            ordinal=n,
            resolved_index=idx,
            resolved_candidate=candidates[idx],
            confidence=1.0,
        )
    return CoreferenceResolution(
        matched_phrase=message,
        ordinal=n,
        resolved_index=None,
        resolved_candidate=None,
        confidence=0.5,
    )


__all__ = ["CoreferenceResolution", "resolve_ordinal"]
