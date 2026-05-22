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
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from kaos_agents.memory.session import SessionMemory
    from kaos_agents.types.memory import MemoryItem, MemoryType

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


def format_coreference_tag(
    resolution: CoreferenceResolution,
    *,
    label_for: Any | None = None,
) -> str:
    """Format a resolution as a ``<context>`` tag for the worker prompt.

    Plan §Issue 8 says: "Inject resolved referent into worker
    prompt as ``<context>`` tag". This helper produces the exact
    string the assemble_context layer drops into the worker's
    thinking_note (or prepends to the worker message).

    Format:

        <context>
        coref: the user wrote "the third NDA" which resolves to
        the document at position 3 (out of 5):
        <referent>doc-C</referent>
        Use this referent unless the user clarifies otherwise.
        </context>

    For low-confidence ambiguous resolutions (e.g. "the next"
    heuristic, or out-of-range numerics), the tag flags the
    ambiguity so the worker can ask for clarification rather than
    silently binding.

    The ``label_for`` callback optionally maps the resolved
    candidate to a display label (e.g. a document's filename
    rather than the raw dict). When ``None``, ``repr(candidate)``
    is used.
    """
    if label_for is not None and resolution.resolved_candidate is not None:
        label = label_for(resolution.resolved_candidate)
    elif resolution.resolved_candidate is not None:
        label = repr(resolution.resolved_candidate)
    else:
        label = "(out of range)"

    phrase = resolution.matched_phrase
    ordinal = resolution.ordinal

    if resolution.resolved_index is None:
        return (
            "<context>\n"
            f'coref: the user wrote "{phrase}" but ordinal '
            f"{ordinal} is out of range. Ask the user to "
            "clarify which item they mean, or refuse with an "
            "explicit message about which ordinals are "
            "available.\n"
            "</context>"
        )

    if resolution.confidence < 0.99:
        return (
            "<context>\n"
            f'coref (low confidence): the user wrote "{phrase}" '
            f"which COULD resolve to:\n"
            f"<referent>{label}</referent>\n"
            "but the binding is heuristic. If the answer hinges "
            "on which item is meant, ask the user to confirm.\n"
            "</context>"
        )

    return (
        "<context>\n"
        f'coref: the user wrote "{phrase}" which resolves to '
        f"the item at position {ordinal}:\n"
        f"<referent>{label}</referent>\n"
        "Use this referent unless the user clarifies otherwise.\n"
        "</context>"
    )


def _default_doc_label(item: MemoryItem) -> str:
    """Default ``label_for`` for a DOCUMENTS-section MemoryItem.

    Prefers the metadata anchor used by the WU-G.2 corpus-handle
    builder (``filename`` / ``uri`` / ``source_uri`` / ``name`` /
    ``source``) so the rendered tag matches the same identifier the
    rest of the system already shows the user.

    Falls back to the first non-empty line of the item content, then
    to a short ``repr``-style anchor so we never emit an empty
    ``<referent></referent>`` block.
    """
    metadata = item.metadata or {}
    anchor = (
        metadata.get("filename")
        or metadata.get("uri")
        or metadata.get("source_uri")
        or metadata.get("name")
        or metadata.get("source")
    )
    if anchor:
        return str(anchor)
    content = item.content or ""
    for line in content.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped[:120]
    return f"item:{item.id}"


def build_coref_context_tag(
    memory: SessionMemory,
    message: str,
    *,
    section: MemoryType | None = None,
    label_for: Any | None = None,
    min_confidence: float = 0.5,
) -> str | None:
    """Compose ``resolve_ordinal`` + ``format_coreference_tag`` against
    the SessionMemory DOCUMENTS section (default) and return a worker-
    prompt-ready ``<context>`` tag, or ``None`` when no ordinal phrase
    fires.

    Plan §Issue 8 acceptance: "Inject resolved referent into worker
    prompt as ``<context>`` tag". This is the integration helper the
    run loop calls AFTER ``assemble_context`` and BEFORE intent
    classification, so the worker sees the resolved referent even
    when the DOCUMENTS section has been BM25-pruned by retrieval.

    The candidates come from the full unpruned DOCUMENTS section
    (``memory.get(DOCUMENTS)``), NOT from the assembled context —
    "the third NDA" refers to the third NDA the user has uploaded
    across the whole session, not the third one that happened to
    survive token-budget trimming this turn.

    Args:
        memory: SessionMemory carrying the DOCUMENTS section.
        message: User's current turn message.
        section: Which memory section to resolve against. Defaults
            to ``MemoryType.DOCUMENTS`` (the deal-room use case).
            Passing e.g. ``MemoryType.FINDINGS`` lets a research
            agent resolve "the third finding" against the same
            primitive.
        label_for: Callback mapping a candidate MemoryItem to a
            display label. Defaults to ``_default_doc_label`` which
            mirrors the corpus-handle metadata anchor (filename /
            uri / ...).
        min_confidence: Below this, the tag is suppressed (returns
            ``None``). Defaults to ``0.5`` so the "the next"
            heuristic (confidence=0.5) still surfaces a clarify-
            ambiguity tag; bump to ``0.99`` to keep only confident
            in-range resolutions.

    Returns:
        The formatted ``<context>`` tag string when a resolution
        fires and meets ``min_confidence``; ``None`` otherwise.
    """
    from kaos_agents.types.memory import MemoryType as _MemoryType

    target_section = section if section is not None else _MemoryType.DOCUMENTS

    if not memory.has_section(target_section):
        return None
    candidates: list[MemoryItem] = list(memory.get(target_section))
    if not candidates:
        return None

    resolution = resolve_ordinal(message, candidates)
    if resolution is None:
        return None
    if resolution.confidence < min_confidence:
        return None

    selected_label_for = label_for if label_for is not None else _default_doc_label
    return format_coreference_tag(resolution, label_for=selected_label_for)


__all__ = [
    "CoreferenceResolution",
    "build_coref_context_tag",
    "format_coreference_tag",
    "resolve_ordinal",
]
