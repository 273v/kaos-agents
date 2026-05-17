"""Refuse to answer when every evidence-gathering tool call failed.

When the user explicitly references attached files (or names that
look like filenames in the message) and **every** tool call the agent
made during the turn returned ``is_error=True``, the agent's LLM-
drafted final answer is no longer grounded in anything. The agent has
two failure modes from here:

1. Say "I couldn't read the files; here's what I tried" — honest.
2. Synthesise an answer anyway from training-data plausibility +
   filename hints — **confidently wrong**.

The kaos-* OSS legal-research bar (273V product policy) ranks (2) as
the WORST class of failure: a user reading that answer and acting on
it = legal / financial harm. This module enforces (1) by replacing the
drafted answer with a structured refusal at the pattern's final-answer
hook point, ahead of any user-facing emit.

Production trigger that motivated this gate: session
``01KRVYAEA3B1HG95DBAG6H0DJ3`` — five NDA .docx files uploaded into the
VFS, every ``kaos-office-parse-docx`` call returned "File not found"
because the tools were VFS-blind (filed in
``vfs-blind-tools-audit-and-fix-plan.md``), and the agent fabricated a
Delaware-vs-Michigan jurisdiction analysis citing files it never read.

Stage 0.2 of the VFS-blind fix plan. The Stage 1-4 tool fixes will
make the failure path far less common; this gate is the belt-and-
suspenders guarantee that even when individual tools regress, the
agent never ships fabricated content as fact.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# Filename-shaped tokens an agent might reference in a question. The
# match list deliberately covers the document-extraction surface
# served by kaos-office / kaos-pdf / kaos-tabular / kaos-source.
# Match the basic ``<base>.<ext>`` (no spaces in base) tokens first.
# Multi-word filenames like ``EMNA Mutual NDA.docx`` are handled by a
# separate pass: we walk backwards from the matched base, accepting
# preceding space-separated words only when they look filename-like
# (TitleCase / ALLCAPS / digit-leading), to avoid swallowing English
# prose ("check the Contract.pdf" → "Contract.pdf", not "check the
# Contract.pdf").
_FILENAME_PATTERN = re.compile(
    r"""
    (?<![\w/.\\-])               # not preceded by a word/path char
    [\w\-]+                       # base name (no spaces in this pass)
    \.
    (?:
        pdf | docx | doc | pptx | ppt | xlsx | xls |
        csv | tsv | parquet | sqlite | sql |
        txt | md | html | htm | xml | json | jsonl |
        eml | mbox | vcf |
        png | jpg | jpeg | gif | tiff | bmp |
        zip | tar | gz | bz2
    )
    \b
    """,
    re.IGNORECASE | re.VERBOSE,
)


def _looks_filename_part(word: str) -> bool:
    """True when ``word`` is plausibly part of a filename basename.

    Used by :func:`extract_referenced_files` to walk backward across
    a multi-word filename like ``EMNA Mutual NDA``. We accept a word
    only when at least one of these holds:

    - wholly uppercase letters/digits with no lowercase ("EMNA", "NDA",
      "MNDA", "C2H6", "Q4"), OR
    - contains a digit anywhere ("Q4-2024", "file_1"), OR
    - contains an underscore ("snake_case"), OR
    - is a single-character hyphen separator ("MNDA - Acme.docx").

    TitleCase single words like "Read", "Contract", "The" are REJECTED
    so the walker doesn't swallow surrounding English prose. That
    occasionally clips a real multi-word filename (e.g. "Mutual"
    in "EMNA Mutual NDA.docx" gets cut), but losing the middle word
    is fine — the gate's job is to refuse with "I couldn't read
    NDA.docx", not to perfectly reconstruct every basename.
    """
    if not word or " " in word:
        return False
    if word == "-":
        return True
    has_lower = any(c.islower() for c in word)
    has_digit = any(c.isdigit() for c in word)
    has_underscore = "_" in word
    if not has_lower:
        # All-caps / digits / hyphens / dots — looks filename-ish.
        return True
    return has_digit or has_underscore


@dataclass(frozen=True, slots=True)
class ToolObservationSummary:
    """Minimal view of a tool call's outcome — what the gate needs.

    The pattern that calls the gate already has the full kaos-llm-core
    ``ToolObservation`` (or the typed ``ToolExecution`` from Act mode);
    this is the projection that survives both shapes so the gate is
    independent of which pattern invoked it.

    ``arguments_preview`` is the JSON-stringified tool-call arguments
    (truncated to a few hundred chars). The gate scans these for
    filename-shaped tokens — if the agent called
    ``kaos-office-parse-docx`` with ``path="sessions/<sid>/files/EMNA
    Mutual NDA.docx"`` and that call failed, the filename in the
    args is the strongest signal that the user wanted that file
    read (stronger than the message text, which often says
    "summarize these" without naming files).
    """

    tool_name: str
    is_error: bool
    result_preview: str = ""
    arguments_preview: str = ""


@dataclass(frozen=True, slots=True)
class NoEvidenceVerdict:
    """The gate's decision for the current turn.

    ``refuse`` is the boolean the pattern should branch on. The other
    fields populate the structured refusal text + the
    ``GroundingRefusalTriggered`` event payload so downstream
    consumers (the SPA's ToolCallBlock, OTel hooks, plan-execute's
    replan logic) can render or react to the refusal honestly.
    """

    refuse: bool
    reason: str = ""
    referenced_files: tuple[str, ...] = ()
    failed_tool_count: int = 0
    failed_tools: tuple[str, ...] = field(default_factory=tuple)
    error_excerpts: tuple[str, ...] = field(default_factory=tuple)


def extract_referenced_files(message: str) -> tuple[str, ...]:
    """Return filename-shaped tokens present in ``message``.

    Used to decide whether the user's question was specifically about
    files (and therefore a "zero-evidence" outcome should refuse) or a
    generic chat where the agent should still respond from priors.

    Conservative match: needs an extension from the known-document
    list. ``"check my file"`` will NOT trigger; ``"check
    Contract.pdf"`` will.
    """
    if not message:
        return ()
    matches = []
    seen: set[str] = set()
    for m in _FILENAME_PATTERN.finditer(message):
        base_token = m.group(0).strip()
        token = _expand_multiword_basename(message, m.start(), base_token)
        if not token or "." not in token:
            continue
        key = token.lower()
        if key in seen:
            continue
        seen.add(key)
        matches.append(token)
    return tuple(matches)


def _expand_multiword_basename(message: str, match_start: int, base_token: str) -> str:
    """Walk backwards across spaces to absorb multi-word filename parts.

    The pattern matches a single contiguous base name like ``NDA.docx``;
    if the prefix immediately before the match is ``EMNA Mutual ``
    (i.e. TitleCase / ALLCAPS / digit words joined by single spaces)
    we absorb those words. Walking is character-level so it stops at
    path separators (``/``, ``\\``) and quote / brace / comma /
    colon characters — common in JSON-encoded tool arguments.
    """
    # Pull up to ~120 chars of prefix; walk backwards collecting only
    # ``[\w\-]`` chars and single spaces between filename-like words.
    stop_chars = "/\\\"'{}[],;:>=\n\t"
    pos = match_start - 1
    chars: list[str] = []
    on_space = False
    word_count_after_match = 0
    while pos >= 0 and (match_start - 1 - pos) < 120:
        ch = message[pos]
        if ch in stop_chars:
            break
        if ch == " ":
            if on_space:
                # Two consecutive spaces — stop; not a real basename.
                break
            on_space = True
        elif ch.isalnum() or ch in "_-.":
            if on_space:
                # We just finished a word (in reverse). Record its
                # start (we are at the LAST char of that earlier
                # word). Check whether the upcoming word starts with
                # a filename-like character. We'll know after we've
                # collected the word; punt the check to after.
                on_space = False
            chars.insert(0, ch)
            pos -= 1
            continue
        else:
            break
        chars.insert(0, ch)
        pos -= 1

    if not chars:
        return base_token

    prefix = "".join(chars).strip()
    # Split into space-joined words and accept the longest tail
    # whose every word is filename-like.
    words = prefix.split(" ")
    absorbed_tail: list[str] = []
    for word in reversed(words):
        if word and _looks_filename_part(word):
            absorbed_tail.insert(0, word)
            word_count_after_match += 1
            if word_count_after_match >= 8:
                break
        else:
            break
    if not absorbed_tail:
        return base_token
    token = " ".join([*absorbed_tail, base_token])
    return re.sub(r"\s+", " ", token).strip()


def evaluate_no_evidence_gate(
    *,
    observations: list[ToolObservationSummary] | tuple[ToolObservationSummary, ...],
    user_message: str = "",
    attached_documents: list[str] | tuple[str, ...] = (),
) -> NoEvidenceVerdict:
    """Decide whether the turn must refuse rather than answer.

    Returns ``NoEvidenceVerdict(refuse=True, ...)`` when ALL of:

    1. At least one tool call was attempted this turn.
    2. Every attempted tool call returned ``is_error=True``.
    3. The user's question referenced files — either by attaching
       documents (``attached_documents``) or by mentioning filename-
       shaped tokens in ``user_message``.

    Returns ``NoEvidenceVerdict(refuse=False)`` otherwise. The
    pattern is then free to assemble its normal LLM-drafted answer.

    Parameters
    ----------
    observations
        One ``ToolObservationSummary`` per tool call attempted during
        the turn. The pattern's adapter to this gate collapses
        ``ToolObservation`` (kaos-llm-core ReAct) or ``ToolExecution``
        (Act / AgenticLoop) into this view.
    user_message
        The user's text message for the turn. Used to detect filename
        tokens. May be empty when the turn was driven by a structured
        input (e.g. a recipe).
    attached_documents
        Names of files explicitly attached to the session
        (``SessionMemory.DOCUMENTS`` or the SPA's per-turn upload
        list). When this is non-empty the gate considers the user as
        having referenced files even if no filename token appears in
        the message text — common when the user asks "summarise these"
        or "what's the key term?" without naming files.
    """
    obs_tuple = tuple(observations)
    if not obs_tuple:
        # No tools attempted — nothing to gate. The agent answered
        # from priors / memory and the LLM gets to ship its text.
        return NoEvidenceVerdict(refuse=False)

    failed = [o for o in obs_tuple if o.is_error]
    if len(failed) != len(obs_tuple):
        # At least one tool succeeded — the agent has SOME evidence.
        # Don't refuse; the LLM can ground on what worked.
        return NoEvidenceVerdict(refuse=False)

    referenced = list(attached_documents)
    if user_message:
        referenced.extend(extract_referenced_files(user_message))
    # Strongest signal: the agent itself reached for these files via
    # tool call arguments. If every call failed, those names are still
    # what the user wanted read.
    for obs in failed:
        if obs.arguments_preview:
            referenced.extend(extract_referenced_files(obs.arguments_preview))
    # Dedupe while preserving order.
    seen: set[str] = set()
    referenced_unique: list[str] = []
    for name in referenced:
        key = name.lower()
        if key in seen:
            continue
        seen.add(key)
        referenced_unique.append(name)

    if not referenced_unique:
        # User asked a generic question and every tool happened to
        # fail (e.g. a SERP outage). Don't force refusal — the agent
        # can still respond from priors. The gate is specifically
        # about refusing fabricated facts ABOUT NAMED FILES.
        return NoEvidenceVerdict(refuse=False)

    failed_tools = tuple(o.tool_name for o in failed)
    error_excerpts = tuple(_excerpt_error(o.result_preview) for o in failed)
    reason = (
        f"All {len(failed)} tool call(s) attempted in this turn returned errors, "
        f"and the user referenced {len(referenced_unique)} file(s) "
        "that the agent therefore did not read"
    )
    return NoEvidenceVerdict(
        refuse=True,
        reason=reason,
        referenced_files=tuple(referenced_unique),
        failed_tool_count=len(failed),
        failed_tools=failed_tools,
        error_excerpts=error_excerpts,
    )


def render_refusal_text(verdict: NoEvidenceVerdict, *, max_files: int = 5) -> str:
    """Compose the final-answer text for a refused turn.

    The output is the literal string the chat pattern emits via
    ``TextDelta`` instead of the LLM's draft. It must be honest about
    the failure mode without being scary: this is a tool-side bug, not
    a model hallucination, and the user should be told what to do
    next.
    """
    if not verdict.refuse:
        return ""

    files = verdict.referenced_files[:max_files]
    file_list = ", ".join(f"`{f}`" for f in files)
    if len(verdict.referenced_files) > max_files:
        file_list += f" (and {len(verdict.referenced_files) - max_files} more)"

    failed_unique = tuple(dict.fromkeys(verdict.failed_tools))
    tool_list = ", ".join(f"`{t}`" for t in failed_unique)

    excerpt_lines: list[str] = []
    seen_excerpts: set[str] = set()
    for excerpt in verdict.error_excerpts:
        if excerpt and excerpt not in seen_excerpts:
            seen_excerpts.add(excerpt)
            excerpt_lines.append(f"- {excerpt}")
        if len(excerpt_lines) >= 3:
            break

    parts: list[str] = []
    parts.append(
        "I tried to read the file(s) you referenced "
        f"({file_list}) but every tool call returned an error, "
        "so I have no evidence to answer your question from."
    )
    parts.append(
        f"Tools attempted: {tool_list}. ({verdict.failed_tool_count} call(s), all failed.)"
    )
    if excerpt_lines:
        parts.append("Errors I saw:\n" + "\n".join(excerpt_lines))
    parts.append(
        "I will NOT fabricate an answer for this question. "
        "This is almost certainly a tool/VFS path-resolution issue "
        "(see `kaos-modules/docs/plans/vfs-blind-tools-audit-and-fix-plan.md`) "
        "rather than a problem with your upload. "
        "Please retry, or report the failure so the tool can be fixed."
    )
    return "\n\n".join(parts)


def _excerpt_error(result_preview: str, *, max_len: int = 160) -> str:
    """Pull a one-line excerpt from a tool error payload.

    The wire-format error envelope is
    ``{"error": true, "message": "<msg>", "locator"?: "<...>"}``;
    we extract ``message`` if available, else strip and clip the raw
    preview. Used in the refusal text + the
    ``GroundingRefusalTriggered`` event payload.
    """
    if not result_preview:
        return ""
    text = result_preview.strip()
    # Best-effort message extraction without importing json (the
    # preview is often truncated mid-JSON so json.loads would fail).
    m = re.search(r'"message"\s*:\s*"((?:[^"\\]|\\.)*)"', text)
    if m:
        text = m.group(1).replace('\\"', '"')
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > max_len:
        text = text[: max_len - 1].rstrip() + "…"
    return text


__all__ = [
    "NoEvidenceVerdict",
    "ToolObservationSummary",
    "evaluate_no_evidence_gate",
    "extract_referenced_files",
    "render_refusal_text",
]
