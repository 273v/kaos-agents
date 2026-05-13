"""Citation aggregation + inline-markdown rendering.

Two pure-function primitives, both side-effect-free:

* :func:`aggregate_citations` — walk FINDINGS via :func:`walk_findings`,
  flatten each Claim's ``supporting_spans``, dedupe via :func:`dedupe_spans`.
  No memory writes; pure read.
* :func:`inline_citations_md` — render a list of :class:`Span` into a
  markdown footer or inline marker string. Pure string transform.

These are the bridge between accumulated agent state (Claims with
typed Spans in FINDINGS) and the deliverable's surface (markdown
text with citation markers + a bibliography). They sit between
:mod:`walks` (which knows about memory) and the composers (which
build deliverables).
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

from kaos_agents.output.types import CitationStyle
from kaos_agents.output.walks import SpanDedupKey, dedupe_spans, walk_findings

if TYPE_CHECKING:
    from kaos_llm_core.signatures.grounding import Span

    from kaos_agents.memory.session import SessionMemory


def aggregate_citations(
    memory: SessionMemory,
    *,
    dedupe_by: SpanDedupKey = "source_uri+char_span",
) -> list[Span]:
    """Walk FINDINGS, flatten supporting spans, dedupe — return ordered citations.

    The deliverable's authoritative citation list. Order is
    first-appearance across FINDINGS items; deduplication respects the
    ``(source_uri, char_span)`` pair by default so the same source
    cited at distinct positions appears multiple times.

    Args:
        memory: Session memory whose FINDINGS section holds the typed
            Claims (written by ``patterns/research.py`` via
            ``Claim.model_dump(mode="json")``).
        dedupe_by: ``"source_uri"`` collapses every span from the same
            source into one entry; ``"source_uri+char_span"`` (default)
            preserves position-distinct spans.

    Returns:
        A list of :class:`Span`s in deduplicated first-appearance order.
        Empty when FINDINGS is empty or absent.
    """
    spans_in_order: list[Span] = []
    for claim in walk_findings(memory):
        spans_in_order.extend(claim.supporting_spans)
    return dedupe_spans(spans_in_order, by=dedupe_by)


def inline_citations_md(
    text: str,
    spans: Iterable[Span],
    *,
    style: CitationStyle = "inline",
) -> str:
    """Render a markdown string with inline citation markers.

    Three styles:

    * ``"inline"`` — append ``[source_uri]`` markers after each
      sentence that mentions the span's quote substring. If the quote
      isn't found verbatim (paraphrased answer), the marker goes at
      the end of the text.
    * ``"footnote"`` — leave inline markers as ``[^N]`` and append a
      footnote section ``[^N]: source_uri (page N)``. N is 1-indexed
      in span-iteration order.
    * ``"bibliography"`` — leave the body unchanged; append a
      ``## References`` section listing every span as a bullet.

    The function is pure. It doesn't mutate ``text`` or ``spans``;
    it returns a fresh string.

    Args:
        text: The body markdown to annotate.
        spans: The citation list. Order matters for footnote numbering.
        style: Citation rendering style.

    Returns:
        Markdown text with citations inlined per the chosen style.
    """
    spans_list = list(spans)
    if not spans_list:
        return text

    if style == "inline":
        return _render_inline(text, spans_list)
    if style == "footnote":
        return _render_footnote(text, spans_list)
    return _render_bibliography(text, spans_list)


def _render_inline(text: str, spans: list[Span]) -> str:
    """Inline ``[source_uri]`` markers after the first occurrence of each quote.

    Falls back to a trailing ``Citations: source_a, source_b`` line if
    no quote is found verbatim — the agent paraphrased.
    """
    body = text
    placed: set[str] = set()
    for span in spans:
        quote = span.quote.strip()
        if not quote:
            continue
        idx = body.find(quote)
        if idx == -1:
            continue
        # Insert the marker after the quote.
        end = idx + len(quote)
        body = body[:end] + f" [{span.source_uri}]" + body[end:]
        placed.add(span.source_uri)
    unplaced = [s for s in spans if s.source_uri not in placed]
    if unplaced:
        # Append a "Citations:" footer for paraphrased / unmatched spans.
        sources = ", ".join(sorted({s.source_uri for s in unplaced}))
        body = body.rstrip() + f"\n\nCitations: {sources}"
    return body


def _render_footnote(text: str, spans: list[Span]) -> str:
    """Append a footnote block; leave body unchanged.

    The agent is responsible for placing ``[^N]`` markers in its text;
    this function just appends the corresponding footnote definitions.
    """
    lines = ["", ""]
    for n, span in enumerate(spans, start=1):
        page_part = f" (page {span.page})" if span.page else ""
        quote_preview = span.quote[:120].rstrip()
        lines.append(f'[^{n}]: {span.source_uri}{page_part} — "{quote_preview}"')
    return text.rstrip() + "\n".join(lines)


def _render_bibliography(text: str, spans: list[Span]) -> str:
    """Append a ``## References`` section listing every span."""
    bib_lines = ["", "", "## References", ""]
    for span in spans:
        page_part = f" p.{span.page}" if span.page else ""
        quote_preview = span.quote[:160].rstrip()
        bib_lines.append(f"- **{span.source_uri}**{page_part}: {quote_preview}")
    return text.rstrip() + "\n".join(bib_lines)
