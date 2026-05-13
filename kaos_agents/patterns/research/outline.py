"""Document outline injection helpers for ``ResearchAgent`` (P7).

When a real lawyer hands an associate a stack of contracts, the
associate's first question is "what's in here?" — a quick scan of the
table of contents before any specific question. This module gives the
agent the same affordance: a compact ``[CORPUS OUTLINE]`` preamble that
lists each document's heading hierarchy. The agent then reasons "for a
confidentiality question, look in §4" or "this corpus is regulatory
docs, not space-mission reports — refuse" before BM25 retrieval kicks
in. We tell the LLM that ABSENCE from the outline does NOT imply
absence from the document — body-text-only content (footnotes,
definitions inside paragraphs, untitled sections) is still searchable;
the outline is structural skeleton only.

Public surface (consumed by ``agent.py``):
- ``_build_outline_text`` — the entry point. Returns the rendered
  preamble or ``""`` to skip injection.
- ``_OUTLINE_MAX_TOTAL_CHARS`` / ``_OUTLINE_MAX_DOCS_AUTO`` — policy
  knobs that the agent passes back through (kept here so the policy
  lives next to the renderer that interprets them).

Everything else (``_DocOutline``, the per-source builders, the
formatters, URI helpers) is private to this module.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

# Per-doc and global caps. Treat as policy knobs, not API.
_OUTLINE_MAX_TOTAL_CHARS = 5000
_OUTLINE_MAX_DOCS_AUTO = 50
_OUTLINE_MAX_HEADINGS_PER_DOC = 40
_OUTLINE_MAX_HEADING_CHARS = 80


def _build_outline_text(
    *,
    documents: Sequence[Any],
    corpus: Any,
    policy: Literal["yes", "no", "auto"],
    max_total_chars: int,
    max_docs_for_auto: int,
) -> str:
    """Render a "[CORPUS OUTLINE]" preamble for the corpus.

    Returns ``""`` when the policy says to skip, when there is nothing to
    render (no documents, no headings), or when ``"auto"`` would
    overflow the size budget.

    Sources of truth, in priority order:
    1. ``documents`` (a list of ``ContentDocument``) — preferred. Yields a
       depth-aware outline by walking ``DocumentView(doc).flat_sections``.
    2. ``corpus`` passages — when documents are unavailable (e.g. the
       agent was constructed with ``corpus=...`` but no ``documents=...``),
       group passages by ``doc_uri`` and emit a single-depth outline
       built from unique ``section_title`` values. Loses depth, but
       still gives the LLM a useful structural skeleton.
    3. Otherwise: empty (no outline; the legacy memory-based path uses
       plain text and has no headings to extract).
    """
    if policy == "no":
        return ""

    doc_outlines: list[_DocOutline] = []
    if documents:
        doc_outlines = _outline_from_documents(documents)
    elif corpus is not None:
        doc_outlines = _outline_from_corpus_passages(corpus)

    # Drop docs that produced no headings — listing a document by name
    # alone isn't worth the prompt tokens.
    doc_outlines = [d for d in doc_outlines if d.headings]
    if not doc_outlines:
        return ""

    n_docs = len(doc_outlines)
    if policy == "auto" and n_docs > max_docs_for_auto:
        return ""

    full = _format_outline(doc_outlines)
    if len(full) <= max_total_chars:
        return full

    # Overflow path. ``auto`` may give up; ``yes`` always degrades.
    degraded = _format_outline(doc_outlines, max_depth=2)
    if len(degraded) <= max_total_chars:
        return degraded
    skeleton = _format_outline_skeleton(doc_outlines)
    if len(skeleton) <= max_total_chars:
        return skeleton
    if policy == "auto":
        return ""
    # ``yes`` policy: hard-truncate the skeleton with a clear marker so the
    # agent doesn't think the corpus ends mid-list.
    truncated = skeleton[: max_total_chars - 32].rstrip()
    return truncated + "\n... (outline truncated)"


# Internal value type — lightweight, doesn't justify a public dataclass.
class _DocOutline:
    """Per-document outline data: name, headings (text, depth, page)."""

    __slots__ = ("display_name", "headings")

    def __init__(
        self,
        display_name: str,
        headings: list[tuple[str, int, int | None]],
    ) -> None:
        self.display_name = display_name
        self.headings = headings


def _outline_from_documents(documents: Sequence[Any]) -> list[_DocOutline]:
    """Build outlines from ``ContentDocument`` instances using DocumentView.

    Groups by ``metadata.source.uri`` so chunks of the same source file
    collapse into one outline entry (chunked corpora register every
    chunk as its own ``ContentDocument`` with the same source URI).
    """
    try:
        from kaos_content.views.document_view import DocumentView
    except ImportError:  # pragma: no cover — kaos-content is a hard dep
        return []

    by_uri: dict[str, _DocOutline] = {}
    seen_heading_keys: dict[str, set[tuple[str, int]]] = {}
    order: list[str] = []

    for doc in documents:
        uri = _doc_display_uri(doc)
        if uri not in by_uri:
            by_uri[uri] = _DocOutline(display_name=_strip_uri_scheme(uri), headings=[])
            seen_heading_keys[uri] = set()
            order.append(uri)
        outline = by_uri[uri]
        seen = seen_heading_keys[uri]

        try:
            view = DocumentView(doc)
        except Exception:
            continue
        if not view.has_sections:
            continue
        for sv in view.flat_sections:
            text = (sv.heading_text or "").strip()
            if not text:
                continue
            depth = sv.depth or 1
            page = sv.page_range[0] if sv.page_range else None
            key = (text, depth)
            if key in seen:
                continue
            seen.add(key)
            outline.headings.append((text, depth, page))
            if len(outline.headings) >= _OUTLINE_MAX_HEADINGS_PER_DOC:
                break

    return [by_uri[u] for u in order]


def _outline_from_corpus_passages(corpus: Any) -> list[_DocOutline]:
    """Fallback: build a single-depth outline from corpus passages.

    Uses ``passage.section_title`` to recover headings. Depth defaults to
    1 because passages don't carry depth info. Page comes from
    ``passage.page`` when available.
    """
    try:
        passages = list(corpus.iter_passages())
    except Exception:
        return []

    by_uri: dict[str, _DocOutline] = {}
    seen_titles: dict[str, set[str]] = {}
    order: list[str] = []

    for p in passages:
        uri = getattr(p, "doc_uri", None) or "doc:unknown"
        # Strip chunk suffixes so ``file:contract.pdf#chunk-0`` and
        # ``file:contract.pdf#chunk-1`` collapse to one outline entry.
        base_uri = uri.split("#chunk-", 1)[0]
        if base_uri not in by_uri:
            by_uri[base_uri] = _DocOutline(display_name=_strip_uri_scheme(base_uri), headings=[])
            seen_titles[base_uri] = set()
            order.append(base_uri)
        outline = by_uri[base_uri]
        seen = seen_titles[base_uri]

        title = getattr(p, "section_title", None)
        if not title:
            continue
        title = title.strip()
        if not title or title in seen:
            continue
        if len(outline.headings) >= _OUTLINE_MAX_HEADINGS_PER_DOC:
            continue
        seen.add(title)
        page = getattr(p, "page", None)
        outline.headings.append((title, 1, page))

    return [by_uri[u] for u in order]


def _doc_display_uri(doc: Any) -> str:
    """Best-effort URI for a ``ContentDocument`` — falls back gracefully."""
    md = getattr(doc, "metadata", None)
    src = getattr(md, "source", None) if md is not None else None
    uri = getattr(src, "uri", None) if src is not None else None
    if uri:
        return str(uri)
    extra = getattr(md, "extra", None) if md is not None else None
    if isinstance(extra, dict):
        for key in ("uri", "source_uri", "source"):
            v = extra.get(key)
            if isinstance(v, str) and v:
                return v
    return "doc:unknown"


def _strip_uri_scheme(uri: str) -> str:
    """Render a URI as a short display name. ``file:contract.pdf`` →
    ``contract.pdf``; ``doc:my-report`` → ``my-report``."""
    for scheme in ("file:", "doc:"):
        if uri.startswith(scheme):
            return uri[len(scheme) :] or uri
    return uri


_OUTLINE_DISCLAIMER = (
    "Notes: this outline lists section headings only. "
    "Content that appears only inside paragraph bodies (definitions, "
    "footnotes, untitled subsections, body text without a heading) is "
    "still searchable but is NOT shown here. Absence from the outline "
    "does NOT mean absence from the document. Use this outline to plan "
    "your search; do not refuse on the basis of the outline alone."
)


def _format_outline(
    docs: Sequence[_DocOutline],
    *,
    max_depth: int | None = None,
) -> str:
    """Render the outline. ``max_depth`` truncates deep nesting."""
    parts: list[str] = [
        f"[CORPUS OUTLINE] ({len(docs)} document(s))",
        "",
    ]
    for d in docs:
        parts.append(f"{d.display_name}:")
        rendered_any = False
        n = 0
        for text, depth, page in d.headings:
            if max_depth is not None and depth > max_depth:
                continue
            n += 1
            # Indent by depth (depth 1 → "  ", depth 2 → "    ", ...).
            indent = "  " * max(1, depth)
            label = text[:_OUTLINE_MAX_HEADING_CHARS]
            page_str = f" (p.{page})" if page else ""
            parts.append(f"{indent}{n}. {label}{page_str}")
            rendered_any = True
        if not rendered_any:
            parts.append("  (no headings)")
        parts.append("")
    parts.append(_OUTLINE_DISCLAIMER)
    return "\n".join(parts)


def _format_outline_skeleton(docs: Sequence[_DocOutline]) -> str:
    """Last-resort outline: doc names + section count only."""
    parts: list[str] = [
        f"[CORPUS OUTLINE] ({len(docs)} document(s))",
        "",
    ]
    for d in docs:
        parts.append(f"{d.display_name}: {len(d.headings)} section(s)")
    parts.append("")
    parts.append(_OUTLINE_DISCLAIMER)
    return "\n".join(parts)
