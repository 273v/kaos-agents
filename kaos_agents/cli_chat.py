"""CLI for conversing with the KAOS agent — interactive REPL or one-shot.

Usage::

    # Interactive REPL (default when no --message is given)
    kaos-agent chat
    kaos-agent chat --session my-session --verbose
    kaos-agent chat --tools "kaos-source-*,kaos-pdf-*" --model anthropic:claude-sonnet-4-6

    # Load documents at startup for RAG Q&A
    kaos-agent chat --files "contracts/*.pdf,memos/*.docx" --verbose
    kaos-agent chat --files "*.pdf" --pattern research

    # One-shot non-interactive mode (scripts, CI, course runnables)
    kaos-agent chat --message "What is 2+2?" --max-cost 0.05
    echo "extract parties from this contract" | kaos-agent chat --message -
    kaos-agent chat --message "summarize" --files "report.pdf" --max-cost 0.20

Slash commands inside the REPL:

    /quit     — exit
    /session  — print current session ID
    /tools    — list registered tools
    /memory   — dump memory section summaries
    /clear    — clear session memory
    /verbose  — toggle verbose event display
    /load <path>  — load file, glob, or folder
                    (PDF, DOCX, PPTX, XLSX, HTML, TXT, MD, CSV, JSON, EML)
    /docs     — list loaded documents

Budget enforcement: ``--max-cost`` (or env ``KAOS_AGENT_MAX_COST_USD``)
sets a hard session cost ceiling in USD. Every turn checks the running
session total against the cap *after* the turn completes; the next
turn is refused when the cap is exceeded, and non-interactive mode
exits with code 2. Set to 0 to disable.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import hashlib
import json
import os
import sys
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Non-interactive mode exit codes. 0 success, 1 runtime error, 2 budget
# exceeded. Chosen to match the convention set by validate-platform.sh
# (2 for "well-formed but intentional refusal").
_EXIT_OK = 0
_EXIT_ERROR = 1
_EXIT_BUDGET = 2


@dataclass
class _ExplainTurn:
    """Per-turn explanation record — the data behind ``/explain``.

    Captures everything the user needs to understand why an agent
    answer landed the way it did: what was retrieved (URI + score per
    citation), what was cited (claim + verifier confidence), per-tool
    latency + cost, total tokens by step. JSON-serializable (all
    primitives + lists) so ``--explain <file>`` can dump it for the
    caller to grep / pipe / save.
    """

    turn_index: int
    user_message: str
    intent: str = ""
    intent_confidence: float = 0.0
    text: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    citations: list[dict[str, Any]] = field(default_factory=list)
    refusals: list[dict[str, Any]] = field(default_factory=list)
    errors: list[dict[str, Any]] = field(default_factory=list)
    tokens_used: int = 0
    cost_usd: float = 0.0
    duration_s: float = 0.0


@dataclass
class _SessionState:
    """Mutable session-level running totals + budget decision state.

    Consolidates what were free-floating locals in ``_run_repl`` so the
    same helper can drive the REPL loop and the one-shot
    ``--message`` path without twinning state-update logic.
    """

    tokens: int = 0
    cost_usd: float = 0.0
    turns: int = 0
    max_cost_usd: float | None = None  # None = no cap
    alert_cost_usd: float | None = None  # None = no alert
    alerted: bool = False  # Latch — only fire the alert once per run.
    # Persistent per-turn explain records — N3 / P9. List indexes turn
    # number 1..N. /explain shows the most recent; /explain N shows
    # turn N. ``--explain <file>`` writes the full list as JSON.
    explain_turns: list[_ExplainTurn] = field(default_factory=list)

    def absorb(self, tokens: int, cost: float) -> None:
        self.tokens += tokens
        self.cost_usd += cost
        self.turns += 1

    def budget_exceeded(self) -> bool:
        """After-the-fact check. We let the current turn finish — budgets
        only block the *next* turn, matching how Budget works in
        planning/compose.py. Set ``max_cost_usd=0`` to disable the cap."""
        return (
            self.max_cost_usd is not None
            and self.max_cost_usd > 0
            and self.cost_usd >= self.max_cost_usd
        )

    def alert_due(self) -> bool:
        """Returns True the first time session cost crosses the alert
        threshold. Latched (``alerted=True``) so we only print once.
        Complement to ``budget_exceeded`` — alert is a soft warning
        with no behavior change; budget hard-stops the next turn."""
        if self.alert_cost_usd is None or self.alert_cost_usd <= 0:
            return False
        if self.alerted:
            return False
        if self.cost_usd >= self.alert_cost_usd:
            self.alerted = True
            return True
        return False


def _explain_to_dict(turn: _ExplainTurn) -> dict[str, Any]:
    """Convert an _ExplainTurn into a plain dict for JSON dumping.

    Kept here (rather than as a method on _ExplainTurn) so the dataclass
    stays pure-data. The shape mirrors what the agent's TurnComplete
    + per-event records carry, with all primitives — no objects whose
    JSON shape might shift across releases.
    """
    return {
        "turn_index": turn.turn_index,
        "user_message": turn.user_message,
        "intent": turn.intent,
        "intent_confidence": turn.intent_confidence,
        "text": turn.text,
        "tool_calls": turn.tool_calls,
        "citations": turn.citations,
        "refusals": turn.refusals,
        "errors": turn.errors,
        "tokens_used": turn.tokens_used,
        "cost_usd": turn.cost_usd,
        "duration_s": turn.duration_s,
    }


def _print_explain(turn: _ExplainTurn) -> None:
    """Pretty-print a per-turn explain record to stdout.

    Layout:
        Turn N — intent (confidence) — Xs, $Y.YYYY, K tokens
        ► User: <message>
        ► Tools (N):
            <tool_name> — <duration>ms — $cost — <error?>
        ► Citations (N):
            ✓ verified (0.95) — <claim 60>
                URI / node_ref / page
        ► Refusals (N):
            ⚠ <reason>
        ► Errors (N):
            ✖ <error_type>: <message>
        ► Answer: <text>
    """
    header = (
        f"Turn {turn.turn_index} — {turn.intent or '?'} "
        f"(confidence={turn.intent_confidence:.2f}) — "
        f"{turn.duration_s:.1f}s, ${turn.cost_usd:.4f}, {turn.tokens_used} tokens"
    )
    print(_c(_ANSI_BOLD, header))
    print(_c(_ANSI_DIM, f"  ► User: {turn.user_message[:200]}"))
    if turn.tool_calls:
        print(_c(_ANSI_CYAN, f"  ► Tools ({len(turn.tool_calls)}):"))
        for tc in turn.tool_calls:
            cost = tc.get("cost_usd", 0.0) or 0.0
            cost_part = f" — ${cost:.4f}" if cost > 0 else ""
            err_part = " — ERROR" if tc.get("is_error") else ""
            preview = (tc.get("preview") or "")[:80]
            print(f"    {tc['tool_name']} ({tc.get('duration_ms', 0):.0f}ms){cost_part}{err_part}")
            if preview:
                print(_c(_ANSI_DIM, f"      → {preview}"))
    if turn.citations:
        print(_c(_ANSI_CYAN, f"  ► Citations ({len(turn.citations)}):"))
        for c in turn.citations:
            v = "✓" if c.get("verified") else "?"
            print(f"    {v} ({c.get('confidence', 0.0):.2f}) — {(c.get('claim') or '')[:80]}")
            uri = c.get("source_uri") or ""
            ref = c.get("node_ref") or ""
            page = c.get("page")
            tail_parts = []
            if uri:
                tail_parts.append(uri)
            if ref:
                tail_parts.append(ref)
            if page is not None:
                tail_parts.append(f"page {page}")
            if tail_parts:
                print(_c(_ANSI_DIM, f"      {' / '.join(tail_parts)}"))
    if turn.refusals:
        print(_c(_ANSI_YELLOW, f"  ► Refusals ({len(turn.refusals)}):"))
        for r in turn.refusals:
            print(f"    ⚠ {r.get('kind', '?')}: {(r.get('reason') or '')[:120]}")
    if turn.errors:
        print(_c(_ANSI_RED, f"  ► Errors ({len(turn.errors)}):"))
        for e in turn.errors:
            print(f"    ✖ {e.get('error_type', '?')}: {e.get('message', '')[:120]}")
    if turn.text:
        print(_c(_ANSI_DIM, f"  ► Answer: {turn.text[:200]}"))


def _resolve_corpus_cache(args: argparse.Namespace) -> Path | None:
    """Resolve the corpus cache directory from CLI args.

    Returns ``None`` when caching is disabled (no ``--corpus-cache``,
    or ``--no-cache`` set, or empty string explicitly passed). Creates
    the directory on first use so the caller doesn't have to. Empty
    string opts out — matches the user-task spec ("--corpus-cache=
    empty as opt-out") and the convention used elsewhere in KAOS for
    "set the env var to '' to disable"."""
    if getattr(args, "no_cache", False):
        return None
    raw = getattr(args, "corpus_cache", None)
    if raw is None or not str(raw).strip():
        return None
    cache_dir = Path(raw).expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def _resolve_max_cost(cli_value: float | None) -> float | None:
    """``--max-cost`` with env-var fallback. Returns ``None`` when
    unset (no cap). Negative / zero disables the cap explicitly."""
    if cli_value is not None:
        return cli_value if cli_value > 0 else None
    raw = os.environ.get("KAOS_AGENT_MAX_COST_USD")
    if raw is None:
        return None
    try:
        value = float(raw)
    except ValueError:
        return None
    return value if value > 0 else None


def _resolve_alert_cost(cli_value: float | None) -> float | None:
    """``--alert-cost`` with env-var fallback. Same precedence rules as
    ``_resolve_max_cost``: CLI > env > None. Negative/zero disables.

    The alert is a soft warning printed once when cumulative session
    cost crosses the threshold. Behavior is unchanged afterwards —
    pair with ``--max-cost`` for a hard ceiling."""
    if cli_value is not None:
        return cli_value if cli_value > 0 else None
    raw = os.environ.get("KAOS_AGENT_ALERT_COST_USD")
    if raw is None:
        return None
    try:
        value = float(raw)
    except ValueError:
        return None
    return value if value > 0 else None


_ANSI_BOLD = "\033[1m"
_ANSI_DIM = "\033[2m"
_ANSI_CYAN = "\033[36m"
_ANSI_YELLOW = "\033[33m"
_ANSI_GREEN = "\033[32m"
_ANSI_RED = "\033[31m"
_ANSI_RESET = "\033[0m"

_SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".docx",
    ".pptx",
    ".xlsx",
    ".html",
    ".htm",
    ".txt",
    ".md",
    ".json",
    ".csv",
    ".eml",
}


def _c(code: str, text: str) -> str:
    """Colorize text if stdout is a TTY."""
    if not sys.stdout.isatty():
        return text
    return f"{code}{text}{_ANSI_RESET}"


def _parse_file_to_document(file_path: Path) -> Any:
    """Parse a file to a ContentDocument AST. Preserves full structure.

    Dispatches by extension using kaos-pdf, kaos-office, kaos-content parsers.
    For plain text formats, wraps in a ContentDocument via parse_plain_text.
    Returns the ContentDocument (not serialized text).
    """
    from kaos_content.model.metadata import SourceRef
    from kaos_content.parsers.plain import parse_plain_text

    ext = file_path.suffix.lower()

    source = SourceRef(uri=file_path.as_uri())

    # PDF → kaos-pdf
    # `merge_column_paragraphs=True` coalesces column-wrap line-rects in
    # multi-column publications (Federal Register, GPO bulletins, court
    # reporters) into flowing paragraphs. Without it, the agent's RAG
    # retrieval sees one rect per visual line and cites garbled
    # cross-column fragments. NLP-driven downstream is the agent CLI's
    # use case, so we default to merging.
    if ext == ".pdf":
        from kaos_pdf import extract_pdf

        return extract_pdf(file_path, merge_column_paragraphs=True)

    # DOCX → kaos-office
    if ext == ".docx":
        from kaos_office import parse_docx

        return parse_docx(file_path)

    # PPTX → kaos-office
    if ext == ".pptx":
        from kaos_office.pptx.reader import parse_pptx

        return parse_pptx(file_path)

    # HTML/HTM → kaos-content html parser
    if ext in (".html", ".htm"):
        from kaos_content.parsers.html import parse_html

        raw = file_path.read_text(encoding="utf-8", errors="replace")
        return parse_html(raw, url=file_path.as_uri())

    # Markdown → kaos-content markdown parser
    if ext == ".md":
        from kaos_content.parsers.markdown import parse_markdown

        raw = file_path.read_text(encoding="utf-8", errors="replace")
        return parse_markdown(raw, source=source)

    # Plain text, CSV, JSON, EML, XLSX — wrap as plain text ContentDocument
    if ext in (".txt", ".csv", ".json", ".eml", ".xlsx"):
        raw = file_path.read_text(encoding="utf-8", errors="replace")
        return parse_plain_text(raw, source=source)

    # Fallback: try reading as plain text
    try:
        raw = file_path.read_text(encoding="utf-8", errors="replace")
        if raw.strip():
            return parse_plain_text(raw, source=source)
    except Exception:
        pass

    msg = (
        f"Could not parse {ext} file. "
        f"Supported formats: {', '.join(sorted(_SUPPORTED_EXTENSIONS))}. "
        f"Alternative: convert to PDF or plain text first."
    )
    raise ValueError(msg)


def _parse_file(file_path: Path) -> tuple[str, str]:
    """Parse a file to plain text. Returns (text, uri).

    Convenience wrapper over _parse_file_to_document for backward compat.
    Prefer _parse_file_to_document to preserve the AST.
    """
    from kaos_content import serialize_text

    doc = _parse_file_to_document(file_path)
    uri = f"file:{file_path.name}"
    return serialize_text(doc), uri


def _default_load_workers() -> int:
    """Default thread count for parallel document loading.

    ``max(2, cpu/2)`` — half the cores on a typical laptop, never less
    than 2. PDFium's global lock means we don't get linear speedup on
    PDF-heavy corpora, but file IO + DOCX/HTML/TXT parsing still
    parallelize cleanly. The agent CLI's actual workloads (deal-room
    PDFs mixed with DOCX exhibits) typically see 2-4x throughput at
    this setting; pushing higher hurts on PDFium-bound batches.
    """
    return max(2, (os.cpu_count() or 4) // 2)


def _hash_file_bytes(file_path: Path, *, chunk_size: int = 1 << 20) -> str:
    """SHA-256 of the file contents. Streamed so 100 MB PDFs don't
    spike memory. Used as the cache key for ``--corpus-cache``."""
    h = hashlib.sha256()
    with file_path.open("rb") as fh:
        while True:
            buf = fh.read(chunk_size)
            if not buf:
                break
            h.update(buf)
    return h.hexdigest()


def _cache_key(file_path: Path, chunk_size: int) -> str:
    """Stable cache key combining content hash and chunk size.

    Keying on bytes (not path/mtime) means re-running the agent on a
    moved or copied file is still a cache hit; renaming
    ``contract.pdf`` → ``contract-final.pdf`` doesn't repay the parse
    cost. Including ``chunk_size`` means flipping ``--chunk-size`` is
    a clean miss instead of returning stale chunks at the wrong grain.
    """
    return f"{_hash_file_bytes(file_path)}:{chunk_size}"


def _cache_paths(cache_dir: Path, key: str) -> tuple[Path, Path]:
    """Return (blob_path, index_path) for a cache key.

    Blobs live under ``<cache>/blobs/<key>.json`` (one JSON document
    per cache entry holding the chunk list). The index lives at
    ``<cache>/INDEX.json`` and is the human-readable manifest.
    """
    blobs_dir = cache_dir / "blobs"
    return blobs_dir / f"{key}.json", cache_dir / "INDEX.json"


# Global lock around index writes — many threads may hit the same
# corpus-cache concurrently and we'd corrupt INDEX.json without it.
_CACHE_INDEX_LOCK = threading.Lock()


def _cache_load(cache_dir: Path, key: str) -> list[Any] | None:
    """Try to load a chunk list from cache. Returns None on miss or
    corruption (a corrupted entry is treated as a miss — we'll re-parse
    and overwrite). Each chunk round-trips through Pydantic v2's
    ``model_validate_json``; the frozen ContentDocument model
    guarantees the deserialized chunks are byte-identical to the
    originals modulo dict ordering inside ``metadata.extra``."""
    from kaos_content.model.document import ContentDocument

    blob_path, _ = _cache_paths(cache_dir, key)
    if not blob_path.exists():
        return None
    try:
        with blob_path.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
        chunks_json = payload.get("chunks", [])
        return [ContentDocument.model_validate_json(c) for c in chunks_json]
    except (json.JSONDecodeError, OSError, ValueError):
        # Corrupted blob — treat as miss so the caller re-parses.
        return None


def _cache_store(
    cache_dir: Path,
    key: str,
    chunks: list[Any],
    *,
    file_uri: str,
    chunk_size: int,
    source_path: Path,
) -> None:
    """Persist a chunk list to cache and update INDEX.json.

    Atomic per-blob via tmp + os.replace, so a Ctrl-C mid-write doesn't
    leave a half-written entry that future runs treat as valid. The
    index write is guarded by ``_CACHE_INDEX_LOCK`` so parallel loads
    don't race."""
    blob_path, index_path = _cache_paths(cache_dir, key)
    blob_path.parent.mkdir(parents=True, exist_ok=True)

    chunks_json = [c.model_dump_json() for c in chunks]
    payload = {
        "key": key,
        "file_uri": file_uri,
        "chunk_size": chunk_size,
        "source": str(source_path),
        "n_chunks": len(chunks),
        "chunks": chunks_json,
    }
    tmp_path = blob_path.with_suffix(".json.tmp")
    with tmp_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh)
    tmp_path.replace(blob_path)

    # Update the human-readable index. Best-effort — failure here
    # doesn't break the cache (blobs are the source of truth).
    with _CACHE_INDEX_LOCK:
        index: dict[str, Any] = {}
        if index_path.exists():
            try:
                with index_path.open("r", encoding="utf-8") as fh:
                    index = json.load(fh)
            except (json.JSONDecodeError, OSError):
                index = {}
        try:
            mtime = source_path.stat().st_mtime
        except OSError:
            mtime = 0.0
        index[key] = {
            "file_uri": file_uri,
            "chunk_size": chunk_size,
            "mtime": mtime,
            "n_chunks": len(chunks),
            "source": str(source_path),
        }
        index_tmp = index_path.with_suffix(".json.tmp")
        with contextlib.suppress(OSError):
            with index_tmp.open("w", encoding="utf-8") as fh:
                json.dump(index, fh, indent=2, sort_keys=True)
            index_tmp.replace(index_path)


def _chunk_document(
    doc: Any,
    *,
    chunk_size: int,
    chunker_available: bool,
) -> list[Any]:
    """Run a parsed ContentDocument through SectionChunker.

    Mirrors the inline logic that used to live in both loader
    functions — pulled out so cache-hit and cache-miss paths share one
    implementation, and so the parallel and serial paths share one
    implementation. Returns ``[doc]`` (the whole document) when
    chunking is disabled or unavailable, matching the legacy
    one-doc-per-file behavior. Never raises ImportError to the caller
    — falls back to the unchunked doc if ``[nlp]`` is missing.
    """
    if not chunker_available or chunk_size <= 0:
        return [doc]
    from kaos_content.chunking import SectionChunker

    try:
        return list(
            SectionChunker.from_outline(
                doc,
                max_chars=chunk_size,
                split_depth=2,
                promote_inferred=True,
            )
        )
    except ImportError:
        # ``[nlp]`` extra missing — promote_inferred raises. Fall back
        # to literal SectionChunker over typed Headings only.
        chunker = SectionChunker(max_chars=chunk_size, split_depth=2)
        return list(chunker.chunk(doc))


def _parse_and_chunk_one(
    fp: Path,
    *,
    chunk_size: int,
    chunker_available: bool,
    cache_dir: Path | None,
) -> tuple[list[Any], bool]:
    """Parse + chunk one file, honoring the cache.

    Returns ``(chunks, cache_hit)``. Raises any underlying parse error
    so the caller can attribute the failure to this specific file
    (parallel loaders need to know which future failed).
    """
    file_uri = f"file:{fp.name}"
    cache_key: str | None = None
    if cache_dir is not None:
        cache_key = _cache_key(fp, chunk_size)
        cached = _cache_load(cache_dir, cache_key)
        if cached is not None:
            return cached, True

    doc = _parse_file_to_document(fp)
    chunks = _chunk_document(doc, chunk_size=chunk_size, chunker_available=chunker_available)

    if cache_dir is not None and cache_key is not None and chunks:
        # Best-effort cache write — don't tank the load if the
        # filesystem is full or read-only.
        with contextlib.suppress(OSError):
            _cache_store(
                cache_dir,
                cache_key,
                chunks,
                file_uri=file_uri,
                chunk_size=chunk_size,
                source_path=fp,
            )
    return chunks, False


# Lock around stdout so parallel loaders don't interleave progress
# lines mid-character. Per-print acquisition is fine — these prints
# are infrequent enough that lock contention is invisible.
_PRINT_LOCK = threading.Lock()


def _safe_print(msg: str) -> None:
    """Thread-safe print. Only used by parallel loader paths."""
    with _PRINT_LOCK:
        print(msg)


def _load_files_to_corpus(
    file_paths: list[Path],
    *,
    verbose: bool = False,
    chunk_size: int = 8000,
    workers: int | None = None,
    cache_dir: Path | None = None,
) -> tuple[Any, list[str]]:
    """Parse files into ContentDocuments, chunk them, and build a
    ContentDocumentCorpus over the chunks.

    Each input file is loaded → run through ``with_inferred_structure``
    (T3c heading promotion via the kaos-nlp-core P7 layer when ``[nlp]``
    is installed) → split via ``SectionChunker.from_outline`` (T4b
    outline-aware chunking) into bounded chunks of ≤ ``chunk_size``
    chars. Each chunk is itself a ``ContentDocument`` carrying:

    * ``metadata.extra['heading_path']`` — the section breadcrumb at the
      chunk's entry point, so the agent's downstream search can show
      ``"Contract A > Section 4 > §4.2"`` instead of just ``"Contract A"``.
    * ``metadata.extra['chunk_index']`` and ``chunk_total``.
    * The chunk's blocks, with their original block_refs preserved (the
      chunk inherits node_refs from the source doc — addresses survive).

    Each chunk is registered in ``ContentDocumentCorpus`` as its own URI:
    ``file:<name>#chunk-<index>``. This makes BM25 retrieval grain-
    appropriate for legal Q&A — a 50KB contract becomes ~30 retrievable
    chunks instead of ~5 unbounded paragraph blocks.

    Set ``chunk_size=0`` to disable chunking (legacy behavior — emit one
    corpus entry per file). Useful when documents are already small or
    when callers want strictly literal one-doc-per-uri retrieval.

    Parallel loading: ``workers`` controls the ``ThreadPoolExecutor``
    size. Defaults to ``max(2, cpu/2)`` per ``_default_load_workers``.
    PDFium calls are serialized through a global lock inside kaos-pdf,
    so PDF-heavy corpora don't see linear speedup, but file IO and
    DOCX/HTML/TXT parsing parallelize cleanly. Order of ``documents``
    and ``uris`` matches the input ``file_paths`` order regardless of
    completion order.

    Persistent cache: when ``cache_dir`` is provided, parsed+chunked
    documents are keyed by ``sha256(file_bytes):chunk_size`` so
    re-running over the same corpus is instant. Cache invalidates
    automatically when file contents change. See ``_cache_load`` /
    ``_cache_store`` for the format.

    Returns (corpus, uris) where corpus is a ContentDocumentCorpus with
    passage-level retrieval, and uris is the list of document URIs (one
    per chunk when chunking is enabled, one per file when disabled).
    Returns (None, []) if no documents loaded.
    """
    from kaos_content.corpus import ContentDocumentCorpus

    n = len(file_paths)
    if n == 0:
        return None, []

    # Lazy-import the chunker — when ``[nlp]`` is missing,
    # ``with_inferred_structure`` raises ``ImportError`` on first call
    # and we fall back to whole-doc emission.
    chunker_available = False
    if chunk_size > 0:
        try:
            from kaos_content.chunking import SectionChunker  # noqa: F401

            chunker_available = True
        except ImportError:
            chunker_available = False

    if workers is None or workers < 1:
        workers = _default_load_workers()
    # No point spinning up more threads than files.
    workers = min(workers, max(1, n))

    # Per-index slot so we can preserve input order regardless of
    # completion order. Each slot ends up holding either a chunk list
    # (success — possibly empty) or None (failure / missing file).
    results: list[list[Any] | None] = [None] * n
    cache_hits: list[bool] = [False] * n

    def _worker(i: int, fp_raw: Path) -> tuple[int, list[Any] | None, bool, str | None]:
        """Returns (index, chunks_or_None, cache_hit, error_msg)."""
        fp = fp_raw.resolve()
        if not fp.exists():
            return i, None, False, f"  File not found: {fp}"
        try:
            chunks, hit = _parse_and_chunk_one(
                fp,
                chunk_size=chunk_size,
                chunker_available=chunker_available,
                cache_dir=cache_dir,
            )
            if verbose:
                tag = " (cache HIT)" if hit else ""
                _safe_print(
                    _c(
                        _ANSI_DIM,
                        f"  Loaded {fp.name}{tag} → {len(chunks)} chunk(s)"
                        f" @ ≤{chunk_size or 'unbounded'} chars",
                    )
                )
            return i, chunks, hit, None
        except ImportError as exc:
            return i, None, False, f"  Missing parser for {fp.suffix}: {exc}"
        except Exception as exc:
            # One bad PDF should not tank the whole load. Log and
            # leave the slot empty; caller proceeds with the rest.
            return i, None, False, f"  Failed {fp.name}: {exc}"

    if workers == 1:
        # Serial path — keeps stack traces tidy when the user
        # explicitly asks for ``--load-workers 1`` to debug an
        # extractor. Functionally equivalent to the parallel path
        # with workers=1 but skips the executor overhead.
        for i, fp_raw in enumerate(file_paths):
            idx, chunks, hit, err = _worker(i, fp_raw)
            results[idx] = chunks
            cache_hits[idx] = hit
            if err is not None:
                print(_c(_ANSI_RED, err))
    else:
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="kaos-load") as pool:
            futures = [pool.submit(_worker, i, fp_raw) for i, fp_raw in enumerate(file_paths)]
            for fut in as_completed(futures):
                idx, chunks, hit, err = fut.result()
                results[idx] = chunks
                cache_hits[idx] = hit
                if err is not None:
                    _safe_print(_c(_ANSI_RED, err))

    documents: list[Any] = []
    uris: list[str] = []
    for i, fp in enumerate(file_paths):
        chunks = results[i]
        if not chunks:
            continue
        file_uri = f"file:{fp.name}"
        if chunker_available and chunk_size > 0:
            for j, chunk in enumerate(chunks):
                documents.append(chunk)
                chunk_idx = chunk.metadata.extra.get("chunk_index", j)
                uris.append(f"{file_uri}#chunk-{chunk_idx}")
        else:
            documents.append(chunks[0])
            uris.append(file_uri)

    if verbose and cache_dir is not None:
        n_hits = sum(1 for h in cache_hits if h)
        if n_hits:
            print(
                _c(
                    _ANSI_DIM,
                    f"  Cache: {n_hits}/{n} hit, {n - n_hits} miss (cache_dir={cache_dir})",
                )
            )

    if not documents:
        return None, []

    corpus = ContentDocumentCorpus(documents, doc_uris=uris)
    return corpus, uris


def _load_files_into_memory(
    file_paths: list[Path],
    memory: Any,
    *,
    verbose: bool = False,
    chunk_size: int = 8000,
    workers: int | None = None,
    cache_dir: Path | None = None,
) -> int:
    """Parse files, chunk them, and add each chunk to DOCUMENTS memory.

    Each input file is loaded → run through ``with_inferred_structure``
    + ``SectionChunker.from_outline`` (when ``[nlp]`` is installed) into
    bounded chunks of ≤ ``chunk_size`` chars; each chunk's text is added
    as a separate ``DOCUMENTS`` memory entry with a unique URI suffix.

    Set ``chunk_size=0`` to disable chunking (legacy behavior — one
    memory entry per file with the full document text).

    Parallel loading + persistent cache (``workers``, ``cache_dir``)
    work the same as ``_load_files_to_corpus`` — see that docstring for
    semantics. Memory writes are still serialized in the main thread
    (after the parallel parse + chunk completes) to avoid touching
    ``SessionMemory`` from worker threads.

    Returns the number of files (not chunks) that produced content.

    .. deprecated::
        Prefer _load_files_to_corpus which preserves the ContentDocument
        AST for passage-level retrieval. This function flattens to text
        strings; consumers that rely on the AST should migrate.
    """
    from kaos_content import serialize_text

    from kaos_agents.memory.types import MemoryType

    n = len(file_paths)
    if n == 0:
        return 0

    chunker_available = chunk_size > 0
    if chunker_available:
        try:
            from kaos_content.chunking import SectionChunker  # noqa: F401
        except ImportError:
            chunker_available = False

    if workers is None or workers < 1:
        workers = _default_load_workers()
    workers = min(workers, max(1, n))

    results: list[list[Any] | None] = [None] * n
    cache_hits: list[bool] = [False] * n

    def _worker(i: int, fp_raw: Path) -> tuple[int, list[Any] | None, bool, str | None]:
        fp = fp_raw.resolve()
        if not fp.exists():
            return i, None, False, f"  File not found: {fp}"
        try:
            chunks, hit = _parse_and_chunk_one(
                fp,
                chunk_size=chunk_size,
                chunker_available=chunker_available,
                cache_dir=cache_dir,
            )
            return i, chunks, hit, None
        except ImportError as exc:
            return i, None, False, f"  Missing parser for {fp.suffix}: {exc}"
        except Exception as exc:
            return i, None, False, f"  Failed {fp.name}: {exc}"

    if workers == 1:
        for i, fp_raw in enumerate(file_paths):
            idx, chunks, hit, err = _worker(i, fp_raw)
            results[idx] = chunks
            cache_hits[idx] = hit
            if err is not None:
                print(_c(_ANSI_RED, err))
    else:
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="kaos-load-mem") as pool:
            futures = [pool.submit(_worker, i, fp_raw) for i, fp_raw in enumerate(file_paths)]
            for fut in as_completed(futures):
                idx, chunks, hit, err = fut.result()
                results[idx] = chunks
                cache_hits[idx] = hit
                if err is not None:
                    _safe_print(_c(_ANSI_RED, err))

    loaded = 0
    for i, fp_raw in enumerate(file_paths):
        chunks = results[i]
        if chunks is None:
            continue
        fp = fp_raw.resolve()
        uri = f"file:{fp.name}"
        chunks_added = 0
        total_chars = 0
        if chunker_available and chunk_size > 0:
            for j, chunk in enumerate(chunks):
                chunk_text = serialize_text(chunk)
                if not chunk_text.strip():
                    continue
                chunk_idx = chunk.metadata.extra.get("chunk_index", j)
                heading_path = chunk.metadata.extra.get("heading_path", [])
                memory.add(
                    MemoryType.DOCUMENTS,
                    chunk_text,
                    metadata={
                        "uri": f"{uri}#chunk-{chunk_idx}",
                        "source": str(fp),
                        "heading_path": heading_path,
                        "chunk_index": chunk_idx,
                    },
                )
                chunks_added += 1
                total_chars += len(chunk_text)
        else:
            doc = chunks[0]
            text = serialize_text(doc)
            if not text.strip():
                print(_c(_ANSI_YELLOW, f"  Empty: {fp.name}"))
                continue
            memory.add(
                MemoryType.DOCUMENTS,
                text,
                metadata={"uri": uri, "source": str(fp)},
            )
            chunks_added = 1
            total_chars = len(text)
        if chunks_added == 0:
            print(_c(_ANSI_YELLOW, f"  Empty: {fp.name}"))
            continue
        loaded += 1
        if verbose:
            tag = " (cache HIT)" if cache_hits[i] else ""
            print(
                _c(
                    _ANSI_DIM,
                    f"  Loaded {fp.name}{tag} ({total_chars:,} chars → {chunks_added} chunk(s))",
                )
            )

    if verbose and cache_dir is not None:
        n_hits = sum(1 for h in cache_hits if h)
        if n_hits:
            print(
                _c(
                    _ANSI_DIM,
                    f"  Cache: {n_hits}/{n} hit, {n - n_hits} miss (cache_dir={cache_dir})",
                )
            )

    return loaded


def _one_shot_message(args: argparse.Namespace) -> str | None:
    """Resolve the one-shot message from CLI args.

    Returns ``None`` when no ``--message`` was supplied (REPL mode).
    Supports ``--message -`` for reading from stdin, matching the
    convention used by ``kaos-office`` writer CLIs. Empty messages
    produce ``None`` — we don't send empty prompts to the agent.
    """
    raw = getattr(args, "message", None)
    if raw is None:
        return None
    if raw == "-":
        raw = sys.stdin.read()
    stripped = raw.strip()
    return stripped or None


async def _run_repl(args: argparse.Namespace) -> _SessionState:
    from kaos_core.registry.container import KaosRuntime

    from kaos_agents.config import Agent
    from kaos_agents.events import (
        CitationFound,
        EvidenceInsufficient,
        GroundingRefusalTriggered,
        IntentClassified,
        PlanProposed,
        RunError,
        StepComplete,
        StepStart,
        TextDelta,
        ThinkingDelta,
        ToolCallResult,
        ToolCallStart,
        TurnComplete,
        TurnStart,
    )
    from kaos_agents.runner import Runner

    runtime = KaosRuntime.default()

    from kaos_agents.memory.session import SessionMemory
    from kaos_agents.memory.types import MemoryType
    from kaos_agents.tools import register_agent_tools

    register_agent_tools(runtime)

    _register_tool_modules(runtime, args)

    session_id = args.session or f"cli-{uuid.uuid4().hex[:8]}"
    verbose = args.verbose
    memory = SessionMemory(session_id)
    corpus = None  # ContentDocumentCorpus for passage-level retrieval

    # Pre-load files from --files flag.
    # Each pattern may be:
    #   • an absolute path to a file (Python 3.13's Path.cwd().glob raises
    #     NotImplementedError on absolute patterns, so we route them
    #     directly without going through cwd).
    #   • a relative glob pattern resolved against the current working
    #     directory.
    #   • a directory path → load all supported files inside it.
    if args.files:
        file_paths: list[Path] = []
        for raw in args.files.split(","):
            pat = raw.strip()
            if not pat:
                continue
            p = Path(pat)
            if p.is_absolute():
                if p.is_dir():
                    file_paths.extend(sorted(c for c in p.rglob("*") if c.is_file()))
                elif p.exists():
                    file_paths.append(p)
                else:
                    # Absolute glob pattern: split into anchor + relative
                    # part. ``Path("/").glob("home/user/*.pdf")`` works
                    # (relative pattern from anchor) where
                    # ``Path.cwd().glob("/home/user/*.pdf")`` doesn't.
                    rel = pat.lstrip("/")
                    file_paths.extend(Path("/").glob(rel))
            else:
                file_paths.extend(Path.cwd().glob(pat))
        if file_paths:
            print(f"Loading {len(file_paths)} file(s)...")
            chunk_size = getattr(args, "chunk_size", 8000)
            workers = getattr(args, "load_workers", None)
            cache_dir = _resolve_corpus_cache(args)
            corpus, uris = _load_files_to_corpus(
                file_paths,
                verbose=verbose,
                chunk_size=chunk_size,
                workers=workers,
                cache_dir=cache_dir,
            )
            if corpus is not None:
                n_passages = corpus.size
                print(f"  {len(uris)} documents → {n_passages} passages (corpus ready)")

    # Auto-select pattern: if corpus loaded, use research
    pattern = args.pattern
    if corpus is not None and pattern == "chat":
        pattern = "research"
        if verbose:
            print(_c(_ANSI_CYAN, f"  Auto-switched to research pattern ({corpus.size} passages)"))

    tools_tuple = tuple(t.strip() for t in args.tools.split(",")) if args.tools else ()
    agent = Agent.create(
        instructions=args.instructions or "You are a helpful assistant.",
        model=args.model,
        pattern=pattern,
        tools=tools_tuple,
    )
    runner = Runner(agent, runtime=runtime, corpus=corpus)

    # Phase 4.2: JSONL event log
    log_file = None
    if args.log:
        log_path = Path(args.log)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_file = log_path.open("a")

    # Phase 4.4 + Phase 5.x: Session-level cost tracking + budget cap.
    # max_cost=None → REPL; one-shot path supplies the cap.
    state = _SessionState(
        max_cost_usd=_resolve_max_cost(getattr(args, "max_cost", None)),
        alert_cost_usd=_resolve_alert_cost(getattr(args, "alert_cost", None)),
    )

    tool_names = sorted(runtime.tools.list_tools())
    print(_c(_ANSI_BOLD, f"KAOS Agent | {args.model} | pattern={pattern}"))
    corpus_size = corpus.size if corpus is not None else 0
    docs_label = f" | Passages: {corpus_size}" if corpus_size > 0 else ""
    print(f"Session: {session_id} | Tools: {len(tool_names)} registered{docs_label}")
    if verbose and tool_names:
        for name in tool_names[:20]:
            print(f"  {_c(_ANSI_DIM, name)}")
        if len(tool_names) > 20:
            print(f"  {_c(_ANSI_DIM, f'... and {len(tool_names) - 20} more')}")
    print()

    # Message source: in REPL mode we loop on ``input()``; in one-shot
    # mode (``--message``) we yield exactly one message and return.
    # Keeping a single inner loop avoids duplicating the turn / event /
    # budget-check plumbing between the two modes.
    one_shot = _one_shot_message(args)
    one_shot_sent = False

    while True:
        if one_shot is not None:
            if one_shot_sent:
                break
            user_input = one_shot
            one_shot_sent = True
        else:
            try:
                user_input = input(_c(_ANSI_GREEN, "> "))
            except (EOFError, KeyboardInterrupt):
                print()
                break

        stripped = user_input.strip()
        if not stripped:
            continue

        if stripped.startswith("/"):
            if stripped == "/quit":
                break
            if stripped == "/session":
                print(f"Session: {session_id}")
                continue
            if stripped == "/verbose":
                verbose = not verbose
                print(f"Verbose: {'ON' if verbose else 'OFF'}")
                continue
            if stripped == "/tools":
                names = sorted(runtime.tools.list_tools())
                for n in names:
                    print(f"  {n}")
                print(f"({len(names)} tools)")
                continue
            if stripped == "/clear":
                print("(session memory cleared)")
                session_id = f"cli-{uuid.uuid4().hex[:8]}"
                continue
            if stripped == "/memory":
                print("(memory dump not yet implemented)")
                continue
            if stripped == "/explain" or stripped.startswith("/explain "):
                # /explain — show the most recent turn.
                # /explain N — show turn N (1-indexed).
                # /explain <path> — write all turns as JSON to <path>.
                rest = stripped[len("/explain") :].strip()
                if rest and (rest.isdigit() or (rest.startswith("-") and rest[1:].isdigit())):
                    n = int(rest)
                    idx = len(state.explain_turns) + n if n < 0 else n - 1
                    if not state.explain_turns:
                        print("  No turns to explain yet.")
                        continue
                    if not 0 <= idx < len(state.explain_turns):
                        print(f"  No such turn: {n}. Have {len(state.explain_turns)} turn(s).")
                        continue
                    _print_explain(state.explain_turns[idx])
                    continue
                if rest:
                    # Treat as output path.
                    out = Path(rest).expanduser().resolve()
                    out.parent.mkdir(parents=True, exist_ok=True)
                    out.write_text(
                        json.dumps(
                            [_explain_to_dict(t) for t in state.explain_turns],
                            indent=2,
                        )
                    )
                    print(f"  Wrote {len(state.explain_turns)} turn(s) to {out}")
                    continue
                if not state.explain_turns:
                    print("  No turns to explain yet.")
                    continue
                _print_explain(state.explain_turns[-1])
                continue
            if stripped.startswith("/load "):
                load_arg = stripped[6:].strip()
                if not load_arg:
                    print("Usage: /load <file_or_glob>")
                    print(f"  Supported: {', '.join(sorted(_SUPPORTED_EXTENSIONS))}")
                    continue
                target = Path(load_arg).expanduser()
                if not target.is_absolute():
                    target = Path.cwd() / target

                if target.is_dir():
                    # Load all supported files from the directory
                    paths = [
                        f
                        for f in sorted(target.iterdir())
                        if f.is_file() and f.suffix.lower() in _SUPPORTED_EXTENSIONS
                    ]
                    if not paths:
                        print(_c(_ANSI_YELLOW, f"  No supported files in {target}"))
                        print(f"  Supported: {', '.join(sorted(_SUPPORTED_EXTENSIONS))}")
                        continue
                elif target.is_file():
                    paths = [target]
                else:
                    # Try as a glob pattern
                    paths = sorted(Path.cwd().glob(load_arg))
                    if not paths:
                        print(_c(_ANSI_RED, f"  No files matching: {load_arg}"))
                        continue
                new_corpus, new_uris = _load_files_to_corpus(
                    paths,
                    verbose=True,
                    chunk_size=getattr(args, "chunk_size", 8000),
                    workers=getattr(args, "load_workers", None),
                    cache_dir=_resolve_corpus_cache(args),
                )
                if new_corpus is not None:
                    corpus = new_corpus
                    print(f"  {len(new_uris)} docs → {corpus.size} passages")
                    # Rebuild runner with the new corpus
                    if agent.pattern.value == "chat":
                        from kaos_agents.config import AgentPattern

                        agent = Agent.create(
                            instructions=agent.instructions,
                            model=agent.effective_model(),
                            pattern=AgentPattern.RESEARCH,
                            tools=tools_tuple,
                        )
                        print(_c(_ANSI_CYAN, "  Switched to research pattern for document Q&A"))
                    runner = Runner(agent, runtime=runtime, corpus=corpus)
                continue
            if stripped == "/load":
                print("Usage: /load <file, glob, or folder>")
                print(f"  Supported: {', '.join(sorted(_SUPPORTED_EXTENSIONS))}")
                print("  Examples:")
                print("    /load ~/deal-room/           (all supported files in folder)")
                print("    /load documents/*.pdf")
                print("    /load report.docx")
                print("    /load *.txt")
                continue
            if stripped == "/docs":
                n_docs = memory.section_item_count(MemoryType.DOCUMENTS)
                if n_docs == 0:
                    print("  No documents loaded. Use /load <path> to add files.")
                else:
                    print(f"  {n_docs} document(s) in session memory")
                    items = memory.get(MemoryType.DOCUMENTS)
                    for item in items[:20]:
                        uri = (item.metadata or {}).get("uri", item.id[:8])
                        chars = len(item.content)
                        print(f"    {uri} ({chars:,} chars)")
                    if len(items) > 20:
                        print(f"    ... and {len(items) - 20} more")
                continue
            print(f"Unknown command: {stripped}")
            continue

        try:
            text_parts: list[str] = []
            # N3 / P9 — capture per-turn explain record. Built up across
            # the event stream and finalized on TurnComplete. The state's
            # ``explain_turns`` list keeps every turn so /explain N can
            # show any past turn, not just the latest.
            import time as _t

            explain = _ExplainTurn(
                turn_index=state.turns + 1,
                user_message=stripped,
            )
            explain_t0 = _t.monotonic()

            async for event in runner.run(stripped, session_id):
                # Phase 4.2: Write every event to JSONL log
                if log_file is not None:
                    from kaos_agents.events import serialize_event_json

                    log_file.write(serialize_event_json(event))
                    log_file.write("\n")
                    log_file.flush()

                if isinstance(event, TurnStart) and verbose:
                    print(_c(_ANSI_DIM, f"[turn:{event.turn_number}]"))

                elif isinstance(event, IntentClassified):
                    explain.intent = event.intent
                    explain.intent_confidence = event.confidence
                    if verbose:
                        print(
                            _c(
                                _ANSI_CYAN,
                                f"[intent] {event.intent} (confidence={event.confidence:.2f})",
                            )
                        )

                elif isinstance(event, PlanProposed) and verbose:
                    print(_c(_ANSI_CYAN, "[plan]"))
                    for step in event.steps:
                        tool = f" ({step.tool_name})" if step.tool_name else ""
                        print(f"  {step.step_id}. {step.description}{tool}")

                elif isinstance(event, StepStart) and verbose:
                    print(_c(_ANSI_CYAN, f"[step:{event.step_id}] {event.description}"))

                elif isinstance(event, StepComplete) and verbose:
                    status = "failed" if event.is_error else "done"
                    print(_c(_ANSI_DIM, f"[step:{event.step_id}] {status}"))

                elif isinstance(event, ToolCallStart):
                    if verbose:
                        args_preview = str(event.arguments)[:120]
                        print(_c(_ANSI_YELLOW, f"[tool:start] {event.tool_name} {args_preview}"))
                    else:
                        print(_c(_ANSI_DIM, f"  [tool: {event.tool_name}]"))

                elif isinstance(event, ToolCallResult):
                    preview = (getattr(event, "result_summary", "") or "")[:100]
                    duration = getattr(event, "duration_ms", 0) or 0
                    explain.tool_calls.append(
                        {
                            "tool_name": event.tool_name,
                            "call_id": event.call_id,
                            "is_error": event.is_error,
                            "duration_ms": float(duration),
                            "preview": preview,
                        }
                    )
                    if verbose:
                        print(_c(_ANSI_DIM, f"[tool:result] {preview} ({duration:.0f}ms)"))

                elif isinstance(event, ThinkingDelta) and verbose:
                    sys.stdout.write(_c(_ANSI_DIM, event.content))
                    sys.stdout.flush()

                elif isinstance(event, TextDelta):
                    sys.stdout.write(event.content)
                    sys.stdout.flush()
                    text_parts.append(event.content)

                elif isinstance(event, CitationFound):
                    explain.citations.append(
                        {
                            "claim": event.claim,
                            "verified": event.verified,
                            "confidence": float(event.confidence),
                            "source_uri": getattr(event, "source_uri", ""),
                            "node_ref": getattr(event, "node_ref", ""),
                            "page": getattr(event, "page", None),
                        }
                    )
                    if verbose:
                        v = "verified" if event.verified else "unverified"
                        print(
                            _c(
                                _ANSI_CYAN,
                                f"[citation] {event.claim[:60]} ({v}, {event.confidence:.2f})",
                            )
                        )

                elif isinstance(event, EvidenceInsufficient):
                    explain.refusals.append(
                        {
                            "reason": event.reason,
                            "kind": "evidence_insufficient",
                        }
                    )
                    print(_c(_ANSI_RED, f"[insufficient] {event.reason}"))

                elif isinstance(event, GroundingRefusalTriggered):
                    explain.refusals.append(
                        {
                            "kind": "grounding_low_confidence",
                            "original_confidence": float(event.original_confidence),
                            "min_confidence": float(event.min_confidence),
                        }
                    )
                    if verbose:
                        print(
                            _c(
                                _ANSI_RED,
                                (
                                    "[refusal] confidence="
                                    f"{event.original_confidence:.2f} < "
                                    f"{event.min_confidence:.2f}"
                                ),
                            )
                        )

                elif isinstance(event, RunError):
                    explain.errors.append(
                        {
                            "error_type": event.error_type,
                            "message": event.message,
                        }
                    )
                    print(_c(_ANSI_RED, f"\n[error] {event.error_type}: {event.message}"))

                elif isinstance(event, TurnComplete):
                    if text_parts:
                        print()
                    # Phase 4.4: Accumulate session cost
                    state.absorb(event.tokens_used, event.cost_usd)
                    n_tools = len(event.tool_calls)
                    # Backfill the per-tool cost into the explain record
                    # before finalizing — TurnComplete carries the
                    # attributed cost the agent layer just computed.
                    cost_by_call_id = {s.call_id: float(s.cost_usd) for s in event.tool_calls}
                    tokens_by_call_id = {
                        s.call_id: int(s.input_tokens + s.output_tokens) for s in event.tool_calls
                    }
                    for tc in explain.tool_calls:
                        cid = tc.get("call_id", "")
                        if cid in cost_by_call_id:
                            tc["cost_usd"] = cost_by_call_id[cid]
                            tc["tokens"] = tokens_by_call_id[cid]
                    explain.text = "".join(text_parts)
                    explain.tokens_used = int(event.tokens_used)
                    explain.cost_usd = float(event.cost_usd)
                    explain.duration_s = _t.monotonic() - explain_t0
                    state.explain_turns.append(explain)
                    if verbose:
                        print(
                            _c(
                                _ANSI_DIM,
                                f"[done] {n_tools} tool(s), {event.tokens_used} tokens, "
                                f"${event.cost_usd:.4f} | session: {state.tokens} tokens, "
                                f"${state.cost_usd:.4f}, {state.turns} turn(s)",
                            )
                        )
                    elif event.cost_usd > 0:
                        print(
                            _c(
                                _ANSI_DIM,
                                f"  [{event.tokens_used} tokens, ${event.cost_usd:.4f}]",
                            )
                        )
                    if state.alert_due():
                        print(
                            _c(
                                _ANSI_YELLOW,
                                f"⚠ Cost alert: session has spent "
                                f"${state.cost_usd:.4f} (threshold ${state.alert_cost_usd:.4f}). "
                                f"Continuing — set --max-cost to hard-stop.",
                            )
                        )
                    print()

        except Exception as exc:
            print(_c(_ANSI_RED, f"\n[fatal] {type(exc).__name__}: {exc}"))
            if verbose:
                import traceback

                traceback.print_exc()

        # Budget cap — check after every turn. The current turn is allowed
        # to complete (it may have been the one that pushed us over); the
        # next one is refused. Exit code 2 in non-interactive mode so
        # CI / course runnables / scripts can tell budget-exceeded apart
        # from real errors (exit 1).
        if state.budget_exceeded():
            print(
                _c(
                    _ANSI_YELLOW,
                    f"\n[budget] session cost ${state.cost_usd:.4f} "
                    f"meets or exceeds cap ${state.max_cost_usd:.4f} — "
                    "refusing further turns.",
                )
            )
            break

    if log_file is not None:
        log_file.close()
        print(f"Event log written to: {args.log}")

    # N3 / P9 — write per-turn explain records to --explain <path>
    # when set. Always-write at session end (REPL or one-shot). Writes
    # an empty list rather than skipping when no turns happened, so
    # downstream tooling has a stable artifact to diff against.
    explain_path = getattr(args, "explain", None)
    if explain_path:
        out = Path(explain_path).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps([_explain_to_dict(t) for t in state.explain_turns], indent=2))
        print(f"Explain records written to: {out} ({len(state.explain_turns)} turn(s))")

    return state


def _register_tool_modules(runtime: Any, args: argparse.Namespace) -> None:
    """Import and register optional tool modules based on CLI flags."""
    modules = []
    if getattr(args, "with_source", False) or getattr(args, "with_all", False):
        modules.append(("kaos_source.tools", "register_source_tools"))
    if getattr(args, "with_pdf", False) or getattr(args, "with_all", False):
        modules.append(("kaos_pdf.tools", "register_pdf_tools"))
    if getattr(args, "with_office", False) or getattr(args, "with_all", False):
        modules.append(("kaos_office.tools", "register_office_tools"))
    if getattr(args, "with_web", False) or getattr(args, "with_all", False):
        modules.append(("kaos_web.tools", "register_web_tools"))
    if getattr(args, "with_citations", False) or getattr(args, "with_all", False):
        modules.append(("kaos_citations.tools", "register_citations_tools"))

    for mod_path, func_name in modules:
        try:
            import importlib

            mod = importlib.import_module(mod_path)
            register_fn = getattr(mod, func_name)
            n = register_fn(runtime)
            print(f"  Loaded {mod_path}: {n} tools")
        except ImportError as exc:
            print(f"  (skip) {mod_path}: {exc}")


def _build_arg_parser() -> argparse.ArgumentParser:
    """Construct the ``kaos-agent`` argparse graph.

    Factored out so tests can exercise the flag schema directly
    without invoking ``main()``.
    """
    parser = argparse.ArgumentParser(
        prog="kaos-agent",
        description="KAOS agent CLI — interactive REPL or one-shot turn.",
    )
    sub = parser.add_subparsers(dest="command")
    chat = sub.add_parser(
        "chat",
        help="Chat with the agent (REPL by default; --message for one-shot).",
    )
    chat.add_argument("--session", help="Session ID (default: auto-generated)")
    from kaos_agents.settings import DEFAULT_MODEL

    chat.add_argument("--model", default=DEFAULT_MODEL)
    chat.add_argument("--pattern", default="chat", choices=["chat", "plan", "research"])
    chat.add_argument("--tools", default="", help="Comma-separated tool name globs")
    chat.add_argument("--instructions", help="System prompt override")
    chat.add_argument("--verbose", "-v", action="store_true", help="Show all events")
    chat.add_argument("--log", help="Write all events to JSONL file")
    chat.add_argument(
        "--files",
        default="",
        help="Comma-separated file paths or globs to pre-load (e.g. 'docs/*.pdf,*.docx')",
    )
    # Phase 5.x — non-interactive mode.
    chat.add_argument(
        "--message",
        help=(
            "Send a single message and exit (non-interactive). Use '-' to "
            "read the message from stdin. Without --message, starts the REPL."
        ),
    )
    # Phase 5.x — session cost ceiling.
    chat.add_argument(
        "--max-cost",
        type=float,
        default=None,
        metavar="USD",
        help=(
            "Stop accepting new turns once cumulative session cost reaches "
            "this value in USD. Falls back to $KAOS_AGENT_MAX_COST_USD when "
            "unset; disable with --max-cost 0. Non-interactive mode exits "
            "with code 2 on budget exceeded (distinct from code 1 for errors)."
        ),
    )
    chat.add_argument(
        "--alert-cost",
        type=float,
        default=None,
        metavar="USD",
        help=(
            "Print a one-time alert when cumulative session cost crosses "
            "this value (USD). Soft warning — does NOT stop the session. "
            "Use with --max-cost for a hard ceiling. Defaults to "
            "$KAOS_AGENT_ALERT_COST_USD when unset; 0 disables."
        ),
    )
    chat.add_argument(
        "--explain",
        type=str,
        default=None,
        metavar="PATH",
        help=(
            "After the session ends, write per-turn explain records "
            "(retrieved passages, citations, refusals, per-tool latency + "
            "cost) as JSON to PATH. In one-shot mode (--message), the "
            "single-turn record is written. In REPL mode, all turns are "
            "written when the REPL exits. Inside the REPL, /explain shows "
            "the most recent turn (or /explain N for turn N, or "
            "/explain <path> to write to a file mid-session)."
        ),
    )
    chat.add_argument("--with-source", action="store_true")
    chat.add_argument("--with-pdf", action="store_true")
    chat.add_argument("--with-office", action="store_true")
    chat.add_argument("--with-web", action="store_true")
    chat.add_argument("--with-citations", action="store_true")
    chat.add_argument("--with-all", action="store_true", help="Register all tool modules")
    # Retrieval / corpus knobs — overrides for power users. These set the
    # corresponding KAOS_AGENT_* env vars so the rest of the agent picks
    # them up via KaosAgentSettings hydration.
    chat.add_argument(
        "--retrieval-threshold",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Minimum memory-section item count that triggers BM25 retrieval "
            "(default 5; legacy was 20). Below this threshold, FIFO is used."
        ),
    )
    chat.add_argument(
        "--chunk-size",
        type=int,
        default=8000,
        metavar="N",
        help=(
            "Maximum chunk size in characters for the SectionChunker that "
            "splits each loaded document before BM25 indexing. Default 8000 "
            "(~2K tokens), matching the SectionChunker default and giving "
            "the model a coherent section's worth of context per chunk. "
            "Was 1500 in the GPT-3.5 era. Set to 0 to disable chunking "
            "(one passage per ContentDocument paragraph block — legacy "
            "behavior, not recommended for legal docs because it "
            "produces unbounded passages)."
        ),
    )
    chat.add_argument(
        "--load-workers",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Parallel workers for document loading. Default = max(2, cpu/2). "
            "Set to 1 to force serial loading (e.g. when debugging extractors)."
        ),
    )
    chat.add_argument(
        "--corpus-cache",
        default=None,
        metavar="DIR",
        help=(
            "Cache parsed+chunked documents under DIR keyed by content hash. "
            "Re-running the agent on the same corpus then skips parse + chunk. "
            "Cache invalidates automatically when file contents change. "
            "Pass an empty string or use --no-cache to disable."
        ),
    )
    chat.add_argument(
        "--no-cache",
        action="store_true",
        help=(
            "Disable the corpus cache even when --corpus-cache (or its env "
            "equivalent) is set. Useful for benchmarks and CI runs that need "
            "to measure cold-start cost."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns an exit code so callers (tests, CI, course
    runnables) can distinguish budget-exceeded (2) from runtime error
    (1) from success (0)."""
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return _EXIT_OK

    # Translate CLI knobs that override settings defaults into env vars
    # so KaosAgentSettings hydration (used everywhere in the agent loop)
    # picks them up consistently. Only set when the user supplied the
    # flag — otherwise the settings-class default wins.
    if args.command == "chat":
        retrieval_threshold = getattr(args, "retrieval_threshold", None)
        if retrieval_threshold is not None:
            os.environ["KAOS_AGENT_RETRIEVAL_THRESHOLD"] = str(retrieval_threshold)

    if args.command == "chat":
        state = asyncio.run(_run_repl(args))
        if state.budget_exceeded():
            return _EXIT_BUDGET
    return _EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
