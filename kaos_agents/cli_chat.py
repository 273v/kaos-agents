"""Interactive CLI REPL for conversing with the KAOS agent.

Usage::

    kaos-agent chat
    kaos-agent chat --session my-session --verbose
    kaos-agent chat --tools "kaos-source-*,kaos-pdf-*" --model anthropic:claude-sonnet-4-6

    # Load documents at startup for RAG Q&A
    kaos-agent chat --files "contracts/*.pdf,memos/*.docx" --verbose
    kaos-agent chat --files "*.pdf" --pattern research

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
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import uuid
from pathlib import Path
from typing import Any

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
    if ext == ".pdf":
        from kaos_pdf import extract_pdf

        return extract_pdf(file_path)

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


def _load_files_to_corpus(
    file_paths: list[Path],
    *,
    verbose: bool = False,
) -> tuple[Any, list[str]]:
    """Parse files into ContentDocuments and build a ContentDocumentCorpus.

    Returns (corpus, uris) where corpus is a ContentDocumentCorpus with
    passage-level retrieval, and uris is the list of document URIs.
    Returns (None, []) if no documents loaded.
    """
    from kaos_content.corpus import ContentDocumentCorpus

    documents: list[Any] = []
    uris: list[str] = []

    for fp in file_paths:
        fp = fp.resolve()
        if not fp.exists():
            print(_c(_ANSI_RED, f"  File not found: {fp}"))
            continue
        try:
            doc = _parse_file_to_document(fp)
            documents.append(doc)
            uris.append(f"file:{fp.name}")
            if verbose:
                from kaos_content.units import iter_paragraph_units

                n_paragraphs = len(iter_paragraph_units(doc))
                print(_c(_ANSI_DIM, f"  Loaded {fp.name} ({n_paragraphs} paragraphs)"))
        except ImportError as exc:
            print(_c(_ANSI_RED, f"  Missing parser for {fp.suffix}: {exc}"))
        except Exception as exc:
            print(_c(_ANSI_RED, f"  Failed {fp.name}: {exc}"))

    if not documents:
        return None, []

    corpus = ContentDocumentCorpus(documents, doc_uris=uris)
    return corpus, uris


def _load_files_into_memory(
    file_paths: list[Path],
    memory: Any,
    *,
    verbose: bool = False,
) -> int:
    """Parse files and add to DOCUMENTS memory section. Returns count loaded.

    .. deprecated::
        Prefer _load_files_to_corpus which preserves the ContentDocument AST
        for passage-level retrieval. This function flattens to text strings.
    """
    from kaos_agents.memory.types import MemoryType

    loaded = 0
    for fp in file_paths:
        fp = fp.resolve()
        if not fp.exists():
            print(_c(_ANSI_RED, f"  File not found: {fp}"))
            continue
        try:
            text, uri = _parse_file(fp)
            if not text.strip():
                print(_c(_ANSI_YELLOW, f"  Empty: {fp.name}"))
                continue
            memory.add(MemoryType.DOCUMENTS, text, metadata={"uri": uri, "source": str(fp)})
            loaded += 1
            if verbose:
                print(_c(_ANSI_DIM, f"  Loaded {fp.name} ({len(text):,} chars)"))
        except ImportError as exc:
            print(_c(_ANSI_RED, f"  Missing parser for {fp.suffix}: {exc}"))
        except Exception as exc:
            print(_c(_ANSI_RED, f"  Failed {fp.name}: {exc}"))
    return loaded


async def _run_repl(args: argparse.Namespace) -> None:
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

    # Pre-load files from --files flag
    if args.files:
        file_paths: list[Path] = []
        for pat in args.files.split(","):
            file_paths.extend(Path.cwd().glob(pat.strip()))
        if file_paths:
            print(f"Loading {len(file_paths)} file(s)...")
            corpus, uris = _load_files_to_corpus(file_paths, verbose=verbose)
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

    # Phase 4.4: Session-level cost tracking
    session_tokens = 0
    session_cost = 0.0
    session_turns = 0

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

    while True:
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
                new_corpus, new_uris = _load_files_to_corpus(paths, verbose=True)
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
            async for event in runner.run(stripped, session_id):
                # Phase 4.2: Write every event to JSONL log
                if log_file is not None:
                    from kaos_agents.events import serialize_event_json

                    log_file.write(serialize_event_json(event))
                    log_file.write("\n")
                    log_file.flush()

                if isinstance(event, TurnStart) and verbose:
                    print(_c(_ANSI_DIM, f"[turn:{event.turn_number}]"))

                elif isinstance(event, IntentClassified) and verbose:
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

                elif isinstance(event, ToolCallResult) and verbose:
                    preview = (getattr(event, "result_summary", "") or "")[:100]
                    duration = getattr(event, "duration_ms", 0) or 0
                    print(_c(_ANSI_DIM, f"[tool:result] {preview} ({duration:.0f}ms)"))

                elif isinstance(event, ThinkingDelta) and verbose:
                    sys.stdout.write(_c(_ANSI_DIM, event.content))
                    sys.stdout.flush()

                elif isinstance(event, TextDelta):
                    sys.stdout.write(event.content)
                    sys.stdout.flush()
                    text_parts.append(event.content)

                elif isinstance(event, CitationFound) and verbose:
                    v = "verified" if event.verified else "unverified"
                    print(
                        _c(
                            _ANSI_CYAN,
                            f"[citation] {event.claim[:60]} ({v}, {event.confidence:.2f})",
                        )
                    )

                elif isinstance(event, EvidenceInsufficient):
                    print(_c(_ANSI_RED, f"[insufficient] {event.reason}"))

                elif isinstance(event, GroundingRefusalTriggered) and verbose:
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
                    print(_c(_ANSI_RED, f"\n[error] {event.error_type}: {event.message}"))

                elif isinstance(event, TurnComplete):
                    if text_parts:
                        print()
                    # Phase 4.4: Accumulate session cost
                    session_tokens += event.tokens_used
                    session_cost += event.cost_usd
                    session_turns += 1
                    n_tools = len(event.tool_calls)
                    if verbose:
                        print(
                            _c(
                                _ANSI_DIM,
                                f"[done] {n_tools} tool(s), {event.tokens_used} tokens, "
                                f"${event.cost_usd:.4f} | session: {session_tokens} tokens, "
                                f"${session_cost:.4f}, {session_turns} turn(s)",
                            )
                        )
                    elif event.cost_usd > 0:
                        print(
                            _c(
                                _ANSI_DIM,
                                f"  [{event.tokens_used} tokens, ${event.cost_usd:.4f}]",
                            )
                        )
                    print()

        except Exception as exc:
            print(_c(_ANSI_RED, f"\n[fatal] {type(exc).__name__}: {exc}"))
            if verbose:
                import traceback

                traceback.print_exc()

    if log_file is not None:
        log_file.close()
        print(f"Event log written to: {args.log}")


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


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="kaos-agent",
        description="Interactive KAOS agent CLI",
    )
    sub = parser.add_subparsers(dest="command")
    chat = sub.add_parser("chat", help="Start an interactive chat session")
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
    chat.add_argument("--with-source", action="store_true")
    chat.add_argument("--with-pdf", action="store_true")
    chat.add_argument("--with-office", action="store_true")
    chat.add_argument("--with-web", action="store_true")
    chat.add_argument("--with-citations", action="store_true")
    chat.add_argument("--with-all", action="store_true", help="Register all tool modules")

    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return

    if args.command == "chat":
        asyncio.run(_run_repl(args))


if __name__ == "__main__":
    main()
