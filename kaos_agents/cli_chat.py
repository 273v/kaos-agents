"""Interactive CLI REPL for conversing with the KAOS agent.

Usage::

    kaos-agent chat
    kaos-agent chat --session my-session --verbose
    kaos-agent chat --tools "kaos-source-*,kaos-pdf-*" --model anthropic:claude-sonnet-4-6

Slash commands inside the REPL:

    /quit     — exit
    /session  — print current session ID
    /tools    — list registered tools
    /memory   — dump memory section summaries
    /clear    — clear session memory
    /verbose  — toggle verbose event display
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import uuid
from typing import Any

_ANSI_BOLD = "\033[1m"
_ANSI_DIM = "\033[2m"
_ANSI_CYAN = "\033[36m"
_ANSI_YELLOW = "\033[33m"
_ANSI_GREEN = "\033[32m"
_ANSI_RED = "\033[31m"
_ANSI_RESET = "\033[0m"


def _c(code: str, text: str) -> str:
    """Colorize text if stdout is a TTY."""
    if not sys.stdout.isatty():
        return text
    return f"{code}{text}{_ANSI_RESET}"


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

    from kaos_agents.tools import register_agent_tools

    register_agent_tools(runtime)

    _register_tool_modules(runtime, args)

    tools_tuple = tuple(t.strip() for t in args.tools.split(",")) if args.tools else ()
    agent = Agent.create(
        instructions=args.instructions or "You are a helpful assistant.",
        model=args.model,
        pattern=args.pattern,
        tools=tools_tuple,
    )
    runner = Runner(agent, runtime=runtime)
    session_id = args.session or f"cli-{uuid.uuid4().hex[:8]}"
    verbose = args.verbose

    # Phase 4.2: JSONL event log
    log_file = None
    if args.log:
        from pathlib import Path

        log_path = Path(args.log)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_file = log_path.open("a")

    # Phase 4.4: Session-level cost tracking
    session_tokens = 0
    session_cost = 0.0
    session_turns = 0

    tool_names = sorted(runtime.list_tools().keys()) if hasattr(runtime, "list_tools") else []
    print(_c(_ANSI_BOLD, f"KAOS Agent | {args.model} | pattern={args.pattern}"))
    print(f"Session: {session_id} | Tools: {len(tool_names)} registered")
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
                names = (
                    sorted(runtime.list_tools().keys()) if hasattr(runtime, "list_tools") else []
                )
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
                    status = "done" if event.success else "failed"
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
                            f"[refusal] confidence={event.original_confidence:.2f} < {event.min_confidence:.2f}",
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
    chat.add_argument("--with-source", action="store_true")
    chat.add_argument("--with-pdf", action="store_true")
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
