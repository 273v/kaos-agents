"""Run the KAOS MCP server with agent tools.

Usage:
    # stdio (for Claude Code / Claude Desktop)
    kaos-agents-serve

    # streamable HTTP
    kaos-agents-serve --http --port 8000

    # with additional tool modules
    kaos-agents-serve --with-source --with-web

    # with debug logging
    kaos-agents-serve --debug
"""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> None:
    """Entry point for the MCP server."""
    parser = argparse.ArgumentParser(description="KAOS MCP Server with agent tools")
    parser.add_argument("--http", action="store_true", help="Use streamable HTTP transport")
    parser.add_argument("--host", default="127.0.0.1", help="HTTP host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="HTTP port (default: 8000)")
    parser.add_argument(
        "--with-source",
        action="store_true",
        help="Also register kaos-source tools (FR, eCFR, EDGAR, etc.)",
    )
    parser.add_argument(
        "--with-web",
        action="store_true",
        help="Also register kaos-web tools (fetch, search, extract)",
    )
    parser.add_argument(
        "--with-pdf",
        action="store_true",
        help="Also register kaos-pdf tools (parse, render, search)",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args(argv)

    try:
        from kaos_core import KaosRuntime
        from kaos_mcp import KaosMCPServer, KaosMCPSettings
    except ImportError:
        print(
            "Error: MCP server requires the 'mcp' extra.\n"
            "Install with: pip install 'kaos-agents[mcp]'",
            file=sys.stderr,
        )
        sys.exit(1)

    from kaos_agents.tools import register_agent_tools

    runtime = KaosRuntime()
    n_tools = register_agent_tools(runtime)
    print(f"Registered {n_tools} agent tools", file=sys.stderr)

    # Optionally register additional tool modules
    if args.with_source:
        try:
            from kaos_source import register_source_tools

            n = register_source_tools(runtime)
            print(f"Registered {n} source tools", file=sys.stderr)
        except ImportError:
            print("Warning: kaos-source not installed, skipping --with-source", file=sys.stderr)

    if args.with_web:
        try:
            from kaos_web import register_web_tools

            n = register_web_tools(runtime)
            print(f"Registered {n} web tools", file=sys.stderr)
        except ImportError:
            print("Warning: kaos-web not installed, skipping --with-web", file=sys.stderr)

    if args.with_pdf:
        try:
            from kaos_pdf import register_pdf_tools

            n = register_pdf_tools(runtime)
            print(f"Registered {n} PDF tools", file=sys.stderr)
        except ImportError:
            print("Warning: kaos-pdf not installed, skipping --with-pdf", file=sys.stderr)

    settings = KaosMCPSettings(
        name="kaos-agents-server",
        transport="streamable-http" if args.http else "stdio",
        host=args.host,
        port=args.port,
        debug=args.debug,
    )

    server = KaosMCPServer(runtime=runtime, settings=settings)

    if args.http:
        print(f"Starting HTTP server on {args.host}:{args.port}/mcp", file=sys.stderr)
        server.run_streamable_http()
    else:
        print("Starting stdio server", file=sys.stderr)
        server.run_stdio()


if __name__ == "__main__":
    main()
