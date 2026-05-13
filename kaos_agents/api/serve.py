"""Run the KAOS agent server (MCP or HTTP API).

Usage:
    # stdio MCP (for Claude Code / Claude Desktop)
    kaos-agents-serve

    # streamable HTTP MCP
    kaos-agents-serve --http --port 8000

    # HTTP API with SSE streaming (for web apps)
    kaos-agents-serve --api --port 8080

    # with additional tool modules
    kaos-agents-serve --with-source --with-web

    # with debug logging
    kaos-agents-serve --debug
"""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> None:
    """Entry point for the agent server."""
    parser = argparse.ArgumentParser(description="KAOS Agent Server (MCP or HTTP API)")
    parser.add_argument("--http", action="store_true", help="Use streamable HTTP MCP transport")
    parser.add_argument(
        "--api",
        action="store_true",
        help="Start HTTP API server with SSE streaming (requires [api] extra)",
    )
    parser.add_argument("--host", default="127.0.0.1", help="HTTP host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="HTTP port (default: 8000)")

    # Optional kaos-* tool modules — flag set is shared with the chat CLI.
    from kaos_agents.tools.optional_modules import add_optional_module_flags

    add_optional_module_flags(parser)

    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args(argv)

    # --- API mode (FastAPI + SSE) ---
    if args.api:
        try:
            import uvicorn

            from kaos_agents.api.server import create_app
        except ImportError:
            print(
                "Error: API server requires the 'api' extra.\n"
                "Install with: pip install 'kaos-agents[api]'",
                file=sys.stderr,
            )
            sys.exit(1)

        # Optional runtime for tool execution
        runtime = None
        try:
            from kaos_core import KaosRuntime

            runtime = KaosRuntime()
        except ImportError:
            print("Warning: kaos-core not available, running without tool runtime", file=sys.stderr)

        app = create_app(runtime=runtime)
        print(f"Starting API server on {args.host}:{args.port}", file=sys.stderr)
        uvicorn.run(
            app, host=args.host, port=args.port, log_level="debug" if args.debug else "info"
        )
        return

    # --- MCP mode (default) ---
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
    from kaos_agents.tools.optional_modules import register_optional_modules

    runtime = KaosRuntime()
    n_tools = register_agent_tools(runtime)
    print(f"Registered {n_tools} agent tools", file=sys.stderr)

    # Optional kaos-* tool modules — single dispatch shared with chat CLI.
    register_optional_modules(runtime, args)

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
