"""Shared helpers for the surface-parity matrix.

Goal: prove the SAME capability works through every surface
(CLI, FastAPI, MCP) with every model family (Anthropic Claude>=4.6,
OpenAI GPT>=5.4). Each test in this directory parametrizes over
(surface, provider) for one capability.

The helpers below abstract the three surfaces so tests look the same
shape regardless of which one they're exercising:

  result = await call_surface(
      surface, message,
      provider="anthropic" | "openai",
      tools=("kaos-source-fr-search",),
      with_source=True,                 # server-side tool registration
      session_id="s9-anthropic",
  )
  text = result.text  # the agent's final answer
  events = result.events  # the typed event stream (where available)

Each surface returns a uniform ``SurfaceResult`` so tests can assert
on text or events without branching on the surface.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

# Pinned models. Mandate: claude >= 4.6 AND gpt >= 5.4.
ANTHROPIC_MODEL = "anthropic:claude-sonnet-4-6"
OPENAI_MODEL = "openai:gpt-5.4-mini"

PROVIDERS = ("anthropic", "openai")
SURFACES = ("cli", "api", "mcp")


def model_for(provider: str) -> str:
    return ANTHROPIC_MODEL if provider == "anthropic" else OPENAI_MODEL


def opposite_provider(provider: str) -> str:
    """For LLM-as-judge: judge with the OPPOSITE family of the author."""
    return "openai" if provider == "anthropic" else "anthropic"


def _require_keys() -> None:
    missing = [k for k in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY") if k not in os.environ]
    if missing:
        pytest.fail(f"surface-parity needs {missing}; no skips")


@dataclass
class SurfaceResult:
    """Uniform return type across CLI / API / MCP surfaces."""

    surface: str
    provider: str
    text: str
    events: list[dict] = field(default_factory=list)
    error: str | None = None
    raw: Any = None


# ---------------------------------------------------------------------------
# CLI surface — subprocess invocation of the real ``kaos-agent`` binary
# ---------------------------------------------------------------------------


def cli_run(
    message: str,
    *,
    provider: str,
    session_id: str,
    with_flags: tuple[str, ...] = (),
    pattern: str | None = None,
    max_cost: float = 0.40,
    extra_args: tuple[str, ...] = (),
    log_path: Path | None = None,
) -> SurfaceResult:
    """Invoke ``kaos-agent chat --message ...`` as a subprocess."""
    _require_keys()
    cmd = [
        "uv",
        "run",
        "--no-sync",
        "kaos-agent",
        "chat",
        "--message",
        message,
        "--session",
        session_id,
        "--model",
        model_for(provider),
        "--max-cost",
        str(max_cost),
        *with_flags,
        *extra_args,
    ]
    if pattern:
        cmd += ["--pattern", pattern]
    if log_path is not None:
        cmd += ["--log", str(log_path)]
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=180,
        cwd=Path(__file__).resolve().parents[3],
    )
    events: list[dict] = []
    if log_path is not None and log_path.exists():
        for line in log_path.read_text().splitlines():
            with contextlib.suppress(Exception):
                events.append(json.loads(line))
    return SurfaceResult(
        surface="cli",
        provider=provider,
        text=proc.stdout,
        events=events,
        error=proc.stderr if proc.returncode != 0 else None,
        raw=proc,
    )


# ---------------------------------------------------------------------------
# API surface — ASGITransport against create_app
# ---------------------------------------------------------------------------


async def api_call(
    message: str,
    *,
    provider: str,
    session_id: str,
    register_source: bool = False,
    register_pdf: bool = False,
    corpus: Any | None = None,
    pattern: str = "chat",
    tools: tuple[str, ...] = (),
    permission_policy: Any | None = None,
    accept: str = "application/json",
    settings: Any | None = None,
) -> SurfaceResult:
    """POST /v1/sessions/{id}/messages against a freshly-built FastAPI app.

    Builds a Runtime with the requested tool sets, hands it to
    ``create_app(runtime=runtime)``, and exercises the route.
    """
    _require_keys()
    from httpx import ASGITransport, AsyncClient
    from kaos_core import KaosRuntime

    from kaos_agents.api.server import create_app

    runtime = KaosRuntime.default()
    if register_source:
        from kaos_source.runtime.tools import register_source_tools

        register_source_tools(runtime)
    if register_pdf:
        from kaos_pdf import register_pdf_tools

        register_pdf_tools(runtime)
    # corpus / permission_policy / settings flow through Runner, not
    # through create_app today. The API server builds a fresh Runner
    # per request from the request body + app-state runtime. Tests
    # that need those features should use the CLI or MCP variant
    # OR exercise the Runner directly (the in-proc ladder).
    # We document the asymmetry so future API tests aren't confused.
    if corpus is not None or permission_policy is not None or settings is not None:
        # Surface this clearly so api_call() callers don't silently
        # ignore the parameters they passed.
        pytest.skip(
            "API path does not accept per-request corpus / permission_policy / "
            "settings overrides — exercise via CLI or MCP."
        )

    app = create_app(runtime=runtime)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            f"/v1/sessions/{session_id}/messages",
            json={
                "message": message,
                "model": model_for(provider),
                "pattern": pattern,
                "tools": list(tools),
            },
            headers={"Accept": accept},
            timeout=180.0,
        )
    if resp.status_code != 200:
        return SurfaceResult(
            surface="api",
            provider=provider,
            text="",
            error=f"HTTP {resp.status_code}: {resp.text[:400]}",
            raw=resp,
        )

    if accept.startswith("text/event-stream"):
        # Parse the SSE body into typed events
        events: list[dict] = []
        text_parts: list[str] = []
        for chunk in resp.text.split("\n\n"):
            for line in chunk.splitlines():
                if line.startswith("data:"):
                    with contextlib.suppress(Exception):
                        ev = json.loads(line[5:].strip())
                        events.append(ev)
                        if ev.get("type") == "text_delta":
                            text_parts.append(ev.get("content", "") or "")
        return SurfaceResult(
            surface="api",
            provider=provider,
            text="".join(text_parts) or _extract_text_from_events(events),
            events=events,
            raw=resp,
        )

    data = resp.json()
    return SurfaceResult(
        surface="api",
        provider=provider,
        text=str(data.get("text", "")),
        events=[],
        raw=data,
    )


def _extract_text_from_events(events: list[dict]) -> str:
    """Pull the final text from a TurnSummary event when text_delta missing."""
    for ev in events:
        if ev.get("type") == "turn_summary":
            return str(ev.get("text", ""))
    return ""


# ---------------------------------------------------------------------------
# MCP surface — stdio_client subprocess
# ---------------------------------------------------------------------------


async def mcp_call(
    tool_name: str,
    *,
    arguments: dict[str, Any],
    server_args: tuple[str, ...] = (),
    timeout: float = 180.0,
) -> SurfaceResult:
    """Spawn ``python -m kaos_agents.api.serve`` over stdio MCP and call a tool.

    server_args lets callers pass --with-source / --with-pdf etc. so
    the spawned server has the right tool surface registered.
    """
    _require_keys()
    from mcp import ClientSession
    from mcp.client.stdio import StdioServerParameters, stdio_client

    server_params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "kaos_agents.api.serve", *server_args],
        env={**os.environ},
    )
    async with (
        stdio_client(server_params) as (read, write),
        ClientSession(read, write) as session,
    ):
        await session.initialize()
        result = await session.call_tool(tool_name, arguments=arguments)

    # The MCP tool result has BOTH:
    #   - content[].text: human-readable summary (kaos-agent-chat
    #     truncates this to 500 chars for terse MCP UIs)
    #   - structuredContent: the FULL result_data dict including the
    #     unredacted "text" field
    # For surface-parity assertions we need the full text; prefer
    # the structured field when present.
    structured = getattr(result, "structuredContent", None)
    full_text = ""
    if isinstance(structured, dict) and isinstance(structured.get("text"), str):
        full_text = structured["text"]
    summary_text = "".join(getattr(c, "text", "") or "" for c in result.content)
    return SurfaceResult(
        surface="mcp",
        provider=arguments.get("_test_provider", "anthropic"),
        text=full_text or summary_text,
        events=[],
        error=None if not getattr(result, "isError", False) else summary_text,
        raw=result,
    )


# ---------------------------------------------------------------------------
# Common regex helpers
# ---------------------------------------------------------------------------


DOC_NUMBER_RE = re.compile(r"\b20\d{2}-\d{4,5}\b")
DATE_RE = re.compile(r"\b20\d{2}-\d{2}-\d{2}\b")


def assert_no_error(result: SurfaceResult) -> None:
    """Fail loudly when a surface returned an error envelope."""
    assert result.error is None, (
        f"{result.surface}/{result.provider} returned error: {result.error[:600]!r}"
    )
