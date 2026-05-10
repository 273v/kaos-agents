"""S6 — citation auto-extraction: model produces a cite, verifier emits CitationFound.

The cite is NOT in the prompt — the model must produce it from
training knowledge. We use the Miranda warning prompt; correct
answers contain "Miranda v. Arizona, 384 U.S. 436 (1966)".

2 surfaces (API + MCP) x 2 providers = 4 tests. CLI variant is
covered by ladder T06.
"""

from __future__ import annotations

import re

import pytest

from tests.integration.surface_parity.conftest import (
    PROVIDERS,
    api_call,
    assert_no_error,
    mcp_call,
    model_for,
)

pytestmark = pytest.mark.live

MESSAGE = (
    "Briefly explain the Miranda warning requirement in 2-3 sentences. "
    "Include the standard case citation (case name + reporter + year)."
)
CASE_CITE_RE = re.compile(r"\b\d+\s+U\.?\s?S\.?\s+\d+|\bMiranda\s+v\.\s+Arizona", re.IGNORECASE)


@pytest.mark.parametrize("provider", PROVIDERS)
@pytest.mark.asyncio
async def test_s6_citation_via_api(provider: str) -> None:
    """API SSE stream must contain a CitationFound event on a real cite."""
    result = await api_call(
        MESSAGE,
        provider=provider,
        session_id=f"s6-api-{provider}",
        accept="text/event-stream",
    )
    assert_no_error(result)

    # The model must have produced a citation
    assert CASE_CITE_RE.search(result.text), (
        f"api/{provider}: model did not produce a recognizable case cite. "
        f"Verifier has nothing to recognize. Answer: {result.text[:400]!r}"
    )

    # CitationFound events must surface in the SSE stream
    cite_events = [e for e in result.events if e.get("type") == "citation_found"]
    assert cite_events, (
        f"api/{provider}: model produced a cite but zero CitationFound "
        f"events surfaced via SSE. The citation_verifier path is broken "
        f"on the API surface OR kaos-citations isn't loaded into the API "
        f"runtime. Answer: {result.text[:400]!r}"
    )


@pytest.mark.parametrize("provider", PROVIDERS)
@pytest.mark.asyncio
async def test_s6_citation_via_mcp(provider: str) -> None:
    """MCP response text must contain the cite (verifier signal harder to expose)."""
    result = await mcp_call(
        "kaos-agent-chat",
        arguments={
            "message": MESSAGE,
            "session_id": f"s6-mcp-{provider}",
            "model": model_for(provider),
        },
    )
    assert_no_error(result)
    assert CASE_CITE_RE.search(result.text), (
        f"mcp/{provider}: model did not produce a case citation. Answer: {result.text[:400]!r}"
    )
