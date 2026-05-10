"""S7 — PDF tool-bridge: agent calls kaos-pdf-extract-page-text via API + MCP.

2 surfaces x 2 providers = 4 tests. CLI covered by ladder T07.

Verifies kaos-pdf bridges through both surfaces. The 1-page DOT
letter fixture has two unique tokens (docket OST-2008-0299 and
Mayor Geoff Dale) — both must surface in the answer, which means
the agent had to call the tool and read its output.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.integration.surface_parity.conftest import (
    PROVIDERS,
    api_call,
    assert_no_error,
    mcp_call,
    model_for,
)

pytestmark = pytest.mark.live

PDF_FIXTURE = Path(__file__).parent.parent / "ladder" / "fixtures" / "sample.pdf"


def _prompt() -> str:
    return (
        f"Use kaos-pdf-extract-page-text on path {PDF_FIXTURE} for page 0. "
        "From the extracted text, report (1) the docket number in the "
        "subject line and (2) the city's Mayor. Format: 'Docket: <number>; "
        "Mayor: <name>'."
    )


def _assert_dot_letter_facts(text: str, label: str) -> None:
    """Ground-truth facts from the fixture PDF (visually inspected)."""
    assert PDF_FIXTURE.exists(), f"missing fixture: {PDF_FIXTURE}"
    assert "OST-2008-0299" in text, f"{label}: missing docket OST-2008-0299. Got: {text[-400:]!r}"
    assert "Geoff Dale" in text or "geoff dale" in text.lower(), (
        f"{label}: missing Mayor 'Geoff Dale'. Got: {text[-400:]!r}"
    )


@pytest.mark.parametrize("provider", PROVIDERS)
@pytest.mark.asyncio
async def test_s7_pdf_via_api(provider: str) -> None:
    """API with register_pdf=True calls the PDF tool and extracts the facts."""
    assert PDF_FIXTURE.exists(), f"missing fixture: {PDF_FIXTURE}"
    result = await api_call(
        _prompt(),
        provider=provider,
        session_id=f"s7-api-{provider}",
        register_pdf=True,
        pattern="plan",  # multi-part question → planner stays in tool-use
        tools=("kaos-pdf-extract-page-text",),
    )
    assert_no_error(result)
    _assert_dot_letter_facts(result.text, f"api/{provider}")


@pytest.mark.parametrize("provider", PROVIDERS)
@pytest.mark.asyncio
async def test_s7_pdf_via_mcp(provider: str) -> None:
    """MCP server launched with --with-pdf invokes the PDF tool."""
    assert PDF_FIXTURE.exists(), f"missing fixture: {PDF_FIXTURE}"
    result = await mcp_call(
        "kaos-agent-plan",  # plan-execute pattern handles multi-step
        arguments={
            "message": _prompt(),
            "session_id": f"s7-mcp-{provider}",
            "model": model_for(provider),
            "tool_filter": "kaos-pdf-extract-page-text",
        },
        server_args=("--with-pdf",),
        timeout=240.0,
    )
    assert_no_error(result)
    _assert_dot_letter_facts(result.text, f"mcp/{provider}")
