"""S1 — smoke (no tools): basic LLM turn returns the right number.

3 surfaces x 2 providers = 6 tests. ~$0.03 total.
"""

from __future__ import annotations

import pytest

from tests.integration.surface_parity.conftest import (
    PROVIDERS,
    SURFACES,
    api_call,
    assert_no_error,
    cli_run,
    mcp_call,
    model_for,
)

pytestmark = pytest.mark.live

MESSAGE = "What is 17 + 25? Answer with just the number."


@pytest.mark.parametrize("provider", PROVIDERS)
@pytest.mark.parametrize("surface", SURFACES)
@pytest.mark.asyncio
async def test_s1_smoke(surface: str, provider: str) -> None:
    """Same prompt, same expected answer, every surface x every provider."""
    session_id = f"s1-{surface}-{provider}"

    if surface == "cli":
        result = cli_run(MESSAGE, provider=provider, session_id=session_id)
    elif surface == "api":
        result = await api_call(MESSAGE, provider=provider, session_id=session_id)
    elif surface == "mcp":
        result = await mcp_call(
            "kaos-agent-chat",
            arguments={
                "message": MESSAGE,
                "session_id": session_id,
                "model": model_for(provider),
            },
        )
    else:
        pytest.fail(f"unknown surface: {surface}")

    assert_no_error(result)
    assert "42" in result.text, (
        f"{surface}/{provider}: answer missing 42; got: {result.text[-400:]!r}"
    )
