"""S3 — memory continuity: 2 turns, turn 2 references turn 1's data.

3 surfaces x 2 providers = 6 tests. ~$0.30 total.

Validates SessionMemory hydrates across calls via shared session_id.
Turn 1 plants a unique token (7919). Turn 2 asks for it — must
retrieve from memory, not hallucinate.
"""

from __future__ import annotations

import os

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

TURN1 = "My favorite color is mauve, and the magic number is 7919. Acknowledge."
TURN2 = "What was my magic number?"


@pytest.mark.parametrize("provider", PROVIDERS)
@pytest.mark.parametrize("surface", SURFACES)
@pytest.mark.asyncio
async def test_s3_memory_continuity(surface: str, provider: str) -> None:
    # Use the PID in session_id so concurrent test runs don't clash
    # in the on-disk SessionStore.
    session_id = f"s3-{surface}-{provider}-{os.getpid()}"

    if surface == "cli":
        r1 = cli_run(TURN1, provider=provider, session_id=session_id)
        assert_no_error(r1)
        r2 = cli_run(TURN2, provider=provider, session_id=session_id)
    elif surface == "api":
        r1 = await api_call(TURN1, provider=provider, session_id=session_id)
        assert_no_error(r1)
        r2 = await api_call(TURN2, provider=provider, session_id=session_id)
    elif surface == "mcp":
        r1 = await mcp_call(
            "kaos-agent-chat",
            arguments={
                "message": TURN1,
                "session_id": session_id,
                "model": model_for(provider),
            },
        )
        assert_no_error(r1)
        r2 = await mcp_call(
            "kaos-agent-chat",
            arguments={
                "message": TURN2,
                "session_id": session_id,
                "model": model_for(provider),
            },
        )
    else:
        pytest.fail(f"unknown surface: {surface}")

    assert_no_error(r2)
    assert "7919" in r2.text, (
        f"{surface}/{provider}: turn 2 missing memory fact 7919. Got: {r2.text[-400:]!r}"
    )
