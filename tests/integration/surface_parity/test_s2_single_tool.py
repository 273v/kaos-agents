"""S2 — single tool call (FR search): real document_number in answer.

3 surfaces x 2 providers = 6 tests. ~$0.50 total.

Real signal: the document_number (YYYY-NNNNN) comes ONLY from the
tool's structuredContent. If the agent answers without calling the
tool — or the bridge drops structuredContent — the regex fails.
"""

from __future__ import annotations

import pytest

from tests.integration.surface_parity.conftest import (
    DOC_NUMBER_RE,
    PROVIDERS,
    SURFACES,
    api_call,
    assert_no_error,
    cli_run,
    mcp_call,
    model_for,
)

pytestmark = pytest.mark.live

MESSAGE = (
    "Use kaos-source-fr-search exactly once: term=PFAS, "
    "agency=environmental-protection-agency, doc_type=NOTICE, per_page=1, "
    "order=newest. From results[0], report ONLY the document_number "
    "(YYYY-NNNNN format). No prose."
)


@pytest.mark.parametrize("provider", PROVIDERS)
@pytest.mark.parametrize("surface", SURFACES)
@pytest.mark.asyncio
async def test_s2_single_tool_call(surface: str, provider: str) -> None:
    session_id = f"s2-{surface}-{provider}"

    if surface == "cli":
        result = cli_run(
            MESSAGE,
            provider=provider,
            session_id=session_id,
            with_flags=("--with-source",),
        )
    elif surface == "api":
        result = await api_call(
            MESSAGE,
            provider=provider,
            session_id=session_id,
            register_source=True,
            tools=("kaos-source-fr-search",),
        )
    elif surface == "mcp":
        result = await mcp_call(
            "kaos-agent-chat",
            arguments={
                "message": MESSAGE,
                "session_id": session_id,
                "model": model_for(provider),
                "tool_filter": "kaos-source-fr-search",
            },
            server_args=("--with-source",),
        )
    else:
        pytest.fail(f"unknown surface: {surface}")

    assert_no_error(result)
    assert DOC_NUMBER_RE.search(result.text), (
        f"{surface}/{provider}: no document_number (YYYY-NNNNN) in output. "
        f"Either the tool wasn't called or the bridge dropped "
        f"structuredContent. Got: {result.text[-500:]!r}"
    )
