"""Sprint-1 #2 — LIVE auth-failure surfacing.

Pin down the **end-to-end production path** against a real provider.

A unit test that monkeypatches ``BaseAgent._classify`` to raise
``KaosLLMAuthError`` proves the wrapper handles the exception
correctly — it does NOT prove the wrapper handles what the real
Anthropic API actually returns for an invalid key. This live test
replaces the real key with a known-invalid sentinel, drives
``AgentChatTool.execute`` through the full kaos-llm-client transport
stack, and asserts the resulting ``ToolResult``:

1. has ``isError=True`` (not the historic silent empty success),
2. contains the credential name (``ANTHROPIC_API_KEY``),
3. names a recovery path (the alternative tool).

This is the bug from skeptic probe 4b — a regulated-industry SOC2 CC7.2
alerting gap. The whole point of this test is that we don't trust
mocked-only evidence for transparency-critical surfaces.

Cost budget: ~$0.000 (the auth check happens before any tokens are
billed). The test still ``KaosRuntime.test_mode()``s its VFS so prior
test state never leaks in.
"""

from __future__ import annotations

import os

import pytest
from kaos_core.registry.container import KaosRuntime

from kaos_agents.errors import ERROR_KIND_AUTH
from kaos_agents.tools import AgentChatTool


@pytest.mark.live
@pytest.mark.skipif(
    "ANTHROPIC_API_KEY" not in os.environ,
    reason="Requires ANTHROPIC_API_KEY in the environment (we corrupt it temporarily).",
)
class TestAuthFailureLive:
    """End-to-end: invalid Anthropic key → ToolResult.isError=True."""

    @pytest.mark.asyncio
    async def test_invalid_anthropic_key_surfaces_as_isError_true(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The smoking gun. Real provider, real transport, real classifier."""
        # Replace ALL forms the env can take so kaos-llm-client's
        # KAOS_LLM_* and legacy fallback resolution paths BOTH see the
        # invalid key.
        for env_name in ("ANTHROPIC_API_KEY", "KAOS_LLM_ANTHROPIC_API_KEY"):
            monkeypatch.setenv(env_name, "sk-ant-garbage-INVALID-FOR-TESTING-PROBE-4B")

        runtime = KaosRuntime.test_mode()
        # Use a context-bound execute so the tool wrapper builds a Runner
        # that uses the test-mode VFS, not a fresh disk-backed one. The
        # default code path tolerates context=None (it falls back to an
        # in-memory VFS via _get_vfs) so we pass None here and rely on
        # the wrapper's fallback. Either way the LLM transport reads the
        # corrupted env var first.
        tool = AgentChatTool()
        result = await tool.execute(
            {
                "message": "What is 2+2?",
                "session_id": "test-auth-failure-live-probe-4b",
                # Force the Anthropic provider so we know which key matters.
                "model": "anthropic:claude-haiku-4-5",
            },
        )
        # Probe 4b: must NOT silently succeed.
        assert result.isError, (
            f"Invalid ANTHROPIC_API_KEY produced isError={result.isError}, "
            f"text={result.text!r}. This is the SOC2 CC7.2 alerting bug "
            f"from skeptic probe 4b — see docs/design/skeptic-prod-ops-findings.md."
        )
        text = result.text or ""
        # The agent-friendly error contract: name the credential and an
        # alternative recovery path.
        assert "ANTHROPIC_API_KEY" in text, (
            f"Error message must name the credential to set; got: {text!r}"
        )
        assert ERROR_KIND_AUTH in text, (
            f"Error message must include the structured error kind; got: {text!r}"
        )
        assert "kaos-llm-core-call" in text, (
            f"Error message must point to the alternative tool; got: {text!r}"
        )
        # Defensive: the resulting runtime is still usable for cleanup.
        del runtime
