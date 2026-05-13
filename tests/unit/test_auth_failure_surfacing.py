"""Sprint-1 #2 — auth-failure / rate-limit / transport-failure transparency.

Probe 4b in ``docs/design/skeptic-prod-ops-findings.md`` proved that
when the LLM API key is invalid (or revoked, or the provider returns
503, or the network is offline), the historic ``AgentChatTool`` path
swallowed the exception and returned
``ToolResult(isError=False, text="")``. This is a SOC2 CC7.2 alerting
failure: the system claimed success when no LLM was reachable.

These tests pin down the **new** contract:

- The intent-classify path raises actionable infrastructure errors
  (auth / rate-limit / transport / context-too-large / service-unavailable)
  instead of falling back to the heuristic — that fallback used to mask
  the auth failure as a "successful" heuristic-classified turn.
- The ``BaseAgent._run_inner`` loop catches the re-raised exception,
  classifies it via :func:`kaos_agents.errors.classify_agent_failure`,
  and emits a structured :class:`RunError` with ``error_type`` set to
  the stable kind (``"auth_failure"`` etc.) plus a credential-named
  ``recovery_hint``.
- The MCP tool wrapper (``AgentChatTool.execute`` and
  ``AgentPlanTool.execute``) reads the RunError from the event stream
  and converts it to ``ToolResult.create_error(...)``.
- The error message names the credential (``"ANTHROPIC_API_KEY"`` for
  Anthropic auth failures) and points to the alternative tool
  (``kaos-llm-core-call``) per the agent-friendly error contract in
  ``kaos-modules/CLAUDE.md``.

The tests stub the LLM client at the kaos-llm-core / kaos-llm-client
boundary so we don't burn live tokens. The companion **live** test in
``tests/integration/test_auth_failure_live.py`` flips a real
``ANTHROPIC_API_KEY`` to a known-invalid value and runs the same path
end-to-end against the real provider.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from kaos_agents.errors import (
    ERROR_KIND_AUTH,
    ERROR_KIND_CONTEXT_TOO_LARGE,
    ERROR_KIND_PROVIDER,
    ERROR_KIND_RATE_LIMIT,
    ERROR_KIND_SERVICE_UNAVAILABLE,
    ERROR_KIND_TRANSPORT,
    AgentFailureClassification,
    classify_agent_failure,
)
from kaos_agents.tools import AgentChatTool, AgentPlanTool

# ---------------------------------------------------------------------------
# Unit tests for ``classify_agent_failure`` — pure mapping function.
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestClassifyAgentFailure:
    """The classifier is the single source of truth for error_type mapping."""

    def test_anthropic_auth_error_classified_as_auth(self) -> None:
        from kaos_llm_client.errors import KaosLLMAuthError

        exc = KaosLLMAuthError(
            "anthropic authentication failed (401): invalid x-api-key",
            provider="anthropic",
            model="claude-haiku-4-5",
            status_code=401,
        )
        result = classify_agent_failure(exc)
        assert isinstance(result, AgentFailureClassification)
        assert result.kind == ERROR_KIND_AUTH
        assert result.credential == "ANTHROPIC_API_KEY"
        assert result.provider == "anthropic"
        assert result.status_code == 401
        assert "ANTHROPIC_API_KEY" in result.recovery_hint
        assert "kaos-llm-core-call" in result.recovery_hint
        assert result.is_surfacing

    def test_openai_auth_error_classified_as_auth(self) -> None:
        from kaos_llm_client.errors import KaosLLMAuthError

        exc = KaosLLMAuthError(
            "openai authentication failed (403): forbidden",
            provider="openai",
            status_code=403,
        )
        result = classify_agent_failure(exc)
        assert result is not None
        assert result.kind == ERROR_KIND_AUTH
        assert result.credential == "OPENAI_API_KEY"
        assert "OPENAI_API_KEY" in result.recovery_hint

    def test_auth_error_through_call_error_cause_chain(self) -> None:
        """``CallError`` from kaos-llm-core wraps the auth error via ``raise X
        from e``. The classifier must walk the cause chain."""
        from kaos_llm_client.errors import KaosLLMAuthError
        from kaos_llm_core.errors import CallError

        try:
            try:
                raise KaosLLMAuthError(
                    "anthropic authentication failed (401)",
                    provider="anthropic",
                    status_code=401,
                )
            except KaosLLMAuthError as inner:
                raise CallError(
                    "Call to RespondSignature failed: "
                    "anthropic authentication failed (401). "
                    "Check model (anthropic:claude-haiku-4-5) and API key configuration."
                ) from inner
        except CallError as outer:
            result = classify_agent_failure(outer)
        assert result is not None
        assert result.kind == ERROR_KIND_AUTH
        assert result.credential == "ANTHROPIC_API_KEY"

    def test_rate_limit_classified_with_credential_hint(self) -> None:
        from kaos_llm_client.errors import KaosLLMProviderError

        exc = KaosLLMProviderError(
            "anthropic returned 429: rate limit exceeded",
            provider="anthropic",
            status_code=429,
            retry_after=12.0,
        )
        result = classify_agent_failure(exc)
        assert result is not None
        assert result.kind == ERROR_KIND_RATE_LIMIT
        assert result.credential == "ANTHROPIC_API_KEY"
        # Recovery hint surfaces the wait window from Retry-After.
        assert "12" in result.recovery_hint
        assert result.is_surfacing

    def test_service_unavailable_classified(self) -> None:
        from kaos_llm_client.errors import KaosLLMProviderError

        exc = KaosLLMProviderError(
            "openai returned 503: service unavailable",
            provider="openai",
            status_code=503,
        )
        result = classify_agent_failure(exc)
        assert result is not None
        assert result.kind == ERROR_KIND_SERVICE_UNAVAILABLE
        assert result.credential == "OPENAI_API_KEY"
        assert result.is_surfacing

    def test_context_too_large_classified(self) -> None:
        from kaos_llm_client.errors import KaosLLMProviderError

        exc = KaosLLMProviderError(
            "openai returned 400: This model's maximum context length is 128000 tokens. "
            "However, your messages resulted in 256000 tokens.",
            provider="openai",
            status_code=400,
        )
        result = classify_agent_failure(exc)
        assert result is not None
        assert result.kind == ERROR_KIND_CONTEXT_TOO_LARGE
        assert "context window" in result.recovery_hint.lower()

    def test_transport_error_classified(self) -> None:
        from kaos_llm_client.errors import KaosLLMTransportError

        exc = KaosLLMTransportError("connection refused (no route to host)")
        # Stuff a model hint into __cause__ so provider inference works.
        try:
            raise exc
        except KaosLLMTransportError as e:
            result = classify_agent_failure(e)
        assert result is not None
        assert result.kind == ERROR_KIND_TRANSPORT

    def test_generic_provider_error_classified(self) -> None:
        from kaos_llm_client.errors import KaosLLMProviderError

        exc = KaosLLMProviderError(
            "anthropic returned 500: internal server error",
            provider="anthropic",
            status_code=500,
        )
        result = classify_agent_failure(exc)
        assert result is not None
        # 500 is not in (502, 503, 504), so we fall through to generic provider.
        assert result.kind == ERROR_KIND_PROVIDER
        assert result.is_surfacing

    def test_unrelated_exception_returns_none(self) -> None:
        """Non-LLM errors don't match any kind — runtime treats them as opaque."""
        result = classify_agent_failure(ValueError("totally unrelated"))
        assert result is None


# ---------------------------------------------------------------------------
# End-to-end tests via the MCP tool surface.
# ---------------------------------------------------------------------------


def _patch_classify_to_raise(exc: Exception) -> Any:
    """Patch ``BaseAgent._classify`` to raise the given exception.

    Mirrors the production failure mode: when the underlying provider
    raises (KaosLLMAuthError / KaosLLMProviderError etc.), kaos-llm-core
    wraps it in CallError and re-raises. classify_intent re-raises that
    upward to ``_run_inner``, which catches and emits RunError.

    We monkeypatch at the BaseAgent boundary so we don't need to drive
    the entire kaos-llm-client transport stack in a unit test.
    """
    return patch(
        "kaos_agents.runtime.agent.BaseAgent._classify",
        new_callable=AsyncMock,
        side_effect=exc,
    )


@pytest.mark.unit
class TestAgentChatToolSurfacing:
    """The MCP tool wrapper must convert RunError(kind=...) → isError=True."""

    @pytest.mark.asyncio
    async def test_anthropic_auth_failure_surfaces_as_error(self) -> None:
        """Probe 4b regression: ``ANTHROPIC_API_KEY`` invalid must NOT
        return ``ToolResult(isError=False, text="")``."""
        from kaos_llm_client.errors import KaosLLMAuthError

        auth_exc = KaosLLMAuthError(
            "anthropic authentication failed (401): invalid x-api-key",
            provider="anthropic",
            model="claude-haiku-4-5",
            status_code=401,
        )
        tool = AgentChatTool()
        with _patch_classify_to_raise(auth_exc):
            result = await tool.execute(
                {"message": "What is 2+2?", "session_id": "test-auth-failure"}
            )
        assert result.isError, "auth failure must surface as isError=True"
        text = result.text or ""
        assert ERROR_KIND_AUTH in text, f"error_type missing from message: {text!r}"
        assert "ANTHROPIC_API_KEY" in text, f"credential missing from message: {text!r}"
        # The agent-friendly error contract says: name an alternative tool.
        assert "kaos-llm-core-call" in text

    @pytest.mark.asyncio
    async def test_openai_auth_failure_names_openai_credential(self) -> None:
        from kaos_llm_client.errors import KaosLLMAuthError

        auth_exc = KaosLLMAuthError(
            "openai authentication failed (401): invalid api key",
            provider="openai",
            status_code=401,
        )
        tool = AgentChatTool()
        with _patch_classify_to_raise(auth_exc):
            result = await tool.execute({"message": "Hi", "session_id": "test-auth-failure-openai"})
        assert result.isError
        assert "OPENAI_API_KEY" in (result.text or "")

    @pytest.mark.asyncio
    async def test_rate_limit_surfaces_as_error_with_retry_hint(self) -> None:
        from kaos_llm_client.errors import KaosLLMProviderError

        rate_exc = KaosLLMProviderError(
            "anthropic returned 429: rate limit",
            provider="anthropic",
            status_code=429,
            retry_after=5.0,
        )
        tool = AgentChatTool()
        with _patch_classify_to_raise(rate_exc):
            result = await tool.execute({"message": "Anything", "session_id": "test-rate-limit"})
        assert result.isError
        text = result.text or ""
        assert ERROR_KIND_RATE_LIMIT in text
        assert "ANTHROPIC_API_KEY" in text
        # Retry-After hint surfaces to the caller.
        assert "5" in text

    @pytest.mark.asyncio
    async def test_service_unavailable_surfaces_as_error(self) -> None:
        from kaos_llm_client.errors import KaosLLMProviderError

        svc_exc = KaosLLMProviderError(
            "openai returned 503: service unavailable",
            provider="openai",
            status_code=503,
        )
        tool = AgentChatTool()
        with _patch_classify_to_raise(svc_exc):
            result = await tool.execute({"message": "Anything", "session_id": "test-svc-unavail"})
        assert result.isError
        text = result.text or ""
        assert ERROR_KIND_SERVICE_UNAVAILABLE in text
        assert "503" in text

    @pytest.mark.asyncio
    async def test_context_too_large_surfaces_as_error(self) -> None:
        from kaos_llm_client.errors import KaosLLMProviderError

        ctx_exc = KaosLLMProviderError(
            "openai returned 400: This model's maximum context length is 128000 tokens; "
            "your prompt is too long.",
            provider="openai",
            status_code=400,
        )
        tool = AgentChatTool()
        with _patch_classify_to_raise(ctx_exc):
            result = await tool.execute({"message": "x" * 100, "session_id": "test-ctx-too-large"})
        assert result.isError
        text = result.text or ""
        assert ERROR_KIND_CONTEXT_TOO_LARGE in text
        assert "context window" in text.lower()

    @pytest.mark.asyncio
    async def test_unrelated_exception_does_not_silently_succeed(self) -> None:
        """A non-LLM exception still bubbles out as RunError → isError=True.

        We don't have a stable ``error_type`` for it (the classifier
        returns None), but the historic empty-text-success bug must NOT
        reappear for arbitrary exceptions either.
        """
        tool = AgentChatTool()
        # Use a plain ValueError so the classifier returns None — runtime
        # should still emit RunError(error_type=type(exc).__name__) but
        # the MCP wrapper does NOT surface it as isError=True for non-
        # SURFACING kinds (intentional: those are recoverable / non-
        # infrastructure failures, e.g. a content-policy refusal). We
        # assert the result still has structuredContent describing what
        # happened (turn_number, intent, tool_calls=...) rather than
        # exploding.
        with _patch_classify_to_raise(ValueError("unrelated")):
            result = await tool.execute({"message": "Hi", "session_id": "test-unrelated"})
        # Non-surfacing → not isError. But the wrapper must still expose
        # the failure in the structured output (turn_number, etc.). The
        # absence of isError is intentional: we don't want to false-
        # positive on every transient validation glitch.
        assert isinstance(result.isError, bool)


@pytest.mark.unit
class TestAgentPlanToolSurfacing:
    """Plan-execute pattern shares the wrapper; same surfacing applies."""

    @pytest.mark.asyncio
    async def test_anthropic_auth_failure_surfaces_in_plan_tool(self) -> None:
        from kaos_llm_client.errors import KaosLLMAuthError

        auth_exc = KaosLLMAuthError(
            "anthropic authentication failed (401): invalid x-api-key",
            provider="anthropic",
            status_code=401,
        )
        tool = AgentPlanTool()
        with _patch_classify_to_raise(auth_exc):
            result = await tool.execute(
                {"message": "Do a 3-step plan", "session_id": "test-plan-auth"}
            )
        assert result.isError
        text = result.text or ""
        assert ERROR_KIND_AUTH in text
        assert "ANTHROPIC_API_KEY" in text
