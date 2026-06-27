"""Exception hierarchy for kaos-agents.

The :class:`AgentFailureClassification` helper at the bottom of this
module is the single source of truth for translating an arbitrary
exception (typically raised by ``kaos-llm-core`` :class:`CallError` or
its ``__cause__`` chain into ``kaos-llm-client`` provider/auth/transport
errors) into a stable agent-facing ``error_type`` + actionable
``recovery_hint``. The runtime emits this on :class:`RunError`, and the
MCP tool wrapper converts it to a ``ToolResult.create_error(...)``.

This exists because a silent ``ToolResult(isError=False, text="")`` on
authentication failure is a **transparency failure** (skeptic probe 4b
in ``docs/design/skeptic-prod-ops-findings.md``): the agent claimed
success when no LLM was reachable. SOC2 CC7.2 alerting depends on the
agent telling the truth about whether the run produced grounded output.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from kaos_core.exceptions import KaosCoreError


class KaosAgentError(KaosCoreError):
    """Base exception for all kaos-agents errors."""


class SessionNotFoundError(KaosAgentError):
    """Raised when a session cannot be found in the store."""


class SessionCorruptedError(KaosAgentError):
    """Raised when a persisted session cannot be deserialized."""


class MemoryBudgetExceededError(KaosAgentError):
    """Raised when a memory operation would exceed the section's token budget."""


class EvictionError(KaosAgentError):
    """Raised when eviction fails to free sufficient space."""


class SectionNotConfiguredError(KaosAgentError, KeyError):
    """Raised when accessing a section not in the memory profile.

    Inherits KeyError for stdlib compatibility (dict-like interface).
    """


class EventSerializationError(KaosAgentError):
    """Raised when an event cannot be serialized to dict/JSON."""


class EventDeserializationError(KaosAgentError):
    """Raised when a dict/JSON payload cannot be deserialized to an KaosEvent.

    Covers: missing ``type`` field, unknown event type, missing required
    fields, type mismatches in nested structures.
    """


class VisionOcrUnavailableError(KaosAgentError):
    """Raised when VLM-based OCR is requested but its dependencies are missing.

    The VLM OCR engine (:class:`kaos_agents.runtime.ocr_engines.VlmOcrEngine`)
    needs ``kaos-llm-core[vision]`` (which pulls ``kaos-content[images]``) plus
    a configured provider API key. The message names the missing piece, the
    install command to fix it, and the Tesseract-only fallback.
    """


# ---------------------------------------------------------------------------
# LLM-failure classification (transparency / SOC2 alerting surface).
# ---------------------------------------------------------------------------

#: Auth-failure kind: API key missing/invalid/revoked or provider returned 401/403.
ERROR_KIND_AUTH = "auth_failure"
#: Rate-limit kind: provider returned 429 (or asked client to back off).
ERROR_KIND_RATE_LIMIT = "rate_limit"
#: Service-unavailable kind: provider returned 503 or other transient 5xx.
ERROR_KIND_SERVICE_UNAVAILABLE = "service_unavailable"
#: Context-too-large kind: provider rejected the request as exceeding context window.
ERROR_KIND_CONTEXT_TOO_LARGE = "context_too_large"
#: Generic transport kind: network/connection failure with no narrower classifier.
ERROR_KIND_TRANSPORT = "transport_error"
#: Generic provider kind: a 4xx/5xx we didn't narrow further.
ERROR_KIND_PROVIDER = "provider_error"

#: Set of "actionable failure" kinds that the MCP tool wrapper MUST surface
#: as ``ToolResult.isError=True`` rather than as a silent empty success.
#: A run that ends in any of these states is not a successful run, even if
#: dispatch produced ``response_text == ""``.
SURFACING_FAILURE_KINDS: frozenset[str] = frozenset(
    {
        ERROR_KIND_AUTH,
        ERROR_KIND_RATE_LIMIT,
        ERROR_KIND_SERVICE_UNAVAILABLE,
        ERROR_KIND_CONTEXT_TOO_LARGE,
        ERROR_KIND_TRANSPORT,
        ERROR_KIND_PROVIDER,
    }
)


@dataclass(frozen=True, slots=True)
class AgentFailureClassification:
    """Typed classification of an LLM/transport failure.

    Produced by :func:`classify_agent_failure`; consumed by the
    runtime ``RunError`` emission path and the MCP tool wrapper. Carries
    just enough structured data to produce a SOC2-grade alert message:
    *what kind of failure it was*, *which credential is implicated*
    (when applicable), and *what the operator should do next*.

    ``kind`` is one of the ``ERROR_KIND_*`` constants. ``credential`` is
    the environment-variable name (e.g. ``"ANTHROPIC_API_KEY"``) when we
    can identify the missing/invalid key — ``None`` for non-auth
    failures or when we couldn't determine the provider.
    """

    kind: str
    credential: str | None
    recovery_hint: str
    provider: str | None = None
    status_code: int | None = None

    @property
    def is_surfacing(self) -> bool:
        """True when this failure must propagate as ``isError=True``."""
        return self.kind in SURFACING_FAILURE_KINDS


# Map provider name (as kaos-llm-client tags errors) → primary credential env var.
# Mirrors the table in kaos-modules CLAUDE.md "Credentials & API Keys".
_PROVIDER_TO_CREDENTIAL: dict[str, str] = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "google": "GOOGLE_API_KEY",
    "xai": "XAI_API_KEY",
    "groq": "GROQ_API_KEY",
    "mistral": "MISTRAL_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
}

# Tokens we look for in stringified exceptions to infer the provider when
# the exception doesn't carry a ``provider`` attribute (e.g. when
# kaos-llm-core has wrapped a kaos-llm-client error inside a generic
# CallError text body).
_PROVIDER_TOKENS = (
    "anthropic",
    "openai",
    "google",
    "gemini",
    "xai",
    "grok",
    "groq",
    "mistral",
    "openrouter",
)

_CONTEXT_TOO_LARGE_PATTERNS = (
    re.compile(r"context[_\s-]?length", re.IGNORECASE),
    re.compile(r"context[_\s-]?window", re.IGNORECASE),
    re.compile(r"maximum context", re.IGNORECASE),
    re.compile(r"too many tokens", re.IGNORECASE),
    re.compile(r"prompt is too long", re.IGNORECASE),
    re.compile(r"\binput too long\b", re.IGNORECASE),
    re.compile(r"max(imum)? tokens? exceeded", re.IGNORECASE),
)


def _walk_cause_chain(exc: BaseException) -> list[BaseException]:
    """Return ``[exc, exc.__cause__, exc.__context__, ...]`` flattened.

    ``kaos-llm-core.CallError`` always wraps the underlying
    ``kaos-llm-client`` exception via ``raise CallError(...) from e``,
    so the auth/rate-limit/etc. classification lives on the cause chain,
    not on the outermost exception. We walk both ``__cause__`` and
    ``__context__`` (some libraries set one but not the other) with a
    cycle guard to be safe.
    """
    chain: list[BaseException] = []
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        chain.append(current)
        # Prefer explicit __cause__ (raise X from Y) over __context__.
        nxt = current.__cause__ or current.__context__
        current = nxt
    return chain


def _attr_or_detail(exc: BaseException, name: str) -> Any:
    """Return ``exc.<name>`` if it's set as a real attribute, else read it
    out of ``exc.details`` (the ``KaosCoreError`` convention).

    ``KaosLLMAuthError`` and friends pass everything through to
    ``KaosCoreError(message, **details)`` without redefining ``__init__``,
    so ``provider`` / ``status_code`` / ``retry_after`` live on
    ``self.details`` rather than as direct attributes. Only
    ``KaosLLMProviderError`` and ``KaosLLMRetryExhaustedError`` promote
    them to attributes. The classifier needs to read both shapes.
    """
    direct = getattr(exc, name, None)
    if direct is not None:
        return direct
    details = getattr(exc, "details", None)
    if isinstance(details, dict):
        return details.get(name)
    return None


def _infer_provider(text: str, attrs: list[BaseException]) -> str | None:
    """Try to recover the provider name from an exception chain.

    Order of precedence:
    1. ``exc.provider`` attribute or ``exc.details["provider"]`` (the
       kaos-llm-client error convention; see :func:`_attr_or_detail`).
    2. ``exc.model`` parsed as ``<provider>:<model_id>``.
    3. Lowercase substring scan of the joined exception text.
    """
    for e in attrs:
        provider = _attr_or_detail(e, "provider")
        if isinstance(provider, str) and provider:
            return provider.lower()
        model = _attr_or_detail(e, "model")
        if isinstance(model, str) and ":" in model:
            return model.split(":", 1)[0].lower()
    lowered = text.lower()
    for token in _PROVIDER_TOKENS:
        if token in lowered:
            # Normalize known aliases.
            if token in ("gemini",):
                return "google"
            if token == "grok":
                return "xai"
            return token
    return None


def _format_call_alternative(credential: str | None) -> str:
    """Return the standard 'alternative tool' hint for actionable errors."""
    if credential:
        return (
            "Alternatively, call kaos-llm-core-call directly with an explicit api_key "
            "argument to bypass the environment variable."
        )
    return "Alternatively, call kaos-llm-core-call directly with explicit credentials."


def classify_agent_failure(exc: BaseException) -> AgentFailureClassification | None:
    """Classify an arbitrary exception as an actionable agent failure.

    Returns ``None`` when the exception does not match any known failure
    surface — the caller should treat it as a generic unexpected error
    (and still surface ``isError=True``, but without the structured
    ``error_type`` envelope this function produces).

    Walks the exception's ``__cause__`` / ``__context__`` chain so that
    a ``CallError("Call to ... failed: anthropic authentication failed
    (401): ...")`` whose cause is ``KaosLLMAuthError`` still classifies
    as :data:`ERROR_KIND_AUTH`. Class-name matching is the primary
    signal (so we don't break when the cause chain is re-wrapped); text
    matching is the secondary signal for in-band CallError messages
    that lost their original cause.
    """
    chain = _walk_cause_chain(exc)
    chain_text = " :: ".join(str(e) for e in chain)
    class_names = {type(e).__name__ for e in chain}
    provider = _infer_provider(chain_text, chain)
    credential = _PROVIDER_TO_CREDENTIAL.get(provider) if provider else None

    # Probe HTTP status code + Retry-After from any KaosLLM*Error in the
    # chain. ``KaosLLMAuthError`` carries them via ``self.details``,
    # ``KaosLLMProviderError`` promotes them to direct attributes — the
    # helper reads either shape.
    status_code: int | None = None
    retry_after: float | None = None
    for e in chain:
        code = _attr_or_detail(e, "status_code")
        if isinstance(code, int) and status_code is None:
            status_code = code
        ra = _attr_or_detail(e, "retry_after")
        if isinstance(ra, int | float) and retry_after is None:
            retry_after = float(ra)

    # --- Authentication failure (401/403, missing key, KaosLLMAuthError) ---
    if "KaosLLMAuthError" in class_names or status_code in (401, 403):
        cred_label = credential or "the provider's API key"
        hint = (
            f"Set a valid {provider.capitalize() if provider else 'LLM'} API key "
            f"in the environment as {cred_label} (or the legacy fallback name "
            f"documented in CLAUDE.md). {_format_call_alternative(credential)}"
        )
        return AgentFailureClassification(
            kind=ERROR_KIND_AUTH,
            credential=credential,
            recovery_hint=hint,
            provider=provider,
            status_code=status_code,
        )

    # --- Rate limit (HTTP 429) ---
    if status_code == 429 or "rate limit" in chain_text.lower():
        wait_hint = (
            f" The provider asked the client to wait ~{retry_after:.0f}s before retrying."
            if retry_after
            else ""
        )
        cred_hint = f" The credential in use is {credential}." if credential else ""
        hint = (
            f"The {provider or 'LLM'} provider is rate-limiting this account.{cred_hint}"
            f"{wait_hint} Retry after a short delay, reduce concurrent calls, or upgrade "
            "the plan. " + _format_call_alternative(credential)
        )
        return AgentFailureClassification(
            kind=ERROR_KIND_RATE_LIMIT,
            credential=credential,
            recovery_hint=hint,
            provider=provider,
            status_code=status_code,
        )

    # --- Service unavailable (HTTP 503 / 502 / 504) ---
    if status_code in (502, 503, 504):
        cred_hint = f" The credential in use is {credential}." if credential else ""
        hint = (
            f"The {provider or 'LLM'} provider is temporarily unavailable "
            f"(HTTP {status_code}).{cred_hint} Retry after a brief delay. "
            + _format_call_alternative(credential)
        )
        return AgentFailureClassification(
            kind=ERROR_KIND_SERVICE_UNAVAILABLE,
            credential=credential,
            recovery_hint=hint,
            provider=provider,
            status_code=status_code,
        )

    # --- Context too large (HTTP 400 with body matching a known phrase) ---
    for pattern in _CONTEXT_TOO_LARGE_PATTERNS:
        if pattern.search(chain_text):
            hint = (
                "The request exceeded the model's context window. Reduce the prompt "
                "size, trim memory sections via kaos-agent-memory-clear, or switch to "
                "a model with a larger context window. " + _format_call_alternative(credential)
            )
            return AgentFailureClassification(
                kind=ERROR_KIND_CONTEXT_TOO_LARGE,
                credential=credential,
                recovery_hint=hint,
                provider=provider,
                status_code=status_code,
            )

    # --- Transport-level failure (no HTTP status; connection/timeout) ---
    if "KaosLLMTransportError" in class_names or "KaosLLMRetryExhaustedError" in class_names:
        hint = (
            f"Network transport to the {provider or 'LLM'} provider failed (no HTTP "
            "response received). Check connectivity, DNS, and firewall rules. "
            + _format_call_alternative(credential)
        )
        return AgentFailureClassification(
            kind=ERROR_KIND_TRANSPORT,
            credential=credential,
            recovery_hint=hint,
            provider=provider,
            status_code=status_code,
        )

    # --- Generic provider failure (HTTP 4xx/5xx we didn't otherwise classify) ---
    if "KaosLLMProviderError" in class_names or (status_code is not None and status_code >= 400):
        hint = (
            f"The {provider or 'LLM'} provider returned HTTP {status_code or '4xx/5xx'}. "
            "Inspect the error message, retry if transient, or fix the request. "
            + _format_call_alternative(credential)
        )
        return AgentFailureClassification(
            kind=ERROR_KIND_PROVIDER,
            credential=credential,
            recovery_hint=hint,
            provider=provider,
            status_code=status_code,
        )

    return None
