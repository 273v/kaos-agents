"""HTTP API settings for kaos-agents (KC17-P0-3).

The FastAPI surface in :mod:`kaos_agents.api.server` previously shipped
with NO auth, NO tenant scoping, and CORS configured as wildcard +
credentials. That was a foot-gun: anyone who could reach the port could
fire ``POST /v1/runs/{run_id}/approve`` and bypass the human-in-the-loop
tool-approval gate, read or delete any session, and exfiltrate session
memory.

This module owns the typed settings that drive the auth + CORS hardening
landed in KC17-P0-3:

- ``KAOS_AGENTS_API_TOKEN`` — bearer token for the ``Authorization: Bearer``
  header. Constant-time compared. When set, the token also drives tenant
  scoping (sessions created by token A are 404 to token B).
- ``KAOS_AGENTS_API_ALLOW_UNAUTH_LOCALHOST`` — only honored when the token
  is unset. Permits requests from 127.0.0.1 / ::1 with a loud per-request
  warning log. Intended strictly for local development.
- ``KAOS_AGENTS_API_CORS_ALLOW_ORIGINS`` — explicit origin list; defaults
  to empty (no CORS). Wildcard ``*`` is silently dropped when credentials
  are enabled because FastAPI/Starlette accepts it but browsers do not.

The startup contract: with neither the token nor the localhost-dev flag
set, ``create_app()`` raises :class:`InsecureApiConfigurationError` with
a remediation message pointing at the env vars + README. There is no
silent insecure default.
"""

from __future__ import annotations

import hashlib
import hmac
import os
from typing import Any

from kaos_core.config import ModuleSettings
from pydantic import Field, SecretStr, model_validator
from pydantic_settings import SettingsConfigDict


class InsecureApiConfigurationError(RuntimeError):
    """Raised by :func:`create_app` when no auth source is configured.

    Either ``KAOS_AGENTS_API_TOKEN`` must be set (bearer-token auth) or
    ``KAOS_AGENTS_API_ALLOW_UNAUTH_LOCALHOST=1`` must be set (local-only
    dev mode). Refusing to start with no auth is intentional — silent
    "0.0.0.0 + no token" was the pre-KC17 footgun.
    """


class KaosAgentsApiSettings(ModuleSettings):
    """Typed configuration for the kaos-agents FastAPI surface.

    All fields are resolved via the standard KAOS settings hierarchy:
    explicit overrides → ``KaosContext._config`` → ``KAOS_AGENTS_API_*``
    env vars → ``.env`` → defaults.
    """

    api_token: SecretStr | None = Field(
        default=None,
        description=(
            "Bearer token required on the Authorization header. When set, "
            "all endpoints require ``Authorization: Bearer <token>``. The "
            "token also derives a tenant id that scopes session and run "
            "lookups so token A cannot read token B's sessions."
        ),
    )

    api_allow_unauth_localhost: bool = Field(
        default=False,
        description=(
            "When True AND api_token is unset, permit unauthenticated "
            "requests from 127.0.0.1 / ::1 only. Emits a loud per-request "
            "warning. Intended for local development; never enable in "
            "production. Has no effect when api_token is set."
        ),
    )

    api_cors_allow_origins: list[str] = Field(
        default_factory=list,
        description=(
            "Explicit CORS origins allowed by the API. Default is empty "
            "(no CORS — browser cross-origin requests blocked). Set via "
            "env var as a comma-separated list, e.g. "
            "'https://app.example.com,https://staging.example.com'. "
            "Wildcard '*' with credentials enabled is rejected — Starlette "
            "permits it but the W3C CORS spec forbids it."
        ),
    )

    model_config = SettingsConfigDict(
        env_prefix="KAOS_AGENTS_API_",
        env_file=".env",
        extra="ignore",
    )

    def __init__(self, **kwargs: Any) -> None:
        # pydantic-settings parses ``list[str]`` env vars as JSON by
        # default. Comma-separated is the ergonomic form for shell-env
        # config, so we intercept the env var before pydantic-settings
        # sees it and pre-split. The kwarg form (constructor call with
        # a real list) still works; env-only callers get CSV parsing.
        if "api_cors_allow_origins" not in kwargs:
            env_raw = os.environ.get("KAOS_AGENTS_API_CORS_ALLOW_ORIGINS")
            if env_raw is not None and env_raw.strip():
                kwargs["api_cors_allow_origins"] = [
                    item.strip() for item in env_raw.split(",") if item.strip()
                ]
        super().__init__(**kwargs)

    @model_validator(mode="before")
    @classmethod
    def _documented_env_names(cls, values: Any) -> Any:
        """Honor the documented env-var names that drop the redundant ``API_`` prefix.

        With ``env_prefix="KAOS_AGENTS_API_"`` and a field literally named
        ``api_token``, pydantic-settings concatenates them to
        ``KAOS_AGENTS_API_API_TOKEN`` (double ``API_``). The README,
        SECURITY.md, the class docstring above, and the
        :meth:`insecure_startup_error` message all advertise the cleaner
        names ``KAOS_AGENTS_API_TOKEN`` and
        ``KAOS_AGENTS_API_ALLOW_UNAUTH_LOCALHOST``. This validator reads
        those documented names from ``os.environ`` and back-fills the
        fields when the caller didn't pass an explicit kwarg, so the docs
        and the runtime agree. Same legacy-fallback shape as kaos-web's
        ``SERPAPI_API_KEY`` / ``BRAVE_API_KEY`` validators.
        """
        if not isinstance(values, dict):
            return values
        if "api_token" not in values:
            token_env = os.environ.get("KAOS_AGENTS_API_TOKEN")
            if token_env:
                values["api_token"] = token_env
        if "api_allow_unauth_localhost" not in values:
            flag_env = os.environ.get("KAOS_AGENTS_API_ALLOW_UNAUTH_LOCALHOST")
            if flag_env is not None:
                values["api_allow_unauth_localhost"] = flag_env.lower() in (
                    "1",
                    "true",
                    "yes",
                    "on",
                )
        return values

    @model_validator(mode="before")
    @classmethod
    def _parse_cors_origins_csv(cls, values: dict[str, Any]) -> dict[str, Any]:
        """Normalize a CSV string passed via the constructor kwarg form.

        Defense in depth alongside ``__init__``: the constructor-side
        interception handles the env-var path; this handles a kwarg
        that snuck in as a string.
        """
        if isinstance(values, dict):
            raw = values.get("api_cors_allow_origins")
            if isinstance(raw, str):
                values["api_cors_allow_origins"] = [
                    item.strip() for item in raw.split(",") if item.strip()
                ]
        return values

    @model_validator(mode="after")
    def _reject_wildcard_with_credentials(self) -> KaosAgentsApiSettings:
        """A wildcard CORS origin combined with credentials is broken.

        The W3C CORS spec forbids ``Access-Control-Allow-Origin: *`` when
        ``Access-Control-Allow-Credentials: true``. Starlette permits it
        but browsers will reject the response. We treat it as a config
        error at startup rather than producing a silently-broken API.
        """
        if "*" in self.api_cors_allow_origins:
            raise ValueError(
                "KAOS_AGENTS_API_CORS_ALLOW_ORIGINS contains '*' which is "
                "incompatible with credentialed requests. List the explicit "
                "origins you want to allow instead, e.g. "
                "'https://app.example.com'. See "
                "https://developer.mozilla.org/en-US/docs/Web/HTTP/CORS/Errors/CORSNotSupportingCredentials"
            )
        return self

    # ------------------------------------------------------------------
    # Auth helpers
    # ------------------------------------------------------------------

    def is_auth_configured(self) -> bool:
        """True when at least one auth source is configured.

        Either a bearer token or the localhost-dev escape hatch must be
        present, or :func:`create_app` refuses to start.
        """
        return self.api_token is not None or self.api_allow_unauth_localhost

    def check_token(self, presented: str | None) -> bool:
        """Constant-time compare ``presented`` to the configured token.

        Returns False when no token is configured or no token is presented.
        """
        if self.api_token is None or not presented:
            return False
        return hmac.compare_digest(self.api_token.get_secret_value(), presented)

    def tenant_id(self) -> str | None:
        """Derive a stable tenant id from the configured token.

        SHA-256(token), first 12 hex chars. Used to namespace session
        and run lookups so token A cannot reach token B's sessions.
        Returns ``None`` when no token is configured (no tenant scoping
        in localhost-dev mode — sessions are unscoped on a single host).
        """
        if self.api_token is None:
            return None
        digest = hashlib.sha256(
            self.api_token.get_secret_value().encode("utf-8"),
        ).hexdigest()
        return digest[:12]

    def insecure_startup_error(self) -> InsecureApiConfigurationError:
        """Build the startup error for missing auth sources."""
        return InsecureApiConfigurationError(
            "kaos-agents HTTP API refuses to start without an auth source. "
            "Set ONE of:\n"
            "  - KAOS_AGENTS_API_TOKEN=<bearer-token> (production: clients "
            "send 'Authorization: Bearer <token>')\n"
            "  - KAOS_AGENTS_API_ALLOW_UNAUTH_LOCALHOST=1 (local development "
            "ONLY: binds 127.0.0.1, emits per-request warning logs)\n"
            "See kaos-agents/SECURITY.md and the README 'HTTP API' section "
            "for the full auth + tenant scoping contract. Pre-KC17-P0-3 "
            "behaviour (no auth + wildcard CORS + credentials) was a "
            "publicly-reachable footgun and is no longer permitted."
        )


__all__ = [
    "InsecureApiConfigurationError",
    "KaosAgentsApiSettings",
    "scope_session_id",
]


def scope_session_id(session_id: str, tenant_id: str | None) -> str:
    """Apply tenant scoping to a session_id.

    When ``tenant_id`` is None (localhost-dev mode), returns the
    session_id unchanged. When set, returns ``f"{tenant_id}:{session_id}"``
    so a token-A session is distinct from a token-B session with the
    same caller-supplied id. Lookups by token B for an A-tenant session
    return 404, not 403, to avoid leaking existence across tenants.

    Note: callers should compute ``effective_id = scope_session_id(id, tenant)``
    once at the API edge and use it everywhere — VFS reads/writes,
    SessionStore.exists/load/delete. The tenant prefix flows through
    transparently because everything is keyed by the same string.
    """
    if tenant_id is None or not session_id:
        return session_id
    if os.sep in session_id or "/" in session_id:
        # Defensive — session ids must not contain path separators or
        # the tenant prefix could be sidestepped by crafted ids.
        raise ValueError(
            "session_id must not contain path separators. Got "
            f"{session_id!r} which would defeat tenant scoping."
        )
    return f"{tenant_id}:{session_id}"
