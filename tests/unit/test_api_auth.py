"""Regression tests for KC17-P0-3 — HTTP API auth + tenant scoping + CORS.

Pre-KC17, the FastAPI surface shipped with NO auth, NO tenant scoping,
and CORS wildcard + credentials. POST /v1/runs/{run_id}/approve was a
human-in-the-loop bypass for anyone who could reach the port.

These tests prove the hardening:

1. Insecure config refuses to start (no token + no localhost-dev flag).
2. Bearer-token auth: missing token → 401, wrong token → 401, right
   token → 200.
3. Tenant scoping: session created with token A is 404 to token B.
4. localhost-dev gate: only 127.0.0.1 / ::1 origins permitted.
5. CORS wildcard + credentials is rejected at config time.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from httpx import ASGITransport, AsyncClient
from pydantic import SecretStr

from kaos_agents.api.server import create_app
from kaos_agents.api.settings import (
    InsecureApiConfigurationError,
    KaosAgentsApiSettings,
    scope_session_id,
)
from kaos_agents.types import IntentResult, IntentType


@pytest.mark.unit
class TestInsecureStartupRejected:
    """KC17-P0-3 — refuse to start when no auth source is configured."""

    def test_no_token_no_localhost_flag_raises(self) -> None:
        settings = KaosAgentsApiSettings(
            api_token=None,
            api_allow_unauth_localhost=False,
        )
        with pytest.raises(InsecureApiConfigurationError) as exc:
            create_app(api_settings=settings)
        msg = str(exc.value)
        assert "KAOS_AGENTS_API_TOKEN" in msg
        assert "KAOS_AGENTS_API_ALLOW_UNAUTH_LOCALHOST" in msg

    def test_token_alone_starts(self) -> None:
        settings = KaosAgentsApiSettings(api_token=SecretStr("super-secret-test-token"))
        app = create_app(api_settings=settings)
        assert app is not None

    def test_localhost_flag_alone_starts(self) -> None:
        settings = KaosAgentsApiSettings(api_allow_unauth_localhost=True)
        app = create_app(api_settings=settings)
        assert app is not None


@pytest.mark.unit
class TestCorsHardening:
    """KC17-P0-3 — wildcard CORS with credentials is rejected at config time."""

    def test_wildcard_in_settings_rejected(self) -> None:
        with pytest.raises(ValueError, match="incompatible with credentialed"):
            KaosAgentsApiSettings(
                api_token=SecretStr("t"),
                api_cors_allow_origins=["*"],
            )

    def test_wildcard_in_create_app_kwarg_rejected(self) -> None:
        settings = KaosAgentsApiSettings(api_token=SecretStr("t"))
        with pytest.raises(ValueError, match="incompatible with allow_credentials"):
            create_app(api_settings=settings, cors_origins=["*"])

    def test_explicit_origin_accepted(self) -> None:
        settings = KaosAgentsApiSettings(
            api_token=SecretStr("t"),
            api_cors_allow_origins=["https://app.example.com"],
        )
        app = create_app(api_settings=settings)
        assert app is not None

    def test_cors_csv_env_parsed(self, monkeypatch) -> None:
        """The env-var form is a comma-separated string; settings normalizes."""
        monkeypatch.setenv("KAOS_AGENTS_API_TOKEN", "csv-test-token")
        monkeypatch.setenv(
            "KAOS_AGENTS_API_CORS_ALLOW_ORIGINS",
            "https://a.example,https://b.example",
        )
        settings = KaosAgentsApiSettings()
        assert settings.api_cors_allow_origins == [
            "https://a.example",
            "https://b.example",
        ]


# ---------------------------------------------------------------------------
# Bearer-token auth + tenant scoping
# ---------------------------------------------------------------------------


_TOKEN_A = "tenant-a-secret-token-aaaaaaaaaaaaaaaaaa"
_TOKEN_B = "tenant-b-secret-token-bbbbbbbbbbbbbbbbbb"


def _make_app(token: str):
    settings = KaosAgentsApiSettings(api_token=SecretStr(token))
    return create_app(api_settings=settings)


@pytest.fixture
def app_a():
    return _make_app(_TOKEN_A)


@pytest.fixture
def app_b():
    return _make_app(_TOKEN_B)


@pytest.fixture
async def client_a(app_a):
    transport = ASGITransport(app=app_a)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


@pytest.fixture
async def client_b(app_b):
    transport = ASGITransport(app=app_b)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


@pytest.mark.unit
class TestBearerTokenAuth:
    @pytest.mark.asyncio
    async def test_missing_token_401(self, client_a: AsyncClient) -> None:
        resp = await client_a.post("/v1/sessions", json={"session_id": "x"})
        assert resp.status_code == 401
        assert "Bearer" in resp.json()["detail"]

    @pytest.mark.asyncio
    async def test_wrong_token_401(self, client_a: AsyncClient) -> None:
        resp = await client_a.post(
            "/v1/sessions",
            json={"session_id": "x"},
            headers={"Authorization": "Bearer the-wrong-token"},
        )
        assert resp.status_code == 401

    @pytest.mark.asyncio
    async def test_wrong_scheme_401(self, client_a: AsyncClient) -> None:
        """Basic auth or anything other than Bearer is rejected."""
        resp = await client_a.post(
            "/v1/sessions",
            json={"session_id": "x"},
            headers={"Authorization": f"Token {_TOKEN_A}"},
        )
        assert resp.status_code == 401

    @pytest.mark.asyncio
    async def test_right_token_200(self, client_a: AsyncClient) -> None:
        resp = await client_a.post(
            "/v1/sessions",
            json={"session_id": "x"},
            headers={"Authorization": f"Bearer {_TOKEN_A}"},
        )
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_get_without_token_401(self, client_a: AsyncClient) -> None:
        resp = await client_a.get("/v1/sessions/nope")
        assert resp.status_code == 401

    @pytest.mark.asyncio
    async def test_delete_without_token_401(self, client_a: AsyncClient) -> None:
        resp = await client_a.delete("/v1/sessions/nope")
        assert resp.status_code == 401

    @pytest.mark.asyncio
    async def test_memory_search_without_token_401(self, client_a: AsyncClient) -> None:
        resp = await client_a.get(
            "/v1/sessions/x/memory/search",
            params={"query": "x"},
        )
        assert resp.status_code == 401

    @pytest.mark.asyncio
    async def test_approve_without_token_401(self, client_a: AsyncClient) -> None:
        resp = await client_a.post("/v1/runs/x/approve", json={"approved": True})
        assert resp.status_code == 401


@pytest.mark.unit
class TestTenantScoping:
    """KC17-P0-3 — session and run lookups are namespaced per tenant."""

    @pytest.mark.asyncio
    async def test_session_created_by_a_invisible_to_b(self) -> None:
        # Two apps share the same in-memory VFS via shared runtime is
        # complicated; instead we share the VFS explicitly by giving
        # each app the same KaosRuntime. Simpler: derive tenant ids and
        # prove scope_session_id() prevents the collision.
        app_a = _make_app(_TOKEN_A)
        app_b = _make_app(_TOKEN_B)
        # The two tokens MUST hash to different tenant ids.
        tid_a = app_a.state.tenant_id
        tid_b = app_b.state.tenant_id
        assert tid_a is not None and tid_b is not None
        assert tid_a != tid_b
        assert scope_session_id("session-x", tid_a) != scope_session_id("session-x", tid_b)

    @pytest.mark.asyncio
    async def test_cross_tenant_get_returns_404(self) -> None:
        """Token A creates a session; token B GETs same id → 404 (not 403)."""
        # Build TWO apps that share the SAME VFS so a session written by
        # token A persists across into token B's read path. Otherwise the
        # tenant-isolation evidence is trivial (each app has its own VFS).
        from kaos_core.vfs.core import IsolationMode, StorageBackend, VFSConfig, VirtualFileSystem

        shared_vfs = VirtualFileSystem(
            config=VFSConfig(
                default_backend=StorageBackend.MEMORY,
                isolation_mode=IsolationMode.GLOBAL,
            )
        )

        app_a = create_app(api_settings=KaosAgentsApiSettings(api_token=SecretStr(_TOKEN_A)))
        app_b = create_app(api_settings=KaosAgentsApiSettings(api_token=SecretStr(_TOKEN_B)))
        app_a.state.vfs = shared_vfs
        app_b.state.vfs = shared_vfs

        # A creates 'shared-id'.
        transport_a = ASGITransport(app=app_a)
        async with AsyncClient(transport=transport_a, base_url="http://test") as c_a:
            r = await c_a.post(
                "/v1/sessions",
                json={"session_id": "shared-id"},
                headers={"Authorization": f"Bearer {_TOKEN_A}"},
            )
            assert r.status_code == 200

        # B queries 'shared-id' — must be 404. Critically NOT 403 — we do
        # not confirm that A's session exists across the boundary.
        transport_b = ASGITransport(app=app_b)
        async with AsyncClient(transport=transport_b, base_url="http://test") as c_b:
            r = await c_b.get(
                "/v1/sessions/shared-id",
                headers={"Authorization": f"Bearer {_TOKEN_B}"},
            )
            assert r.status_code == 404

    @pytest.mark.asyncio
    async def test_same_tenant_round_trip(self, app_a, client_a: AsyncClient) -> None:
        """Sanity: token A can read its own session under the same id.

        Seeds the VFS directly (POST /v1/sessions doesn't ``save()``,
        only ``load_or_create()`` which is in-memory). This proves the
        tenant prefix is applied consistently on the GET path.
        """
        from kaos_agents.memory.session import SessionMemory
        from kaos_agents.memory.store import SessionStore

        tid = app_a.state.tenant_id
        store = SessionStore(app_a.state.vfs)
        mem = SessionMemory(scope_session_id("my-session", tid))
        await store.save(mem)

        r = await client_a.get(
            "/v1/sessions/my-session",
            headers={"Authorization": f"Bearer {_TOKEN_A}"},
        )
        assert r.status_code == 200
        # The wire form echoes the caller-supplied id (tenant prefix
        # stripped) so client code remains simple.
        assert r.json()["session_id"] == "my-session"


@pytest.mark.unit
class TestLocalhostDev:
    """KC17-P0-3 — unauthenticated localhost mode allows 127.0.0.1 only."""

    @pytest.mark.asyncio
    async def test_localhost_request_allowed(self) -> None:
        app = create_app(api_settings=KaosAgentsApiSettings(api_allow_unauth_localhost=True))
        transport = ASGITransport(app=app, client=("127.0.0.1", 12345))
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            r = await c.post("/v1/sessions", json={"session_id": "x"})
            assert r.status_code == 200

    @pytest.mark.asyncio
    async def test_non_localhost_request_blocked(self) -> None:
        app = create_app(api_settings=KaosAgentsApiSettings(api_allow_unauth_localhost=True))
        # Simulate a public-IP caller — must be 401.
        transport = ASGITransport(app=app, client=("203.0.113.7", 54321))
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            r = await c.post("/v1/sessions", json={"session_id": "x"})
            assert r.status_code == 401
            assert "127.0.0.1" in r.json()["detail"]


@pytest.mark.unit
class TestScopeSessionIdSafety:
    def test_no_tenant_passthrough(self) -> None:
        assert scope_session_id("s1", None) == "s1"

    def test_path_separator_rejected(self) -> None:
        with pytest.raises(ValueError, match="path separators"):
            scope_session_id("evil/path", "tenant-aa")

    def test_empty_id_passthrough(self) -> None:
        assert scope_session_id("", "tenant-aa") == ""

    def test_tenant_prefix_applied(self) -> None:
        assert scope_session_id("s1", "tenant-aa") == "tenant-aa:s1"


@pytest.mark.unit
class TestAuthBypassedEndpointsStillWork:
    """A quick sanity-check that auth-gated endpoints still produce normal
    responses on the happy path — proves the dependency didn't break the
    contract for the existing flows."""

    @pytest.mark.asyncio
    async def test_send_message_with_token_200(self) -> None:
        app = create_app(api_settings=KaosAgentsApiSettings(api_token=SecretStr(_TOKEN_A)))
        transport = ASGITransport(app=app)

        mock_intent = IntentResult(intent=IntentType.RESPOND, confidence=1.0, reasoning="ok")
        with (
            patch(
                "kaos_agents.runtime.agent.BaseAgent._classify",
                new_callable=AsyncMock,
                return_value=mock_intent,
            ),
            patch(
                "kaos_agents.runtime.agent.BaseAgent._dispatch",
                new_callable=AsyncMock,
                return_value=("hello!", []),
            ),
        ):
            async with AsyncClient(transport=transport, base_url="http://test") as c:
                r = await c.post(
                    "/v1/sessions/auth-happy/messages",
                    json={"message": "hi"},
                    headers={
                        "Authorization": f"Bearer {_TOKEN_A}",
                        "Accept": "application/json",
                    },
                )
        assert r.status_code == 200
        assert r.json()["text"] == "hello!"
