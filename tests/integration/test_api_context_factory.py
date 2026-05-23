"""Integration tests for the ``context_factory`` parameter on ``create_app``.

Pre-0.1.14 ``create_app`` did not expose a hook for hosts that scope
the on-disk VFS namespace per session/tenant (e.g. the single-user-chat
SPA's ``sessions/{tenant}:{sid}/files/`` layout). The only way to wire
a tenant-aware ``KaosContext`` into each per-request ``Runner`` was a
monkey-patch on ``Runner.__init__`` at host import time.

0.1.14 promotes the contract: ``create_app`` accepts
``context_factory: Callable[[str], KaosContext] | None`` and threads it
into the ``Runner`` built by ``/v1/sessions/{id}/messages`` and
``/v1/sessions/{id}/runs/{rid}/resume``.

These tests pin:

1. Default (``None``) preserves existing behaviour — ``app.state.context_factory`` is ``None``.
2. Explicit factory is stored on ``app.state.context_factory`` and remains callable.
"""

from __future__ import annotations

import pytest
from kaos_core.base.context import KaosContext
from kaos_core.registry.container import KaosRuntime

from kaos_agents.api.server import create_app
from kaos_agents.api.settings import KaosAgentsApiSettings

pytestmark = pytest.mark.integration


def _make_settings() -> KaosAgentsApiSettings:
    return KaosAgentsApiSettings(api_allow_unauth_localhost=True)


def test_default_context_factory_is_none() -> None:
    """When the host does not supply a factory, ``app.state.context_factory`` is ``None``."""
    app = create_app(
        runtime=KaosRuntime.test_mode(),
        api_settings=_make_settings(),
        context_factory=None,
    )
    assert app.state.context_factory is None


def test_explicit_context_factory_is_stored_and_callable() -> None:
    """An explicit factory is stored on ``app.state.context_factory`` and remains callable.

    Hosts that scope VFS layout per session/tenant supply a factory of
    shape ``(session_id) -> KaosContext`` so the agent's tool calls
    resolve against the same prefix the host wrote to, without
    monkey-patching ``Runner.__init__``.
    """

    def factory(session_id: str) -> KaosContext:
        return KaosContext(
            session_id=session_id,
            default_vfs_namespace=f"sessions/{session_id}/files/",
        )

    app = create_app(
        runtime=KaosRuntime.test_mode(),
        api_settings=_make_settings(),
        context_factory=factory,
    )

    stored = app.state.context_factory
    assert stored is factory
    assert callable(stored)

    # Exercise the factory to prove it returns a usable KaosContext
    # whose namespace embeds the session id (the SPA's contract).
    ctx = stored("sess-cf-1")
    assert isinstance(ctx, KaosContext)
    assert ctx.session_id == "sess-cf-1"
    assert ctx.default_vfs_namespace == "sessions/sess-cf-1/files/"
