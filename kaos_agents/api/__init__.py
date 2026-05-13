"""kaos-agents HTTP / wire surface — Track 5 chunk T5-3 consolidation.

Subpackage layout:

- :mod:`kaos_agents.api.server` — FastAPI ``create_app`` + REST
  routes for ``/v1/sessions/...`` (was ``kaos_agents.api``).  **Only
  importable when the ``[api]`` extra is installed** (it imports
  ``fastapi``).  Resolved lazily via PEP 562 ``__getattr__`` so the
  base install can still ``import kaos_agents.api``.
- :mod:`kaos_agents.api.serve`  — ``kaos-agents-serve`` CLI entry
  point (was ``kaos_agents.serve``).
- :mod:`kaos_agents.api.wire`   — SSE / JSONL / WebSocket event
  serialisers (was ``kaos_agents.wire``).  No ``fastapi`` dependency
  — eagerly re-exported here.

Top-level entry point (registered via ``[project.scripts]``):

- ``kaos-agents-serve = kaos_agents.api.serve:main``

Public re-exports keep the most-used names accessible at the
subpackage level::

    from kaos_agents.api import create_app, events_to_sse, events_to_jsonl

``create_app`` is resolved on first attribute access; consumers
without the ``[api]`` extra get a clear install-hint ``ImportError``
only when they actually use the FastAPI surface.

Pre-T5 leaf modules (``kaos_agents.api`` as a flat module,
``kaos_agents.serve``, ``kaos_agents.wire``) are gone.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from kaos_agents.api.wire import events_to_jsonl, events_to_sse, events_to_ws

if TYPE_CHECKING:  # pragma: no cover — typing-only re-export
    from kaos_agents.api.server import create_app

__all__ = [
    "create_app",
    "events_to_jsonl",
    "events_to_sse",
    "events_to_ws",
]


def __getattr__(name: str) -> Any:
    """PEP 562 lazy attribute access for the ``[api]``-only surface.

    :func:`create_app` lives in :mod:`kaos_agents.api.server`, which imports
    ``fastapi`` + ``starlette`` at module load.  Those ship only with the
    ``[api]`` extra, so eagerly importing them from this package's
    ``__init__`` would break the base install.

    Resolving :func:`create_app` lazily here keeps ``import kaos_agents.api``
    working without ``[api]`` and only raises (with a clear install hint) at
    the point where a consumer actually touches the name.
    """
    if name == "create_app":
        try:
            from kaos_agents.api.server import create_app as _create_app
        except ImportError as exc:  # pragma: no cover — exercised by smoke test
            raise ImportError(
                "kaos_agents.api.create_app requires the [api] extra "
                "(FastAPI + starlette). Install with: "
                "`pip install 'kaos-agents[api]'` (or "
                "`uv pip install 'kaos-agents[api]'`). "
                f"Underlying ImportError: {exc}"
            ) from exc
        return _create_app
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
