"""Gated imports for the optional [llm] extra.

kaos-llm-client and kaos-llm-core are optional dependencies declared
under [project.optional-dependencies] llm = [...]. These helpers
provide clear ImportError messages with install guidance per
docs/python/design/dependencies.md.
"""

from __future__ import annotations

_INSTALL_HINT = "Install the LLM extra: uv add 'kaos-agents[llm]' or pip install 'kaos-agents[llm]'"


def require_llm_core() -> None:
    """Raise ImportError with install guidance if kaos-llm-core is not available."""
    try:
        import kaos_llm_core  # noqa: F401
    except ImportError:
        raise ImportError(
            f"kaos-llm-core is required for this feature but not installed. {_INSTALL_HINT}"
        ) from None


def require_llm_client() -> None:
    """Raise ImportError with install guidance if kaos-llm-client is not available."""
    try:
        import kaos_llm_client  # noqa: F401
    except ImportError:
        raise ImportError(
            f"kaos-llm-client is required for this feature but not installed. {_INSTALL_HINT}"
        ) from None
