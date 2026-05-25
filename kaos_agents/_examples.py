"""Few-shot ``Example`` pools loaded from TOML config.

The kaos-llm-core idiom treats :class:`~kaos_llm_core.types.Example` as
data, not code: it is the primary "learnable parameter" that optimizers
(MIPRO / BootstrapFewShot) read + write. So every Signature in
kaos-agents grounds itself by loading a labelled pool from one
human-editable TOML file under ``kaos_agents/_assets/examples/``.

Convention — one file per Signature, shape::

    # _assets/examples/goal_check.toml
    [[example]]
    name = "fresh_with_tools"
    description = "Worker called tools, got grounded data, cited it."

    [example.inputs]
    user_message = "..."
    agent_response = "..."
    tool_calls_made = [{name = "kaos-web-search", is_error = false, summary_excerpt = "..."}]
    elevation_trail = []
    available_groups = ["web", "source"]
    iteration = 1

    [example.outputs]
    result = {kind = "satisfied", confidence = 0.9, rationale = "..."}

Usage::

    from kaos_agents._examples import load_examples
    from kaos_llm_core import Call

    call = Call(MySignature, model=..., examples=load_examples("my_sig"))

Optimizers may write back to the same files; the runtime + the
optimizer share one canonical source of truth.
"""

from __future__ import annotations

import tomllib
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from kaos_llm_core.types import Example

_ASSETS_ROOT = Path(__file__).parent / "_assets" / "examples"


@lru_cache(maxsize=64)
def load_examples(name: str) -> list[Example]:
    """Load the labelled ``Example`` pool for the named Signature.

    ``name`` is the TOML filename without the ``.toml`` suffix. The
    returned list is cached per name; mutate the underlying TOML to
    extend the pool (the cache reads the file on first call only — for
    optimizer-driven hot-edits, call :func:`reload_examples`).

    Raises ``FileNotFoundError`` if the file doesn't exist (callers
    should treat this as a contract violation, not a runtime
    fallback).
    """
    from kaos_llm_core import Example

    path = _ASSETS_ROOT / f"{name}.toml"
    data = tomllib.loads(path.read_text())
    return [Example(inputs=ex["inputs"], outputs=ex["outputs"]) for ex in data.get("example", [])]


def reload_examples(name: str | None = None) -> None:
    """Clear the loader cache.

    Called by optimizer runs that just rewrote one or more TOML files
    on disk. Pass ``name`` to invalidate one entry; omit to clear all.
    """
    if name is None:
        load_examples.cache_clear()
    else:
        # functools.lru_cache has no per-key eviction — clear all.
        # Optimizer hot-reload is rare; clearing the whole cache is fine.
        load_examples.cache_clear()


__all__ = ["load_examples", "reload_examples"]
