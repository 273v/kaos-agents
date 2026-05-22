"""Tool-result staleness gate primitives (plan §Issue 8 / B1.4).

A long deal-room session may surface a cached web-fetch result from
Day 1 on Day 3 even though the source (FR rule / EDGAR filing /
web page) could have changed between fetches. The staleness gate
tags items past their TTL so the context-assembly layer can mark
``needs_reverification=True`` in the next thinking note.

Two primitives ship here:

- :func:`is_stale` — pure check: given a ``fetched_at`` POSIX
  timestamp and a TTL, returns ``True`` when the item is stale.
- :func:`mark_stale_items` — best-effort scanner over a list of
  memory items, returning the subset that has aged past the TTL.

The wider integration (writing ``needs_reverification`` into the
worker thinking note) is wired by ``kaos_agents/context/assemble.py``
in a follow-on commit; this module exposes the primitives so the
unit tests can pin contract first.
"""

from __future__ import annotations

import time
from collections.abc import Iterable
from typing import Any


def is_stale(
    fetched_at: float | None,
    *,
    ttl_seconds: float,
    now: float | None = None,
) -> bool:
    """Return ``True`` when an item fetched at ``fetched_at`` has
    aged past ``ttl_seconds``.

    Semantics:
    - ``ttl_seconds <= 0`` → False (gate disabled).
    - ``fetched_at is None`` → False (no recorded fetch time, can't
      decide).
    - ``now is None`` → uses ``time.time()``.

    Boundary: the check uses ``>=`` so an item fetched exactly
    ``ttl_seconds`` ago counts as stale. This is the conservative
    direction — a real attorney would rather re-verify one extra
    time than serve a 24h-old EDGAR filing on a deal-day session.

    The ``now`` parameter lets tests pin "current time" without
    monkey-patching ``time.time()``.
    """
    if ttl_seconds <= 0.0:
        return False
    if fetched_at is None:
        return False
    reference = now if now is not None else time.time()
    return (reference - fetched_at) >= ttl_seconds


def mark_stale_items(
    items: Iterable[Any],
    *,
    ttl_seconds: float,
    now: float | None = None,
) -> list[Any]:
    """Return the subset of ``items`` whose ``fetched_at`` attribute
    (or ``"fetched_at"`` dict key) has aged past ``ttl_seconds``.

    Best-effort: items that don't expose a ``fetched_at`` are
    treated as "no recorded fetch time" → not stale. Operators
    backfill ``fetched_at`` at tool-call time (in a follow-on
    commit) to engage the gate.
    """
    if ttl_seconds <= 0.0:
        return []
    reference = now if now is not None else time.time()
    stale: list[Any] = []
    for item in items:
        fetched_at = _get_fetched_at(item)
        if is_stale(fetched_at, ttl_seconds=ttl_seconds, now=reference):
            stale.append(item)
    return stale


def format_staleness_hint(
    stale_items: list[Any],
    *,
    ttl_seconds: float,
    label_for: Any | None = None,
    max_items: int = 5,
) -> str | None:
    """Format a worker-prompt ``<context>`` tag warning the model
    that one or more tool-result items have aged past their TTL.

    Companion to :func:`format_coreference_tag` in
    ``kaos_agents/context/coreference.py``. The agent loop calls
    this after :func:`mark_stale_items` finds expired items in the
    ACTIONS / FINDINGS sections and prepends the returned tag to
    the worker's thinking_note so the model knows to re-verify
    rather than rely on cached data.

    Returns ``None`` when ``stale_items`` is empty — the caller
    skips the prompt-injection step entirely. Otherwise returns a
    single tag block with:

    - the number of stale items found,
    - the TTL the operator configured (so the model sees the
      magnitude of the freshness contract),
    - up to ``max_items`` item labels (truncated with a
      ``"... and N more"`` suffix when the list exceeds the cap),
    - an instruction to re-verify by re-running the tool that
      produced the stale data.

    ``label_for`` is an optional callback that maps an item to a
    display label. When ``None``, falls back to a content-based
    label that uses the item's ``name`` / ``filename`` / ``source``
    metadata or the first non-empty content line, with
    ``"item:<id>"`` as the last resort.

    Plan §Issue 8 / B1.4 acceptance: "When surfacing past-TTL item,
    mark `needs_reverification=True` in next thinking_note". This
    helper produces that thinking-note text. The caller emits the
    structured `needs_reverification=True` field separately.
    """
    if not stale_items:
        return None
    if ttl_seconds <= 0.0:
        return None

    selected_label_for = label_for if label_for is not None else _default_staleness_label

    n_stale = len(stale_items)
    visible = stale_items[:max_items]
    labels = [str(selected_label_for(it)) for it in visible]
    overflow = n_stale - len(visible)
    suffix = f" (+ {overflow} more)" if overflow > 0 else ""

    item_lines = "\n".join(f"- {label}" for label in labels)

    return (
        "<context>\n"
        f"staleness: {n_stale} tool-result item(s) have aged past the "
        f"configured TTL of {ttl_seconds:.0f}s and may no longer be "
        "authoritative. Re-verify by re-running the original tool "
        "before relying on this data for the user's answer:\n"
        f"{item_lines}{suffix}\n"
        "needs_reverification=True\n"
        "</context>"
    )


def _default_staleness_label(item: Any) -> str:
    """Best-effort label for a stale item.

    Mirrors the precedence used by the corpus-handle anchor in
    ``assemble_context``: metadata-name keys first, then a content
    excerpt, then a synthetic ``item:<id>`` fallback. Operates on
    both attribute-style items (dataclass / pydantic) AND dict-style
    items (JSONL-deserialised) — staleness items come from both
    paths."""

    def _attr(key: str) -> Any:
        if isinstance(item, dict):
            return item.get(key)
        return getattr(item, key, None)

    metadata = _attr("metadata") or {}
    for key in ("filename", "uri", "source_uri", "name", "source"):
        anchor = metadata.get(key) if isinstance(metadata, dict) else None
        if anchor:
            return str(anchor)
    # Fall back to first non-empty content line.
    content = _attr("content") or ""
    if isinstance(content, str):
        for line in content.splitlines():
            stripped = line.strip()
            if stripped:
                return stripped[:120]
    # Last resort.
    item_id = _attr("id") or _attr("item_id") or "<unknown>"
    return f"item:{item_id}"


def _get_fetched_at(item: Any) -> float | None:
    """Best-effort lookup of ``fetched_at`` on a memory item.

    Accepts both attribute access (``item.fetched_at`` on a
    dataclass / pydantic model) and dict access (``item["fetched_at"]``
    on a JSONL-deserialised payload).
    """
    value = item.get("fetched_at") if isinstance(item, dict) else getattr(item, "fetched_at", None)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


__all__ = ["is_stale", "mark_stale_items"]
