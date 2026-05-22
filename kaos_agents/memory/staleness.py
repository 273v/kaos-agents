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
