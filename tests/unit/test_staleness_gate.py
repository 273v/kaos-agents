"""Unit tests for the tool-result staleness gate (plan §Issue 8 / B1.4).

Acceptance row from §Issue 8:

    3-turn fixture: turn 1 fetches URL with TTL=10s; turn 3 (after
    11s) → staleness flag in turn 3's thinking_note; re-fetch
    attempted.

This pack pins the primitives the next-step ``assemble_context``
integration will call. The integration itself (write
``needs_reverification`` into the thinking note) follows in a
separate commit.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from kaos_agents.memory.staleness import is_stale, mark_stale_items
from kaos_agents.types.session_policy import (
    DEFAULT_TOOL_STALENESS_TTL_SECONDS,
    SessionPolicy,
)

# ── SessionPolicy field ──────────────────────────────────────────────


@pytest.mark.unit
def test_session_policy_default_disables_staleness_gate() -> None:
    """Default 0.0 preserves the historic behaviour — long sessions
    that worked pre-fix continue to work."""
    assert DEFAULT_TOOL_STALENESS_TTL_SECONDS == 0.0
    policy = SessionPolicy()
    assert policy.tool_staleness_ttl_seconds == 0.0


@pytest.mark.unit
def test_session_policy_explicit_ttl_round_trips() -> None:
    """Operators tighten the cap via ``SessionPolicy(...)``."""
    policy = SessionPolicy(tool_staleness_ttl_seconds=3600.0)
    assert policy.tool_staleness_ttl_seconds == 3600.0


# ── is_stale ─────────────────────────────────────────────────────────


@pytest.mark.unit
def test_is_stale_returns_false_when_ttl_is_zero() -> None:
    """The gate is OFF when TTL is 0.0 — even an ancient fetch passes."""
    assert is_stale(0.0, ttl_seconds=0.0, now=1_000_000_000.0) is False
    # And explicit negative TTLs also disable (defensive).
    assert is_stale(0.0, ttl_seconds=-1.0, now=1_000_000_000.0) is False


@pytest.mark.unit
def test_is_stale_returns_false_when_fetched_at_is_none() -> None:
    """Items without a recorded fetch time can't be decided — fail
    safe (treat as not stale rather than always-stale)."""
    assert is_stale(None, ttl_seconds=10.0) is False


@pytest.mark.unit
def test_is_stale_fires_at_exact_ttl_boundary() -> None:
    """``>=`` boundary: an item fetched EXACTLY ttl_seconds ago is
    counted as stale. This is the conservative direction — a real
    attorney would rather re-verify one extra time than serve a
    stale EDGAR filing."""
    assert is_stale(1000.0, ttl_seconds=10.0, now=1010.0) is True


@pytest.mark.unit
def test_is_stale_fires_when_aged_past_ttl() -> None:
    """An item 11s old with TTL=10s is stale (the §Issue 8 fixture)."""
    assert is_stale(1000.0, ttl_seconds=10.0, now=1011.0) is True


@pytest.mark.unit
def test_is_stale_does_not_fire_when_within_ttl() -> None:
    """An item 9s old with TTL=10s is fresh."""
    assert is_stale(1000.0, ttl_seconds=10.0, now=1009.0) is False


@pytest.mark.unit
def test_is_stale_uses_time_time_by_default() -> None:
    """``now=None`` falls back to ``time.time()``."""
    import time as _time

    fresh = _time.time() - 1.0
    ancient = _time.time() - 10_000.0
    assert is_stale(fresh, ttl_seconds=60.0) is False
    assert is_stale(ancient, ttl_seconds=60.0) is True


# ── mark_stale_items ────────────────────────────────────────────────


@pytest.mark.unit
def test_mark_stale_items_empty_when_ttl_disabled() -> None:
    """TTL <= 0 → nothing stale, regardless of input."""
    items = [{"fetched_at": 0.0, "tool_name": "x"}]
    assert mark_stale_items(items, ttl_seconds=0.0) == []


@pytest.mark.unit
def test_mark_stale_items_handles_dict_items() -> None:
    """Memory items deserialised from JSONL come back as dicts."""
    items = [
        {"fetched_at": 1000.0, "tool_name": "fresh"},
        {"fetched_at": 800.0, "tool_name": "old"},
    ]
    stale = mark_stale_items(items, ttl_seconds=100.0, now=1005.0)
    assert len(stale) == 1
    assert stale[0]["tool_name"] == "old"


@pytest.mark.unit
def test_mark_stale_items_handles_dataclass_items() -> None:
    """Items used in-memory may be dataclasses; attribute access
    must also work for the same gate."""

    @dataclass
    class _Item:
        tool_name: str
        fetched_at: float | None = None

    items = [
        _Item("fresh", fetched_at=1000.0),
        _Item("old", fetched_at=800.0),
        _Item("unknown", fetched_at=None),
    ]
    stale = mark_stale_items(items, ttl_seconds=100.0, now=1005.0)
    assert len(stale) == 1
    assert stale[0].tool_name == "old"


@pytest.mark.unit
def test_mark_stale_items_skips_items_without_fetched_at() -> None:
    """An item missing ``fetched_at`` is "no recorded fetch time"
    and falls through (fail-safe direction — see is_stale docstring)."""
    items = [
        {"tool_name": "no-timestamp"},
        {"fetched_at": "not-a-number", "tool_name": "garbage"},
    ]
    assert mark_stale_items(items, ttl_seconds=10.0, now=1_000_000.0) == []


@pytest.mark.unit
def test_mark_stale_items_full_three_turn_fixture() -> None:
    """Plan §Issue 8 acceptance fixture, faithfully replicated:
    turn 1 fetches at t=1000.0 with TTL=10.0; turn 3 evaluates at
    t=1011.0 → item is stale + flagged. This pins the contract the
    follow-on assemble_context wire-up must honour."""
    turn_1_fetched_at = 1000.0
    ttl = 10.0
    turn_3_now = 1011.0

    item = {"fetched_at": turn_1_fetched_at, "tool_name": "kaos-source-fetch-url"}
    stale = mark_stale_items([item], ttl_seconds=ttl, now=turn_3_now)
    assert len(stale) == 1
    assert stale[0] is item
