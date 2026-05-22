"""Shape-contract tests for :class:`BudgetExceeded` event
(plan §Issue 9 — Cost overshoot mid-iteration).

Plan §Issue 9 acceptance row: "tool call about to push over budget
→ BudgetExceeded raised; tool not dispatched". The SPA's
RunInspector cost panel + plan-execute replan logic + audit JSONL
all consume the BudgetExceeded event. If a future refactor renames
or drops one of the kind strings, the downstream consumers
silently lose their dispatch.

These tests pin the field-shape contract: the five canonical
``kind`` strings (cost / tokens / steps / replans / wall_clock)
and the (limit, actual, reason) audit triple.

Plan: ``kaos-modules/docs/plans/2026-05-22-launch-blocker-top-10.md``
§Issue 9 — per-tool + per-loop cost gates emit BudgetExceeded.
"""

from __future__ import annotations

import pytest

from kaos_agents.events.budget import BudgetExceeded
from kaos_agents.types.plan import StopReason

# ── Canonical kind strings (the StopReason alignment) ──────────────


_CANONICAL_KINDS: tuple[str, ...] = (
    "cost",
    "tokens",
    "steps",
    "replans",
    "wall_clock",
)


@pytest.mark.unit
@pytest.mark.parametrize("kind", _CANONICAL_KINDS)
def test_canonical_kind_strings_construct_cleanly(kind: str) -> None:
    """Each canonical kind string constructs a valid BudgetExceeded
    event with the expected kind field."""
    e = BudgetExceeded(
        timestamp=0.0,
        sequence=0,
        session_id="s",
        run_id="r",
        kind=kind,
        limit=0.25,
        actual=0.27,
        reason="x",
    )
    assert e.kind == kind
    assert e.limit == 0.25
    assert e.actual == 0.27
    assert e.reason == "x"


@pytest.mark.unit
def test_kind_strings_align_with_stop_reason_via_max_prefix() -> None:
    """The BudgetExceeded.kind values use bare names (``cost``,
    ``tokens``, ``steps``, ``replans``, ``wall_clock``) while the
    StopReason enum uses the ``max_*`` prefix. Pin the alignment
    so plan-execute can map between them without ambiguity.

    A future drift in either direction (e.g. BudgetExceeded says
    ``"costs"`` plural, or StopReason drops the ``max_`` prefix) is
    a silent dispatch bug — this test catches it."""
    stop_reason_values = {sr.value for sr in StopReason}
    # Map kind → expected StopReason value (the prefix-add convention).
    expected = {
        "cost": "max_cost",
        "tokens": "max_tokens",
        "steps": "max_steps",
        "replans": "max_replans",
        "wall_clock": "max_wall_clock",
    }
    for kind in _CANONICAL_KINDS:
        sr_val = expected[kind]
        assert sr_val in stop_reason_values, (
            f"Canonical BudgetExceeded.kind={kind!r} is expected to "
            f"map to StopReason.{sr_val!r}. That mapping is broken; "
            f"add the StopReason enum value or update the test's "
            f"expected dict to reflect the new convention."
        )


# ── Field defaults ─────────────────────────────────────────────────


@pytest.mark.unit
def test_default_construction_yields_safe_zero_defaults() -> None:
    """Defaults are zero/empty so a partially-constructed event
    doesn't crash a consumer that walks all fields. Pin so a future
    refactor that switches to None defaults trips this test."""
    e = BudgetExceeded(timestamp=0.0, sequence=0, session_id="s", run_id="r")
    assert e.kind == ""
    assert e.limit == 0.0
    assert e.actual == 0.0
    assert e.reason == ""


@pytest.mark.unit
def test_actual_exceeds_limit_as_canonical_overshoot_shape() -> None:
    """The canonical event shape: ``actual > limit`` indicates an
    overshoot. Constructing with ``actual <= limit`` is legal (a
    future "almost-exceeded" warning use case) but the named
    invariant in the SPA RunInspector keys on ``actual > limit``
    for the orange/red threshold."""
    e = BudgetExceeded(
        timestamp=0.0,
        sequence=0,
        session_id="s",
        run_id="r",
        kind="cost",
        limit=0.25,
        actual=0.27,
        reason="overshoot",
    )
    assert e.actual > e.limit
    # The audit row carries the overshoot magnitude — pin it so the
    # SPA's cost-overshoot warning has stable math.
    overshoot = e.actual - e.limit
    assert overshoot == pytest.approx(0.02)


# ── Audit-trail field presence ─────────────────────────────────────


@pytest.mark.unit
def test_audit_triple_all_present_on_constructed_event() -> None:
    """The (limit, actual, reason) audit triple lets an operator
    re-derive the original budget config from the event log. Pin
    that all three fields land on the event instance."""
    e = BudgetExceeded(
        timestamp=0.0,
        sequence=0,
        session_id="s",
        run_id="r",
        kind="cost",
        limit=0.25,
        actual=0.40,
        reason="32-tool runaway on Sonnet with $0.25 cap",
    )
    # The reason field is the human-readable explanation — should
    # echo enough context that an operator reading the audit log
    # tomorrow can identify the failure mode without re-running.
    assert "Sonnet" in e.reason
    assert "cap" in e.reason
    assert e.limit > 0.0
    assert e.actual > 0.0


# ── Distinct events compare distinct ───────────────────────────────


@pytest.mark.unit
def test_distinct_kinds_produce_distinct_events() -> None:
    """A cost overshoot is not the same event class as a wall-clock
    overshoot. Pin the per-kind dispatch contract."""
    cost = BudgetExceeded(
        timestamp=0.0,
        sequence=0,
        session_id="s",
        run_id="r",
        kind="cost",
        limit=0.25,
        actual=0.27,
        reason="cost",
    )
    wall = BudgetExceeded(
        timestamp=0.0,
        sequence=0,
        session_id="s",
        run_id="r",
        kind="wall_clock",
        limit=60.0,
        actual=75.0,
        reason="time",
    )
    assert cost.kind != wall.kind
    # Field-value comparison (event types are equality-compared by
    # value via pydantic).
    assert cost != wall
