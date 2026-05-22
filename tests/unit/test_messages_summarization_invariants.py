"""MESSAGES summarization invariants (plan §Issue 8 / B0.3 acceptance).

Plan §Issue 8 acceptance row: "30-turn synthetic session →
``MESSAGES.item_count`` plateaus by turn 25; summary items present
in section".

B0.3 (the MESSAGES summarization wire-up shipped in kaos-agents
0.1.6) installs the summarization-on-turn hook so a long deal-
room session doesn't OOM the planner prompt budget by turn 25.

The full live integration test lives in
``tests/integration/test_messages_summarization_at_25.py`` (gated
on an LLM key — runs the real summarizer). This file is the
deterministic-tier sibling that pins the **structural invariants**
the live test depends on:

- ``SummarizationPolicy.ON_TURN`` and ``ON_OVERFLOW`` are stable
  enum values (audit consumers + section configs key on them);
- ``SessionMemory.end_turn`` increments the turn counter
  monotonically;
- ``SectionConfig.summarization_policy`` is readable on every
  section that the canonical-turn append loop consults;
- the ``summarize_turn`` async method exists with the expected
  signature.

A regression to any of these short-circuits the live test before
it has a chance to fire — pin them here so a refactor fails the
gate at unit-tier latency.
"""

from __future__ import annotations

import pytest

from kaos_agents.memory.session import SessionMemory
from kaos_agents.types.memory import MemoryType, SummarizationPolicy

# ── Enum-shape pin ──────────────────────────────────────────────────


@pytest.mark.unit
def test_summarization_policy_canonical_values() -> None:
    """The four canonical policy values are stable strings.
    Auditors + section configs index on them; a rename is a
    silent-breakage class for downstream consumers."""
    assert SummarizationPolicy.NEVER.value == "never"
    assert SummarizationPolicy.ON_OVERFLOW.value == "on_overflow"
    assert SummarizationPolicy.ON_TURN.value == "on_turn"
    # MANUAL exists for sections that summarize on explicit request only.
    assert SummarizationPolicy.MANUAL.value == "manual"


@pytest.mark.unit
def test_summarization_policy_set_membership_stable() -> None:
    """The full enum value set is locked. Adding a new policy
    (e.g. ``ON_TIME``) is a breaking change — pin so the gate
    catches it at unit-tier review."""
    values = {p.value for p in SummarizationPolicy}
    # AUTO is the heuristic policy (overflow OR unused-for-N-turns)
    # — keep it in the locked set so a refactor that drops it trips.
    assert values == {"never", "on_overflow", "on_turn", "manual", "auto"}


# ── Turn-counter monotonic invariant ───────────────────────────────


@pytest.mark.unit
def test_end_turn_increments_turn_counter() -> None:
    """``end_turn()`` is what the canonical-turn append loop calls
    after every assistant completion (per ``api/server.py:740``).
    Pin that it increments monotonically so the summarization
    hook fires deterministically on the right boundary."""
    m = SessionMemory("test")
    assert m._turn_count == 0
    m.end_turn()
    assert m._turn_count == 1
    m.end_turn()
    assert m._turn_count == 2


@pytest.mark.unit
def test_end_turn_does_not_skip_or_reset() -> None:
    """Across 30 sequential calls, the counter increments by 1
    each time — no off-by-one, no silent reset. Matches the
    canonical synthetic-30-turn scenario from the plan acceptance
    row."""
    m = SessionMemory("test")
    for expected in range(1, 31):
        m.end_turn()
        assert m._turn_count == expected


# ── Section-config / summarize_turn surface presence ────────────────


@pytest.mark.unit
def test_messages_section_present_in_default_memory() -> None:
    """The MESSAGES section MUST be configured by default —
    summarization-on-turn only fires for sections that exist."""
    m = SessionMemory("test")
    assert m.has_section(MemoryType.MESSAGES)


@pytest.mark.unit
def test_summarize_turn_method_exists_and_is_async() -> None:
    """The ``summarize_turn`` async coroutine must remain on
    SessionMemory. ``api/server.py:append_memory_turn`` awaits it
    before ``end_turn()``; a rename or sync conversion would break
    the wire-up. Pin both the name AND the async-ness."""
    import inspect

    method = getattr(SessionMemory, "summarize_turn", None)
    assert method is not None, "SessionMemory.summarize_turn missing"
    assert inspect.iscoroutinefunction(method), (
        "summarize_turn must be async; canonical-turn append loop awaits it"
    )


@pytest.mark.unit
def test_summarize_turn_accepts_model_kwarg() -> None:
    """The B0.3 wire-up passes the session's model to
    summarize_turn so the summarizer uses the same provider as
    the agent. Pin the kwarg name so a refactor doesn't silently
    re-route to a default model."""
    import inspect

    sig = inspect.signature(SessionMemory.summarize_turn)
    assert "model" in sig.parameters, (
        f"summarize_turn must accept 'model' kwarg; got params {list(sig.parameters)}"
    )


# ── Synthetic 30-turn end_turn sweep ────────────────────────────────


@pytest.mark.unit
def test_30_turn_end_turn_sweep_produces_no_regression() -> None:
    """Plan-acceptance shape: 30 sequential ``end_turn()`` calls
    on a fresh SessionMemory increment the counter deterministically
    to 30. This is the synthetic fixture the live
    ``test_messages_summarization_at_25.py`` builds on.

    The deterministic tier asserts the counter; the live tier
    asserts that summary items appear in the section AND
    ``item_count`` plateaus. Both layers needed for full
    acceptance coverage."""
    m = SessionMemory("synthetic-30-turn")
    for _ in range(30):
        # Add a small message so MESSAGES has something to summarize
        # (the live tier will exercise the real summarizer; here we
        # just pin the structural turn-counter contract).
        m.add(
            MemoryType.MESSAGES,
            content="user: hello\nassistant: hi",
        )
        m.end_turn()
    assert m._turn_count == 30
    # The section accumulated 30 items in the absence of a real
    # summarization call. The live test confirms the count plateaus
    # below 30 when summarization fires.
    assert m.section_item_count(MemoryType.MESSAGES) == 30


# ── Section-config field shape ─────────────────────────────────────


@pytest.mark.unit
def test_section_config_carries_summarization_policy_field() -> None:
    """SectionConfig.summarization_policy is the field
    ``begin_turn`` / ``end_turn`` consult to decide whether to
    invoke the summarizer. Pin its presence."""
    from kaos_agents.types.memory import SectionConfig

    cfg = SectionConfig(
        memory_type=MemoryType.MESSAGES,
        budget_tokens=1000,
        summarization_policy=SummarizationPolicy.ON_TURN,
    )
    assert cfg.summarization_policy is SummarizationPolicy.ON_TURN
    # Default summarization policy at the SectionConfig level is
    # NEVER — opt-in by section.
    default_cfg = SectionConfig(
        memory_type=MemoryType.MESSAGES,
        budget_tokens=1000,
    )
    assert default_cfg.summarization_policy is SummarizationPolicy.NEVER
