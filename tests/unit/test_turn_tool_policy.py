"""Unit tests for `kaos_agents.planning.policy`.

PRD `kaos-modules/docs/internal/dynamic-tool-planning-completion-plan.md`
§3 Stage A — the promoted TurnToolPolicy Program. Ported from
``kaos-ui/examples/single-user-chat/backend/tests/unit/test_turn_tool_policy.py``
and extended with 5 new cases pinning the v2 Signature inputs:

- ``corpus_kinds`` — content-type labels for uploaded files
- ``session_intent`` — preset chip selection
- ``raw_turn_groups`` — last turn's wanted set for cross-turn coherence
- ``kept_groups`` / ``dropped_groups`` split — the "wanted but blocked" UX
- ``wanted_groups`` (planner output) is now unconstrained — the caller
  filters

We stub ``Call.invoke`` rather than hit a real provider — the contract
we want to pin is input → output behavior, not the LLM's actual
reasoning. The kaos-llm-core integration is covered separately in
``tests/integration/test_turn_tool_policy_live.py``.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any
from unittest.mock import patch

import pytest

from kaos_agents.planning.policy import (
    TurnToolPolicy,
    TurnToolPolicyResult,
    plan_turn_tool_policy,
)

pytestmark = pytest.mark.unit


@dataclass
class _FakeUsage:
    cost_usd: float = 0.0001


@dataclass
class _FakeOutput:
    policy: TurnToolPolicyResult


@dataclass
class _FakeInvocation:
    output: _FakeOutput
    usage: _FakeUsage


def _stub_invoke(policy: TurnToolPolicyResult, cost_usd: float = 0.0001) -> Any:
    """Patchable replacement for ``Call.invoke`` returning a fixed
    policy. Method signature: ``self`` is the bound Call instance.
    """

    async def _impl(self: Any, **_kwargs: Any) -> _FakeInvocation:
        return _FakeInvocation(
            output=_FakeOutput(policy=policy),
            usage=_FakeUsage(cost_usd=cost_usd),
        )

    return _impl


# ─── happy path ──────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_returns_narrowed_groups_when_planner_is_confident() -> None:
    """Confident planner output → kept_groups is the intersection with ceiling."""
    stub = TurnToolPolicyResult(
        wanted_groups=["documents", "citations"],
        rationale="The user is asking about an uploaded contract.",
        confidence=0.85,
    )
    with patch("kaos_llm_core.Call.invoke", new=_stub_invoke(stub)):
        policy = await plan_turn_tool_policy(
            user_message="What does section 7 of the PDF say?",
            recent_turns="",
            corpus_headlines="contract.pdf - 250kb, application/pdf",
            corpus_kinds=["pdf"],
            ceiling_groups=["documents", "citations", "vfs", "web"],
            available_groups=["documents", "citations", "vfs", "web"],
        )

    assert isinstance(policy, TurnToolPolicy)
    assert policy.kept_groups == frozenset({"documents", "citations"})
    assert policy.dropped_groups == frozenset()
    assert policy.fell_back_to_ceiling is False
    assert policy.confidence == pytest.approx(0.85)
    assert policy.cost_usd == pytest.approx(0.0001)
    # Back-compat alias still works.
    assert policy.turn_groups == policy.kept_groups


# ─── ceiling intersection + dropped_groups ───────────────────────────


@pytest.mark.asyncio
async def test_planner_wants_outside_ceiling_drops_to_dropped_groups() -> None:
    """Planner wants `web` but ceiling forbids — kept omits, dropped includes."""
    stub = TurnToolPolicyResult(
        wanted_groups=["documents", "web"],  # web is NOT in the ceiling below
        rationale="The user asked to search a website + their PDF.",
        confidence=0.9,
    )
    with patch("kaos_llm_core.Call.invoke", new=_stub_invoke(stub)):
        policy = await plan_turn_tool_policy(
            user_message="Compare the PDF to what's on https://sec.gov",
            recent_turns="",
            corpus_headlines="filing.pdf - 1mb",
            corpus_kinds=["pdf"],
            ceiling_groups=["documents", "citations"],  # no web!
            available_groups=["documents", "citations", "vfs", "web"],
        )

    assert policy.kept_groups == frozenset({"documents"})
    assert policy.dropped_groups == frozenset({"web"})
    assert policy.fell_back_to_ceiling is False


@pytest.mark.asyncio
async def test_planner_wanted_set_disjoint_from_ceiling_falls_back() -> None:
    """No overlap → fall back to full ceiling."""
    stub = TurnToolPolicyResult(
        wanted_groups=["web", "browser"],
        rationale="Live data lookup.",
        confidence=0.9,
    )
    with patch("kaos_llm_core.Call.invoke", new=_stub_invoke(stub)):
        policy = await plan_turn_tool_policy(
            user_message="Look this up live.",
            recent_turns="",
            corpus_headlines="",
            ceiling_groups=["documents", "citations"],  # disjoint from wanted
            available_groups=["documents", "citations", "web", "browser"],
        )

    assert policy.kept_groups == frozenset({"documents", "citations"})
    assert policy.dropped_groups == frozenset({"web", "browser"})
    assert policy.fell_back_to_ceiling is True


# ─── low-confidence fallback ─────────────────────────────────────────


@pytest.mark.asyncio
async def test_low_confidence_falls_back_to_full_ceiling() -> None:
    """`confidence < threshold` → expand to ceiling, mark fell_back."""
    stub = TurnToolPolicyResult(
        wanted_groups=["documents"],
        rationale="Could be anything.",
        confidence=0.4,  # below default 0.6 threshold
    )
    with patch("kaos_llm_core.Call.invoke", new=_stub_invoke(stub)):
        policy = await plan_turn_tool_policy(
            user_message="hmm",
            recent_turns="",
            corpus_headlines="",
            ceiling_groups=["documents", "citations", "vfs", "web"],
            available_groups=["documents", "citations", "vfs", "web"],
        )

    assert policy.kept_groups == frozenset({"documents", "citations", "vfs", "web"})
    assert policy.fell_back_to_ceiling is True
    assert "expanded to ceiling" in policy.rationale


@pytest.mark.asyncio
async def test_caller_can_override_confidence_threshold() -> None:
    """Caller-specified threshold beats settings default."""
    stub = TurnToolPolicyResult(
        wanted_groups=["documents"],
        rationale="Reasonable.",
        confidence=0.5,
    )
    with patch("kaos_llm_core.Call.invoke", new=_stub_invoke(stub)):
        # threshold=0.4 → 0.5 is above; planner output is honored.
        policy = await plan_turn_tool_policy(
            user_message="Read the doc.",
            recent_turns="",
            corpus_headlines="",
            ceiling_groups=["documents", "citations"],
            available_groups=["documents", "citations"],
            confidence_threshold=0.4,
        )

    assert policy.kept_groups == frozenset({"documents"})
    assert policy.fell_back_to_ceiling is False


# ─── short-circuits + exception handling ─────────────────────────────


@pytest.mark.asyncio
async def test_empty_ceiling_short_circuits_no_llm_call() -> None:
    """Empty ceiling → no Call, immediate empty policy."""
    # No patch needed — we assert no Call was constructed by passing
    # ceiling_groups=[] and not touching `kaos_llm_core.Call.invoke`.
    policy = await plan_turn_tool_policy(
        user_message="ignored",
        recent_turns="",
        corpus_headlines="",
        ceiling_groups=[],
        available_groups=["documents"],
    )

    assert policy.kept_groups == frozenset()
    assert policy.dropped_groups == frozenset()
    assert policy.fell_back_to_ceiling is False
    assert policy.cost_usd == 0.0
    assert policy.latency_ms == 0.0


@pytest.mark.asyncio
async def test_provider_exception_falls_back_to_ceiling() -> None:
    """Any exception from the LLM call → fell_back to ceiling."""

    async def _raise(self: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("provider 503")

    with patch("kaos_llm_core.Call.invoke", new=_raise):
        policy = await plan_turn_tool_policy(
            user_message="anything",
            recent_turns="",
            corpus_headlines="",
            ceiling_groups=["documents", "citations"],
            available_groups=["documents", "citations"],
        )

    assert policy.kept_groups == frozenset({"documents", "citations"})
    assert policy.dropped_groups == frozenset()
    assert policy.fell_back_to_ceiling is True
    assert policy.confidence == 0.0
    assert "Planner unavailable" in policy.rationale
    # Latency is measured even on exception.
    assert policy.latency_ms >= 0.0


# ─── v2 Signature inputs (corpus_kinds, session_intent, raw_turn_groups) ──


@pytest.mark.asyncio
async def test_corpus_kinds_is_passed_to_invoke() -> None:
    """The new `corpus_kinds` input reaches the LLM call unchanged."""
    captured: dict[str, Any] = {}

    async def _capture(self: Any, **kwargs: Any) -> _FakeInvocation:
        captured.update(kwargs)
        return _FakeInvocation(
            output=_FakeOutput(
                policy=TurnToolPolicyResult(
                    wanted_groups=["documents"],
                    rationale="ok",
                    confidence=0.9,
                )
            ),
            usage=_FakeUsage(),
        )

    with patch("kaos_llm_core.Call.invoke", new=_capture):
        await plan_turn_tool_policy(
            user_message="Summarize the spreadsheet.",
            corpus_kinds=["spreadsheet", "pdf"],
            ceiling_groups=["documents", "citations"],
            available_groups=["documents", "citations"],
        )

    assert captured["corpus_kinds"] == ["spreadsheet", "pdf"]


@pytest.mark.asyncio
async def test_session_intent_is_passed_to_invoke() -> None:
    """The new `session_intent` input reaches the LLM call."""
    captured: dict[str, Any] = {}

    async def _capture(self: Any, **kwargs: Any) -> _FakeInvocation:
        captured.update(kwargs)
        return _FakeInvocation(
            output=_FakeOutput(
                policy=TurnToolPolicyResult(
                    wanted_groups=["authoring"],
                    rationale="drafting persona",
                    confidence=0.9,
                )
            ),
            usage=_FakeUsage(),
        )

    with patch("kaos_llm_core.Call.invoke", new=_capture):
        await plan_turn_tool_policy(
            user_message="Draft the memo.",
            session_intent="drafting",
            ceiling_groups=["documents", "authoring"],
            available_groups=["documents", "authoring"],
        )

    assert captured["session_intent"] == "drafting"


@pytest.mark.asyncio
async def test_raw_turn_groups_carries_prior_turn_for_coherence() -> None:
    """`raw_turn_groups` (last turn's wanted set) reaches the LLM call."""
    captured: dict[str, Any] = {}

    async def _capture(self: Any, **kwargs: Any) -> _FakeInvocation:
        captured.update(kwargs)
        return _FakeInvocation(
            output=_FakeOutput(
                policy=TurnToolPolicyResult(
                    wanted_groups=["web"],
                    rationale="follow-up to prior turn",
                    confidence=0.9,
                )
            ),
            usage=_FakeUsage(),
        )

    with patch("kaos_llm_core.Call.invoke", new=_capture):
        await plan_turn_tool_policy(
            user_message="continue",
            raw_turn_groups=["web", "documents"],
            ceiling_groups=["web", "documents"],
            available_groups=["web", "documents"],
        )

    # The list is preserved.
    assert captured["raw_turn_groups"] == ["web", "documents"]


@pytest.mark.asyncio
async def test_omitted_optional_inputs_default_to_safe_values() -> None:
    """corpus_kinds=None → empty list, session_intent=None → None,
    raw_turn_groups=None → None (not empty list — distinguishes
    'first turn' from 'planner saw an empty wanted set')."""
    captured: dict[str, Any] = {}

    async def _capture(self: Any, **kwargs: Any) -> _FakeInvocation:
        captured.update(kwargs)
        return _FakeInvocation(
            output=_FakeOutput(
                policy=TurnToolPolicyResult(
                    wanted_groups=["documents"],
                    rationale="default",
                    confidence=0.9,
                )
            ),
            usage=_FakeUsage(),
        )

    with patch("kaos_llm_core.Call.invoke", new=_capture):
        await plan_turn_tool_policy(
            user_message="anything",
            ceiling_groups=["documents"],
            available_groups=["documents"],
        )

    assert captured["corpus_kinds"] == []
    assert captured["session_intent"] is None
    assert captured["raw_turn_groups"] is None


# ─── value-type shape ────────────────────────────────────────────────


class TestTurnToolPolicyValueType:
    """Frozen + slotted dataclass; back-compat alias exposes turn_groups."""

    def test_frozen_dataclass_cannot_be_mutated(self) -> None:
        p = TurnToolPolicy(
            kept_groups=frozenset({"documents"}),
            dropped_groups=frozenset(),
            rationale="test",
            confidence=0.9,
            fell_back_to_ceiling=False,
            cost_usd=0.0001,
            latency_ms=120.0,
        )
        with pytest.raises((AttributeError, dataclasses.FrozenInstanceError)):
            p.rationale = "mutated"  # ty: ignore[invalid-assignment]

    def test_turn_groups_alias_returns_kept_groups(self) -> None:
        kept = frozenset({"documents", "vfs"})
        p = TurnToolPolicy(
            kept_groups=kept,
            dropped_groups=frozenset({"web"}),
            rationale="test",
            confidence=0.9,
            fell_back_to_ceiling=False,
            cost_usd=0.0001,
            latency_ms=120.0,
        )
        assert p.turn_groups == kept
        assert p.turn_groups is p.kept_groups
