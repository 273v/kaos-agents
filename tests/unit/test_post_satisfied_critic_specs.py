"""Unit tests for the unified post-satisfied grounding-critic surface.

The M2/M3/M4 critics that run after a *satisfied* GoalCheck used to be
three near-identical ~55-line blocks inside ``run_agentic_turn``. They
are now described by ``_GroundingCritic`` specs and share one
``_process_critic`` runner. These tests pin the invariants that the
unification is responsible for and that the loop-level integration tests
(``test_agentic_loop_m2.py``) do not directly assert:

* ``_critic_override`` gating semantics (confidence floor, fell_back,
  unknown label).
* The per-critic confidence-floor values — M2/M4 share the floor, M3 is
  deliberately ungated (0.0). This asymmetry is intentional; the test
  exists so it cannot change silently. See
  docs/plans/2026-05-29-kaos-agents-simplification-diary.md.
* The override directives are jargon-free LLM-visible prose — no internal
  critic id ("M2") and no rubric label ("contradicts_reasoning") leaks
  into the prompt threaded back to the worker
  (feedback_llm_visible_prompt_prose).
"""

from __future__ import annotations

import pytest

from kaos_agents.patterns.agentic_loop import (
    _CRITIC_OVERRIDE_CONFIDENCE_FLOOR,
    _M2_CRITIC,
    _M3_CRITIC,
    _M4_CRITIC,
    _critic_override,
)
from kaos_agents.planning.judge import JudgeVerdict

pytestmark = pytest.mark.unit


def _verdict(label: str, confidence: float, *, fell_back: bool = False) -> JudgeVerdict:
    return JudgeVerdict(
        label=label,
        confidence=confidence,
        reasoning="because",
        cost_usd=0.0001,
        latency_ms=10.0,
        fell_back=fell_back,
    )


# ── Confidence-floor configuration ──────────────────────────────────


def test_m2_and_m4_share_the_named_floor() -> None:
    assert _M2_CRITIC.confidence_floor == _CRITIC_OVERRIDE_CONFIDENCE_FLOOR
    assert _M4_CRITIC.confidence_floor == _CRITIC_OVERRIDE_CONFIDENCE_FLOOR


def test_m3_is_deliberately_ungated() -> None:
    """M3 (document-grounding fabrication) overrides on any trustworthy
    fabrication label at any confidence — the highest-severity failure
    family. If this ever needs to change, change it as an explicit
    decision, not by accident."""
    assert _M3_CRITIC.confidence_floor == 0.0


# ── _critic_override gating ─────────────────────────────────────────


def test_override_fires_for_failing_label_at_floor() -> None:
    note = _critic_override(_M2_CRITIC, _verdict("contradicts_reasoning", 0.85))
    assert note  # non-empty directive
    assert note == _M2_CRITIC.directives["contradicts_reasoning"]


def test_override_suppressed_below_floor() -> None:
    assert _critic_override(_M2_CRITIC, _verdict("contradicts_tool_results", 0.84)) == ""


def test_override_suppressed_when_fell_back() -> None:
    # Even a high-confidence verdict must not override if the critic
    # fell back (provider error / disallowed label).
    assert (
        _critic_override(_M2_CRITIC, _verdict("contradicts_reasoning", 0.99, fell_back=True)) == ""
    )


def test_override_suppressed_for_passing_label() -> None:
    # "consistent" / "grounded" / "complete" carry no directive.
    assert _critic_override(_M2_CRITIC, _verdict("consistent", 1.0)) == ""
    assert _critic_override(_M3_CRITIC, _verdict("grounded", 1.0)) == ""
    assert _critic_override(_M4_CRITIC, _verdict("complete", 1.0)) == ""


def test_m3_overrides_at_low_confidence() -> None:
    """The ungated M3 floor means a low-confidence fabrication label
    still overrides — distinct from M2/M4."""
    note = _critic_override(_M3_CRITIC, _verdict("fabricated_without_admission", 0.10))
    assert note == _M3_CRITIC.directives["fabricated_without_admission"]


def test_m4_partial_at_floor_overrides() -> None:
    note = _critic_override(_M4_CRITIC, _verdict("partial", 0.85))
    assert note == _M4_CRITIC.directives["partial"]


# ── Directives are jargon-free LLM-visible prose ────────────────────

_ALL_DIRECTIVES = [
    text for spec in (_M2_CRITIC, _M3_CRITIC, _M4_CRITIC) for text in spec.directives.values()
]

# Substrings that must NEVER appear in a string threaded into a worker
# prompt: internal critic ids and the rubric labels they emit.
_FORBIDDEN_JARGON = (
    "M2",
    "M3",
    "M4",
    "critic flagged",
    "contradicts_reasoning",
    "contradicts_tool_results",
    "fabricated_with_admission",
    "fabricated_without_admission",
)


@pytest.mark.parametrize("directive", _ALL_DIRECTIVES)
def test_directive_is_nonempty_and_actionable(directive: str) -> None:
    assert directive.strip()
    # Plain-English directives describe the fix; "Re-write" / "EITHER" /
    # "continue" all appear. Anchor on the common imperative.
    assert "Re-write" in directive or "continue searching" in directive


@pytest.mark.parametrize("directive", _ALL_DIRECTIVES)
def test_directive_carries_no_internal_jargon(directive: str) -> None:
    for token in _FORBIDDEN_JARGON:
        assert token not in directive, (
            f"LLM-visible directive leaks internal jargon {token!r}: {directive!r}"
        )
