"""Unit tests for P0.4 — M4 silent-incompleteness critic.

The 2026-05-23 corpus stress S12 reproduced a confident-partial
failure: agent found 4 of 5 planted needles, declared partial
victory, did NOT drill for the 5th, and did NOT acknowledge that
one was missing. M2 (reasoning-action consistency) and M3
(document-grounding fabrication) both pass — what's returned IS
grounded; the defect is in completeness.

The fix composes the generic JudgeSignature from
``kaos_agents.planning.judge`` with an M4 COMPLETENESS rubric and
wires it into the AgenticLoop's force-elevate chain after M3.

These tests exercise the M4 rubric directly via
:func:`judge_completeness` on three canonical cases:

* ``partial`` — user asked for "all 5 risks", response lists 4 with
  no caveat
* ``exhausted`` — user asked for "all 5 risks", response lists 4 +
  explicit "I reviewed all 50 documents and only found 4"
* ``not_quantified`` — user asked an open-ended question

Live tests, marker ``@pytest.mark.live``. Model
``anthropic:claude-sonnet-4-6`` per kaos-agents test discipline.
"""

from __future__ import annotations

import os

import pytest

from kaos_agents.planning.m4_completeness import (
    M4_ALLOWED_LABELS,
    judge_completeness,
)

DEFAULT_LIVE_MODEL = "anthropic:claude-sonnet-4-6"


def _have_anthropic_key() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("KAOS_LLM_ANTHROPIC_API_KEY"))


# ── 1. partial — definite quantifier, fewer returned, no exhaustion ──


@pytest.mark.live
@pytest.mark.asyncio
async def test_m4_partial_when_definite_quantifier_and_fewer_returned() -> None:
    """User asked for 'all 5 risks'; response returns only 4 + does
    NOT acknowledge exhaustion. M4 must flag this as ``partial`` at
    confidence >= 0.85 — this is the S12 confident-partial-answer
    failure mode."""
    if not _have_anthropic_key():
        pytest.skip("ANTHROPIC_API_KEY not set")
    user_prompt = (
        "Please list all 5 risk factors disclosed in the attached 10-K. I need every one of them."
    )
    response_text = (
        "Here are the risk factors I identified:\n\n"
        "1. Cybersecurity incidents and data breaches.\n"
        "2. Macro-economic conditions affecting consumer spending.\n"
        "3. Foreign currency exchange rate fluctuations.\n"
        "4. Dependence on third-party suppliers for critical components.\n\n"
        "These represent the principal risks disclosed in the filing."
    )
    verdict = await judge_completeness(
        user_prompt=user_prompt,
        response_text=response_text,
        model=DEFAULT_LIVE_MODEL,
    )
    assert not verdict.fell_back, (
        f"M4 fell back unexpectedly: label={verdict.label!r} reasoning={verdict.reasoning!r}"
    )
    assert verdict.label in M4_ALLOWED_LABELS, f"unknown label {verdict.label!r}"
    assert verdict.label == "partial", (
        f"expected label=partial (4 of 5 with no exhaustion clause), "
        f"got label={verdict.label!r} confidence={verdict.confidence:.2f} "
        f"reasoning={verdict.reasoning!r}"
    )
    assert verdict.confidence >= 0.85, (
        f"expected confidence >= 0.85 for an unambiguous 4-of-5 "
        f"silent-incompleteness, got {verdict.confidence:.2f}. "
        f"reasoning={verdict.reasoning!r}"
    )


# ── 2. exhausted — fewer returned BUT explicit exhaustion proof ──────


@pytest.mark.live
@pytest.mark.asyncio
async def test_m4_exhausted_when_fewer_returned_with_explicit_exhaustion() -> None:
    """User asked for 'all 5 risks'; response returns 4 AND explicitly
    states the search/corpus was exhausted. M4 must NOT flag this as
    ``partial`` — exhaustion is the honest fallback and should pass
    through as ``exhausted``."""
    if not _have_anthropic_key():
        pytest.skip("ANTHROPIC_API_KEY not set")
    user_prompt = (
        "Please list all 5 risk factors disclosed in the attached 10-K. I need every one of them."
    )
    response_text = (
        "I reviewed all 50 pages of the attached 10-K filing and found "
        "only 4 distinct risk factors disclosed in the Item 1A section, "
        "not 5 as your prompt suggested. Here are the 4 I identified:\n\n"
        "1. Cybersecurity incidents and data breaches.\n"
        "2. Macro-economic conditions affecting consumer spending.\n"
        "3. Foreign currency exchange rate fluctuations.\n"
        "4. Dependence on third-party suppliers for critical components.\n\n"
        "No additional risk factors appear in the filing. If you have "
        "another source that mentions a 5th, please share it."
    )
    verdict = await judge_completeness(
        user_prompt=user_prompt,
        response_text=response_text,
        model=DEFAULT_LIVE_MODEL,
    )
    assert not verdict.fell_back, (
        f"M4 fell back: label={verdict.label!r} reasoning={verdict.reasoning!r}"
    )
    assert verdict.label in M4_ALLOWED_LABELS
    assert verdict.label != "partial", (
        f"expected NOT partial — response explicitly acknowledges "
        f"exhaustion ('I reviewed all 50 pages... found only 4'). "
        f"Got label={verdict.label!r} confidence={verdict.confidence:.2f} "
        f"reasoning={verdict.reasoning!r}"
    )
    # The honest label is ``exhausted``; accept either ``exhausted``
    # or ``complete`` (model may treat the explicit cap-and-justify as
    # a valid completion). Both are non-failing.
    assert verdict.label in ("exhausted", "complete"), (
        f"expected label in (exhausted, complete), got {verdict.label!r}. "
        f"reasoning={verdict.reasoning!r}"
    )


# ── 3. not_quantified — open-ended Q, no completeness expectation ────


@pytest.mark.live
@pytest.mark.asyncio
async def test_m4_not_quantified_for_open_ended_question() -> None:
    """User asked an open-ended qualitative question — no definite
    quantifier, no enumeration. M4 must emit ``not_quantified`` and
    NOT over-fire on conversational turns."""
    if not _have_anthropic_key():
        pytest.skip("ANTHROPIC_API_KEY not set")
    user_prompt = "What's the company's primary business?"
    response_text = (
        "The company's primary business is the design, manufacture, and "
        "sale of consumer electronics — primarily smartphones, tablets, "
        "wearables, and accessories — together with a growing services "
        "segment that includes digital content, cloud storage, and "
        "advertising."
    )
    verdict = await judge_completeness(
        user_prompt=user_prompt,
        response_text=response_text,
        model=DEFAULT_LIVE_MODEL,
    )
    assert not verdict.fell_back, (
        f"M4 fell back: label={verdict.label!r} reasoning={verdict.reasoning!r}"
    )
    assert verdict.label in M4_ALLOWED_LABELS
    assert verdict.label == "not_quantified", (
        f"expected label=not_quantified for an open-ended question "
        f"with no quantifier, got label={verdict.label!r} "
        f"confidence={verdict.confidence:.2f} reasoning={verdict.reasoning!r}"
    )
