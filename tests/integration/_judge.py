"""Opus-as-judge layer for integration tests.

Closes the gap where regex assertions rubber-stamp confident-partial
answers. The judge composes the generic ``JudgeSignature`` from
``kaos_agents.planning.judge`` with a per-scenario rubric; verdict
is a hard-gated ``pytest.fail()`` when the label doesn't match.

The 2026-05-23 verifier sub-agent run proved this layer is necessary:
5/5 web-tools regex assertions passed, but Opus ground-truthing
revealed 2 partial-failures (FR climate fabricated after tool-failure;
10b-5 citation hygiene). This file makes that level of verification
a CI-time gate, not a one-off audit.

Design constraints (from
``kaos-modules/docs/plans/2026-05-24-corpus-stress-suite-followup.md``
§2.1):

- Compose ``judge_with_rubric`` from ``kaos_agents.planning.judge`` —
  do NOT invent a new Signature. The rubric carries all the semantic
  logic and the LLM reads + judges. No regex / keyword grading on the
  response side; the rubric is load-bearing.
- Default model is ``anthropic:claude-opus-4-7`` (per design doc +
  ``feedback_test_model_floor.md``). Override only via
  ``KAOS_TEST_JUDGE_MODEL``.
- Hard-gate via ``pytest.fail()`` on judge failure — this is a real
  assertion, not advisory. Judge fall-back (model error / timeout)
  also fails the test; do not silently green-light.
- Telemetry: write the verdict into ``judge_state`` (a dict fixture)
  regardless of pass/fail so the per-test SUMMARY.jsonl line captures
  both directions.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import pytest

# Per the model-floor rule, Opus is the right model for the JUDGE.
# Override with KAOS_TEST_JUDGE_MODEL when Anthropic API is unavailable.
JUDGE_MODEL = os.environ.get("KAOS_TEST_JUDGE_MODEL", "anthropic:claude-opus-4-7")


@dataclass(frozen=True, slots=True)
class JudgeOutcome:
    """Typed outcome of one ``judge_response()`` call.

    Mirrors ``kaos_agents.planning.judge.JudgeVerdict`` plus the
    book-keeping the test layer cares about. Frozen + slotted to match
    the kaos-agents value-type discipline.
    """

    label: str
    confidence: float
    reasoning: str
    cost_usd: float
    latency_ms: float
    fell_back: bool


async def judge_response(
    *,
    rubric: str,
    user_prompt: str,
    response_text: str,
    tool_trace: str = "",
    model: str = JUDGE_MODEL,
) -> JudgeOutcome:
    """Run the generic ``JudgeSignature`` with the supplied rubric.

    The ``rubric`` MUST enumerate the labels the judge may emit (e.g.,
    'grounded / partial / fabricated / refused'). The judge reads the
    user prompt, the agent's response, and optionally the tool trace.
    Returns a typed verdict; the test decides what to assert.

    For convenience, ``tool_trace`` is appended as context so the
    judge can ground its verdict in WHAT THE AGENT ACTUALLY DID, not
    just what it claimed in the response. This is the key piece for
    catching fabrication: the tool trace shows whether a fetch
    succeeded, what came back, and whether the response cites it.

    Args:
        rubric: Free-form evaluation criteria; must enumerate labels.
        user_prompt: The user's original question — gives the judge
            the goal the agent was trying to satisfy.
        response_text: The agent's response text being judged.
        tool_trace: Optional rendered trace of tool calls (one per
            line, with name + error flag + result preview). When
            non-empty, appended to the judge's context.
        model: ``provider:model`` string; defaults to
            ``KAOS_TEST_JUDGE_MODEL`` env (falls back to
            ``anthropic:claude-opus-4-7``).
    """
    # Lazy import keeps test collection cheap and matches the
    # kaos-agents [llm]-optional invariant of judge.py itself.
    from kaos_agents.planning.judge import judge_with_rubric

    context = f"User prompt:\n{user_prompt}\n\nAgent response:\n{response_text}\n\n"
    if tool_trace:
        context += f"Tool trace (for grounding the verdict):\n{tool_trace}\n"

    verdict = await judge_with_rubric(
        rubric=rubric,
        input_text=response_text,
        context=context,
        model=model,
    )
    return JudgeOutcome(
        label=verdict.label,
        confidence=verdict.confidence,
        reasoning=verdict.reasoning,
        cost_usd=verdict.cost_usd,
        latency_ms=verdict.latency_ms,
        fell_back=verdict.fell_back,
    )


def assert_judge_passes(
    outcome: JudgeOutcome,
    *,
    passing_labels: tuple[str, ...],
    test_id: str,
    judge_state: dict | None = None,
) -> None:
    """Hard-gate the test on the judge verdict.

    Records the verdict into ``judge_state`` (a dict the conftest
    recorder picks up for per-test JSONL) regardless of pass/fail,
    so the audit trail captures both directions.

    Failing labels emit a ``pytest.fail`` with the judge's reasoning so
    the dev can see WHY the judge dropped the gate, not just THAT.
    """
    # Always write to judge_state — the per-test SUMMARY.jsonl line
    # needs the verdict even when we're about to fail the test, so
    # the audit trail captures the WHY of every CI red.
    if judge_state is not None:
        judge_state.update(
            test_id=test_id,
            judge_label=outcome.label,
            judge_confidence=outcome.confidence,
            judge_cost_usd=outcome.cost_usd,
            judge_latency_ms=outcome.latency_ms,
            judge_fell_back=outcome.fell_back,
            judge_reasoning=outcome.reasoning[:600],
        )

    if outcome.fell_back:
        # Judge errored or returned an unknown label. Surface the
        # fall-back loudly; a silent pass here would defeat the whole
        # point of the judge layer.
        pytest.fail(
            f"[{test_id}] Judge fell back (model error / unknown label). "
            f"label={outcome.label!r} reasoning={outcome.reasoning!r}"
        )

    passing_lower = {p.strip().lower() for p in passing_labels}
    if outcome.label.strip().lower() not in passing_lower:
        pytest.fail(
            f"[{test_id}] Judge verdict: {outcome.label!r} "
            f"(conf={outcome.confidence:.2f}, not in passing={passing_labels}).\n"
            f"Judge reasoning: {outcome.reasoning}"
        )


__all__ = [
    "JUDGE_MODEL",
    "JudgeOutcome",
    "assert_judge_passes",
    "judge_response",
]
