"""Shared fixtures + helpers for the use-case ladder.

Intentionally narrow surface — every tier file only needs a handful
of helpers. Bigger machinery (tracing, validators, etc.) belongs in
the framework, not in tier-specific helpers.

The ladder hard-fails if API keys are missing rather than skipping,
per the project's no-skips-in-CI policy. Fix the env, don't dodge
the test.
"""

from __future__ import annotations

import os
import time as _time
from collections.abc import AsyncIterator, Callable
from typing import Any

import pytest

from kaos_agents.events import (
    CitationFound,
    IntentClassified,
    KaosEvent,
    Span,
    SpanPhase,
    SpanSubject,
    TurnSummary,
)

# Pinned model strings — source of truth is
# kaos-llm-client/tests/integration/test_live.py header. Mandate:
# Claude >= 4.6 AND GPT >= 5.4. Odd-numbered ladder tiers run on
# Anthropic, even on OpenAI for cross-provider coverage.
ANTHROPIC_DEFAULT = "anthropic:claude-sonnet-4-6"
OPENAI_DEFAULT = "openai:gpt-5.4-mini"


def model_for_tier(tier: int) -> str:
    """Pick the model for a given tier — odd→Anthropic, even→OpenAI."""
    return ANTHROPIC_DEFAULT if tier % 2 == 1 else OPENAI_DEFAULT


@pytest.fixture(scope="session", autouse=True)
def _require_api_keys(request: pytest.FixtureRequest) -> None:
    """Hard-fail (do not skip) when required keys are missing.

    The no-skips policy treats silent skipping as a worse outcome
    than a loud failure: skipped tests get ignored in CI dashboards.
    A missing key means the dev environment isn't set up — fix it.

    Scoped to live-marked items only: deterministic offline tests in
    this directory (e.g. test_baseline_drift) must remain runnable in
    min-deps / compat lanes that exclude ``live`` via ``-m "not live"``.
    Without this gate the autouse fixture trips on every run that
    happens to include any ladder test in its selection, even when
    that selection contains no live-API tests at all.
    """
    needs_live = any(item.get_closest_marker("live") is not None for item in request.session.items)
    if not needs_live:
        return
    missing = [k for k in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY") if k not in os.environ]
    if missing:
        pytest.fail(
            f"use-case ladder requires {missing}. Wire credentials "
            "into the dev environment + CI; do not skip."
        )


async def collect_events(events: AsyncIterator[KaosEvent]) -> list[KaosEvent]:
    """Drain an async event iterator into a list.

    Tests typically want to assert on the full event sequence, so
    materialise upfront. Each ladder turn is bounded enough (~5-30s,
    < 10K tokens) that buffering all events is cheap.
    """
    out: list[KaosEvent] = []
    async for event in events:
        out.append(event)
    return out


def assert_within_budget(
    events: list[KaosEvent], *, budget_usd: float, label: str = "tier"
) -> None:
    """Hard cost gate — assert sum of TurnSummary costs ≤ 2x budget.

    The 2x slack absorbs provider-pricing drift (e.g. a model's
    output rate ticking up 30%) without making the gate noisy.
    Catastrophic regressions (a turn looping into a million tool
    calls) blow well past 2x and fail loudly here, surfacing
    silent-cost-amplification bugs that would otherwise inflate
    the bill unnoticed.
    """
    summaries = [e for e in events if isinstance(e, TurnSummary)]
    if not summaries:
        pytest.fail(f"{label}: no TurnSummary event in the stream")
    total = sum(float(getattr(s, "cost_usd", 0.0) or 0.0) for s in summaries)
    cap = 2.0 * budget_usd
    assert total <= cap, (
        f"{label}: total cost ${total:.4f} exceeded 2x budget ${cap:.4f} "
        f"(target was ${budget_usd:.4f}). Likely regression — investigate "
        f"runaway tool calls, exploded prompt sizes, or a planner replan loop."
    )


def turn_spans(events: list[KaosEvent]) -> list[Span]:
    """Return all turn-subject spans (start + complete)."""
    return [e for e in events if isinstance(e, Span) and e.subject == SpanSubject.TURN]


def tool_call_starts(events: list[KaosEvent]) -> list[Span]:
    """Return all tool_call/start spans."""
    return [
        e
        for e in events
        if isinstance(e, Span) and e.subject == SpanSubject.TOOL_CALL and e.phase == SpanPhase.START
    ]


def text_from_summary(events: list[KaosEvent]) -> str:
    """Concatenate the TurnSummary.text from every turn in the stream."""
    summaries = [e for e in events if isinstance(e, TurnSummary)]
    return "\n".join(getattr(s, "text", "") or "" for s in summaries)


def intent_events(events: list[KaosEvent]) -> list[IntentClassified]:
    """Return all IntentClassified events."""
    return [e for e in events if isinstance(e, IntentClassified)]


def citation_events(events: list[KaosEvent]) -> list[CitationFound]:
    """Return all CitationFound events."""
    return [e for e in events if isinstance(e, CitationFound)]


@pytest.fixture
def ladder_runner_factory() -> Callable[..., Any]:
    """Factory for one-off Runners scoped to a single tier.

    Each tier builds its own agent + runtime so tier-specific
    configuration (tools registered, pattern, permission policy,
    max_cost) doesn't leak across tiers. The factory caches nothing —
    a fresh KaosRuntime + Agent per call.

    Returns:
        A callable ``factory(*, agent_kwargs, register_source=False,
        register_pdf=False, permission_policy=None) -> (runner, runtime)``.
    """
    from kaos_core import KaosRuntime

    from kaos_agents.config import Agent
    from kaos_agents.runtime.runner import Runner

    def _factory(
        *,
        agent_kwargs: dict[str, Any] | None = None,
        register_source: bool = False,
        register_pdf: bool = False,
        permission_policy: Any | None = None,
        corpus: Any | None = None,
    ) -> tuple[Runner, KaosRuntime]:
        runtime = KaosRuntime.default()
        if register_source:
            from kaos_source.runtime.tools import register_source_tools

            register_source_tools(runtime)
        if register_pdf:
            from kaos_pdf import register_pdf_tools

            register_pdf_tools(runtime)
        agent = Agent.create(**(agent_kwargs or {}))
        runner = Runner(
            agent,
            runtime=runtime,
            permission_policy=permission_policy,
            corpus=corpus,
        )
        return runner, runtime

    return _factory


@pytest.fixture
def turn_session_id() -> str:
    """A unique-per-test session id so memory-tier tests don't collide."""
    return f"ladder-{int(_time.monotonic() * 1000):x}"
