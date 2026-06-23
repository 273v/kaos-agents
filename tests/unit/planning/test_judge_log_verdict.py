"""Unit tests for the shared ``log_verdict`` critic-logging helper.

``log_verdict`` (``kaos_agents.planning.judge``) is the single
observability format the M2/M3/M4 ``judge_*`` wrappers now share. These
tests pin that format so the dedup can't silently drift: a WARNING when
the critic fell back, otherwise a DEBUG carrying the verdict + the
caller's ``char_counts`` rendered as space-separated ``key=value`` pairs
in insertion order.

The successful-verdict line is DEBUG (not INFO): a critic verdict fires on
every critic invocation in the agentic loop, so it is routine per-call
telemetry that a library must not emit at INFO. Only the fallback case
(which signals a degraded critic) stays at WARNING.
"""

from __future__ import annotations

import logging

import pytest

from kaos_agents.planning.judge import JudgeVerdict, log_verdict

pytestmark = pytest.mark.unit


def _verdict(*, fell_back: bool) -> JudgeVerdict:
    return JudgeVerdict(
        label="" if fell_back else "consistent",
        confidence=0.91,
        reasoning="some reasoning",
        cost_usd=0.0012,
        latency_ms=84.0,
        fell_back=fell_back,
    )


def test_trusted_verdict_logs_debug_with_char_counts(caplog: pytest.LogCaptureFixture) -> None:
    log = logging.getLogger("test.critic.info")
    with caplog.at_level(logging.DEBUG, logger="test.critic.info"):
        log_verdict(
            log,
            "M2",
            _verdict(fell_back=False),
            model="anthropic:claude-haiku-4-5",
            char_counts={"response_chars": 812, "tool_results_chars": 1190},
        )
    rec = caplog.records[-1]
    assert rec.levelno == logging.DEBUG
    msg = rec.getMessage()
    assert msg.startswith("M2 verdict label=consistent")
    assert "response_chars=812 tool_results_chars=1190" in msg
    assert "anthropic:claude-haiku-4-5" in msg


def test_fell_back_verdict_logs_warning(caplog: pytest.LogCaptureFixture) -> None:
    log = logging.getLogger("test.critic.warn")
    with caplog.at_level(logging.WARNING, logger="test.critic.warn"):
        log_verdict(
            log,
            "M3",
            _verdict(fell_back=True),
            model="m",
            char_counts={"response_chars": 0, "tool_results_chars": 0},
        )
    rec = caplog.records[-1]
    assert rec.levelno == logging.WARNING
    assert rec.getMessage().startswith("M3 verdict fell back")


def test_char_counts_render_in_insertion_order(caplog: pytest.LogCaptureFixture) -> None:
    """M4 passes different count keys; the helper renders exactly what the
    caller supplies, in insertion order — so each critic keeps its own
    historic field set and ordering."""
    log = logging.getLogger("test.critic.m4")
    with caplog.at_level(logging.DEBUG, logger="test.critic.m4"):
        log_verdict(
            log,
            "M4",
            _verdict(fell_back=False),
            model="m",
            char_counts={"user_prompt_chars": 40, "response_chars": 600},
        )
    assert "user_prompt_chars=40 response_chars=600" in caplog.records[-1].getMessage()
