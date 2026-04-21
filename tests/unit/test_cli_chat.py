"""Tests for ``kaos_agents.cli_chat`` — non-interactive mode + budget cap.

Two features under test, both shipped after the KAOS training course
discovered the gaps at curriculum-authoring time:

- ``--message`` / ``--message -`` for one-shot / stdin-fed turns so
  CI, scripts, and course runnables can drive the agent without a TTY.
- ``--max-cost USD`` (with ``$KAOS_AGENT_MAX_COST_USD`` env fallback)
  so a runaway chat can't burn arbitrary money. Non-interactive mode
  exits with code 2 on budget exceeded to distinguish it from code 1
  for actual errors.

Tests stay at the CLI contract surface: argparse, the session-state
dataclass, the one-shot-message resolver, the env-var fallback. We
don't spin up a real Runner here — integration tests cover that.
"""

from __future__ import annotations

import io

import pytest

from kaos_agents.cli_chat import (
    _EXIT_BUDGET,
    _EXIT_ERROR,
    _EXIT_OK,
    _build_arg_parser,
    _one_shot_message,
    _resolve_max_cost,
    _SessionState,
)


class TestArgParser:
    """The new flags parse correctly and preserve the existing ones."""

    def test_message_flag_parses(self) -> None:
        parser = _build_arg_parser()
        args = parser.parse_args(["chat", "--message", "hello world"])
        assert args.message == "hello world"

    def test_message_stdin_sentinel_parses(self) -> None:
        parser = _build_arg_parser()
        args = parser.parse_args(["chat", "--message", "-"])
        assert args.message == "-"

    def test_max_cost_is_float(self) -> None:
        parser = _build_arg_parser()
        args = parser.parse_args(["chat", "--max-cost", "0.05"])
        assert args.max_cost == pytest.approx(0.05)

    def test_defaults_leave_flags_unset(self) -> None:
        """Pre-existing users who never pass the new flags get None."""
        parser = _build_arg_parser()
        args = parser.parse_args(["chat"])
        assert args.message is None
        assert args.max_cost is None

    def test_existing_flags_still_wire(self) -> None:
        """Regression guard: the pre-5.x flag graph is intact."""
        parser = _build_arg_parser()
        args = parser.parse_args(
            [
                "chat",
                "--session",
                "s1",
                "--model",
                "anthropic:claude-haiku-4-5",
                "--pattern",
                "research",
                "--verbose",
                "--with-all",
            ]
        )
        assert args.session == "s1"
        assert args.model == "anthropic:claude-haiku-4-5"
        assert args.pattern == "research"
        assert args.verbose is True
        assert args.with_all is True


class TestOneShotMessage:
    """``_one_shot_message`` is the single source of truth for "what
    message do we send in non-interactive mode". Keeps stdin + CLI +
    empty-message handling in one place."""

    def test_none_when_message_unset(self) -> None:
        parser = _build_arg_parser()
        args = parser.parse_args(["chat"])
        assert _one_shot_message(args) is None

    def test_returns_stripped_message(self) -> None:
        parser = _build_arg_parser()
        args = parser.parse_args(["chat", "--message", "  hi there  "])
        assert _one_shot_message(args) == "hi there"

    def test_reads_stdin_on_dash(self, monkeypatch: pytest.MonkeyPatch) -> None:
        parser = _build_arg_parser()
        args = parser.parse_args(["chat", "--message", "-"])
        monkeypatch.setattr("sys.stdin", io.StringIO("piped message\n"))
        assert _one_shot_message(args) == "piped message"

    def test_empty_message_returns_none(self) -> None:
        """Don't send empty prompts to the agent — empty input should
        collapse to the REPL-vs-one-shot-decision equivalent of "no one-
        shot, go interactive"."""
        parser = _build_arg_parser()
        args = parser.parse_args(["chat", "--message", "   "])
        assert _one_shot_message(args) is None

    def test_empty_stdin_returns_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        parser = _build_arg_parser()
        args = parser.parse_args(["chat", "--message", "-"])
        monkeypatch.setattr("sys.stdin", io.StringIO(""))
        assert _one_shot_message(args) is None


class TestSessionStateBudget:
    """``_SessionState.budget_exceeded`` is the single decision point
    for 'should we refuse the next turn'. Tests lock the semantics."""

    def test_no_cap_never_exceeds(self) -> None:
        s = _SessionState(max_cost_usd=None)
        s.absorb(tokens=1_000_000, cost=999.0)
        assert s.budget_exceeded() is False

    def test_zero_cap_disables(self) -> None:
        """Explicit ``--max-cost 0`` means ``off`` not ``infinitely
        strict``. Flipping this would surprise anyone upgrading from
        a pre-flag setup who sets the env var to 0 to "disable."""
        s = _SessionState(max_cost_usd=0.0)
        s.absorb(tokens=1, cost=0.50)
        assert s.budget_exceeded() is False

    def test_below_cap_is_ok(self) -> None:
        s = _SessionState(max_cost_usd=0.10)
        s.absorb(tokens=10, cost=0.05)
        assert s.budget_exceeded() is False

    def test_at_cap_exceeds(self) -> None:
        """We use >= so a session that exactly meets the cap stops on
        the next turn. One cent spent vs. one cent allowed should not
        be "you can spend one more"."""
        s = _SessionState(max_cost_usd=0.10)
        s.absorb(tokens=10, cost=0.10)
        assert s.budget_exceeded() is True

    def test_over_cap_exceeds(self) -> None:
        s = _SessionState(max_cost_usd=0.10)
        s.absorb(tokens=10, cost=0.11)
        assert s.budget_exceeded() is True

    def test_absorb_tracks_turns(self) -> None:
        s = _SessionState()
        s.absorb(tokens=5, cost=0.01)
        s.absorb(tokens=3, cost=0.02)
        assert s.turns == 2
        assert s.tokens == 8
        assert s.cost_usd == pytest.approx(0.03)


class TestResolveMaxCost:
    """CLI > env > None precedence for the max-cost ceiling."""

    def test_cli_wins_over_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("KAOS_AGENT_MAX_COST_USD", "0.50")
        assert _resolve_max_cost(0.05) == pytest.approx(0.05)

    def test_env_when_cli_unset(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("KAOS_AGENT_MAX_COST_USD", "0.25")
        assert _resolve_max_cost(None) == pytest.approx(0.25)

    def test_none_when_both_unset(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("KAOS_AGENT_MAX_COST_USD", raising=False)
        assert _resolve_max_cost(None) is None

    def test_zero_cli_disables(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("KAOS_AGENT_MAX_COST_USD", "0.25")
        assert _resolve_max_cost(0.0) is None  # explicit off

    def test_zero_env_disables(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("KAOS_AGENT_MAX_COST_USD", "0")
        assert _resolve_max_cost(None) is None

    def test_malformed_env_treated_as_unset(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A typo in the env var shouldn't crash startup — treat it as
        'no cap' with the reasoning that crashing at CLI-parse time
        would be worse than silently proceeding without a cap."""
        monkeypatch.setenv("KAOS_AGENT_MAX_COST_USD", "not-a-number")
        assert _resolve_max_cost(None) is None


class TestExitCodeConstants:
    """Lock the exit-code contract so scripts/CI can rely on distinct
    values for success / error / budget."""

    def test_exit_codes_distinct(self) -> None:
        assert len({_EXIT_OK, _EXIT_ERROR, _EXIT_BUDGET}) == 3

    def test_budget_is_two(self) -> None:
        """Documented in the cli_chat module docstring — change with
        care (tools invoking this CLI may encode 2 == budget)."""
        assert _EXIT_BUDGET == 2

    def test_success_is_zero(self) -> None:
        assert _EXIT_OK == 0
