"""Test-model resolution for kaos-agents live integration tests.

Single source of truth for which models live tests use, with hard-floor
enforcement so nothing further down the stack can silently downgrade
to a weaker reasoner and let the suite false-green on the legal-research
bar contracts.

Four roles, each independently configurable via env var, each with its
own floor:

* ``KAOS_TEST_RESPOND_MODEL`` — the agent's primary respond/reason
  model. Floor: ``openai:gpt-5.4-mini`` or ``anthropic:claude-sonnet-4-6``
  (the documented Anthropic+OpenAI tier-1 floor for legal research).
  Haiku is **rejected** here. Defaults to Sonnet 4.6.

* ``KAOS_TEST_CRITIC_MODEL`` — sub-role for goal-check, M2/M3 critics,
  Reflexion critic, intent classifier. Floor: same as respond.
  Defaults to respond model. **Haiku is rejected** here too — the
  critic must be at least as strong as the respond model per the
  Reflexion contract (the critic should not be the bottleneck).

* ``KAOS_TEST_JUDGE_MODEL`` — the test-suite grounding judge. Floor:
  ``anthropic:claude-opus-4-7`` or stronger. Used by
  ``_judge_and_record`` for grounding rubric verdicts. Defaults to
  Opus.

* ``KAOS_TEST_PROBE_MODEL`` — intentional weak-model probe. Use this
  ONLY when a test deliberately exercises Haiku to verify Haiku
  *fails* (e.g. injection-defense matrix testing whether Haiku falls
  for a payload that Sonnet rejects). **NO floor enforcement** —
  arbitrary models accepted. Defaults to ``anthropic:claude-haiku-4-5``.

The floor lists are deliberately small. If a test needs a model not on
the list, that's an architectural signal: either the model is too weak
(reject) or the floor list needs an org-level update (don't paper over
it in the test).

Per ``feedback_test_model_floor.md`` (memory): live integration tests
must use Sonnet 4.6 or gpt-5.4-mini as the floor. Haiku is NOT a valid
fallback when OpenAI is rate-limited. This module enforces that
contract once, not 50 times.

Usage::

    from tests.integration._models import (
        respond_model,
        critic_model,
        judge_model,
        probe_model,
        requires_provider_for,
    )

    MODEL = respond_model()          # validated, floor-enforced
    CRITIC = critic_model()          # validated, floor-enforced
    JUDGE = judge_model()            # validated, opus-floor
    HAIKU_PROBE = probe_model()      # opt-in weak target

    requires_provider = requires_provider_for(MODEL)

    @requires_provider
    @pytest.mark.live
    async def test_something():
        agent = ChatAgent(model=MODEL, ...)
        ...

Parametrize over multiple models when intentional::

    from tests.integration._models import respond_model, probe_model

    @pytest.mark.parametrize("model", [respond_model(), probe_model()])
    async def test_injection_defense_matrix(model):
        # Sonnet should refuse the payload; Haiku may fall for it.
        ...

Do NOT hardcode model names in test files. The whole point of this
module is to centralize the policy.
"""

from __future__ import annotations

import os

import pytest

# ─── Floor definitions ──────────────────────────────────────────────
#
# The lists are deliberately small. They define the org's contract:
# "what counts as a competent enough model to validate kaos-agents
# behavior end to end against the legal-research bar." Edits require
# explicit org-level sign-off, not test-author convenience.

_RESPOND_FLOOR = frozenset(
    {
        "openai:gpt-5.4-mini",
        "openai:gpt-5.4",
        "openai:gpt-5.5",
        "anthropic:claude-sonnet-4-6",
        "anthropic:claude-opus-4-7",
    }
)
"""Models acceptable as the agent's primary respond/reason path.

Haiku is forbidden here per feedback_test_model_floor.md. If OpenAI is
rate-limited, fall over to Sonnet 4.6 — NOT down to Haiku.
"""

_CRITIC_FLOOR = _RESPOND_FLOOR
"""Models acceptable as critics (goal-check, M2/M3, Reflexion, intent).

The critic must be at least as strong as the respond model — otherwise
the critic becomes the bottleneck and may approve answers that the
respond model would have rejected on re-evaluation.
"""

_JUDGE_FLOOR = frozenset(
    {
        "anthropic:claude-opus-4-7",
        "openai:gpt-5.5",
    }
)
"""Models acceptable as the test-suite grounding judge.

The judge reads agent responses and emits a verdict against a rubric.
It must be at least as strong as the respond model — ideally stronger,
since the judge is the ground-truth oracle for "did the agent satisfy
the rubric." Opus is the canonical kaos-* judge.
"""

# Probe model floor is intentionally empty — any model accepted for
# adversarial probes that deliberately target weaker reasoners.

# ─── Defaults ───────────────────────────────────────────────────────

_DEFAULT_RESPOND = "anthropic:claude-sonnet-4-6"
_DEFAULT_CRITIC: str | None = None  # defaults to respond model
_DEFAULT_JUDGE = "anthropic:claude-opus-4-7"
_DEFAULT_PROBE = "anthropic:claude-haiku-4-5"


# ─── Env-var contract ───────────────────────────────────────────────
#
# Test authors set these env vars in CI / local dev:
#
#   KAOS_TEST_MODEL          (legacy alias for RESPOND)
#   KAOS_TEST_RESPOND_MODEL  (preferred)
#   KAOS_TEST_CRITIC_MODEL
#   KAOS_TEST_JUDGE_MODEL
#   KAOS_TEST_PROBE_MODEL    (opt-in adversarial probe target)
#
# The legacy ``KAOS_TEST_MODEL`` is honored for backward compat with
# CI files + the existing test_corpus_stress_suite.py pattern.


def _read_env(*names: str, default: str | None = None) -> str | None:
    """Return the first env var that is set, or default."""
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return default


def _enforce_floor(model: str, floor: frozenset[str], role: str) -> str:
    """Reject the model with a loud error if not on the floor list.

    The error names the offending env var, the floor list, and the
    feedback memory file so the operator can self-serve the fix.
    """
    if model not in floor:
        env_hint = {
            "respond": "KAOS_TEST_RESPOND_MODEL (or legacy KAOS_TEST_MODEL)",
            "critic": "KAOS_TEST_CRITIC_MODEL",
            "judge": "KAOS_TEST_JUDGE_MODEL",
        }.get(role, f"KAOS_TEST_{role.upper()}_MODEL")
        raise RuntimeError(
            f"{env_hint}={model!r} is not on the {role}-floor list. "
            f"Allowed: {sorted(floor)}. Per feedback_test_model_floor.md, "
            "Haiku is forbidden as a fallback for kaos-agents live tests; "
            "the critic and judge floors are tighter still. To probe "
            "intentionally-weak models, use probe_model() instead."
        )
    return model


# ─── Public resolvers ───────────────────────────────────────────────


def respond_model() -> str:
    """The agent's primary respond/reason model. Floor-enforced.

    Reads ``KAOS_TEST_RESPOND_MODEL`` first, falls back to legacy
    ``KAOS_TEST_MODEL``, then to ``anthropic:claude-sonnet-4-6``.

    Raises RuntimeError if the resolved model is not on the legal-floor
    list. Tests that need a deliberately weaker target use
    :func:`probe_model` instead.
    """
    model = _read_env("KAOS_TEST_RESPOND_MODEL", "KAOS_TEST_MODEL", default=_DEFAULT_RESPOND)
    assert model is not None  # _DEFAULT_RESPOND is non-None
    return _enforce_floor(model, _RESPOND_FLOOR, role="respond")


def critic_model() -> str:
    """Critic/goal-check/Reflexion model. Floor-enforced.

    Reads ``KAOS_TEST_CRITIC_MODEL``, falls back to :func:`respond_model`.
    The critic must be at least as strong as the respond model per the
    Reflexion contract — otherwise the critic is the bottleneck and may
    approve responses that the respond model would have rejected on
    re-evaluation.
    """
    model = _read_env("KAOS_TEST_CRITIC_MODEL", default=respond_model())
    assert model is not None
    return _enforce_floor(model, _CRITIC_FLOOR, role="critic")


def judge_model() -> str:
    """Test-suite grounding judge. Floor-enforced (Opus or stronger).

    Reads ``KAOS_TEST_JUDGE_MODEL``, falls back to
    ``anthropic:claude-opus-4-7``. The judge is the test-suite's
    ground-truth oracle — it reads agent responses and emits verdicts
    against a rubric. It must be at least as strong as the respond
    model; Opus is the canonical kaos-* judge.
    """
    model = _read_env("KAOS_TEST_JUDGE_MODEL", default=_DEFAULT_JUDGE)
    assert model is not None
    return _enforce_floor(model, _JUDGE_FLOOR, role="judge")


def probe_model() -> str:
    """Intentional weak-model probe target. NO floor enforcement.

    Reads ``KAOS_TEST_PROBE_MODEL``, falls back to
    ``anthropic:claude-haiku-4-5``. Use only for tests that
    deliberately exercise a weaker reasoner — e.g. an injection-defense
    matrix that verifies Haiku falls for a payload that Sonnet rejects.
    Do NOT use as a general fallback when the primary respond path
    needs cost savings; use :func:`respond_model` and pick a cheaper
    model from the floor list (gpt-5.4-mini is the cheapest legal
    option).
    """
    model = _read_env("KAOS_TEST_PROBE_MODEL", default=_DEFAULT_PROBE)
    assert model is not None
    return model


# ─── Provider gates ─────────────────────────────────────────────────


def _provider_env_var(model: str) -> str:
    """The env var that holds the API key for the model's provider."""
    prov = model.split(":", 1)[0] if ":" in model else "anthropic"
    return {
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "google": "GOOGLE_API_KEY",
    }.get(prov, "ANTHROPIC_API_KEY")


def requires_provider_for(model: str) -> pytest.MarkDecorator:
    """Skip-marker that requires the provider's API key.

    Usage::

        MODEL = respond_model()
        requires_provider = requires_provider_for(MODEL)

        @requires_provider
        @pytest.mark.live
        async def test_foo(): ...
    """
    env_var = _provider_env_var(model)
    return pytest.mark.skipif(env_var not in os.environ, reason=f"{env_var} missing")


__all__ = [
    "critic_model",
    "judge_model",
    "probe_model",
    "requires_provider_for",
    "respond_model",
]
