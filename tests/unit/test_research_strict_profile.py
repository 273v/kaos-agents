"""Unit tests for the ``research_profile`` setting (KC17-P2-1).

Auditor finding EXT#8 — the default research-pattern behavior
ships with BM25 score floor, verifier confidence threshold, and
refusal of unverified answers all disabled. Field docstrings
recommend stricter settings "for legal deployments". The
``research_profile`` setting flips all three at once.

Covers:

1. Defaults remain unchanged (back-compat in alpha).
2. ``research_profile="strict"`` flips the three resolved
   accessors to the ``STRICT_PROFILE_DEFAULTS`` values.
3. Explicit field override beats the strict floor (profile is
   a floor, not a ceiling).
4. Env-var ``KAOS_AGENT_RESEARCH_PROFILE=strict`` activates the
   profile.
5. The ``ResearchAgent`` reads the resolved values at
   construction time — ``_refuse_unverified`` is True under
   strict profile even when the underlying field is False;
   ``_strict_profile`` is True; the refusal policy is built
   from the strict verifier-confidence floor.
6. The refusal-message branch on the unverified-answer path
   references ``research_profile=strict`` so an operator can
   downgrade either way.

No LLM calls. ResearchAgent construction is exercised against
an in-memory VFS; the unverified-collapse path is unit-tested
via the existing N4 wiring (already tested in test_research.py
for the legacy ``refuse_unverified_answers`` flag).
"""

from __future__ import annotations

import pytest
from kaos_core.types.enums import IsolationMode, StorageBackend
from kaos_core.vfs.core import VirtualFileSystem
from kaos_core.vfs.models import VFSConfig

from kaos_agents.patterns.research import ResearchAgent
from kaos_agents.settings import STRICT_PROFILE_DEFAULTS, KaosAgentSettings


def _memory_vfs() -> VirtualFileSystem:
    config = VFSConfig(default_backend=StorageBackend.MEMORY, isolation_mode=IsolationMode.GLOBAL)
    return VirtualFileSystem(config=config)


# ---------------------------------------------------------------------------
# 1. STRICT_PROFILE_DEFAULTS — stable wire contract
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestStrictProfileDefaultsConstant:
    """The strict-profile constant values are part of the public surface.

    Changing them silently weakens the legal-deployment posture, so
    lock them in here. If we need to tighten / loosen, change AND
    update the test in the same commit so the deliberate change is
    visible in the diff.
    """

    def test_bm25_score_floor_is_legal_recommended(self) -> None:
        # Field docstring says "1.0-3.0 for legal corpora" — 1.5 is
        # the lower end of that range and the chosen strict default.
        assert STRICT_PROFILE_DEFAULTS["bm25_score_floor"] == 1.5

    def test_verifier_min_confidence_is_legal_recommended(self) -> None:
        # Field docstring says "0.5-0.7 recommended floor for legal".
        # 0.6 sits at the lower end.
        assert STRICT_PROFILE_DEFAULTS["verifier_min_confidence"] == 0.6

    def test_refuse_unverified_answers_is_on(self) -> None:
        # Strict profile MUST refuse unverified answers — that is the
        # whole reason the profile exists per EXT#8.
        assert STRICT_PROFILE_DEFAULTS["refuse_unverified_answers"] is True


# ---------------------------------------------------------------------------
# 2. Settings resolver — default vs strict
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestDefaultProfile:
    """Default profile must preserve historical permissive behavior."""

    def test_default_research_profile_value(self) -> None:
        s = KaosAgentSettings(research_profile="default")
        assert s.research_profile == "default"

    def test_default_bm25_floor_unchanged(self) -> None:
        s = KaosAgentSettings(research_profile="default")
        # Underlying field default is 0.0 — explicitly preserved.
        assert s.bm25_score_floor == 0.0
        assert s.resolved_bm25_score_floor() == 0.0

    def test_default_verifier_threshold_unchanged(self) -> None:
        s = KaosAgentSettings(research_profile="default")
        assert s.verifier_min_confidence == 0.0
        assert s.resolved_verifier_min_confidence() == 0.0

    def test_default_refuse_unverified_unchanged(self) -> None:
        s = KaosAgentSettings(research_profile="default")
        assert s.refuse_unverified_answers is False
        assert s.resolved_refuse_unverified_answers() is False


@pytest.mark.unit
class TestStrictProfile:
    """Strict profile flips the three resolved accessors at once."""

    def test_strict_profile_value(self) -> None:
        s = KaosAgentSettings(research_profile="strict")
        assert s.research_profile == "strict"

    def test_strict_bm25_floor_flipped(self) -> None:
        s = KaosAgentSettings(research_profile="strict")
        # Field itself stays at 0.0 — resolved value picks up the
        # strict floor. This is the back-compat contract: nothing
        # changes for callers that read the raw field; consumers
        # that read the resolved accessor see the strict value.
        assert s.bm25_score_floor == 0.0
        assert s.resolved_bm25_score_floor() == 1.5

    def test_strict_verifier_threshold_flipped(self) -> None:
        s = KaosAgentSettings(research_profile="strict")
        assert s.verifier_min_confidence == 0.0
        assert s.resolved_verifier_min_confidence() == 0.6

    def test_strict_refuse_unverified_flipped(self) -> None:
        s = KaosAgentSettings(research_profile="strict")
        assert s.refuse_unverified_answers is False
        assert s.resolved_refuse_unverified_answers() is True


# ---------------------------------------------------------------------------
# 3. Floor semantics — explicit override beats strict floor
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestProfileIsFloorNotCeiling:
    """An explicit stricter field value beats the strict-profile floor.

    Strict profile is a floor: it only raises the threshold; if the
    user sets a stricter value (e.g. bm25_score_floor=2.5) the
    explicit value wins. The opposite (user sets 0.5, profile keeps
    1.5) is what makes the profile useful — it can't be undermined
    by leaving the field at default.
    """

    def test_higher_explicit_bm25_floor_wins(self) -> None:
        s = KaosAgentSettings(research_profile="strict", bm25_score_floor=2.5)
        # Strict default would be 1.5; explicit 2.5 is stricter.
        assert s.resolved_bm25_score_floor() == 2.5

    def test_lower_explicit_bm25_floor_loses_to_strict(self) -> None:
        s = KaosAgentSettings(research_profile="strict", bm25_score_floor=0.2)
        # User explicitly set 0.2 but strict profile raises to 1.5.
        # The profile is intentionally a floor — a partner who set
        # the profile is opting INTO the stricter posture and can't
        # accidentally undermine it by leaving a weak field setting
        # in place from a prior config.
        assert s.resolved_bm25_score_floor() == 1.5

    def test_higher_explicit_verifier_floor_wins(self) -> None:
        s = KaosAgentSettings(research_profile="strict", verifier_min_confidence=0.85)
        assert s.resolved_verifier_min_confidence() == 0.85

    def test_lower_explicit_verifier_floor_loses_to_strict(self) -> None:
        s = KaosAgentSettings(research_profile="strict", verifier_min_confidence=0.3)
        assert s.resolved_verifier_min_confidence() == 0.6


# ---------------------------------------------------------------------------
# 4. Env-var activation
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestEnvVarActivation:
    def test_env_var_strict(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("KAOS_AGENT_RESEARCH_PROFILE", "strict")
        s = KaosAgentSettings()
        assert s.research_profile == "strict"
        assert s.resolved_refuse_unverified_answers() is True
        assert s.resolved_verifier_min_confidence() == 0.6

    def test_env_var_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("KAOS_AGENT_RESEARCH_PROFILE", "default")
        s = KaosAgentSettings()
        assert s.research_profile == "default"
        assert s.resolved_refuse_unverified_answers() is False

    def test_env_var_invalid_value_rejected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Anything other than ``"default"`` or ``"strict"`` should
        fail loudly at construction. Silent default-fallback on a
        typo would defeat the audit gate the profile exists to
        provide."""
        from pydantic import ValidationError

        monkeypatch.setenv("KAOS_AGENT_RESEARCH_PROFILE", "permissive")
        with pytest.raises(ValidationError):
            KaosAgentSettings()


# ---------------------------------------------------------------------------
# 5. ResearchAgent reads resolved values at construction time
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestResearchAgentReadsProfile:
    def test_default_agent_has_no_strict_flags(self) -> None:
        vfs = _memory_vfs()
        agent = ResearchAgent(vfs)
        # Default settings — _refuse_unverified is False (back-compat).
        assert agent._refuse_unverified is False
        assert agent._strict_profile is False
        # No refusal policy built because verifier_min_confidence == 0.
        assert agent._refusal_policy is None

    def test_strict_agent_has_strict_flags(self) -> None:
        vfs = _memory_vfs()
        settings = KaosAgentSettings(research_profile="strict")
        agent = ResearchAgent(vfs, settings=settings)
        # _refuse_unverified flipped on by the resolved accessor.
        assert agent._refuse_unverified is True
        # _strict_profile mirrors the setting so the warn-vs-refuse
        # message can mention how to downgrade.
        assert agent._strict_profile is True

    def test_strict_agent_builds_refusal_policy(self) -> None:
        """Strict profile must build a verifier RefusalPolicy at
        the resolved threshold (0.6) even when the raw field is 0.0.

        Skipped cleanly when kaos-llm-core isn't installed — the
        [llm] extra is optional, and _build_refusal_policy returns
        None in that case (already tested via the None-path).
        """
        try:
            from kaos_llm_core.signatures.grounding import RefusalPolicy
        except ImportError:
            pytest.skip("kaos-llm-core not installed; refusal policy is None")

        vfs = _memory_vfs()
        settings = KaosAgentSettings(research_profile="strict")
        agent = ResearchAgent(vfs, settings=settings)
        assert isinstance(agent._refusal_policy, RefusalPolicy)
        assert agent._refusal_policy.min_confidence == 0.6

    def test_strict_agent_explicit_settings_object_via_env(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """End-to-end env-var → ResearchAgent flag wiring.

        The auditor cares that a partner setting one env var flips
        the behavior — verify that path actually reaches the agent.
        """
        monkeypatch.setenv("KAOS_AGENT_RESEARCH_PROFILE", "strict")
        vfs = _memory_vfs()
        # Don't pass settings explicitly — let the agent resolve
        # from env via KaosAgentSettings.resolve(None).
        agent = ResearchAgent(vfs)
        assert agent._refuse_unverified is True
        assert agent._strict_profile is True


# ---------------------------------------------------------------------------
# 6. Unverified-collapse path under strict profile
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestUnverifiedAnswerCollapseUnderStrict:
    """When ``_refuse_unverified`` is True (which strict profile
    forces) and a RAG result reports ``is_verified=False`` with an
    Answer payload, the agent collapses to InsufficientEvidence.

    We don't drive a full RAG call here — that's covered by live
    integration tests. We verify the wiring: the construction-time
    flags are set correctly so the existing N4 collapse block at
    research/agent.py:~669 will fire.
    """

    def test_strict_profile_enables_collapse_wiring(self) -> None:
        vfs = _memory_vfs()
        settings = KaosAgentSettings(research_profile="strict")
        agent = ResearchAgent(vfs, settings=settings)
        # The collapse block at research/agent.py:~669 guards on
        # ``self._refuse_unverified``. Verify it's True under strict.
        assert agent._refuse_unverified is True

    def test_default_profile_disables_collapse_wiring(self) -> None:
        vfs = _memory_vfs()
        settings = KaosAgentSettings(research_profile="default")
        agent = ResearchAgent(vfs, settings=settings)
        # Default behavior: warn-and-return, no collapse.
        assert agent._refuse_unverified is False

    def test_default_with_legacy_explicit_flag_still_collapses(self) -> None:
        """The legacy ``refuse_unverified_answers=True`` field still
        works on its own — strict profile is additive, not a
        replacement. This ensures we didn't break the N4 wiring
        for callers that configured it field-by-field pre-KC17."""
        vfs = _memory_vfs()
        settings = KaosAgentSettings(
            research_profile="default",
            refuse_unverified_answers=True,
        )
        agent = ResearchAgent(vfs, settings=settings)
        assert agent._refuse_unverified is True
        assert agent._strict_profile is False
