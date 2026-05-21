"""Unit tests for :mod:`kaos_agents.planning.tool_fitness`.

These tests pin the rubric SHAPE (Signature docstring) — the live
ranker behavior tests are in
``tests/integration/test_tool_fitness_live.py``.
"""

from __future__ import annotations

import re

import pytest

from kaos_agents.planning.tool_fitness import ToolFitnessSignature

pytestmark = pytest.mark.unit


class TestRubricShape:
    """The rubric must enumerate every documented decision rule + the
    0.1.1 atomic-over-composite carve-out (#549.A)."""

    @staticmethod
    def _flat() -> str:
        return re.sub(r"\s+", " ", (ToolFitnessSignature.__doc__ or "").lower())

    def test_rubric_calls_out_domain_fit(self) -> None:
        # Rule 2: rank by domain fit, not name keyword overlap.
        flat = self._flat()
        assert "domain fit" in flat or "domain match" in flat

    def test_rubric_calls_out_atomic_over_composite_pattern(self) -> None:
        # 0.1.1 (#549.A): atomic-over-composite carve-out.
        flat = self._flat()
        assert "atomic" in flat and "composite" in flat
        # Worked example exemplar — Meridian-style, anchors the rule
        # to a concrete scenario the model can pattern-match on.
        assert "kaos-web-domain-profile" in flat
        assert "kaos-web-dns-enumerate" in flat
        # The carve-out must say "prefer atomic" affirmatively, not
        # bury the rule in caveats.
        assert "prefer the atomic" in flat or "prefer atomic" in flat

    def test_rubric_documents_composite_identification(self) -> None:
        # The rubric must give the model a deterministic way to spot
        # composite tools — keyword list + axis-count heuristic.
        flat = self._flat()
        # Keyword markers
        assert "profile" in flat and "snapshot" in flat
        # The "≥2 axes" carve-out exists so the model doesn't over-
        # narrow on cases where the composite is genuinely the right
        # call (a "security snapshot" query needs DNS + WHOIS + TLS).
        assert "snapshot" in flat or "security snapshot" in flat
