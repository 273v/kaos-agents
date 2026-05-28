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
    """The rubric must enumerate every documented decision rule and
    the atomic-over-composite preference. Anchor on rule SHAPE, not
    on specific tool-name strings — tool names are catalog data, not
    rubric content."""

    @staticmethod
    def _flat() -> str:
        return re.sub(r"\s+", " ", (ToolFitnessSignature.__doc__ or "").lower())

    def test_rubric_calls_out_domain_fit(self) -> None:
        # Rule 2: rank by domain fit, not name keyword overlap.
        flat = self._flat()
        assert "domain fit" in flat or "domain match" in flat

    def test_rubric_calls_out_atomic_over_composite_pattern(self) -> None:
        # The rubric must (a) name the atomic-vs-composite axis,
        # (b) tell the model to prefer atomic for single-axis
        # questions, and (c) tell the model how to identify a
        # composite from its description. Tool-name examples in the
        # rubric are catalog-coupled and brittle — the rule body
        # is what we pin.
        flat = self._flat()
        assert "atomic" in flat and "composite" in flat
        assert "prefer the atomic" in flat or "prefer atomic" in flat
        assert "single-axis" in flat or "single axis" in flat

    def test_rubric_documents_composite_identification(self) -> None:
        # The rubric must give the model a deterministic way to spot
        # composite tools — keyword markers, or an axis-count
        # heuristic. Wildcard tool-shape references like
        # ``*-domain-profile`` are acceptable; named-tool anchors
        # are not.
        flat = self._flat()
        # Keyword markers
        assert "profile" in flat and "snapshot" in flat
        # Either keyword markers OR an axis-count heuristic must be
        # documented so the model has a deterministic signal.
        has_axis_count = "axes" in flat or "axis" in flat
        has_keyword_list = (
            "profile" in flat
            and "summary" in flat
            and ("snapshot" in flat or "intel" in flat or "overview" in flat)
        )
        assert has_axis_count or has_keyword_list
