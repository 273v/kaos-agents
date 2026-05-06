"""Composers — orchestrators that build :class:`Deliverable`s from agent state.

Three composers, one per Deliverable shape:

* :func:`compose_tabular` — extraction-shaped deliverable from FINDINGS cells.
* :func:`compose_narrative` — report-shaped deliverable via per-section LLM calls.
* :func:`compose_hybrid` — narrative sections with embedded tables.

Every composer is pure orchestration over the lower-layer primitives
(``walks``, ``citations``, ``signatures``) plus existing kaos-content +
kaos-llm-core primitives. Composers don't write to memory; they read.
"""

from __future__ import annotations

from kaos_agents.output.composers.hybrid import HybridComposeResult, compose_hybrid
from kaos_agents.output.composers.narrative import (
    NarrativeComposeResult,
    compose_narrative,
)
from kaos_agents.output.composers.tabular import compose_tabular

__all__ = [
    "HybridComposeResult",
    "NarrativeComposeResult",
    "compose_hybrid",
    "compose_narrative",
    "compose_tabular",
]
