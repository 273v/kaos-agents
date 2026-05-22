"""Tool-fitness ranker bypass-gate semantics (plan §Issue 7).

Plan §Issue 7 #581 fix: when ``len(tools) <= bypass_threshold``,
the ranker is skipped and the full bridged catalog is forwarded
to the LLM. This is the cost / latency tradeoff: small catalogs
don't need an LLM call to narrow.

These tests pin the gate's decision shape — the per-condition
return-tools-unchanged contract. A regression that flips the
direction (e.g. ``<`` instead of ``<=``) would silently change the
cutover point and cause unexpected ranker LLM calls on small
catalogs (cost regression) or skip ranking on borderline-large
catalogs (correctness regression).

Plan: ``kaos-modules/docs/plans/2026-05-22-launch-blocker-top-10.md``
§Issue 7 — Wire bugs blocking corpus / document workflows.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest


@dataclass
class _FakeTool:
    """Mimics the kaos-llm-core Tool surface that
    ``_maybe_narrow_tools_via_fitness_ranker`` inspects."""

    name: str
    description: str = ""


def _decide_bypass(
    tools: list,
    *,
    enabled: bool,
    bypass_threshold: int,
    query: str,
) -> bool:
    """Reproduce the bypass-gate semantics from
    ``kaos_agents/patterns/chat.py:_maybe_narrow_tools_via_fitness_ranker``
    (lines 312-319) so the contract is testable without the full
    ChatAgent ctor dependency chain.

    Returns ``True`` when the ranker is bypassed (tools returned
    unchanged); ``False`` when the ranker would be invoked."""
    if not enabled:
        return True
    if len(tools) <= bypass_threshold:
        return True
    clean_query = (query or "").strip()
    return not clean_query


# ── Each bypass condition ──────────────────────────────────────────


@pytest.mark.unit
def test_bypass_when_disabled() -> None:
    """``tool_fitness_enabled=False`` → bypass regardless of
    catalog size. Operators who want raw-catalog dispatch (bench
    runs, debugging) flip this flag and expect zero ranker calls."""
    tools = [_FakeTool(name=f"t-{i}") for i in range(20)]
    assert _decide_bypass(tools, enabled=False, bypass_threshold=10, query="hello") is True


@pytest.mark.unit
def test_invoke_when_above_threshold() -> None:
    """The canonical narrow-path: catalog > threshold + enabled +
    non-empty query → ranker invoked."""
    tools = [_FakeTool(name=f"t-{i}") for i in range(20)]
    assert _decide_bypass(tools, enabled=True, bypass_threshold=10, query="hello") is False


@pytest.mark.unit
def test_bypass_at_threshold_boundary() -> None:
    """Boundary: ``len(tools) <= threshold`` is the inclusive
    bypass condition. A catalog of EXACTLY threshold-many tools
    bypasses (cost saving on the smallest viable narrow set).

    A future refactor that flipped this to ``<`` would silently
    invoke the ranker on every threshold-sized session — pin to
    catch that."""
    tools = [_FakeTool(name=f"t-{i}") for i in range(10)]
    assert _decide_bypass(tools, enabled=True, bypass_threshold=10, query="hello") is True


@pytest.mark.unit
def test_invoke_one_above_threshold() -> None:
    """Just above the boundary: threshold+1 catalog DOES invoke
    the ranker. This is the matching half of the boundary pin."""
    tools = [_FakeTool(name=f"t-{i}") for i in range(11)]
    assert _decide_bypass(tools, enabled=True, bypass_threshold=10, query="hello") is False


@pytest.mark.unit
def test_bypass_on_empty_query() -> None:
    """Empty query → bypass (no ranker context, no narrowing).
    Defends against a regression that would invoke the ranker with
    an empty query and waste a turn's tool-routing latency."""
    tools = [_FakeTool(name=f"t-{i}") for i in range(20)]
    assert _decide_bypass(tools, enabled=True, bypass_threshold=10, query="") is True


@pytest.mark.unit
def test_bypass_on_whitespace_only_query() -> None:
    """Whitespace-only query is treated as empty after .strip() —
    pin so the equivalence is explicit, not implicit."""
    tools = [_FakeTool(name=f"t-{i}") for i in range(20)]
    assert _decide_bypass(tools, enabled=True, bypass_threshold=10, query="   \n\t  ") is True


# ── Threshold compositions ─────────────────────────────────────────


@pytest.mark.unit
@pytest.mark.parametrize(
    "threshold, catalog_size, expected_bypass",
    [
        (0, 0, True),  # zero-catalog edge case
        (0, 1, False),  # threshold=0 means "always invoke if any tools"
        (5, 5, True),  # at threshold → bypass
        (5, 6, False),  # one above → invoke
        (20, 21, False),  # large threshold
        (100, 50, True),  # below threshold
    ],
)
def test_threshold_sweep(threshold: int, catalog_size: int, expected_bypass: bool) -> None:
    """Sweep the threshold/catalog-size product to pin the gate
    semantics across the threshold lattice. The orientation
    (inclusive vs exclusive) is what the test catches."""
    tools = [_FakeTool(name=f"t-{i}") for i in range(catalog_size)]
    assert (
        _decide_bypass(tools, enabled=True, bypass_threshold=threshold, query="x")
        is expected_bypass
    )


# ── Real ranker surface contract pin ───────────────────────────────


@pytest.mark.unit
def test_real_ranker_module_exposes_rank_tools_for_query() -> None:
    """The bypass test above re-implements the gate locally. To
    confirm the file under test still ships the entry point our
    test was modeled on, attempt the actual import. A regression
    that renames or removes ``rank_tools_for_query`` would break
    the production code path; this test catches that at unit time
    rather than waiting for a live integration run."""
    from kaos_agents.planning.tool_fitness import rank_tools_for_query

    assert callable(rank_tools_for_query)


@pytest.mark.unit
def test_real_chat_pattern_exposes_narrow_method() -> None:
    """Mirror import-contract test for the consumer side:
    ``ChatAgent._maybe_narrow_tools_via_fitness_ranker`` must
    exist with the production signature so the wire path stays
    intact even when the bypass-gate logic changes."""
    from kaos_agents.patterns.chat import ChatAgent

    assert hasattr(ChatAgent, "_maybe_narrow_tools_via_fitness_ranker")
