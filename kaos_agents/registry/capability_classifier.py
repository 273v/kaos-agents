"""Derive :class:`Capability` from existing :class:`ToolMetadata`.

Plan §16 phase 1.3: most existing tools don't have an explicit
``Capability`` declaration. To get the Capability layer working
without forcing a tool-by-tool migration, we derive a Capability from
each tool's ``ToolMetadata`` using a small truth table over the same
fields the SessionToolSet group classifier already consumes
(``category`` / ``capability`` / ``annotations`` / ``tags`` /
``module_name``).

This is **derived view**, not new ground truth — adding a new tool
with sane metadata auto-classifies on the next runtime walk; no
kaos-agents release needed. Authors who want a richer Capability
shape (e.g., a unified ``retrieve`` capability that aggregates
multiple tools) register that capability explicitly and the runtime's
:class:`CapabilityRegistry` keeps both declarative and derived entries
side-by-side.

The mapping intentionally produces ONE Capability per tool. Aggregating
multiple tools into one Capability (the "kaos-retrieve federates over
web + source + content" pattern in plan §1) is a separate phase that
ships an explicit Capability declaration on top of these auto-derived
entries.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from kaos_core.types import (
    Capability,
    CapabilityKind,
    CostClass,
    LatencyClass,
    ToolCapability,
    ToolCategory,
)

if TYPE_CHECKING:
    from kaos_core import KaosRuntime
    from kaos_core.types.metadata import ToolMetadata

    from kaos_agents.registry.capability_registry import CapabilityRegistry


# Map (ToolCategory, ToolCapability) → CapabilityKind. The truth table
# is intentionally small — first-match-wins on capability, then category
# falls through to a sensible default.
#
# kaos-core's ``ToolCapability`` is the verb-level classifier (EXTRACT /
# ANALYZE / TRANSFORM / QUERY / GENERATE / VALIDATE); kaos-core's
# ``ToolCategory`` is the domain classifier (DOCUMENT / TEXT / DATA /
# MEDIA / INTEGRATION / UTILITY / AGENT). We compose them.
_CAPABILITY_TO_KIND: dict[ToolCapability, CapabilityKind] = {
    ToolCapability.EXTRACT: CapabilityKind.EXTRACT,
    ToolCapability.ANALYZE: CapabilityKind.COMPUTE,
    ToolCapability.TRANSFORM: CapabilityKind.EXTRACT,
    ToolCapability.QUERY: CapabilityKind.SEARCH,
    ToolCapability.GENERATE: CapabilityKind.DRAFT,
    ToolCapability.VALIDATE: CapabilityKind.JUDGE,
}


def _derive_kind(meta: ToolMetadata) -> CapabilityKind:
    """Pick a :class:`CapabilityKind` for the tool.

    Priority (first-match-wins):

    1. ``"retrieval"`` tag → SEARCH (BM25 / dense / rerank)
    2. ``"browser"`` tag → READ (Playwright fetches and renders pages)
    3. ``module_name == "kaos-llm-core"`` AND not in extract verbs → JUDGE
       (alpha-* extractors are EXTRACT; everything else is judging-ish)
    4. ``module_name == "kaos-graph"`` → GRAPH
    5. ``ToolCategory.AGENT`` → META (agent introspection / dispatch)
    6. Map ``ToolCapability`` via :data:`_CAPABILITY_TO_KIND`
    7. ``side_effects=True`` OR ``destructiveHint=True`` → MUTATE
    8. Fallback: COMPUTE (the most defensible default for an unknown tool)
    """
    tags = set(meta.tags)
    if "retrieval" in tags:
        return CapabilityKind.SEARCH
    if "browser" in tags:
        return CapabilityKind.READ
    if meta.module_name == "kaos-llm-core":
        # alpha-* extractors are EXTRACT; everything else (Call, ReAct,
        # judges) is JUDGE-class.
        if meta.name.startswith(("kaos-llm-core-alpha-",)):
            return CapabilityKind.EXTRACT
        return CapabilityKind.JUDGE
    if meta.module_name == "kaos-graph":
        return CapabilityKind.GRAPH
    if meta.category == ToolCategory.AGENT:
        return CapabilityKind.META
    if meta.capability in _CAPABILITY_TO_KIND:
        return _CAPABILITY_TO_KIND[meta.capability]
    annotations = meta.annotations
    if meta.side_effects or (annotations is not None and annotations.destructiveHint):
        return CapabilityKind.MUTATE
    return CapabilityKind.COMPUTE


def _derive_cost_class(meta: ToolMetadata) -> CostClass:
    """Pick a :class:`CostClass`.

    Heuristic:

    - ``estimated_duration`` < 0.5s AND no LLM → FREE
    - ``estimated_duration`` < 5s → CHEAP
    - ``module_name == "kaos-llm-core"`` OR tagged ``"browser"`` → EXPENSIVE
    - default → MODERATE
    """
    if meta.module_name == "kaos-llm-core":
        return CostClass.EXPENSIVE
    if "browser" in set(meta.tags):
        return CostClass.EXPENSIVE
    duration = meta.estimated_duration
    if duration is not None:
        if duration < 0.5 and meta.module_name in {"kaos-core", "kaos-nlp-core"}:
            return CostClass.FREE
        if duration < 5.0:
            return CostClass.CHEAP
    return CostClass.MODERATE


def _derive_latency_class(meta: ToolMetadata) -> LatencyClass:
    """Pick a :class:`LatencyClass` from ``estimated_duration`` + tags."""
    if "browser" in set(meta.tags):
        return LatencyClass.SLOW
    duration = meta.estimated_duration
    if duration is None:
        return LatencyClass.MODERATE
    if duration < 0.1:
        return LatencyClass.INSTANT
    if duration < 1.0:
        return LatencyClass.FAST
    if duration < 10.0:
        return LatencyClass.MODERATE
    return LatencyClass.SLOW


def derive_capability(meta: ToolMetadata) -> Capability:
    """Auto-derive a :class:`Capability` from a tool's ``ToolMetadata``.

    The result has ``backing_tool_names = (meta.name,)`` — one capability
    per tool. Aggregation (multiple tools under one capability) is a
    separate, explicit registration step.

    The derived Capability uses ``meta.name`` as its capability name so
    every tool gets a stable, unique capability identifier. This lets
    consumers reason about capabilities by name without changing the
    underlying tool naming.

    Args:
        meta: Source ``ToolMetadata``. Must have a non-empty ``name``
            and ``description``.

    Returns:
        A frozen ``Capability`` that the registry can ingest directly.

    Raises:
        ValueError: If ``meta.description`` is empty (the planner
            depends on this string).
    """
    if not meta.description or not meta.description.strip():
        raise ValueError(
            f"Tool {meta.name!r} has empty description — cannot derive "
            "Capability. Set a non-empty ToolMetadata.description."
        )

    annotations = meta.annotations
    side_effects = bool(meta.side_effects) or (
        annotations is not None and annotations.destructiveHint
    )
    # MUTATE kind requires side_effects=True (Capability.__post_init__).
    kind = _derive_kind(meta)
    if kind == CapabilityKind.MUTATE:
        side_effects = True

    return Capability(
        name=meta.name,
        kind=kind,
        description=meta.description,
        inputs=tuple(p.name for p in meta.input_schema),
        outputs=(),  # output schema is a free-form dict; not extracted here
        cost_class=_derive_cost_class(meta),
        latency_class=_derive_latency_class(meta),
        side_effects=side_effects,
        preconditions=(),
        tags=tuple(meta.tags),
        backing_tool_names=(meta.name,),
    )


def register_capabilities_from_runtime(
    runtime: KaosRuntime,
    *,
    registry: CapabilityRegistry | None = None,
    force: bool = False,
) -> int:
    """Walk every tool registered on ``runtime`` and register a derived
    :class:`Capability` for each.

    Mirrors :func:`register_kaos_tool_groups`. Idempotent when called
    repeatedly with the same tool surface.

    Args:
        runtime: The runtime whose ``tools`` registry to walk.
        registry: Target capability registry. Defaults to the module's
            ``default_capability_registry`` singleton.
        force: When ``True``, replace existing capabilities of the same
            name (rare; usually False so re-walking is idempotent).

    Returns:
        Count of capabilities registered (including those that already
        existed when ``force=True``).
    """
    from kaos_agents.registry.capability_registry import (
        default_capability_registry,
    )

    if registry is None:
        registry = default_capability_registry

    count = 0
    for tool_name in runtime.tools.list_tools():
        tool = runtime.tools.get_tool(tool_name)
        if tool is None:
            continue
        meta = tool.metadata
        try:
            capability = derive_capability(meta)
        except ValueError:
            # Tools with empty descriptions are skipped silently — the
            # registry can't reason about them anyway. A separate
            # validate-tool-metadata pass should flag these.
            continue
        registry.register(capability, force=force)
        count += 1
    return count


__all__ = [
    "derive_capability",
    "register_capabilities_from_runtime",
]
