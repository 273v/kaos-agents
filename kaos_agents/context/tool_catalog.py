"""Tool-catalog rendering — Track 4 chunk T4-4.

Composes :class:`FieldSet` (T4-3), :class:`ToolGroup` (T4-2), and
:class:`DataType` (T4-1) into the prompt-time view the agent sees.

Three rendering modes match the three operational stances:
- ``flat`` — bullet list at MEDIUM verbosity. Default. Right when the
  agent has all tools available and needs to scan once.
- ``grouped`` — group-by-ToolGroup, then bullet list inside each
  group. Right for 2-level discovery: "I'll use the extraction
  tools" → drill in.
- ``compact`` — comma-separated names at SMALL verbosity. Right
  when the prompt budget is tight or the catalogue is just an
  audit trail rather than a decision aid.

The renderer is **prompt-text** — plain strings, not markdown. Agents
embed the output verbatim in system / user prompts; markdown
rendering happens at the client.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, Any, Literal

from kaos_agents.registry import default_tool_group_registry
from kaos_agents.types.field_set import FieldSet, project

if TYPE_CHECKING:
    from kaos_agents.registry.tool_group_registry import ToolGroupRegistry

CatalogMode = Literal["flat", "grouped", "compact", "full"]


def _tool_metadata_iter(tools: Sequence[Any]) -> Iterable[Any]:
    """Yield :class:`ToolMetadata` instances from a mixed input.

    Accepts either :class:`KaosTool` instances (we read ``.metadata``)
    or raw :class:`ToolMetadata` objects. Anything else is silently
    skipped — the renderer must never raise on prompt assembly.
    """
    for t in tools:
        if hasattr(t, "metadata") and not callable(getattr(t, "metadata", None)):
            # KaosTool — metadata is a property, not a method
            yield t.metadata
        elif callable(getattr(t, "metadata", None)):
            # Some KaosTool overrides — metadata is callable
            yield t.metadata()
        elif hasattr(t, "name") and hasattr(t, "description"):
            # Raw ToolMetadata-like shape
            yield t


def _render_flat(metas: list[Any], field_set: FieldSet) -> str:
    """Bullet list of tools at the given tier, one per line."""
    lines: list[str] = []
    for meta in metas:
        d = project(meta, field_set)
        name = d.get("name", "?")
        if field_set == FieldSet.SMALL:
            lines.append(f"- {name}")
        else:
            description = d.get("description", "") or d.get("display_name", "")
            line = f"- {name}: {description}" if description else f"- {name}"
            lines.append(line)
    return "\n".join(lines)


def _render_compact(metas: list[Any]) -> str:
    """Comma-separated tool names — minimal token cost."""
    names = [project(meta, FieldSet.SMALL).get("name", "?") for meta in metas]
    return ", ".join(names)


def _render_full(metas: list[Any]) -> str:
    """JSON array of full tool dumps. For debugging / audit, not prompts."""
    payload = [project(meta, FieldSet.ALL) for meta in metas]
    return json.dumps(payload, indent=2, default=str)


def _render_grouped(
    metas: list[Any],
    group_registry: ToolGroupRegistry,
    field_set: FieldSet,
) -> str:
    """Group by :class:`ToolGroup`, then bullet list within each group.

    Tools whose name is in no registered group land in a synthetic
    "Other" group — keeps the agent from missing tools that lawyers
    forgot to bundle.
    """
    by_name: dict[str, Any] = {meta.name: meta for meta in metas}

    sections: list[str] = []
    grouped_names: set[str] = set()

    for group in group_registry.groups():
        # Tools in this group AND in our input
        group_metas = [by_name[name] for name in group.tool_names if name in by_name]
        if not group_metas:
            continue
        sections.append(f"## {group.name}: {group.description}")
        sections.append(_render_flat(group_metas, field_set))
        grouped_names.update(meta.name for meta in group_metas)

    # Catch-all for tools in the input but in no registered group
    ungrouped = [m for n, m in by_name.items() if n not in grouped_names]
    if ungrouped:
        sections.append("## other: Tools not in any registered group.")
        sections.append(_render_flat(ungrouped, field_set))

    return "\n".join(sections)


def render_tool_categories_for_classifier(
    runtime: Any,
    *,
    group_registry: ToolGroupRegistry | None = None,
) -> str:
    """Render the runtime's tool catalog as one-line-per-category text.

    Output shape, one category per line::

        <group-name>: <group-description>

    Plus one trailing line per ungrouped tool ``<tool-name>: <description>``
    so capabilities that haven't been bundled into a registered group
    still surface to the classifier. Returns the empty string when the
    runtime has no registered tools (preserves the pre-fix behavior on
    the classifier side: no catalog means no new constraint, the
    docstring's catalog-aware bullets become no-ops).

    This is the canonical input for the dynamic-catalog awareness on
    :class:`~kaos_agents.context.classify.ClassifyIntentSignature` and
    :class:`~kaos_agents.intent.signature.IntentSignature` — both
    classifiers accept the produced string as an InputField so they can
    reason about which tool *categories* fit the user's question
    without any hardcoded tool / category names baked into their
    prompts. The 2026-05-19 senator-question regression (session
    01KS0R64Q744DTVZ53KCS9VC7M) is the canonical driver: the agent
    classified a current-factual question as ``respond`` because the
    classifier never saw that a relevant tool group was registered.

    The shape is deliberately compact (group line only — no per-tool
    enumeration inside each group) so the classifier prompt stays
    small. A planner that needs the per-tool breakdown should call
    :func:`render_tool_catalog` with ``mode="grouped"`` instead.

    Args:
        runtime: A :class:`~kaos_core.runtime.KaosRuntime`-like object
            exposing ``runtime.tools.list_tool_objects()``. Passing
            ``None`` (or a runtime with no ``.tools``) yields the
            empty string — the classifier docstring's catalog-aware
            bullets all gate on a non-empty catalog string, so an
            absent runtime keeps the pre-fix behavior unchanged.
        group_registry: Tool-group registry to consult. Default is
            :data:`default_tool_group_registry`. Tests can pass an
            isolated registry to assert a stable output string.

    Returns:
        A single string. Empty when no tools / no runtime; otherwise a
        newline-separated list of ``"<name>: <description>"`` lines,
        one per group present in the catalog, with ungrouped tools
        appended after the registered groups.
    """
    if runtime is None:
        return ""
    tools_attr = getattr(runtime, "tools", None)
    if tools_attr is None:
        return ""
    list_objects = getattr(tools_attr, "list_tool_objects", None)
    list_names = getattr(tools_attr, "list_tools", None)
    tool_names: set[str]
    if callable(list_objects):
        try:
            objects = list_objects()
        except Exception:
            objects = []
        tool_names = {
            str(getattr(getattr(t, "metadata", None), "name", "") or getattr(t, "name", ""))
            for t in objects
        }
    elif callable(list_names):
        try:
            tool_names = set(list_names())
        except Exception:
            tool_names = set()
    else:
        return ""
    tool_names.discard("")
    if not tool_names:
        return ""

    registry = group_registry or default_tool_group_registry

    lines: list[str] = []
    grouped: set[str] = set()
    for group in registry.groups():
        present = [name for name in group.tool_names if name in tool_names]
        if not present:
            continue
        description = group.description or group.name
        lines.append(f"{group.name}: {description}")
        grouped.update(present)

    # Ungrouped tools — surface each one as its own line so a capability
    # that hasn't been bundled into a group still reaches the classifier.
    ungrouped_names = sorted(tool_names - grouped)
    if ungrouped_names and callable(list_objects):
        by_name: dict[str, Any] = {}
        try:
            for t in list_objects():
                meta = getattr(t, "metadata", None)
                n = str(getattr(meta, "name", "") or getattr(t, "name", ""))
                if n:
                    by_name[n] = meta if meta is not None else t
        except Exception:
            by_name = {}
        for name in ungrouped_names:
            meta = by_name.get(name)
            desc = ""
            if meta is not None:
                desc = str(getattr(meta, "description", "") or "")
            lines.append(f"{name}: {desc}" if desc else name)
    else:
        for name in ungrouped_names:
            lines.append(name)

    return "\n".join(lines)


def render_tool_catalog(
    tools: Sequence[Any],
    *,
    mode: CatalogMode = "flat",
    field_set: FieldSet | None = None,
    group_registry: ToolGroupRegistry | None = None,
) -> str:
    """Render a tool catalogue as a single prompt-ready string.

    Args:
        tools: Sequence of :class:`KaosTool` instances or raw
            :class:`ToolMetadata` records. Items lacking a name are
            skipped.
        mode: One of:
            - ``"flat"`` (default): bullet list at the chosen
              ``field_set`` (defaults to MEDIUM).
            - ``"grouped"``: bullet list grouped by
              :class:`ToolGroup`; tools without a group land in a
              synthetic ``other`` section.
            - ``"compact"``: comma-separated names at SMALL verbosity.
              Best for tight token budgets.
            - ``"full"``: pretty-printed JSON dump at ALL verbosity.
              For debugging / audit — *not* recommended in prompts.
        field_set: Override the per-mode default verbosity tier.
            ``compact`` always uses SMALL; ``full`` always uses ALL.
            For ``flat`` and ``grouped`` this picks the tier within
            the layout (default MEDIUM).
        group_registry: Where to look up tool groups for
            ``mode="grouped"``. Default is the module-level
            :data:`default_tool_group_registry`.

    Returns:
        Single string — empty when ``tools`` yields nothing renderable.
    """
    metas = list(_tool_metadata_iter(tools))
    if not metas:
        return ""

    if mode == "compact":
        return _render_compact(metas)

    if mode == "full":
        return _render_full(metas)

    fs = field_set or FieldSet.MEDIUM

    if mode == "flat":
        return _render_flat(metas, fs)

    if mode == "grouped":
        registry = group_registry or default_tool_group_registry
        return _render_grouped(metas, registry, fs)

    # Defensive — keep type checkers happy and surface bad modes loudly
    raise ValueError(
        f"Unknown catalog mode {mode!r}. Expected one of: 'flat', 'grouped', 'compact', 'full'."
    )


__all__ = [
    "CatalogMode",
    "render_tool_catalog",
    "render_tool_categories_for_classifier",
]
