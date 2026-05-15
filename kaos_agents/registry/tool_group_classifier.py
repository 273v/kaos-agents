"""Canonical KAOS tool-group derivation + registry population.

PRD `kaos-modules/docs/internal/dynamic-tool-planning-completion-plan.md` §2
(v2) replaces the v1 prefix-pattern taxonomy with a **derivation function**
over the existing kaos-core ``ToolMetadata`` fields:

- ``category`` (:class:`~kaos_core.types.enums.ToolCategory`) — 7-value enum
- ``capability`` (:class:`~kaos_core.types.enums.ToolCapability`) — 6-value enum
- ``annotations`` (:class:`~kaos_core.types.annotations.ToolAnnotations`) —
  ``readOnlyHint``, ``destructiveHint``, ``idempotentHint``, ``openWorldHint``,
  ``humanConfirmationRequired``
- ``tags`` (``list[str]``) — free-form labels, with a small recognized
  allowlist (:data:`RECOGNIZED_TAGS`) for cases ``category`` + ``annotations``
  alone don't disambiguate
- ``module_name`` — repo-of-origin (``kaos-core``, ``kaos-pdf``, etc.)

The 11 SessionToolSet groups are **derived views** over those fields, not new
ground truth. Adding a new tool elsewhere with sane metadata auto-classifies
on the next runtime walk — zero kaos-agents release needed. Third-party
tools self-declare via the same path.

Recognized tags (the only narrowing signal that lives outside enum/annotation
state):

- ``"browser"`` — sandboxed Chromium / Playwright surface
- ``"netinfra"`` — DNS / WHOIS / TLS / TCP / UDP / cert intel
- ``"forensics"`` — offline byte-processing tools not covered by
  ``category=DOCUMENT`` alone (PACER, email, vCard, image / file metadata,
  filesystem discovery)
- ``"retrieval"`` — BM25 / dense / hybrid search surfaces

Tools that omit these tags fall into the broader category-derived bucket;
setting one of them is a *narrowing* signal. No enum, no breaking change.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from kaos_core.types.enums import ToolCapability, ToolCategory

from kaos_agents.types.tool_group import ToolGroup

if TYPE_CHECKING:
    from kaos_core import KaosRuntime
    from kaos_core.types.metadata import ToolMetadata

    from kaos_agents.registry.tool_group_registry import ToolGroupRegistry


# The four tag values the derivation function recognizes as narrowing
# signals. Tools may carry additional free-form tags for cross-cutting
# concerns (``"experimental"``, ``"deprecated"``, domain labels) — those
# are passed through unmolested but don't influence group assignment.
RECOGNIZED_TAGS: frozenset[str] = frozenset({"browser", "netinfra", "forensics", "retrieval"})


# The 4 self-recursive kaos-agents tools — registered in the agents group
# but always denied via SessionToolSet.DEFAULT_DENIED_TOOLS so an accidental
# group opt-in doesn't trigger infinite recursion.
_SELF_RECURSIVE_AGENT_TOOLS: frozenset[str] = frozenset(
    {
        "kaos-agent-chat",
        "kaos-agent-plan",
        "kaos-agent-findings",
        "kaos-agent-corpus-filter",
    }
)


# Group descriptions surfaced in the SettingsSheet Tool Policy section + the
# `render_tool_catalog(mode="grouped")` prompt projection.
KAOS_TOOL_GROUP_DESCRIPTIONS: dict[str, str] = {
    "web": (
        "Network-egress fetch + search surface. HTTP/HTTPS GETs, web "
        "search engines, public-data APIs (Federal Register, eCFR, "
        "EDGAR, GovInfo, GLEIF, CourtListener). Granted when the "
        "session permits outbound traffic."
    ),
    "browser": (
        "Sandboxed Chromium (Playwright) — full JS-rendering, click + "
        "fill + type, screenshots, network capture. Opt-in (CPU + RAM "
        "cost; requires the kaos-web[browser] extra at runtime)."
    ),
    "netinfra": (
        "Network-infrastructure introspection — DNS lookups, WHOIS, "
        "TLS certificate inspection, TCP / UDP banner grabbing, HTTP "
        "header analysis. Default off; opt-in for compliance / "
        "diligence / abuse-investigation sessions."
    ),
    "documents": (
        "Read-only document parsers — PDF / DOCX / PPTX / XLSX text + "
        "structure + metadata + BM25 search. Produces ContentDocument "
        "or TabularDocument AST nodes; never mutates the source file."
    ),
    "citations": (
        "Legal-citation parsing, linking, and reporter normalization. "
        "Builds on the documents group."
    ),
    "vfs": (
        "Per-session virtual filesystem + artifact store. Read / "
        "write / list operations on the session-scoped sandbox; never "
        "escapes the session root."
    ),
    "forensics": (
        "Offline byte-processing — filesystem discovery, archive "
        "inspection, PACER docket parsing, email / vCard / image "
        "metadata extraction. Read-only on bytes the session already "
        "controls; no network egress."
    ),
    "retrieval": (
        "Corpus-search surface — BM25 / dense / hybrid over uploaded "
        "files + the RetrievalAgent sub-agent tools (synonyms, HyDE, "
        "evaluate, rerank, answer)."
    ),
    "authoring": (
        "Document writers — produces or mutates a DOCX / PPTX / XLSX "
        "/ PDF artifact. Opt-in: a session that wants drafting "
        "workflows toggles this on."
    ),
    "programs": (
        "kaos-llm-core typed-program wrappers (Call, ReAct, Refine, "
        "Judge, optimizers, codecs, batch ops) plus the 6 "
        "deterministic alpha-* extractors (date, duration, entity, "
        "money, number, percent). Opt-in — power-user surface."
    ),
    "agents": (
        "Self-recursive agent dispatch (chat / plan / findings / "
        "corpus-filter). Default denied at the ceiling AND in "
        "DEFAULT_DENIED_TOOLS — opting into the agents group is "
        "not enough; the per-tool deny must also be cleared. Reserved "
        "for explicit multi-agent topologies."
    ),
}


def derive_group(meta: ToolMetadata) -> str | None:
    """Derive a SessionToolSet group from a tool's ``ToolMetadata``.

    Pure function over ``category`` + ``capability`` + ``annotations`` +
    ``tags`` + ``module_name``. First-match-wins on a small truth table;
    returns ``None`` for tools that don't fit any group (the caller can
    treat them as either "ungrouped/other" or filter them out).

    The 11 groups in declaration order:

    1. **agents** — the 4 self-recursive kaos-agent dispatch tools (by
       name allowlist, since they're indistinguishable from other
       ``category=AGENT`` tools by metadata alone)
    2. **programs** — every other ``module_name=kaos-llm-core`` agent tool
    3. **browser** — tools tagged ``"browser"`` (Playwright surface)
    4. **netinfra** — tools tagged ``"netinfra"`` (DNS / WHOIS / TLS)
    5. **forensics** — tools tagged ``"forensics"`` (offline parsers)
    6. **retrieval** — tools tagged ``"retrieval"`` (BM25 / dense search)
    7. **citations** — ``module_name=kaos-citations`` read-only tools
    8. **vfs** — ``module_name=kaos-core`` + ``category=UTILITY``
    9. **authoring** — ``capability in {GENERATE, TRANSFORM}`` AND not read-only
    10. **web** — ``openWorldHint=True`` AND ``readOnlyHint=True`` (after the
        tag-based browser/netinfra carve-outs)
    11. **documents** — ``category=DOCUMENT`` AND ``openWorldHint=False``

    Order matters: tag-based rules (3-6) run before category-based rules so
    a kaos-source PACER parser (`category=DOCUMENT`, `tags=["forensics"]`)
    lands in ``forensics``, not ``documents``.
    """
    name = meta.name
    module_name = meta.module_name
    tags = set(meta.tags)
    annotations = meta.annotations

    # 1. agents — self-recursive kaos-agent dispatch tools (name allowlist)
    if name in _SELF_RECURSIVE_AGENT_TOOLS:
        return "agents"

    # 2. programs — kaos-llm-core typed-program + alpha-* extractors
    if module_name == "kaos-llm-core":
        return "programs"

    # 3-6. Tag-based narrowings (run before category-based rules so a
    # DOCUMENT-category tool tagged "forensics" lands in forensics, not
    # documents).
    if "browser" in tags:
        return "browser"
    if "netinfra" in tags:
        return "netinfra"
    if "forensics" in tags:
        return "forensics"
    if "retrieval" in tags:
        return "retrieval"

    # 7. citations — module-scoped read-only tools
    if module_name == "kaos-citations":
        return "citations"

    # 8. vfs — kaos-core utility tools (filesystem primitives + artifact
    # store). UTILITY is currently kaos-core-only by convention.
    if module_name == "kaos-core" and meta.category == ToolCategory.UTILITY:
        return "vfs"

    # 9. authoring — writers / mutators with side-effects. Must come before
    # the web/documents rules so a hypothetical openWorld-true writer still
    # lands in authoring.
    if (
        meta.capability in {ToolCapability.GENERATE, ToolCapability.TRANSFORM}
        and annotations is not None
        and annotations.readOnlyHint is False
        and annotations.destructiveHint is False
    ):
        return "authoring"

    # 10. web — network-egress + read-only (after browser/netinfra carve-outs)
    if (
        annotations is not None
        and annotations.openWorldHint is True
        and annotations.readOnlyHint is True
    ):
        return "web"

    # 11. documents — offline read-only document parsing
    if (
        meta.category == ToolCategory.DOCUMENT
        and annotations is not None
        and annotations.openWorldHint is False
        and annotations.readOnlyHint is True
    ):
        return "documents"

    return None


def register_kaos_tool_groups(
    runtime: KaosRuntime,
    registry: ToolGroupRegistry | None = None,
) -> dict[str, int]:
    """Walk a runtime + register every derived group.

    Reads every tool currently registered on the runtime, calls
    :func:`derive_group` on each, and writes one :class:`ToolGroup` per
    non-empty bucket into the registry (with ``force=True`` so repeated
    calls — e.g. across test sessions — stay idempotent).

    Args:
        runtime: The runtime whose tools to inspect.
        registry: Target registry; defaults to
            :data:`default_tool_group_registry`.

    Returns:
        ``{group_name: tool_count}`` for every group with at least one
        matched tool.
    """
    if registry is None:
        from kaos_agents.registry.tool_group_registry import (
            default_tool_group_registry,
        )

        registry = default_tool_group_registry

    by_group: dict[str, list[str]] = {}
    for tool in runtime.tools.list_tool_objects():
        meta = tool.metadata
        group = derive_group(meta)
        if group is None:
            continue
        by_group.setdefault(group, []).append(meta.name)

    counts: dict[str, int] = {}
    for group_name, names in by_group.items():
        description = KAOS_TOOL_GROUP_DESCRIPTIONS.get(group_name, f"{group_name} tool group")
        registry.register(
            ToolGroup(
                name=group_name,
                description=description,
                tool_names=tuple(sorted(names)),
            ),
            force=True,
        )
        counts[group_name] = len(names)
    return counts


__all__ = [
    "KAOS_TOOL_GROUP_DESCRIPTIONS",
    "RECOGNIZED_TAGS",
    "derive_group",
    "register_kaos_tool_groups",
]
