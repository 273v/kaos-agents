"""Canonical KAOS tool-group taxonomy + classifier.

PRD `kaos-modules/docs/internal/dynamic-tool-planning-prd.md` §4 (PR 2)
moves the prefix-pattern taxonomy from `kaos-ui/kaos_ui/agents.py`
into kaos-agents proper, so the agent runtime owns the canonical
group catalogue used by:

- :class:`SessionToolSet` ceiling enforcement (which groups a
  session may invoke)
- The per-turn :class:`TurnToolPolicy` planner (which subset of the
  ceiling this turn actually needs)
- :class:`@273v/kaos-ui-react` UI surfaces (the SettingsSheet Tool
  Policy section, the preset chip row, the CostStrip planner row)

The taxonomy here is 11 groups:

**9 default-visible groups** — registered in the runtime and visible
to the planner / SessionToolSet UI:

================  ========  ==========================================
group             default   tool prefix patterns
================  ========  ==========================================
web               on        kaos-source-fr-* / -ecfr-* / -edgar-* /
                            -govinfo-* / -gleif-* / -fetch-url,
                            kaos-web-fetch-* / -get-* / -search-* /
                            -batch-fetch / -crawl-site / -discover-urls,
                            kaos-content-fetch-*, kaos-citations-cl-*
browser           on        kaos-web-browser-*
netinfra          off       kaos-web-tcp-* / -tls-* / -http-* /
                            -service-* / -dns-* / -whois-* / -domain-* /
                            -extract-org / -fingerprint-* / -udp-*
documents         on        kaos-pdf-extract-* / -render-* / -metadata /
                            -search-document / -get-outline /
                            -classify-page, kaos-office-parse-* /
                            -get-* / -list-* / -search* / -metadata,
                            kaos-content-extract-* / -summarize-*
citations         on        kaos-citations-*
vfs               on        kaos-core-vfs-*, kaos-core-artifacts-*
forensics         on        kaos-source-discover / -describe / -preview /
                            -materialize / -inspect-archive / -pacer-* /
                            -vcard-* / -parse-eml / -parse-mbox /
                            -email-forensics / -file-metadata /
                            -image-metadata
retrieval         on        kaos-agents-retrieval-* + the 3 BM25
                            surfaces (kaos-source-bm25-search,
                            kaos-nlp-core-bm25-search,
                            kaos-agents-retrieval-bm25)
authoring         opt-in    kaos-pdf-write-*, kaos-office-write-*
================  ========  ==========================================

**2 conditional opt-in groups** — registered but explicitly denied at
the default ceiling (callers opt in per-session):

================  ==========  ========================================
group             default     tool prefix patterns
================  ==========  ========================================
programs          deny        kaos-llm-core-* (Call, ReAct, Refine,
                              optimizers, codecs, batch, alpha-*)
agents            deny        kaos-agent-chat / -plan / -findings /
                              -corpus-filter (self-recursive — see
                              :data:`DEFAULT_DENIED_TOOLS`)
================  ==========  ========================================

The classifier is **prefix-based**, not metadata-based. A new tool
under one of the registered prefixes automatically lands in the right
group on the next runtime restart — no per-tool annotation required.
Tools that don't match any prefix end up in the *unclassified* bucket;
the caller decides whether to register that as a group, ignore it, or
treat it as an opt-in surface.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

from kaos_agents.types.tool_group import ToolGroup

if TYPE_CHECKING:
    from kaos_core import KaosRuntime

    from kaos_agents.registry.tool_group_registry import ToolGroupRegistry


# Order matters: more-specific prefixes must come BEFORE less-specific
# ones. e.g. ``kaos-source-fr-`` must be checked before
# ``kaos-source-`` (which would otherwise swallow Federal Register
# tools into the ``forensics`` bucket).
KAOS_TOOL_GROUP_PREFIXES: tuple[tuple[str, str], ...] = (
    # web (network egress — HTTP fetch + search + remote APIs)
    ("kaos-source-fetch-url", "web"),
    ("kaos-source-fr-", "web"),
    ("kaos-source-ecfr-", "web"),
    ("kaos-source-edgar-", "web"),
    ("kaos-source-govinfo-", "web"),
    ("kaos-source-gleif-", "web"),
    ("kaos-web-fetch-", "web"),
    ("kaos-web-get-", "web"),
    ("kaos-web-search", "web"),
    ("kaos-web-batch-fetch", "web"),
    ("kaos-web-crawl-site", "web"),
    ("kaos-web-discover-urls", "web"),
    ("kaos-content-fetch-", "web"),
    ("kaos-citations-cl-", "web"),
    # browser (sandboxed Chromium — opt-in CPU + RAM cost)
    ("kaos-web-browser-", "browser"),
    # netinfra (DNS / WHOIS / TLS / TCP banner / UDP probe / HTTP headers
    # / cert / org-extract / service-detect / fingerprint)
    ("kaos-web-tcp-", "netinfra"),
    ("kaos-web-tls-", "netinfra"),
    ("kaos-web-http-", "netinfra"),
    ("kaos-web-service-", "netinfra"),
    ("kaos-web-dns-", "netinfra"),
    ("kaos-web-whois-", "netinfra"),
    ("kaos-web-domain-", "netinfra"),
    ("kaos-web-extract-org", "netinfra"),
    ("kaos-web-fingerprint-", "netinfra"),
    ("kaos-web-udp-", "netinfra"),
    # documents (read-only extractors / parsers / metadata / search)
    ("kaos-pdf-extract-", "documents"),
    ("kaos-pdf-render-", "documents"),
    ("kaos-pdf-metadata", "documents"),
    ("kaos-pdf-search-document", "documents"),
    ("kaos-pdf-get-outline", "documents"),
    ("kaos-pdf-classify-page", "documents"),
    ("kaos-office-parse-", "documents"),
    ("kaos-office-get-", "documents"),
    ("kaos-office-list-", "documents"),
    ("kaos-office-search", "documents"),
    ("kaos-office-metadata", "documents"),
    ("kaos-office-xlsx-metadata", "documents"),
    ("kaos-content-extract-", "documents"),
    ("kaos-content-summarize-", "documents"),
    # citations (legal-citation parsing / linking)
    ("kaos-citations-", "citations"),
    # vfs (session storage primitives)
    ("kaos-core-vfs-", "vfs"),
    ("kaos-core-artifacts-", "vfs"),
    # forensics (offline byte-processing — discovery + parsers)
    ("kaos-source-discover", "forensics"),
    ("kaos-source-describe", "forensics"),
    ("kaos-source-preview", "forensics"),
    ("kaos-source-materialize", "forensics"),
    ("kaos-source-inspect-archive", "forensics"),
    ("kaos-source-pacer-", "forensics"),
    ("kaos-source-vcard-", "forensics"),
    ("kaos-source-parse-eml", "forensics"),
    ("kaos-source-parse-mbox", "forensics"),
    ("kaos-source-email-forensics", "forensics"),
    ("kaos-source-file-metadata", "forensics"),
    ("kaos-source-image-metadata", "forensics"),
    # retrieval (BM25 surfaces + the kaos-agents RetrievalAgent tools)
    ("kaos-agents-retrieval-", "retrieval"),
    ("kaos-source-bm25-search", "retrieval"),
    ("kaos-nlp-core-bm25-search", "retrieval"),
    # authoring (writers / mutators / redactors — opt-in)
    ("kaos-pdf-write-", "authoring"),
    ("kaos-office-write-", "authoring"),
    # programs (LLM-program wrappers + alpha-* extractors — opt-in)
    ("kaos-llm-core-", "programs"),
    # agents (self-recursive agent dispatch — opt-in + still
    # DEFAULT_DENIED so accidental opt-in doesn't trigger recursion)
    ("kaos-agent-chat", "agents"),
    ("kaos-agent-plan", "agents"),
    ("kaos-agent-findings", "agents"),
    ("kaos-agent-corpus-filter", "agents"),
)


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
        "Corpus-search surface — BM25 over uploaded files + the "
        "RetrievalAgent sub-agent tools (synonyms, HyDE, evaluate)."
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


def classify_tool_group(tool_name: str) -> str | None:
    """Return the group name for ``tool_name`` or ``None``.

    Walks :data:`KAOS_TOOL_GROUP_PREFIXES` in order; the first
    matching prefix wins. Tools with no matching prefix return
    ``None`` — the caller decides whether to register them under a
    fallback bucket or leave them ungrouped.
    """
    for prefix, group in KAOS_TOOL_GROUP_PREFIXES:
        if tool_name.startswith(prefix):
            return group
    return None


def _group_tool_names(
    tool_names: Iterable[str],
) -> dict[str, list[str]]:
    """Partition tool names into ``{group_name: [tool_name, ...]}``."""
    by_group: dict[str, list[str]] = {}
    for name in tool_names:
        group = classify_tool_group(name)
        if group is None:
            continue
        by_group.setdefault(group, []).append(name)
    return by_group


def register_kaos_tool_groups(
    runtime: KaosRuntime,
    registry: ToolGroupRegistry | None = None,
) -> dict[str, int]:
    """Walk a runtime + register every prefix-classified group.

    Reads every tool currently registered on the runtime, partitions
    by :func:`classify_tool_group`, and writes one :class:`ToolGroup`
    per non-empty group into the registry (with ``force=True`` so
    repeated calls — e.g. across test sessions — stay idempotent).

    Args:
        runtime: The runtime whose tools to inspect.
        registry: Target registry; defaults to
            :data:`default_tool_group_registry`.

    Returns:
        ``{group_name: tool_count}`` for every group with at least
        one matched tool.
    """
    if registry is None:
        from kaos_agents.registry.tool_group_registry import (
            default_tool_group_registry,
        )

        registry = default_tool_group_registry

    all_names = [tool.metadata.name for tool in runtime.tools.list_tool_objects()]
    by_group = _group_tool_names(all_names)
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
    "KAOS_TOOL_GROUP_PREFIXES",
    "classify_tool_group",
    "register_kaos_tool_groups",
]
