"""Seed personas — Research / Drafting / Forensics.

Plan §12 of
``kaos-modules/docs/plans/2026-05-19-lateral-redesign-capability-layer.md``.

These three personas mirror the trio the single-user-chat SPA has
historically inlined as hardcoded UI. Lifting them into kaos-agents
gives every consumer of the Runner (CLI, MCP host, future TUI) the
same canonical definitions, so the policy stops drifting between
deployments.

The names ``"research"`` / ``"drafting"`` / ``"forensics"`` are
**registry entries**, not identifiers any business-logic branch keys
off of. Downstream code resolves them via
:data:`kaos_agents.registry.persona_registry.default_persona_registry`,
not by importing :data:`RESEARCH`/`DRAFTING`/`FORENSICS` directly.

The seeded group-name choices mirror the ``RESEARCH_SOFT_CEILING``,
``DRAFTING_SOFT_CEILING``, and ``FORENSICS_SOFT_CEILING`` constants
in :mod:`kaos_agents.types.session_policy` so the persona-as-runtime
primitive and the session-policy ceiling stay consistent. The
group-name strings themselves are validated structurally by the
``KaosPersona`` ceiling-superset rule and operationally by the
tool-group registry; ``KaosPersona`` deliberately doesn't crosscheck
the strings against the live tool-group registry so personas can
reference groups that aren't yet populated in a fresh runtime.
"""

from __future__ import annotations

from kaos_agents.registry.persona_registry import PersonaRegistry
from kaos_agents.types.persona import KaosPersona

# ─── Canonical model role preferences ────────────────────────────────
#
# Shared across all three personas. Specific personas may override
# either role independently. The role names ("classify" / "respond")
# are the conventional kaos-llm-core / kaos-agents role names.

_DEFAULT_CLASSIFY_MODEL: str = "openai:gpt-5.4-mini"
_DEFAULT_RESPOND_MODEL: str = "anthropic:claude-sonnet-4-6"

# Conservative per-turn loop budgets. Tuned to the same scale as
# ``DEFAULT_MAX_LOOP_COST_USD`` in :mod:`kaos_agents.types.session_policy`
# but free-form so a persona can override without a type bump.
_DEFAULT_BUDGETS: dict[str, float] = {
    "max_loop_cost_usd": 0.50,
    "max_tool_calls": 12.0,
}


# ─── Seeded personas ─────────────────────────────────────────────────

#: Read-mostly research persona. Pure information gathering — web,
#: browser, sources, documents, citations, vfs, retrieval. No drafting
#: surface, no forensics-only carve-outs.
RESEARCH: KaosPersona = KaosPersona.build(
    name="research",
    description=(
        "Read-mostly research. Web + browser + structured sources + "
        "documents + citations + VFS + retrieval. No drafting, no "
        "forensics-only surfaces."
    ),
    allowed_groups=(
        "web",
        "browser",
        "sources",
        "documents",
        "citations",
        "vfs",
        "retrieval",
    ),
    ceiling_groups=(
        "web",
        "browser",
        "sources",
        "documents",
        "citations",
        "vfs",
        "retrieval",
    ),
    model_role_preferences={
        "classify": _DEFAULT_CLASSIFY_MODEL,
        "respond": _DEFAULT_RESPOND_MODEL,
    },
    default_budgets=_DEFAULT_BUDGETS,
)


#: Drafting persona — Research surface plus authoring (DOCX / PPTX /
#: XLSX / PDF writers) and programs (kaos-llm-core typed-program
#: surface). Used when the user wants to redline or generate
#: deliverables.
DRAFTING: KaosPersona = KaosPersona.build(
    name="drafting",
    description=(
        "Research surface plus authoring (writers) and programs "
        "(typed-program / extractor surface). Used when the user "
        "wants to draft / redline / synthesize deliverables."
    ),
    allowed_groups=(
        "web",
        "browser",
        "sources",
        "documents",
        "citations",
        "vfs",
        "retrieval",
        "authoring",
        "programs",
    ),
    ceiling_groups=(
        "web",
        "browser",
        "sources",
        "documents",
        "citations",
        "vfs",
        "retrieval",
        "authoring",
        "programs",
    ),
    model_role_preferences={
        "classify": _DEFAULT_CLASSIFY_MODEL,
        "respond": _DEFAULT_RESPOND_MODEL,
    },
    default_budgets=_DEFAULT_BUDGETS,
)


#: Tight-lane forensics persona — local-byte processing only. No
#: outbound web, no browser, no public-data sources. Used for
#: confidentiality-sensitive review where surprise egress is itself
#: a compliance bug (DISCO Cecilia / Relativity aiR for Review
#: pattern).
FORENSICS: KaosPersona = KaosPersona.build(
    name="forensics",
    description=(
        "Tight-lane forensics. Local-byte processing only — documents, "
        "citations, VFS, forensics. No web, no browser, no sources. "
        "Used when surprise network egress is itself a compliance bug."
    ),
    allowed_groups=(
        "documents",
        "citations",
        "vfs",
        "forensics",
    ),
    ceiling_groups=(
        "documents",
        "citations",
        "vfs",
        "forensics",
    ),
    model_role_preferences={
        "classify": _DEFAULT_CLASSIFY_MODEL,
        "respond": _DEFAULT_RESPOND_MODEL,
    },
    default_budgets=_DEFAULT_BUDGETS,
)


_BUILTIN_PERSONAS: tuple[KaosPersona, ...] = (RESEARCH, DRAFTING, FORENSICS)


def register_builtin_personas(registry: PersonaRegistry) -> tuple[KaosPersona, ...]:
    """Register the three canonical personas into ``registry``.

    Idempotent: uses ``force=True`` so repeated calls (the package
    ``__init__`` runs this on every import) keep the seeded set in
    sync with the source-of-truth definitions even if a downstream
    deployment replaced one of them in the same process.

    Returns:
        The tuple of registered personas in declaration order
        (Research, Drafting, Forensics).
    """
    for persona in _BUILTIN_PERSONAS:
        registry.register(persona, force=True)
    return _BUILTIN_PERSONAS


__all__ = [
    "DRAFTING",
    "FORENSICS",
    "RESEARCH",
    "register_builtin_personas",
]
