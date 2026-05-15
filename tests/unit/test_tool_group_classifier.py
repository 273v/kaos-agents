"""Tests for the derivation-based KAOS tool-group classifier.

PRD `kaos-modules/docs/internal/dynamic-tool-planning-completion-plan.md`
§2.3 (v2) defines the 11 groups as derived views over
``ToolMetadata.category`` + ``capability`` + ``annotations`` + ``tags`` +
``module_name``. These tests pin the truth-table behavior so a regression
surfaces here before it breaks the planner / SettingsSheet UI.
"""

from __future__ import annotations

import pytest
from kaos_core.types.annotations import ToolAnnotations
from kaos_core.types.enums import ToolCapability, ToolCategory
from kaos_core.types.metadata import ToolMetadata

from kaos_agents.registry import (
    KAOS_TOOL_GROUP_DESCRIPTIONS,
    RECOGNIZED_TAGS,
    ToolGroupRegistry,
    derive_group,
    register_kaos_tool_groups,
)

# ---------------------------------------------------------------------------
# Helpers — build a minimal ToolMetadata for each truth-table row
# ---------------------------------------------------------------------------


def _meta(
    *,
    name: str = "kaos-test-tool",
    module_name: str = "kaos-source",
    category: ToolCategory = ToolCategory.DOCUMENT,
    capability: ToolCapability = ToolCapability.EXTRACT,
    tags: list[str] | None = None,
    read_only: bool = True,
    destructive: bool = False,
    idempotent: bool = True,
    open_world: bool = False,
) -> ToolMetadata:
    """Build a ToolMetadata with all classification-relevant fields set."""
    return ToolMetadata(
        name=name,
        description="test fixture",
        category=category,
        capability=capability,
        tags=list(tags or []),
        module_name=module_name,
        version="0.0.0-test",
        annotations=ToolAnnotations(
            readOnlyHint=read_only,
            destructiveHint=destructive,
            idempotentHint=idempotent,
            openWorldHint=open_world,
        ),
    )


# ---------------------------------------------------------------------------
# RECOGNIZED_TAGS surface
# ---------------------------------------------------------------------------


class TestRecognizedTags:
    """The four tags the derivation function reads as narrowing signals."""

    def test_recognized_tags_are_documented(self) -> None:
        assert frozenset({"browser", "netinfra", "forensics", "retrieval"}) == RECOGNIZED_TAGS

    def test_descriptions_cover_all_11_groups(self) -> None:
        assert set(KAOS_TOOL_GROUP_DESCRIPTIONS) == {
            "web",
            "browser",
            "netinfra",
            "documents",
            "citations",
            "vfs",
            "forensics",
            "retrieval",
            "authoring",
            "programs",
            "agents",
        }


# ---------------------------------------------------------------------------
# derive_group truth table — one parametrized test per row
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("description", "meta", "expected"),
    [
        # 1. agents — self-recursive dispatch tools (name allowlist)
        (
            "kaos-agent-chat → agents",
            _meta(
                name="kaos-agent-chat",
                module_name="kaos-agents",
                category=ToolCategory.AGENT,
                capability=ToolCapability.GENERATE,
            ),
            "agents",
        ),
        (
            "kaos-agent-plan → agents",
            _meta(
                name="kaos-agent-plan",
                module_name="kaos-agents",
                category=ToolCategory.AGENT,
                capability=ToolCapability.GENERATE,
            ),
            "agents",
        ),
        (
            "kaos-agent-findings → agents",
            _meta(
                name="kaos-agent-findings",
                module_name="kaos-agents",
                category=ToolCategory.AGENT,
                capability=ToolCapability.EXTRACT,
            ),
            "agents",
        ),
        (
            "kaos-agent-corpus-filter → agents",
            _meta(
                name="kaos-agent-corpus-filter",
                module_name="kaos-agents",
                category=ToolCategory.AGENT,
                capability=ToolCapability.QUERY,
            ),
            "agents",
        ),
        # 2. programs — every other kaos-llm-core agent tool
        (
            "kaos-llm-core-call → programs",
            _meta(
                name="kaos-llm-core-call",
                module_name="kaos-llm-core",
                category=ToolCategory.AGENT,
                capability=ToolCapability.GENERATE,
            ),
            "programs",
        ),
        (
            "kaos-llm-core-alpha-date (programs even though deterministic)",
            _meta(
                name="kaos-llm-core-alpha-date",
                module_name="kaos-llm-core",
                category=ToolCategory.TEXT,
                capability=ToolCapability.EXTRACT,
            ),
            "programs",
        ),
        # 3. browser — tagged
        (
            "kaos-web-browser-navigate → browser (tag)",
            _meta(
                name="kaos-web-browser-navigate",
                module_name="kaos-web",
                category=ToolCategory.DOCUMENT,
                capability=ToolCapability.EXTRACT,
                tags=["browser"],
                open_world=True,
            ),
            "browser",
        ),
        # 4. netinfra — tagged
        (
            "kaos-web-dns-lookup → netinfra (tag)",
            _meta(
                name="kaos-web-dns-lookup",
                module_name="kaos-web",
                category=ToolCategory.INTEGRATION,
                capability=ToolCapability.QUERY,
                tags=["netinfra"],
                open_world=True,
            ),
            "netinfra",
        ),
        # 5. forensics — tagged (overrides documents)
        (
            "kaos-source-pacer-parse → forensics (tag beats DOCUMENT category)",
            _meta(
                name="kaos-source-pacer-parse",
                module_name="kaos-source",
                category=ToolCategory.DOCUMENT,
                capability=ToolCapability.EXTRACT,
                tags=["forensics"],
                open_world=False,
            ),
            "forensics",
        ),
        (
            "kaos-source-discover → forensics (tag, even though it's a discovery tool)",
            _meta(
                name="kaos-source-discover",
                module_name="kaos-source",
                category=ToolCategory.DOCUMENT,
                capability=ToolCapability.QUERY,
                tags=["forensics"],
                open_world=False,
            ),
            "forensics",
        ),
        # 6. retrieval — tagged
        (
            "kaos-retrieval-bm25 → retrieval (tag)",
            _meta(
                name="kaos-retrieval-bm25",
                module_name="kaos-agents",
                category=ToolCategory.DATA,
                capability=ToolCapability.QUERY,
                tags=["retrieval"],
                open_world=False,
            ),
            "retrieval",
        ),
        (
            "kaos-source-bm25-search → retrieval (tag, cross-repo)",
            _meta(
                name="kaos-source-bm25-search",
                module_name="kaos-source",
                category=ToolCategory.DATA,
                capability=ToolCapability.QUERY,
                tags=["retrieval"],
                open_world=False,
            ),
            "retrieval",
        ),
        # 7. citations — module-scoped
        (
            "kaos-citations-parse → citations (module)",
            _meta(
                name="kaos-citations-parse",
                module_name="kaos-citations",
                category=ToolCategory.TEXT,
                capability=ToolCapability.EXTRACT,
                read_only=True,
                open_world=False,
            ),
            "citations",
        ),
        # 8. vfs — kaos-core UTILITY
        (
            "kaos-core-vfs-read → vfs (module + UTILITY)",
            _meta(
                name="kaos-core-vfs-read",
                module_name="kaos-core",
                category=ToolCategory.UTILITY,
                capability=ToolCapability.QUERY,
            ),
            "vfs",
        ),
        (
            "kaos-core-artifacts-list → vfs (module + UTILITY)",
            _meta(
                name="kaos-core-artifacts-list",
                module_name="kaos-core",
                category=ToolCategory.UTILITY,
                capability=ToolCapability.QUERY,
            ),
            "vfs",
        ),
        # 9. authoring — writer with no destructive side-effects
        (
            "kaos-office-write-docx → authoring (GENERATE, not read-only)",
            _meta(
                name="kaos-office-write-docx",
                module_name="kaos-office",
                category=ToolCategory.DOCUMENT,
                capability=ToolCapability.GENERATE,
                read_only=False,
                destructive=False,
            ),
            "authoring",
        ),
        (
            "kaos-pdf-write-merge → authoring (TRANSFORM, not read-only)",
            _meta(
                name="kaos-pdf-write-merge",
                module_name="kaos-pdf",
                category=ToolCategory.DOCUMENT,
                capability=ToolCapability.TRANSFORM,
                read_only=False,
                destructive=False,
            ),
            "authoring",
        ),
        # 10. web — openWorld + read-only (after tag carve-outs)
        (
            "kaos-source-fr-search → web (openWorld + read-only)",
            _meta(
                name="kaos-source-fr-search",
                module_name="kaos-source",
                category=ToolCategory.DOCUMENT,
                capability=ToolCapability.QUERY,
                open_world=True,
                read_only=True,
            ),
            "web",
        ),
        (
            "kaos-source-fetch-url → web",
            _meta(
                name="kaos-source-fetch-url",
                module_name="kaos-source",
                category=ToolCategory.DOCUMENT,
                capability=ToolCapability.EXTRACT,
                open_world=True,
                read_only=True,
            ),
            "web",
        ),
        (
            "kaos-web-fetch-page → web",
            _meta(
                name="kaos-web-fetch-page",
                module_name="kaos-web",
                category=ToolCategory.DOCUMENT,
                capability=ToolCapability.EXTRACT,
                open_world=True,
                read_only=True,
            ),
            "web",
        ),
        (
            "kaos-citations-cl-search → web (openWorld beats citations module rule)",
            _meta(
                name="kaos-citations-cl-search",
                module_name="kaos-citations",
                category=ToolCategory.DOCUMENT,
                capability=ToolCapability.QUERY,
                open_world=True,
                read_only=True,
            ),
            # Per the rule order: citations comes AFTER tag-based rules but
            # BEFORE web. So citations-module + read-only lands in citations,
            # not web. (Confirms ordering.)
            "citations",
        ),
        # 11. documents — DOCUMENT category + read-only + offline
        (
            "kaos-pdf-extract-parse → documents",
            _meta(
                name="kaos-pdf-extract-parse",
                module_name="kaos-pdf",
                category=ToolCategory.DOCUMENT,
                capability=ToolCapability.EXTRACT,
                open_world=False,
                read_only=True,
            ),
            "documents",
        ),
        (
            "kaos-office-parse-docx → documents",
            _meta(
                name="kaos-office-parse-docx",
                module_name="kaos-office",
                category=ToolCategory.DOCUMENT,
                capability=ToolCapability.EXTRACT,
                open_world=False,
                read_only=True,
            ),
            "documents",
        ),
        # Fallthroughs — tools with no matching rule return None
        (
            "MEDIA category + ANALYZE (no rule matches) → None",
            _meta(
                module_name="kaos-anomaly",
                category=ToolCategory.MEDIA,
                capability=ToolCapability.ANALYZE,
                open_world=False,
                read_only=True,
            ),
            None,
        ),
        (
            "third-party tool with no metadata signal → None",
            _meta(
                name="acme-mystery-tool",
                module_name="acme-tools",
                category=ToolCategory.UTILITY,
                capability=ToolCapability.ANALYZE,
                open_world=False,
                read_only=True,
            ),
            None,
        ),
    ],
)
def test_derive_group_truth_table(
    description: str, meta: ToolMetadata, expected: str | None
) -> None:
    actual = derive_group(meta)
    assert actual == expected, f"{description}: expected {expected!r}, got {actual!r}"


# ---------------------------------------------------------------------------
# Rule ordering — pin the precedence rules
# ---------------------------------------------------------------------------


class TestRuleOrdering:
    """Pin the precedence rules where two derivation rules could both fire."""

    def test_self_recursive_name_beats_module_name(self) -> None:
        """`kaos-agent-chat` is in kaos-agents module — both the name
        allowlist (rule 1) and the kaos-llm-core rule (rule 2) could fire if
        we got the module wrong. Confirm the name allowlist wins."""
        meta = _meta(
            name="kaos-agent-chat",
            module_name="kaos-agents",  # NOT kaos-llm-core
            category=ToolCategory.AGENT,
            capability=ToolCapability.GENERATE,
        )
        assert derive_group(meta) == "agents"

    def test_tag_beats_category(self) -> None:
        """A tool with `category=DOCUMENT` but `tags=["forensics"]` lands
        in forensics. Without the tag rule, it would land in documents."""
        meta = _meta(
            module_name="kaos-source",
            category=ToolCategory.DOCUMENT,
            capability=ToolCapability.EXTRACT,
            tags=["forensics"],
            open_world=False,
            read_only=True,
        )
        assert derive_group(meta) == "forensics"

    def test_citations_module_beats_web(self) -> None:
        """`kaos-citations-cl-search` is openWorld + read-only — under the
        web rule it would be web. But the citations module rule fires first."""
        meta = _meta(
            name="kaos-citations-cl-search",
            module_name="kaos-citations",
            category=ToolCategory.DOCUMENT,
            capability=ToolCapability.QUERY,
            open_world=True,
            read_only=True,
        )
        assert derive_group(meta) == "citations"

    def test_authoring_beats_documents(self) -> None:
        """A DOCUMENT-category tool with `capability=GENERATE` + `readOnly=False`
        is authoring, not documents (which requires `readOnly=True`)."""
        meta = _meta(
            module_name="kaos-office",
            category=ToolCategory.DOCUMENT,
            capability=ToolCapability.GENERATE,
            open_world=False,
            read_only=False,
            destructive=False,
        )
        assert derive_group(meta) == "authoring"

    def test_unknown_extra_tag_passes_through(self) -> None:
        """Tools may carry tags outside RECOGNIZED_TAGS — those don't affect
        classification."""
        meta = _meta(
            module_name="kaos-pdf",
            category=ToolCategory.DOCUMENT,
            capability=ToolCapability.EXTRACT,
            tags=["experimental", "high-cost"],  # neither is recognized
            open_world=False,
            read_only=True,
        )
        assert derive_group(meta) == "documents"


# ---------------------------------------------------------------------------
# register_kaos_tool_groups — runtime walker
# ---------------------------------------------------------------------------


class _NameOnlyTool:
    """Minimal stand-in for a KaosTool that carries only `metadata`."""

    def __init__(self, meta: ToolMetadata) -> None:
        self.metadata = meta


class _FakeToolsRegistry:
    def __init__(self, tools: list[_NameOnlyTool]) -> None:
        self._tools = tools

    def list_tool_objects(self) -> list[_NameOnlyTool]:
        return self._tools


class _FakeRuntime:
    def __init__(self, tools: list[ToolMetadata]) -> None:
        self.tools = _FakeToolsRegistry([_NameOnlyTool(m) for m in tools])


class TestRegisterKaosToolGroups:
    """End-to-end happy path: a runtime with a representative tool
    inventory partitions into the expected groups."""

    def test_representative_runtime_partitions_correctly(self) -> None:
        runtime = _FakeRuntime(
            [
                # web
                _meta(
                    name="kaos-source-fetch-url",
                    module_name="kaos-source",
                    open_world=True,
                    read_only=True,
                ),
                _meta(
                    name="kaos-source-fr-search",
                    module_name="kaos-source",
                    open_world=True,
                    read_only=True,
                ),
                # browser
                _meta(
                    name="kaos-web-browser-navigate",
                    module_name="kaos-web",
                    tags=["browser"],
                    open_world=True,
                ),
                # netinfra
                _meta(
                    name="kaos-web-dns-lookup",
                    module_name="kaos-web",
                    category=ToolCategory.INTEGRATION,
                    capability=ToolCapability.QUERY,
                    tags=["netinfra"],
                    open_world=True,
                ),
                # forensics
                _meta(
                    name="kaos-source-pacer-parse",
                    module_name="kaos-source",
                    tags=["forensics"],
                ),
                # documents
                _meta(name="kaos-pdf-extract-parse", module_name="kaos-pdf"),
                # citations
                _meta(
                    name="kaos-citations-parse",
                    module_name="kaos-citations",
                    category=ToolCategory.TEXT,
                ),
                # vfs
                _meta(
                    name="kaos-core-vfs-read",
                    module_name="kaos-core",
                    category=ToolCategory.UTILITY,
                ),
                # authoring
                _meta(
                    name="kaos-office-write-docx",
                    module_name="kaos-office",
                    capability=ToolCapability.GENERATE,
                    read_only=False,
                ),
                # retrieval
                _meta(
                    name="kaos-retrieval-bm25",
                    module_name="kaos-agents",
                    category=ToolCategory.DATA,
                    capability=ToolCapability.QUERY,
                    tags=["retrieval"],
                ),
                # programs
                _meta(
                    name="kaos-llm-core-call",
                    module_name="kaos-llm-core",
                    category=ToolCategory.AGENT,
                    capability=ToolCapability.GENERATE,
                ),
                # agents
                _meta(
                    name="kaos-agent-chat",
                    module_name="kaos-agents",
                    category=ToolCategory.AGENT,
                    capability=ToolCapability.GENERATE,
                ),
                # falls outside the taxonomy
                _meta(
                    name="acme-mystery-tool",
                    module_name="acme",
                    category=ToolCategory.MEDIA,
                    capability=ToolCapability.ANALYZE,
                ),
            ]
        )
        registry = ToolGroupRegistry()
        counts = register_kaos_tool_groups(
            runtime,  # ty: ignore[invalid-argument-type]
            registry=registry,
        )
        assert counts == {
            "web": 2,
            "browser": 1,
            "netinfra": 1,
            "forensics": 1,
            "documents": 1,
            "citations": 1,
            "vfs": 1,
            "authoring": 1,
            "retrieval": 1,
            "programs": 1,
            "agents": 1,
        }
        # Each populated group carries the canonical description.
        for group in registry.groups():
            assert group.description == KAOS_TOOL_GROUP_DESCRIPTIONS[group.name]
        # Ungrouped tools never land in the registry.
        for group in registry.groups():
            assert "acme-mystery-tool" not in group.tool_names
