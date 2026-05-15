"""Tests for the canonical KAOS tool-group taxonomy + classifier.

PRD `kaos-modules/docs/internal/dynamic-tool-planning-prd.md` §4 (PR 2)
moves the prefix-pattern taxonomy from kaos-ui into kaos-agents. Pins
the classification for every tool name a session is realistically
going to see — a regression here surfaces as planner mis-routing.
"""

from __future__ import annotations

import pytest

from kaos_agents.registry import (
    KAOS_TOOL_GROUP_DESCRIPTIONS,
    KAOS_TOOL_GROUP_PREFIXES,
    ToolGroupRegistry,
    classify_tool_group,
    register_kaos_tool_groups,
)


class TestPrefixTaxonomy:
    """Pin every group that's documented as part of the 11-group
    taxonomy. A drift here breaks downstream UI presets + planner
    few-shot prompts."""

    def test_eleven_groups_in_descriptions(self) -> None:
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

    def test_every_prefix_resolves_to_a_documented_group(self) -> None:
        documented = set(KAOS_TOOL_GROUP_DESCRIPTIONS)
        for _prefix, group in KAOS_TOOL_GROUP_PREFIXES:
            assert group in documented, f"prefix maps to undocumented group {group!r}"


class TestClassifyToolGroup:
    """Pin classifier output for representative tool names per group."""

    @pytest.mark.parametrize(
        ("tool_name", "expected_group"),
        [
            # web (network egress)
            ("kaos-source-fetch-url", "web"),
            ("kaos-source-fr-search", "web"),
            ("kaos-source-ecfr-titles", "web"),
            ("kaos-source-edgar-search", "web"),
            ("kaos-source-govinfo-search", "web"),
            ("kaos-source-gleif-search", "web"),
            ("kaos-web-fetch-page", "web"),
            ("kaos-web-get-text", "web"),
            ("kaos-web-search", "web"),
            ("kaos-web-batch-fetch", "web"),
            ("kaos-web-crawl-site", "web"),
            ("kaos-web-discover-urls", "web"),
            ("kaos-citations-cl-search", "web"),
            # browser
            ("kaos-web-browser-navigate", "browser"),
            ("kaos-web-browser-screenshot", "browser"),
            # netinfra
            ("kaos-web-dns-lookup", "netinfra"),
            ("kaos-web-whois-lookup", "netinfra"),
            ("kaos-web-tls-inspect", "netinfra"),
            ("kaos-web-extract-org", "netinfra"),
            # documents (read-only)
            ("kaos-pdf-extract-parse", "documents"),
            ("kaos-pdf-render-page", "documents"),
            ("kaos-pdf-metadata", "documents"),
            ("kaos-pdf-search-document", "documents"),
            ("kaos-office-parse-docx", "documents"),
            ("kaos-office-get-text", "documents"),
            ("kaos-office-search", "documents"),
            ("kaos-office-xlsx-metadata", "documents"),
            ("kaos-content-extract-blocks", "documents"),
            # citations
            ("kaos-citations-parse", "citations"),
            # vfs
            ("kaos-core-vfs-read", "vfs"),
            ("kaos-core-artifacts-list", "vfs"),
            # forensics (offline byte ops)
            ("kaos-source-discover", "forensics"),
            ("kaos-source-describe", "forensics"),
            ("kaos-source-preview", "forensics"),
            ("kaos-source-materialize", "forensics"),
            ("kaos-source-inspect-archive", "forensics"),
            ("kaos-source-pacer-parse", "forensics"),
            ("kaos-source-vcard-parse", "forensics"),
            ("kaos-source-parse-eml", "forensics"),
            ("kaos-source-parse-mbox", "forensics"),
            ("kaos-source-email-forensics", "forensics"),
            ("kaos-source-file-metadata", "forensics"),
            ("kaos-source-image-metadata", "forensics"),
            # retrieval
            ("kaos-agents-retrieval-bm25", "retrieval"),
            ("kaos-agents-retrieval-synonyms", "retrieval"),
            ("kaos-source-bm25-search", "retrieval"),
            ("kaos-nlp-core-bm25-search", "retrieval"),
            # authoring (writers — opt-in)
            ("kaos-pdf-write-merge", "authoring"),
            ("kaos-office-write-docx", "authoring"),
            ("kaos-office-write-pptx", "authoring"),
            ("kaos-office-write-xlsx", "authoring"),
            # programs (kaos-llm-core typed-program + alpha-*)
            ("kaos-llm-core-call", "programs"),
            ("kaos-llm-core-react", "programs"),
            ("kaos-llm-core-alpha-date", "programs"),
            ("kaos-llm-core-alpha-money", "programs"),
            # agents (self-recursive — also in DEFAULT_DENIED_TOOLS)
            ("kaos-agent-chat", "agents"),
            ("kaos-agent-plan", "agents"),
            ("kaos-agent-findings", "agents"),
            ("kaos-agent-corpus-filter", "agents"),
        ],
    )
    def test_tool_name_classifies_to_expected_group(
        self, tool_name: str, expected_group: str
    ) -> None:
        assert classify_tool_group(tool_name) == expected_group, (
            f"{tool_name!r} did not classify as {expected_group!r}"
        )

    def test_unknown_tool_name_returns_none(self) -> None:
        assert classify_tool_group("totally-unrelated-tool") is None
        assert classify_tool_group("") is None

    def test_kaos_source_fr_beats_forensics_ordering(self) -> None:
        """`kaos-source-fr-*` is online (web) and must NOT be swallowed
        into forensics by a less-specific `kaos-source-` prefix that
        appears later in the list."""
        assert classify_tool_group("kaos-source-fr-search") == "web"
        assert classify_tool_group("kaos-source-pacer-parse") == "forensics"


class _NameOnly:
    """Stand-in for a KaosTool — exposes only ``metadata.name``,
    which is the only attribute :func:`register_kaos_tool_groups`
    actually reads. Avoids building a full ``ToolMetadata`` (which
    would require category / capability / module_name / version)."""

    def __init__(self, name: str) -> None:
        self.metadata = type("Meta", (), {"name": name})()


class _FakeToolsRegistry:
    """Stand-in for ``runtime.tools`` — only needs
    :meth:`list_tool_objects`."""

    def __init__(self, tools: list[_NameOnly]) -> None:
        self._tools = tools

    def list_tool_objects(self) -> list[_NameOnly]:
        return self._tools


class _FakeRuntime:
    """Stand-in for KaosRuntime — only the ``tools`` attribute is
    touched."""

    def __init__(self, names: list[str]) -> None:
        self.tools = _FakeToolsRegistry([_NameOnly(name) for name in names])


class TestRegisterKaosToolGroups:
    """The end-to-end happy path: a runtime with a representative
    tool inventory partitions into the expected groups."""

    def test_partitions_a_representative_runtime(self) -> None:
        runtime = _FakeRuntime(
            [
                "kaos-source-fetch-url",
                "kaos-source-fr-search",
                "kaos-source-pacer-parse",
                "kaos-web-fetch-page",
                "kaos-web-browser-navigate",
                "kaos-web-dns-lookup",
                "kaos-pdf-extract-parse",
                "kaos-office-write-docx",
                "kaos-citations-parse",
                "kaos-core-vfs-read",
                "kaos-llm-core-alpha-date",
                "kaos-agent-chat",
                "totally-unrelated-tool",  # falls outside the taxonomy
            ]
        )
        registry = ToolGroupRegistry()
        counts = register_kaos_tool_groups(
            runtime,  # ty: ignore[invalid-argument-type]
            registry=registry,
        )
        assert counts == {
            "web": 3,  # fetch-url + fr-search + fetch-page
            "browser": 1,
            "netinfra": 1,
            "documents": 1,
            "citations": 1,
            "vfs": 1,
            "forensics": 1,  # pacer-parse (discover/etc. not in this stub set)
            "authoring": 1,  # write-docx
            "programs": 1,
            "agents": 1,
        }
        assert "totally-unrelated-tool" not in [
            name for group in registry.groups() for name in group.tool_names
        ]
        # Every populated group carries the canonical description.
        for group in registry.groups():
            assert group.description == KAOS_TOOL_GROUP_DESCRIPTIONS[group.name]
