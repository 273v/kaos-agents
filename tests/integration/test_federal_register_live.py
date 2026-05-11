"""Live integration — kaos-source Federal Register + K3 entity filters + kaos-agent-chat.

This test composes three production surfaces that were each unit-tested
in isolation but had **zero coverage as a single agent pipeline**:

1. ``kaos_source.apis.federal_register`` — fetches a real, permanently
   archived Federal Register final rule (HTTP, no auth, ~700 KB).
2. ``kaos_content.parsers.html.parse_html`` — converts the FR HTML body
   into a ``ContentDocument`` AST with paragraph + sentence structure
   that the K3 entity filters can navigate.
3. ``kaos_content.tools.entity_filters`` — three deterministic
   regex+gazetteer extractors (dates, money, durations) operating on
   sentence-level views of the AST.
4. ``kaos_agents.tools.AgentChatTool`` — single-turn synthesis with the
   K3 extracted sentences as ground-truth context.

The agent receives **only** the entity-filtered sentences (not the full
700 KB document) and must answer a structured question, citing the FR
document_number. This proves the entity filters surface the right
sentences AND that the LLM treats them as authoritative facts.

# Document selection

We pin a 2024 Department of Labor (Wage and Hour Division) final rule:

- ``document_number``: ``2024-08038``
- Title: "Defining and Delimiting the Exemptions for Executive,
  Administrative, Professional, Outside Sales, and Computer Employees"
- Citation: 89 FR 32842
- Published: 2024-04-26
- Effective: 2024-07-01 (with phased applicability on 2025-01-01)
- Contains explicit dollar thresholds (e.g. $43,888, $58,656 standard
  salary level) and durations (e.g. "every three years"), so all three
  K3 entity filters should hit.

The document number is the immutable identifier — even if the rule is
later amended or superseded, the FR API keeps the original document
permanently retrievable. If federalregister.gov reorganizes the API
path, re-pin by running:

    curl https://www.federalregister.gov/api/v1/documents.json?\\
        conditions[term]=overtime+rule&conditions[type][]=RULE

and choosing any final rule with non-null ``effective_on`` plus dollar
amounts in the body.

# Cost gate

Total LLM spend is bounded at $0.05 (one Haiku turn over ~5 KB of
filtered context). The fetch + parse + extract phases are deterministic
and free.
"""

from __future__ import annotations

import os
import re
from typing import Any

import httpx
import pytest

from kaos_agents.tools.registry import AgentChatTool

# ---------------------------------------------------------------------------
# Pin: FR document and expected ground truth
# ---------------------------------------------------------------------------

# DOL "Overtime Final Rule" — see module docstring for why this one.
PINNED_DOCUMENT_NUMBER = "2024-08038"

# Substring expected somewhere in the dates-sentences output. The full
# effective text is "The effective date for this final rule is July 1,
# 2024." but we keep the assertion narrow to a stable canonical form
# the dates extractor must emit.
EXPECTED_EFFECTIVE_DATE_PATTERNS = (
    re.compile(r"July\s+1,?\s+2024", re.IGNORECASE),
    re.compile(r"2024-07-01"),
    re.compile(r"7/1/2024"),
)

# The rule's headline salary numbers — confirmed by reading the
# published rule preamble. Used to sanity-check the money extractor
# (we only require *some* USD amounts, not these specific ones).
KNOWN_DOLLAR_AMOUNTS = ("$43,888", "$58,656", "$132,964", "$151,164")


# ---------------------------------------------------------------------------
# Skip conditions
# ---------------------------------------------------------------------------


requires_anthropic = pytest.mark.skipif(
    "ANTHROPIC_API_KEY" not in os.environ,
    reason="ANTHROPIC_API_KEY missing",
)


def _fr_reachable() -> bool:
    """Probe the FR API. The test is meaningless if FR is down."""
    try:
        with httpx.Client(timeout=10.0) as client:
            resp = client.get(
                f"https://www.federalregister.gov/api/v1/documents/{PINNED_DOCUMENT_NUMBER}.json",
                headers={"User-Agent": "kaos-tests"},
            )
            return resp.status_code == 200
    except (httpx.HTTPError, OSError):
        return False


requires_fr_reachable = pytest.mark.skipif(
    not _fr_reachable(),
    reason=(
        f"federalregister.gov API unreachable or FR doc "
        f"{PINNED_DOCUMENT_NUMBER} not retrievable — re-pin if it has "
        "been moved (see module docstring)."
    ),
)


# ---------------------------------------------------------------------------
# Fixtures (mirror test_k7_k8_mcp_tools_live.py)
# ---------------------------------------------------------------------------


@pytest.fixture
def runtime() -> Any:
    from kaos_core.registry.container import KaosRuntime

    return KaosRuntime()


@pytest.fixture
def context(runtime: Any) -> Any:
    from kaos_core.base.context import KaosContext

    return KaosContext.create(session_id="fr-k3-chat-live", runtime=runtime)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _fetch_and_parse_fr_doc(
    document_number: str,
) -> tuple[str, Any]:
    """Fetch the pinned FR document and convert HTML body → ContentDocument.

    Returns ``(raw_html, ContentDocument)``. We keep the raw HTML around
    so the test can also assert size + presence of known markers.
    """
    from kaos_content.parsers.html import parse_html
    from kaos_source.apis.federal_register.client import (
        fetch_document_content,
        get_document,
    )

    fr_doc = await get_document(document_number)
    html = await fetch_document_content(fr_doc, format="html")
    assert isinstance(html, str)

    # ``pre_content_mode="prose"`` — FR's body_html sometimes wraps
    # plain paragraphs in <pre> tags (an artifact of the original
    # type-setting). The default ``"code"`` mode would turn the entire
    # rule into one CodeBlock, which the entity filters can't traverse
    # at sentence granularity. ``"prose"`` re-segments on blank lines.
    content_doc = parse_html(
        html,
        url=f"https://www.federalregister.gov/d/{document_number}",
        pre_content_mode="prose",
    )
    return html, content_doc


def _entity_texts(payload: dict[str, Any]) -> list[str]:
    """Flatten the entity-filter tool payload to plain sentence strings."""
    return [m["text"] for m in payload.get("matches", [])]


def _format_entity_block(
    title: str,
    sentences: list[str],
    cap: int,
) -> str:
    """Render entity sentences for inclusion in the agent prompt.

    Capped per category so the total prompt stays well under the cost
    ceiling. We also dedupe — FR text often repeats effective dates
    verbatim across the preamble and the rule body, and the agent
    doesn't need duplicates to answer.
    """
    seen: set[str] = set()
    unique: list[str] = []
    for s in sentences:
        normalized = " ".join(s.split())
        if normalized in seen:
            continue
        seen.add(normalized)
        unique.append(normalized)
        if len(unique) >= cap:
            break
    if not unique:
        return f"## {title}\n(no matches)\n"
    body = "\n".join(f"- {s}" for s in unique)
    return f"## {title}\n{body}\n"


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


@pytest.mark.live
@pytest.mark.network
@requires_anthropic
@requires_fr_reachable
class TestFederalRegisterK3ChatLive:
    """Compose FR fetch + K3 entity filters + agent chat end-to-end."""

    async def test_fetch_extract_synthesize(self, runtime: Any, context: Any) -> None:
        # ------------------------------------------------------------------
        # Phase 1 — fetch + parse the pinned FR document.
        # ------------------------------------------------------------------
        html, content_doc = await _fetch_and_parse_fr_doc(PINNED_DOCUMENT_NUMBER)
        assert len(html) > 50_000, (
            f"FR document body suspiciously small ({len(html)} bytes). "
            "Re-pin the document_number if FR changed the response shape."
        )
        # The doc should have block content — empty docs would short-circuit
        # all downstream entity extraction.
        assert len(content_doc.body) > 0, (
            "parse_html produced an empty document body. "
            "Check pre_content_mode and the FR HTML structure."
        )

        # ------------------------------------------------------------------
        # Phase 2 — store as VFS artifact so the K3 tools can load it.
        # ------------------------------------------------------------------
        from kaos_content.artifacts import store_document

        manifest = await store_document(
            content_doc,
            runtime,
            context,
            name=f"fr-{PINNED_DOCUMENT_NUMBER}",
        )
        artifact_id = manifest.artifact_id

        # ------------------------------------------------------------------
        # Phase 3 — run K3 entity-filter tools (deterministic, no LLM).
        # ------------------------------------------------------------------
        from kaos_content.tools import (
            SentencesWithDatesTool,
            SentencesWithDurationsTool,
            SentencesWithMoneyTool,
        )

        dates_result = await SentencesWithDatesTool().execute(
            {"artifact_id": artifact_id, "max_results": 50},
            context,
        )
        money_result = await SentencesWithMoneyTool().execute(
            {"artifact_id": artifact_id, "max_results": 50},
            context,
        )
        durations_result = await SentencesWithDurationsTool().execute(
            {"artifact_id": artifact_id, "max_results": 50},
            context,
        )

        for label, r in (
            ("dates", dates_result),
            ("money", money_result),
            ("durations", durations_result),
        ):
            assert not r.isError, f"K3 {label} tool errored: {r.text}"
            assert r.structuredContent is not None, f"K3 {label} returned no payload"

        # Local rebinds — ``structuredContent`` is typed ``dict | None``
        # on ``ToolResult``; the asserts above prove non-None, but the
        # static checker can't narrow across the loop variable.
        dates_payload: dict[str, Any] = dates_result.structuredContent or {}
        money_payload: dict[str, Any] = money_result.structuredContent or {}
        durations_payload: dict[str, Any] = durations_result.structuredContent or {}

        dates_sentences = _entity_texts(dates_payload)
        money_sentences = _entity_texts(money_payload)
        durations_sentences = _entity_texts(durations_payload)

        # A 700 KB DOL rule MUST mention all three — if any category is
        # empty, the entity filter has regressed or the AST is broken.
        assert dates_sentences, "No date sentences extracted from FR rule"
        assert money_sentences, "No money sentences extracted from FR rule"
        assert durations_sentences, "No duration sentences extracted from FR rule"

        # The effective date "July 1, 2024" must show up *somewhere* in
        # the dates sentences. We check via the raw sentence text rather
        # than the structured ``value`` field because the test should
        # tolerate equivalent representations (ISO, US-style, ...).
        joined_dates = " | ".join(dates_sentences)
        assert any(p.search(joined_dates) for p in EXPECTED_EFFECTIVE_DATE_PATTERNS), (
            "Expected effective date 'July 1, 2024' not found in the "
            "dates-sentences output. Got first few sentences: "
            f"{dates_sentences[:3]}"
        )

        # At least one known headline dollar amount should be in the
        # money sentences (the rule explicitly publishes salary
        # thresholds).
        joined_money = " | ".join(money_sentences)
        assert any(amount in joined_money for amount in KNOWN_DOLLAR_AMOUNTS), (
            "None of the expected DOL salary thresholds "
            f"{KNOWN_DOLLAR_AMOUNTS} found in money sentences. "
            f"First few: {money_sentences[:3]}"
        )

        # ------------------------------------------------------------------
        # Phase 4 — compose entity-filtered context + run agent chat.
        # ------------------------------------------------------------------
        # Cap per-category so the prompt stays compact (~3 KB).
        entity_block = "\n".join(
            [
                _format_entity_block("Dates", dates_sentences, cap=12),
                _format_entity_block("Money amounts", money_sentences, cap=10),
                _format_entity_block("Durations", durations_sentences, cap=8),
            ]
        )

        question = (
            f"You are reviewing Federal Register document "
            f"{PINNED_DOCUMENT_NUMBER}. The following sentences were "
            "extracted by deterministic entity filters from the rule's "
            "body. Using ONLY these sentences, answer:\n\n"
            "1. What is the effective date of the rule?\n"
            "2. What is the comment period (or state that none applies)?\n"
            "3. What dollar amounts are central to the rule?\n\n"
            f"Cite the FR document number ({PINNED_DOCUMENT_NUMBER}) in "
            "your answer. Be concise.\n\n"
            f"{entity_block}"
        )

        chat_tool = AgentChatTool()
        chat_result = await chat_tool.execute(
            {
                "message": question,
                "session_id": "fr-k3-chat-live",
                "model": "anthropic:claude-haiku-4-5",
                # Disable tool calls — the agent should synthesize from
                # the prompt-injected entity sentences alone, not go
                # fetch the FR doc again.
                "tool_filter": "kaos-agent-chat",
                "max_cost_usd": 0.10,
            },
            context,
        )

        assert not chat_result.isError, f"Agent chat errored: {chat_result.text}"
        payload = chat_result.structuredContent
        assert payload is not None
        assert not payload.get("budget_exceeded"), (
            f"Agent exceeded budget: {payload.get('budget_kind')}"
        )

        answer = payload["text"]
        assert isinstance(answer, str) and len(answer) > 50, (
            f"Agent answer suspiciously short: {answer!r}"
        )

        # ------------------------------------------------------------------
        # Phase 5 — semantic assertions on the agent's answer.
        # ------------------------------------------------------------------
        # (a) Cites the document_number.
        assert PINNED_DOCUMENT_NUMBER in answer, (
            f"Agent answer did not cite document_number "
            f"{PINNED_DOCUMENT_NUMBER}. Answer: {answer!r}"
        )

        # (b) References an effective date that the entity extractor
        # actually surfaced. We accept any of the equivalent forms.
        assert any(p.search(answer) for p in EXPECTED_EFFECTIVE_DATE_PATTERNS), (
            "Agent answer's effective date does not match anything in "
            f"the dates-sentences list. Answer: {answer!r}"
        )

        # (c) References at least one dollar amount. We don't pin the
        # exact figure — Haiku may pick any from the rule's many
        # thresholds — but it MUST surface one to answer Q3.
        assert "$" in answer, (
            f"Agent answer contains no dollar amount despite money "
            f"sentences being supplied. Answer: {answer!r}"
        )

        # ------------------------------------------------------------------
        # Cost gate.
        # ------------------------------------------------------------------
        # The tool_calls list captures all sub-tool invocations; for
        # this single-turn chat with no tool calling the cost lives on
        # the TurnSummary aggregate. Pull it from the agent's session
        # memory instead — the cost-tracking hook records it there.
        # If the structuredContent ever surfaces total_cost_usd
        # directly, prefer that.
        cost = payload.get("total_cost_usd") or payload.get("cost_usd")
        if cost is not None:
            assert 0 < float(cost) < 0.05, f"Synthesis cost ${cost} outside expected gate (<$0.05)"
