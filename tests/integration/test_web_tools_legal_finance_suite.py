"""Live-API integration suite — kaos-web + kaos-source against real legal/finance research.

This is the answer to the November-December debug-cycle question:
"why aren't these things being tested at a lower level — kaos-core,
kaos-agents, something?!" Mocked + small-scope tests miss whole
classes of integration failure (VFS namespace mismatch, artifact-
registry gaps, agent tool-routing drift, search-backend silent
zero-result, web-search returning 0 results in one session but 10
in another). This file exercises 20 open-ended legal + finance
research scenarios a lawyer or analyst would hit on day one,
through the full ``ChatAgent + ReAct + tool-bridge + real web/source
tools + real LLM`` pipeline.

Architecture (per the 2026-05-23 design doc):

- ``KaosRuntime.test_mode()`` — in-memory + GLOBAL-isolated VFS so
  no test can false-green off another run's session memory. This
  is the discipline documented in kaos-agents CLAUDE.md
  "Isolation patterns for live tests" — the Excel sub-agent went
  from 25-40% flake rate to 8/8 stable after switching to
  ``test_mode()``.
- Full tool surface: ``register_web_all_tools`` (kaos-web: search,
  fetch-page, get-markdown, get-links, get-tables, fetch-feed, ...)
  + ``register_federal_register_tools`` + ``register_ecfr_tools``
  + ``register_edgar_tools`` + ``register_govinfo_tools`` from
  kaos-source. Source connectors are wrapped in
  ``contextlib.suppress(ImportError)`` mirroring the SPA's pattern
  in ``kaos-ui/examples/single-user-chat/backend/app/main.py``.
- ``ChatAgent`` (not raw Runner) so the test sees the full
  production agent loop: M1 fitness ranker, M2 consistency critic,
  circuit breaker, ReAct iterations, real tool dispatch.
- Default model ``openai:gpt-5.4-mini`` — cheap, fast, good
  tool-calling. Skip-if missing ``OPENAI_API_KEY``.
- Ground-truth assertions per scenario: regex + substring on
  ``response.text`` that name the missing signal in the failure
  message. If a test fails, the test is the messenger — the bug
  is in the tool stack, do not loosen the assertion.
- Cost guard per scenario: ``response.cost_usd < 0.50`` catches
  runaway ReAct loops.
- Tool-call sanity: at least one tool call is non-error AND the
  expected tool family was invoked.
- Mark every test ``@pytest.mark.live`` + ``@pytest.mark.network``.

Run:

    cd kaos-agents
    uv run pytest tests/integration/test_web_tools_legal_finance_suite.py \\
        -v -m live --no-cov --timeout=120

Cost ceiling: 20 tests * ~$0.05 budget = ~$1 total. A single
test exceeding $0.50 is a regression in the cost guard, not a
reason to raise the cap.
"""

from __future__ import annotations

import contextlib
import os
import re

import pytest
from kaos_core.registry.container import KaosRuntime
from kaos_source import (
    register_ecfr_tools,
    register_edgar_tools,
    register_federal_register_tools,
    register_govinfo_tools,
)
from kaos_web import register_web_all_tools

from kaos_agents.patterns.chat import ChatAgent
from kaos_agents.settings import KaosAgentSettings
from kaos_agents.types.response import AgentResponse
from tests.integration._judge import assert_judge_passes, judge_response
from tests.integration.conftest import record_tool_telemetry

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODEL = os.environ.get("KAOS_TEST_MODEL", "openai:gpt-5.4-mini")
"""Default model. Cheap, fast, current-gen, strong tool-calling.

Don't pick a stronger model "to give the agent a chance" — the
suite is verifying the whole production stack, and the production
stack uses cheap models for the chat lane. Upgrading here would
hide regressions that only fire on weaker reasoners (see the
KFM-B05 M1-fitness-ranker history in kaos-agents/CLAUDE.md).

Override with ``KAOS_TEST_MODEL=anthropic:claude-haiku-4-5`` when
OpenAI is rate-limited or unavailable so the suite still exercises
the tool surface (the failure mode we care about is web/source +
agent loop + dispatch, not which provider is up).
"""


def _requires_provider(model: str) -> pytest.MarkDecorator:
    """Return the right skipif for the provider implied by ``model``."""
    prov = model.split(":", 1)[0] if ":" in model else "openai"
    env_var = {
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "google": "GOOGLE_API_KEY",
    }.get(prov, "OPENAI_API_KEY")
    return pytest.mark.skipif(env_var not in os.environ, reason=f"{env_var} missing")


PER_TEST_COST_CAP_USD = 0.50
"""Catches runaway ReAct loops. Raise only after investigation."""

requires_openai = _requires_provider(MODEL)


# ---------------------------------------------------------------------------
# Runtime fixture — in-memory, GLOBAL-isolated, full web + source surface
# ---------------------------------------------------------------------------


@pytest.fixture
def runtime() -> KaosRuntime:
    """Per-test fresh runtime with the full web + source tool catalog.

    ``test_mode()`` installs a fresh in-memory ``VirtualFileSystem``
    with ``IsolationMode.GLOBAL`` so session memory cannot leak
    across pytest invocations. Without this, a test that asks
    "what's the FOMC's last decision?" would record the answer in
    SessionMemory on the first run and on the second run answer
    from memory WITHOUT calling any tool — the tool-call assertion
    would silently false-green.

    The optional source registrations are wrapped in
    ``contextlib.suppress(ImportError)`` to mirror the SPA backend's
    behavior in ``kaos-ui/examples/single-user-chat/backend/app/main.py``
    — if a connector is uninstalled, the rest of the suite still
    runs. Web tools are mandatory (any failure to import is a real
    environment problem worth surfacing loudly).
    """
    rt = KaosRuntime.test_mode()
    # kaos-web: search, fetch-page, get-markdown, get-text, get-links,
    # get-tables, get-images, get-metadata, search-page, fetch-feed,
    # plus browser/crawl/domain tool groups.
    register_web_all_tools(rt)
    # kaos-source connectors — all four are dep-light enough to be
    # mandatory in production setups, but we still suppress ImportError
    # to match the SPA pattern.
    with contextlib.suppress(ImportError):
        register_federal_register_tools(rt)
    with contextlib.suppress(ImportError):
        register_ecfr_tools(rt)
    with contextlib.suppress(ImportError):
        register_edgar_tools(rt)
    with contextlib.suppress(ImportError):
        register_govinfo_tools(rt)
    return rt


def _settings() -> KaosAgentSettings:
    """Settings tuned for live integration — generous tool budget,
    cost-bounded planning, fast enough timeout for real APIs.
    """
    return KaosAgentSettings(
        default_llm_model=MODEL,
        planning_llm_model=MODEL,
        max_tools=80,
        max_react_iterations=15,
        tool_timeout_seconds=60.0,
        plan_max_steps=15,
        plan_max_replans=2,
        plan_max_cost_usd=PER_TEST_COST_CAP_USD,
        plan_max_wall_clock_seconds=100.0,
    )


def _make_agent(runtime: KaosRuntime) -> ChatAgent:
    """Construct a ChatAgent over the full runtime catalog.

    No ``tool_filter`` — we want the agent to pick from the entire
    registered surface, which is exactly what the SPA does. The
    M1 fitness ranker (auto-enabled when catalog > 10 tools) will
    narrow the worker's catalog per query.
    """
    return ChatAgent(
        runtime.vfs,
        runtime=runtime,
        model=MODEL,
        settings=_settings(),
    )


# ---------------------------------------------------------------------------
# Assertion helpers
# ---------------------------------------------------------------------------


def _tool_names(response: AgentResponse) -> list[str]:
    """The tool names invoked during the turn."""
    return [tc.tool_name for tc in response.tool_calls]


def _assert_cost_and_tools(response: AgentResponse, *, families: tuple[str, ...]) -> None:
    """Cost guard + tool-call sanity shared across every scenario.

    1. ``cost_usd < PER_TEST_COST_CAP_USD`` — runaway-loop detector.
    2. At least one tool call must have been recorded and at least
       one must NOT be an error (a 100%-error tool run is the
       silent zero-result signature from December).
    3. At least one tool from the expected family was used. The
       family is a substring match (e.g. ``"search"`` matches
       ``kaos-web-search`` and ``kaos-source-fr-search``).
    """
    assert response.cost_usd < PER_TEST_COST_CAP_USD, (
        f"cost ${response.cost_usd:.4f} exceeded per-test cap "
        f"${PER_TEST_COST_CAP_USD} — investigate cost-guard regression "
        f"before raising the cap. tools={_tool_names(response)}"
    )
    names = _tool_names(response)
    assert names, (
        f"agent recorded zero tool calls — the question requires live data "
        f"and a memory-only answer is a regression. response.text head: "
        f"{response.text[:200]!r}"
    )
    assert any(not tc.is_error for tc in response.tool_calls), (
        f"every tool call errored — silent zero-result regression in tool stack? tools={names}"
    )
    assert any(any(fam in n for fam in families) for n in names), (
        f"agent did not invoke any tool from the expected families {families}. "
        f"Invoked: {names}. response.text head: {response.text[:200]!r}"
    )


# ---------------------------------------------------------------------------
# Opus-as-judge wiring (P1.1)
# ---------------------------------------------------------------------------


def _render_tool_trace(response: AgentResponse, *, max_chars: int = 300) -> str:
    """Flatten ``response.tool_calls`` into a judge-readable string.

    One line per tool call: ``[N] <name> ok|ERR -> <result preview>``.
    Truncated so a long get-markdown dump doesn't blow the judge's
    context — the judge is grounding "did this tool actually succeed
    and return content" rather than re-parsing the page.
    """
    lines: list[str] = []
    for i, tc in enumerate(response.tool_calls):
        flag = "ERR" if tc.is_error else "ok"
        preview = (tc.result_summary or "").strip().replace("\n", " ")
        if len(preview) > max_chars:
            preview = preview[:max_chars] + "..."
        lines.append(f"[{i + 1}] {tc.tool_name} {flag} -> {preview}")
    return "\n".join(lines) if lines else "(no tool calls)"


_WEB_GROUNDED_RUBRIC_TEMPLATE = """\
Evaluate whether the agent's response satisfies the user's research
goal AND grounds its claims in actual successful tool output.

This is the failure mode the layer exists to catch: the 2026-05-23
verifier sub-agent run found responses that regex-passed but had
fabricated content after tool errors (FR climate after 200-OK
anti-bot pages; 10b-5 citation hygiene). The judge MUST treat
"agent confidently asserts X" + "tool trace shows zero successful
retrievals of X" as fabrication, not grounding.

Labels (emit one verbatim):

- ``grounded``: the response satisfies the goal — it contains the
  expected entities (numbers, names, citations, URLs the user asked
  for) AND the tool trace shows at least one successful retrieval
  (no error flag) of a relevant source. URLs in the response should
  match domains the tool trace actually fetched. This is the
  passing label.
- ``partial``: the response addresses the topic but is missing one
  or more requested entities (e.g. user asked for 3 numbers, got 2;
  user asked for verbatim quote, got paraphrase only). Confident-
  partial is NOT grounded — it's the failure mode the judge layer
  exists to catch.
- ``fabricated``: the response confidently asserts specific values,
  citations, or URLs that the tool trace does NOT show coming back
  from a successful fetch. This includes: citing a URL the tool
  never fetched; citing a number after every tool call errored;
  paraphrasing as if quoting verbatim text the agent never retrieved.
- ``refused``: response declines to answer or acknowledges
  insufficient information. Acceptable ONLY if the tool trace shows
  the agent made a real effort (tried multiple tools) AND the user
  prompt did not contain enough to answer offline.

For this scenario:

  Scenario: {scenario}
  Acceptance criteria: {acceptance}

Strict: only ``grounded`` is the passing label. Confident-partial
and confident-wrong both fail the gate — that's the failure mode
this judge layer exists to catch.
"""


def _web_rubric(*, scenario: str, acceptance: str) -> str:
    """Build a web-tools grounding rubric from the template."""
    return _WEB_GROUNDED_RUBRIC_TEMPLATE.format(
        scenario=scenario,
        acceptance=acceptance,
    )


async def _judge_and_record(
    *,
    response: AgentResponse,
    user_prompt: str,
    rubric: str,
    test_id: str,
    passing_labels: tuple[str, ...],
    judge_state: dict,
    request: pytest.FixtureRequest,
) -> None:
    """Run the judge, record tool telemetry, hard-gate on verdict."""
    outcome = await judge_response(
        rubric=rubric,
        user_prompt=user_prompt,
        response_text=response.text,
        tool_trace=_render_tool_trace(response),
    )
    record_tool_telemetry(
        request.node.nodeid,
        tool_names=_tool_names(response),
        n_tool_errors=sum(1 for tc in response.tool_calls if tc.is_error),
    )
    assert_judge_passes(
        outcome,
        passing_labels=passing_labels,
        test_id=test_id,
        judge_state=judge_state,
    )


# ===========================================================================
# Class 1 — Legal: primary-source research (10)
# ===========================================================================


@pytest.mark.live
@pytest.mark.network
@requires_openai
class TestLegalResearch:
    """Scenarios 1-10: primary-source legal research workflows."""

    async def test_scenario_01_occ_bank_capital_rule(
        self,
        runtime: KaosRuntime,
        judge_state: dict,
        request: pytest.FixtureRequest,
    ) -> None:
        """Scenario 1 — OCC bank-capital rule (FR primary-source research).

        Ground truth: response must contain an FR document-number
        pattern (``\\d{4}-\\d{5}``), a ``federalregister.gov`` URL,
        and at least one successful tool call from the search family.

        What this guards: regression in kaos-source FR search OR
        kaos-web search + fetch chain. If both fail the agent has
        nothing to ground on.
        """
        agent = _make_agent(runtime)
        prompt = (
            "Find the most recent Office of the Comptroller of the Currency "
            "(OCC) rule on bank capital requirements. Cite the Federal "
            "Register document number and the effective date. Include the "
            "Federal Register URL."
        )
        response = await agent.turn(prompt, session_id="scenario-01-occ-bank-capital")
        assert re.search(r"\d{4}-\d{5}", response.text), (
            f"no FR document-number pattern in response. head: {response.text[:200]!r}"
        )
        assert "federalregister.gov" in response.text.lower(), (
            f"no federalregister.gov URL. head: {response.text[:200]!r}"
        )
        _assert_cost_and_tools(response, families=("search", "fetch", "fr-"))
        await _judge_and_record(
            response=response,
            user_prompt=prompt,
            rubric=_web_rubric(
                scenario="W01 OCC bank-capital rule on Federal Register",
                acceptance=(
                    "Response cites a specific Federal Register document "
                    "number (NNNN-NNNNN) AND a federalregister.gov URL the "
                    "tool trace actually fetched. Effective date is a real "
                    "date from the rule, not a guess."
                ),
            ),
            test_id="W01",
            passing_labels=("grounded",),
            judge_state=judge_state,
            request=request,
        )

    async def test_scenario_02_bostock_holding(
        self,
        runtime: KaosRuntime,
        judge_state: dict,
        request: pytest.FixtureRequest,
    ) -> None:
        """Scenario 2 — Bostock v. Clayton County holding.

        Ground truth: the literal holding phrasing — "discriminat*"
        + "sex" + ("homosexual" OR "transgender"). This is the
        Court's actual one-sentence holding, not paraphrase.

        What this guards: web-search + fetch-page chain on case-law
        material. If the agent paraphrases without retrieving, the
        literal phrasing won't appear.
        """
        agent = _make_agent(runtime)
        prompt = (
            "Look up the holding of Bostock v. Clayton County, 590 U.S. 644 "
            "(2020). Quote the one-sentence holding verbatim."
        )
        response = await agent.turn(prompt, session_id="scenario-02-bostock")
        text = response.text.lower()
        # Bostock's actual one-sentence holding (Gorsuch, J.) opens:
        # "An employer who fires an individual merely for being gay or
        # transgender violates Title VII." So we accept any holding-equivalent
        # verb (fires / discharges / violates / discriminat*) plus the
        # protected-class hook (gay / homosexual / transgender) plus the statute.
        assert any(v in text for v in ("fires", "discharg", "violates", "discriminat")), (
            f"no holding-verb (fires/discharges/violates/discriminat*) — "
            f"paraphrase not literal holding. head: {response.text[:200]!r}"
        )
        assert any(p in text for p in ("gay", "homosexual", "transgender")), (
            f"no protected-class term (gay/homosexual/transgender) — "
            f"verbatim holding is missing. head: {response.text[:200]!r}"
        )
        assert "title vii" in text, f"no 'Title VII' citation. head: {response.text[:200]!r}"
        _assert_cost_and_tools(response, families=("search", "fetch", "get-"))
        await _judge_and_record(
            response=response,
            user_prompt=prompt,
            rubric=_web_rubric(
                scenario="W02 Bostock v. Clayton County holding verbatim",
                acceptance=(
                    "Response quotes the Court's actual one-sentence "
                    "holding (containing a fires/discharges/violates verb, "
                    "a gay/homosexual/transgender protected-class term, "
                    "and 'Title VII'). The verbatim quote must come from "
                    "a source the tool trace actually fetched — paraphrase "
                    "or 'I remember it says...' without retrieval is "
                    "fabrication."
                ),
            ),
            test_id="W02",
            passing_labels=("grounded",),
            judge_state=judge_state,
            request=request,
        )

    async def test_scenario_03_apple_10k_accession(
        self,
        runtime: KaosRuntime,
        judge_state: dict,
        request: pytest.FixtureRequest,
    ) -> None:
        """Scenario 3 — Apple 10-K accession number from EDGAR.

        Ground truth: Apple's CIK is 0000320193; accession numbers
        for Apple filings match ``0000320193-\\d{2}-\\d{6}``.

        What this guards: kaos-source EDGAR connectors OR the
        web-fetch-page chain hitting sec.gov.
        """
        agent = _make_agent(runtime)
        prompt = (
            "Find Apple Inc.'s most recent 10-K filing on EDGAR (CIK "
            "0000320193). Return the accession number and the filing date."
        )
        response = await agent.turn(prompt, session_id="scenario-03-apple-10k")
        assert re.search(r"0000320193-\d{2}-\d{6}", response.text), (
            f"no Apple 10-K accession pattern (0000320193-NN-NNNNNN). head: {response.text[:200]!r}"
        )
        _assert_cost_and_tools(response, families=("edgar", "search", "fetch"))
        await _judge_and_record(
            response=response,
            user_prompt=prompt,
            rubric=_web_rubric(
                scenario="W03 Apple 10-K accession number from EDGAR",
                acceptance=(
                    "Response contains an EDGAR accession pattern "
                    "(0000320193-NN-NNNNNN) AND a filing date, both "
                    "from a tool-trace fetch of EDGAR or sec.gov. "
                    "Numbers fabricated without a successful retrieval "
                    "fail the gate."
                ),
            ),
            test_id="W03",
            passing_labels=("grounded",),
            judge_state=judge_state,
            request=request,
        )

    async def test_scenario_04_rule_10b5_verbatim(
        self,
        runtime: KaosRuntime,
        judge_state: dict,
        request: pytest.FixtureRequest,
    ) -> None:
        """Scenario 4 — 17 CFR 240.10b-5 verbatim text.

        Ground truth: "manipulative or deceptive device or contrivance"
        is the literal statutory phrase. Plus an ``ecfr.gov`` URL.

        What this guards: kaos-source eCFR connector OR the
        web-fetch-page chain on ecfr.gov.
        """
        agent = _make_agent(runtime)
        prompt = (
            "Quote the text of 17 CFR 240.10b-5 (the SEC's anti-fraud rule) "
            "verbatim. Cite the eCFR URL."
        )
        response = await agent.turn(prompt, session_id="scenario-04-rule-10b5")
        # 17 CFR 240.10b-5 (rule, NOT statute 15 USC 78j(b)) reads:
        #   § 240.10b-5 Employment of manipulative and deceptive devices.
        #   It shall be unlawful for any person, directly or indirectly, by
        #   the use of any means or instrumentality of interstate commerce…
        #   (a) To employ any device, scheme, or artifice to defraud,
        #   (b) To make any untrue statement of a material fact …
        # The phrase "manipulative or deceptive device or contrivance" is
        # from the statute; the rule's signature phrase is "device, scheme,
        # or artifice to defraud".
        text_lower = response.text.lower()
        assert any(
            p in text_lower
            for p in (
                "manipulative and deceptive devices",  # rule title
                "device, scheme, or artifice to defraud",  # rule body (a)
                "untrue statement of a material fact",  # rule body (b)
                "operate as a fraud or deceit",  # rule body (c)
            )
        ), (
            f"missing a signature phrase from 17 CFR 240.10b-5 (title or "
            f"any of subparts a/b/c). head: {response.text[:200]!r}"
        )
        # Accept any authoritative host for the rule OR a bare CFR citation.
        # Limiting to ecfr.gov rejected agent answers that quoted from
        # sec.gov / govinfo.gov / law.cornell.edu (all authoritative)
        # — the verbatim-text + tool-trace checks above/below already
        # ground the answer; this is a citation-presence floor, not a
        # host prescription.
        _RULE_CITATION_HOSTS = (
            "ecfr.gov",
            "cfr.gov",
            "govinfo.gov",
            "sec.gov",
            "law.cornell.edu",
            "justia.com",
        )
        _RULE_CITATION_TOKENS = ("17 cfr 240.10b-5", "§240.10b-5", "§ 240.10b-5", "rule 10b-5")
        assert any(h in text_lower for h in _RULE_CITATION_HOSTS) or any(
            t in text_lower for t in _RULE_CITATION_TOKENS
        ), f"no authoritative-source URL and no CFR citation token. head: {response.text[:200]!r}"
        _assert_cost_and_tools(response, families=("ecfr", "search", "fetch"))
        await _judge_and_record(
            response=response,
            user_prompt=prompt,
            rubric=_web_rubric(
                scenario="W04 17 CFR 240.10b-5 verbatim from eCFR",
                acceptance=(
                    "Response quotes one of the rule's signature phrases "
                    "('manipulative and deceptive devices' / 'device, "
                    "scheme, or artifice to defraud' / 'untrue statement "
                    "of a material fact' / 'operate as a fraud or "
                    "deceit') from an ecfr.gov page the tool trace "
                    "actually fetched. Confusing the statute (15 USC "
                    "78j(b)) with the rule is partial, not grounded."
                ),
            ),
            test_id="W04",
            passing_labels=("grounded",),
            judge_state=judge_state,
            request=request,
        )

    async def test_scenario_05_loper_bright_followon(
        self,
        runtime: KaosRuntime,
        judge_state: dict,
        request: pytest.FixtureRequest,
    ) -> None:
        """Scenario 5 — Loper Bright follow-on SCOTUS deference shift.

        Ground truth: a 2024+ case-name + the words 'Loper' AND
        'Chevron' AND ('defer' or 'deference').

        What this guards: multi-step web-search → fetch-page chain
        for evolving case-law research.
        """
        agent = _make_agent(runtime)
        prompt = (
            "Find a Supreme Court decision (June 2024 or later) that cites "
            "Loper Bright Enterprises v. Raimondo and explains how the "
            "decision changes the prior Chevron deference framework. Name "
            "the case and summarize the deference shift."
        )
        response = await agent.turn(prompt, session_id="scenario-05-loper-bright")
        text_lower = response.text.lower()
        assert "loper" in text_lower, f"no 'Loper' in response. head: {response.text[:200]!r}"
        assert "chevron" in text_lower, f"no 'Chevron' in response. head: {response.text[:200]!r}"
        assert "defer" in text_lower, (
            f"no 'defer*' (deference) discussion. head: {response.text[:200]!r}"
        )
        _assert_cost_and_tools(response, families=("search", "fetch", "get-"))
        await _judge_and_record(
            response=response,
            user_prompt=prompt,
            rubric=_web_rubric(
                scenario="W05 Loper Bright follow-on Chevron deference shift",
                acceptance=(
                    "Response names a specific post-June-2024 case AND "
                    "summarizes the deference shift away from Chevron, "
                    "with the case/explanation grounded in a successful "
                    "tool fetch (search → fetch → get-text chain showing "
                    "the named case)."
                ),
            ),
            test_id="W05",
            passing_labels=("grounded",),
            judge_state=judge_state,
            request=request,
        )

    async def test_scenario_06_section_10b(
        self,
        runtime: KaosRuntime,
        judge_state: dict,
        request: pytest.FixtureRequest,
    ) -> None:
        """Scenario 6 — Section 10(b) of the Securities Exchange Act.

        Ground truth: the literal phrase "unlawful for any person" AND
        ("means or instrumentality" OR "instrumentality of interstate
        commerce").

        What this guards: kaos-source govinfo OR web-search for
        statutory text (15 U.S.C. § 78j(b)).
        """
        agent = _make_agent(runtime)
        prompt = (
            "Get the text of Section 10(b) of the Securities Exchange Act of "
            "1934, codified at 15 U.S.C. § 78j(b). Quote the operative "
            "language verbatim and cite the source URL."
        )
        response = await agent.turn(prompt, session_id="scenario-06-section-10b")
        text_lower = response.text.lower()
        assert "unlawful for any person" in text_lower, (
            f"missing 'unlawful for any person'. head: {response.text[:200]!r}"
        )
        assert "instrumentality" in text_lower, (
            f"missing 'instrumentality' (interstate-commerce hook). head: {response.text[:200]!r}"
        )
        _assert_cost_and_tools(response, families=("search", "fetch", "govinfo"))
        await _judge_and_record(
            response=response,
            user_prompt=prompt,
            rubric=_web_rubric(
                scenario="W06 15 USC 78j(b) (Section 10(b)) verbatim",
                acceptance=(
                    "Response quotes the operative statutory text "
                    "('unlawful for any person' + 'means or "
                    "instrumentality of interstate commerce') AND cites "
                    "a source URL the tool trace actually fetched."
                ),
            ),
            test_id="W06",
            passing_labels=("grounded",),
            judge_state=judge_state,
            request=request,
        )

    async def test_scenario_07_aba_opinion_512(
        self,
        runtime: KaosRuntime,
        judge_state: dict,
        request: pytest.FixtureRequest,
    ) -> None:
        """Scenario 7 — ABA Formal Opinion 512 (GenAI lawyer obligations).

        Ground truth: response mentions 'Opinion 512' (or just '512' near
        ABA context), 'Generative' or 'GenAI', and at least two of the
        three obligation areas (confidentiality / competence / supervision).

        What this guards: web-search + fetch-page chain against
        americanbar.org.
        """
        agent = _make_agent(runtime)
        prompt = (
            "Find ABA Formal Opinion 512 (the August 2024 generative-AI "
            "opinion). State the title, the publication date, and identify "
            "the three main areas of lawyer obligation the opinion addresses."
        )
        response = await agent.turn(prompt, session_id="scenario-07-aba-512")
        text_lower = response.text.lower()
        assert "512" in response.text, (
            f"no '512' in response — opinion number missing. head: {response.text[:200]!r}"
        )
        assert "generative" in text_lower or "genai" in text_lower, (
            f"neither 'generative' nor 'GenAI' in response — subject matter "
            f"missing. head: {response.text[:200]!r}"
        )
        obligation_hits = sum(
            1 for w in ("confidentiality", "competence", "supervision") if w in text_lower
        )
        assert obligation_hits >= 2, (
            f"fewer than two of the three obligation areas (confidentiality, "
            f"competence, supervision) appear. head: {response.text[:200]!r}"
        )
        _assert_cost_and_tools(response, families=("search", "fetch", "get-"))
        await _judge_and_record(
            response=response,
            user_prompt=prompt,
            rubric=_web_rubric(
                scenario="W07 ABA Formal Opinion 512 (GenAI lawyer obligations)",
                acceptance=(
                    "Response identifies Opinion 512 by number AND the "
                    "GenAI subject AND at least two of (confidentiality, "
                    "competence, supervision) — all from americanbar.org "
                    "or similar source the tool trace fetched. Hallucinated "
                    "obligation areas not in the actual opinion fail."
                ),
            ),
            test_id="W07",
            passing_labels=("grounded",),
            judge_state=judge_state,
            request=request,
        )

    async def test_scenario_08_state_bar_advertising_diff(
        self,
        runtime: KaosRuntime,
        judge_state: dict,
        request: pytest.FixtureRequest,
    ) -> None:
        """Scenario 8 — California Rule 7.1 vs Texas Rule 7.02 advertising diff.

        Ground truth: both rule references must appear ("7.1" + "7.02"
        OR "Rule 7.02"); response must contain at least one of
        ('false', 'misleading', 'communication') showing actual rule
        text was retrieved, not paraphrased.

        What this guards: two parallel web-search chains + synthesis.
        """
        agent = _make_agent(runtime)
        prompt = (
            "Compare California Rule of Professional Conduct 7.1 "
            "(communications about a lawyer's services) and Texas "
            "Disciplinary Rule of Professional Conduct 7.02. What is the "
            "key textual difference between them?"
        )
        response = await agent.turn(prompt, session_id="scenario-08-state-bar-ads")
        text = response.text
        text_lower = text.lower()
        assert "7.1" in text, f"no California 'Rule 7.1' marker. head: {text[:200]!r}"
        assert "7.02" in text, f"no Texas 'Rule 7.02' marker. head: {text[:200]!r}"
        assert any(w in text_lower for w in ("false", "misleading", "communication")), (
            f"none of false/misleading/communication appear — rule text "
            f"was not actually retrieved. head: {text[:200]!r}"
        )
        _assert_cost_and_tools(response, families=("search", "fetch", "get-"))
        await _judge_and_record(
            response=response,
            user_prompt=prompt,
            rubric=_web_rubric(
                scenario="W08 CA Rule 7.1 vs TX Rule 7.02 advertising diff",
                acceptance=(
                    "Response identifies BOTH rule numbers (7.1 and 7.02) "
                    "AND describes a concrete textual difference grounded "
                    "in two successful tool fetches (one per jurisdiction). "
                    "Generic 'both regulate lawyer advertising' without "
                    "specifics is partial."
                ),
            ),
            test_id="W08",
            passing_labels=("grounded",),
            judge_state=judge_state,
            request=request,
        )

    async def test_scenario_09_cfpb_enforcement_2025(
        self,
        runtime: KaosRuntime,
        judge_state: dict,
        request: pytest.FixtureRequest,
    ) -> None:
        """Scenario 9 — Recent CFPB enforcement action in 2025.

        Ground truth: at least one dollar amount with '$' or 'million'
        or 'billion', a 2025 date reference, and a
        'consumerfinance.gov' URL.

        What this guards: web-search + fetch chain on
        consumerfinance.gov press releases.
        """
        agent = _make_agent(runtime)
        prompt = (
            "Find a CFPB enforcement action announced in 2025. Identify the "
            "respondent, summarize the violation, and report the monetary "
            "penalty. Cite the CFPB press release URL."
        )
        response = await agent.turn(prompt, session_id="scenario-09-cfpb-2025")
        text_lower = response.text.lower()
        assert "$" in response.text or "million" in text_lower or "billion" in text_lower, (
            f"no dollar / million / billion in response — penalty figure "
            f"missing. head: {response.text[:200]!r}"
        )
        assert "2025" in response.text, (
            f"no '2025' anywhere — wrong year or memory-only answer. head: {response.text[:200]!r}"
        )
        assert "consumerfinance.gov" in text_lower, (
            f"no consumerfinance.gov URL. head: {response.text[:200]!r}"
        )
        _assert_cost_and_tools(response, families=("search", "fetch", "get-"))
        await _judge_and_record(
            response=response,
            user_prompt=prompt,
            rubric=_web_rubric(
                scenario="W09 CFPB enforcement action in 2025",
                acceptance=(
                    "Response names a specific respondent + violation + "
                    "monetary penalty + a 2025 announcement date, all "
                    "grounded in a successful fetch of a consumerfinance.gov "
                    "press release URL the tool trace shows actually loaded."
                ),
            ),
            test_id="W09",
            passing_labels=("grounded",),
            judge_state=judge_state,
            request=request,
        )

    async def test_scenario_10_ofac_lazarus(
        self,
        runtime: KaosRuntime,
        judge_state: dict,
        request: pytest.FixtureRequest,
    ) -> None:
        """Scenario 10 — OFAC SDN list check for 'Lazarus Group'.

        Ground truth: 'Lazarus' appears in response (case-insensitive)
        and either 'SDN' or 'OFAC' or 'sanction*' appears, AND a
        treasury.gov / ofac URL appears.

        What this guards: web search + treasury.gov fetch for
        sanctions-list lookups (OFAC publishes SDN as a flat file).
        """
        agent = _make_agent(runtime)
        prompt = (
            "Check the OFAC Specially Designated Nationals (SDN) list for "
            "any entity matching the name 'Lazarus Group'. If found, return "
            "the relevant SDN list entry block; if not, explain where it "
            "appears in OFAC's cyber-related designations."
        )
        response = await agent.turn(prompt, session_id="scenario-10-ofac-lazarus")
        text_lower = response.text.lower()
        assert "lazarus" in text_lower, f"no 'Lazarus' in response. head: {response.text[:200]!r}"
        assert any(w in text_lower for w in ("sdn", "ofac", "sanction")), (
            f"none of SDN/OFAC/sanction* in response — sanctions context "
            f"missing. head: {response.text[:200]!r}"
        )
        assert "treasury.gov" in text_lower or "ofac" in text_lower, (
            f"no treasury.gov or ofac URL. head: {response.text[:200]!r}"
        )
        _assert_cost_and_tools(response, families=("search", "fetch", "get-"))
        await _judge_and_record(
            response=response,
            user_prompt=prompt,
            rubric=_web_rubric(
                scenario="W10 OFAC SDN check for Lazarus Group",
                acceptance=(
                    "Response either quotes a real SDN entry for Lazarus "
                    "Group OR correctly explains where it lives in OFAC's "
                    "cyber designations — grounded in a treasury.gov fetch. "
                    "Generic 'OFAC sanctions Lazarus' without citation or "
                    "fetch evidence is partial."
                ),
            ),
            test_id="W10",
            passing_labels=("grounded",),
            judge_state=judge_state,
            request=request,
        )


# ===========================================================================
# Class 2 — Finance: markets + filings (6)
# ===========================================================================


@pytest.mark.live
@pytest.mark.network
@requires_openai
class TestFinanceMarkets:
    """Scenarios 11-16: finance / markets / filings workflows."""

    async def test_scenario_11_friday_market_close(
        self,
        runtime: KaosRuntime,
        judge_state: dict,
        request: pytest.FixtureRequest,
    ) -> None:
        """Scenario 11 — Friday market close (baseline finance-news case).

        Ground truth: at least two of (Dow, S&P, Nasdaq) appear; at
        least one numeric pattern matching a typical index close
        level (4-5 digit number, possibly with comma); a finance-news
        domain URL.

        What this guards: baseline web-search + fetch-page chain.
        This scenario was already verified outside the suite; it
        stays in as the canary — if it fails everything is broken.
        """
        agent = _make_agent(runtime)
        prompt = (
            "How did the Dow Jones Industrial Average, S&P 500, and Nasdaq "
            "Composite close this past Friday? Give the closing levels and "
            "cite the source URL."
        )
        response = await agent.turn(prompt, session_id="scenario-11-friday-close")
        text_lower = response.text.lower()
        index_hits = sum(1 for marker in ("dow", "s&p", "nasdaq") if marker in text_lower)
        assert index_hits >= 2, (
            f"fewer than two index names found — finance-news fetch failed. "
            f"head: {response.text[:200]!r}"
        )
        # A typical close level looks like "5,234.12" or "38,901".
        assert re.search(r"\d{1,2}[,\.]\d{3}", response.text) or re.search(
            r"\b\d{4,5}\b", response.text
        ), f"no numeric close-level pattern. head: {response.text[:200]!r}"
        # Common finance-news domains.
        # Accept any major finance OR general-news domain that covers
        # market closes — limiting to specialty finance domains misses
        # AP / Reuters wire / BBC / Axios who all report Friday closes.
        finance_domains = (
            "reuters.com",
            "bloomberg.com",
            "wsj.com",
            "cnbc.com",
            "marketwatch.com",
            "ft.com",
            "yahoo.com",
            "google.com/finance",
            "nasdaq.com",
            "barrons.com",
            "investors.com",
            "investopedia.com",
            "apnews.com",
            "ap.org",
            "axios.com",
            "bbc.com",
            "abcnews.go.com",
            "foxbusiness.com",
            "businessinsider.com",
            "forbes.com",
            "morningstar.com",
        )
        assert any(d in text_lower for d in finance_domains), (
            f"no finance/news domain URL. head: {response.text[:200]!r}"
        )
        _assert_cost_and_tools(response, families=("search", "fetch", "get-"))
        await _judge_and_record(
            response=response,
            user_prompt=prompt,
            rubric=_web_rubric(
                scenario="W11 Friday market close — Dow / S&P / Nasdaq",
                acceptance=(
                    "Response reports closing levels for at least two of "
                    "(Dow, S&P, Nasdaq) with numeric values consistent "
                    "with reality (4-5 digit index levels), grounded in "
                    "a successful fetch of a finance-news article. "
                    "Stale or made-up numbers fail."
                ),
            ),
            test_id="W11",
            passing_labels=("grounded",),
            judge_state=judge_state,
            request=request,
        )

    async def test_scenario_12_nvidia_latest_quarter(
        self,
        runtime: KaosRuntime,
        judge_state: dict,
        request: pytest.FixtureRequest,
    ) -> None:
        """Scenario 12 — Nvidia latest quarterly results.

        Ground truth: at least two of (revenue / EPS / data center)
        terms appear; at least one dollar / billion figure; an
        Nvidia or nasdaq investor-relations URL.

        What this guards: web-search → fetch press release on
        nvidianews.nvidia.com.
        """
        agent = _make_agent(runtime)
        prompt = (
            "What was Nvidia's most recent quarterly revenue, GAAP EPS, and "
            "Data Center segment revenue? Cite the press-release URL."
        )
        response = await agent.turn(prompt, session_id="scenario-12-nvidia-quarter")
        text_lower = response.text.lower()
        topic_hits = sum(1 for w in ("revenue", "eps", "data center") if w in text_lower)
        assert topic_hits >= 2, (
            f"fewer than two of (revenue, eps, data center) in response. "
            f"head: {response.text[:200]!r}"
        )
        assert "$" in response.text or "billion" in text_lower, (
            f"no dollar or billion in response — figures missing. head: {response.text[:200]!r}"
        )
        assert "nvidia" in text_lower, f"no 'nvidia' in response. head: {response.text[:200]!r}"
        _assert_cost_and_tools(response, families=("search", "fetch", "get-"))
        await _judge_and_record(
            response=response,
            user_prompt=prompt,
            rubric=_web_rubric(
                scenario="W12 Nvidia latest-quarter revenue + EPS + Data Center",
                acceptance=(
                    "Response provides three specific numeric figures "
                    "(revenue, GAAP EPS, Data Center segment) AND cites "
                    "a press-release URL grounded in a successful fetch. "
                    "Two-of-three is partial."
                ),
            ),
            test_id="W12",
            passing_labels=("grounded",),
            judge_state=judge_state,
            request=request,
        )

    async def test_scenario_13_fomc_last_decision(
        self,
        runtime: KaosRuntime,
        judge_state: dict,
        request: pytest.FixtureRequest,
    ) -> None:
        """Scenario 13 — Most recent FOMC decision.

        Ground truth: contains 'FOMC' OR 'Federal Open Market
        Committee'; at least one percentage figure (rate target
        like "5.25" or "4.50%"); 'federalreserve.gov' URL.

        What this guards: web-search → fetch federalreserve.gov
        press-release chain.
        """
        agent = _make_agent(runtime)
        prompt = (
            "What did the Federal Open Market Committee (FOMC) announce at "
            "its most recent meeting? Give the rate decision, the vote count "
            "if available, and any change to forward-guidance language. "
            "Cite the FOMC statement URL."
        )
        response = await agent.turn(prompt, session_id="scenario-13-fomc")
        text_lower = response.text.lower()
        assert "fomc" in text_lower or "federal open market committee" in text_lower, (
            f"no FOMC reference. head: {response.text[:200]!r}"
        )
        # The Fed quotes rates as "4-1/4 to 4-1/2 percent" (vulgar fractions)
        # in the official statement, OR as "4.25%-4.50%" in news coverage,
        # OR as "4.25 to 4.50 percent" in some recaps.
        # Accept four flavors of Fed rate notation: bare percent,
        # decimal range, vulgar-fraction range (the Fed's official
        # form), and "X to Y percent" prose. The en-dash hex U+2013
        # is the dash in the Fed's published press releases.
        _DASH = "[-–]"  # noqa: RUF001
        rate_pattern = (
            r"\d+(\.\d+)?\s*%"
            r"|\d+\.\d+\s*" + _DASH + r"\s*\d+\.\d+"
            r"|\d+-\d+/\d+\s*(to|" + _DASH + r")\s*\d+-\d+/\d+"
            r"|\d+(\.\d+)?\s*(to|" + _DASH + r")\s*\d+(\.\d+)?\s*percent"
        )
        assert re.search(rate_pattern, response.text, re.IGNORECASE), (
            f"no rate / percentage pattern. head: {response.text[:200]!r}"
        )
        assert "federalreserve.gov" in text_lower, (
            f"no federalreserve.gov URL. head: {response.text[:200]!r}"
        )
        _assert_cost_and_tools(response, families=("search", "fetch", "get-"))
        await _judge_and_record(
            response=response,
            user_prompt=prompt,
            rubric=_web_rubric(
                scenario="W13 FOMC most recent decision",
                acceptance=(
                    "Response reports the rate decision with a specific "
                    "rate range matching the most recent FOMC statement, "
                    "AND cites federalreserve.gov URL grounded in a "
                    "successful fetch. Stale rates from prior meetings "
                    "or rates not in the trace are fabrication."
                ),
            ),
            test_id="W13",
            passing_labels=("grounded",),
            judge_state=judge_state,
            request=request,
        )

    async def test_scenario_14_treasury_10y_2y_spread(
        self,
        runtime: KaosRuntime,
        judge_state: dict,
        request: pytest.FixtureRequest,
    ) -> None:
        """Scenario 14 — 10Y/2Y Treasury yield spread.

        Ground truth: 'treasury' or 'yield' in response; '10' and '2'
        as year markers; at least two percent figures (`X.XX%`).

        What this guards: web-search + fetch + light synthesis with
        explicit numeric computation.
        """
        agent = _make_agent(runtime)
        prompt = (
            "What is the current US 10-year Treasury yield and 2-year "
            "Treasury yield? Compute the 10Y minus 2Y spread. Cite the "
            "source URL."
        )
        response = await agent.turn(prompt, session_id="scenario-14-treasury-spread")
        text_lower = response.text.lower()
        assert "treasury" in text_lower or "yield" in text_lower, (
            f"no 'treasury' or 'yield' in response. head: {response.text[:200]!r}"
        )
        # Tenor markers must be explicit ("10-year" / "10Y" / "10 yr"),
        # not just the bare characters "10" and "2" — every numeric
        # answer contains those, so the prior check trivially passed.
        _has_10y = bool(re.search(r"\b10[-\s]?[Yy](?:ear|r)?\b", response.text))
        _has_2y = bool(re.search(r"\b2[-\s]?[Yy](?:ear|r)?\b", response.text))
        assert _has_10y and _has_2y, (
            f"missing explicit 10Y/2Y tenor markers (need '10-year/10Y/10yr' "
            f"AND '2-year/2Y/2yr'). head: {response.text[:200]!r}"
        )
        percent_hits = re.findall(r"\d+(?:\.\d+)?\s*%", response.text)
        assert len(percent_hits) >= 2, (
            f"fewer than two percentage figures — both yields not shown. "
            f"head: {response.text[:200]!r}"
        )
        _assert_cost_and_tools(response, families=("search", "fetch", "get-"))
        await _judge_and_record(
            response=response,
            user_prompt=prompt,
            rubric=_web_rubric(
                scenario="W14 10Y / 2Y Treasury yield spread",
                acceptance=(
                    "Response provides both 10Y and 2Y yields with "
                    "specific values AND a computed spread that matches "
                    "the math (10Y - 2Y). Both yields grounded in a "
                    "successful fetch. Made-up yields or arithmetic "
                    "errors fail."
                ),
            ),
            test_id="W14",
            passing_labels=("grounded",),
            judge_state=judge_state,
            request=request,
        )

    async def test_scenario_15_apple_form_4(
        self,
        runtime: KaosRuntime,
        judge_state: dict,
        request: pytest.FixtureRequest,
    ) -> None:
        """Scenario 15 — Recent Apple Form 4 insider filing.

        Ground truth: 'Form 4' appears; at least one transaction code
        letter from the SEC Section 16 set ({P,S,M,A,G,F,D,I,X,W}) is
        mentioned in context; a share-count number; an SEC/EDGAR URL.

        What this guards: kaos-source EDGAR OR web-fetch chain against
        EDGAR full-text search.
        """
        agent = _make_agent(runtime)
        prompt = (
            "Find a recent Form 4 (insider trading filing) on EDGAR by any "
            "executive officer of Apple Inc. (CIK 0000320193). Report the "
            "filer's name, the transaction date, the transaction code, and "
            "the number of shares involved. Cite the EDGAR URL."
        )
        response = await agent.turn(prompt, session_id="scenario-15-apple-form-4")
        text = response.text
        text_lower = text.lower()
        assert "form 4" in text_lower, f"no 'Form 4' phrase in response. head: {text[:200]!r}"
        # SEC §16 transaction codes — look for a standalone capital letter
        # in the code set. The presence of these single-letter codes is the
        # signature that the agent actually read the filing.
        assert re.search(r"\b[PSMAGFDIWX]\b", text) or "transaction code" in text_lower, (
            f"no transaction-code letter (P/S/M/A/G/F/...) and no "
            f"'transaction code' phrase. head: {text[:200]!r}"
        )
        # Share-count grounding: the agent must mention BOTH a 3+ digit
        # number AND the word "share" / "shares" / "shrs" somewhere in
        # the answer. The prior regex required the literal "NNN shares"
        # adjacency which rejected valid answers formatted as
        # "87,135 (transaction code S)" or "shares of common stock —
        # quantity 87,135". The judge rubric still grades the four-field
        # ground truth (filer + date + code + count); this is the
        # programmatic floor.
        _has_share_word = bool(re.search(r"\bshares?\b|\bshrs?\b", text_lower))
        _has_share_count = bool(re.search(r"\d[\d,]{2,}", text))
        assert _has_share_word and _has_share_count, (
            f"missing share-count grounding (need a 3+ digit number AND a "
            f"share/shrs word). head: {text[:200]!r}"
        )
        assert "sec.gov" in text_lower or "edgar" in text_lower, (
            f"no sec.gov / EDGAR URL. head: {text[:200]!r}"
        )
        _assert_cost_and_tools(response, families=("edgar", "search", "fetch"))
        await _judge_and_record(
            response=response,
            user_prompt=prompt,
            rubric=_web_rubric(
                scenario="W15 Apple Form 4 insider filing details",
                acceptance=(
                    "Response identifies a real Apple-officer Form 4 "
                    "filing with filer name + transaction date + "
                    "transaction code + share count, all from a "
                    "successful EDGAR/sec.gov fetch. Missing any of "
                    "those four fields is partial."
                ),
            ),
            test_id="W15",
            passing_labels=("grounded",),
            judge_state=judge_state,
            request=request,
        )

    async def test_scenario_16_ma_news_this_week(
        self,
        runtime: KaosRuntime,
        judge_state: dict,
        request: pytest.FixtureRequest,
    ) -> None:
        """Scenario 16 — Largest M&A deal this week.

        Ground truth: response contains '$' or 'billion'; the words
        'acqui*' or 'merger' or 'deal'; an article URL from a major
        finance-news domain.

        What this guards: open-ended web-search + fetch on
        breaking-news.
        """
        agent = _make_agent(runtime)
        prompt = (
            "Find the largest publicly-announced M&A deal this week (any "
            "sector, deal value over $1 billion). Identify the acquirer, "
            "the target, the deal value, and the announcement date. Cite "
            "the source URL."
        )
        response = await agent.turn(prompt, session_id="scenario-16-ma-news")
        text_lower = response.text.lower()
        assert "$" in response.text or "billion" in text_lower, (
            f"no $ or billion in response. head: {response.text[:200]!r}"
        )
        assert any(w in text_lower for w in ("acqui", "merger", "deal")), (
            f"none of acqui*/merger/deal in response. head: {response.text[:200]!r}"
        )
        ma_domains = (
            "reuters.com",
            "bloomberg.com",
            "wsj.com",
            "ft.com",
            "cnbc.com",
            "businesswire.com",
            "prnewswire.com",
            "marketwatch.com",
            "axios.com",
        )
        assert any(d in text_lower for d in ma_domains), (
            f"no major finance-news domain URL. head: {response.text[:200]!r}"
        )
        _assert_cost_and_tools(response, families=("search", "fetch", "get-"))
        await _judge_and_record(
            response=response,
            user_prompt=prompt,
            rubric=_web_rubric(
                scenario="W16 Largest M&A deal this week",
                acceptance=(
                    "Response names a real announced M&A deal with "
                    "acquirer + target + deal value (≥$1B) + "
                    "announcement date this week, all grounded in a "
                    "successful fetch of a finance-news article."
                ),
            ),
            test_id="W16",
            passing_labels=("grounded",),
            judge_state=judge_state,
            request=request,
        )


# ===========================================================================
# Class 3 — Web mechanics (4): exercise specific tool corners
# ===========================================================================


@pytest.mark.live
@pytest.mark.network
@requires_openai
class TestWebMechanics:
    """Scenarios 17-20: page-mechanics tests that exercise specific
    web tool surfaces (markdown, link-extract, table-extract, RSS).
    """

    async def test_scenario_17_wikipedia_sox_404(
        self,
        runtime: KaosRuntime,
        judge_state: dict,
        request: pytest.FixtureRequest,
    ) -> None:
        """Scenario 17 — Wikipedia Sarbanes-Oxley Section 404.

        Ground truth: 'Section 404' appears; 'internal control' AND
        'management' AND 'annual' all appear (the canonical SOX-404
        Wikipedia phrasing is "annual Exchange Act report", not the
        bare phrase "annual report" — the assertion tracks the actual
        page text); an en.wikipedia.org URL pointing to Sarbanes-Oxley.

        What this guards: kaos-web-get-markdown (Wikipedia
        readability → markdown) and section-targeted fetch.
        """
        agent = _make_agent(runtime)
        prompt = (
            "Fetch the Wikipedia page for 'Sarbanes-Oxley Act'. In the "
            "section titled 'Section 404', what does the text say about "
            "internal control reports? Quote the first paragraph of that "
            "section."
        )
        response = await agent.turn(prompt, session_id="scenario-17-sox-404")
        text_lower = response.text.lower()
        assert "section 404" in text_lower, (
            f"no 'Section 404' in response. head: {response.text[:200]!r}"
        )
        # The first paragraph of Wikipedia's SOX §404 section is fluid
        # (Wikipedia edits change which sentence opens it). What's stable
        # across all revisions is: it mentions "internal control" AND
        # discusses the §404 implementation/reporting/audit obligation.
        # So require "internal control" plus any one of the secondary
        # terms that characterize the §404 obligation.
        assert "internal control" in text_lower, (
            f"required SOX-404 phrase 'internal control' missing. head: {response.text[:200]!r}"
        )
        assert any(
            t in text_lower
            for t in ("management", "auditor", "annual", "report", "assess", "implement", "costly")
        ), (
            f"no §404-obligation term (management/auditor/annual/report/"
            f"assess/implement/costly). head: {response.text[:200]!r}"
        )
        # Accept any wikipedia.org host (en, simple, or a mirror) — the
        # tool-trace assertion below independently verifies the fetch
        # actually happened. Conjoining `and "sarbanes"` in the URL
        # check rejected valid answers where the URL was URL-encoded
        # (`Sarbanes%E2%80%93Oxley_Act`) or used the SOX abbreviation
        # in the path.
        assert "wikipedia.org" in text_lower, (
            f"no wikipedia.org URL — Section-404 prompt explicitly asked "
            f"to fetch the Wikipedia page. head: {response.text[:200]!r}"
        )
        _assert_cost_and_tools(response, families=("get-markdown", "fetch", "search", "get-text"))
        await _judge_and_record(
            response=response,
            user_prompt=prompt,
            rubric=_web_rubric(
                scenario="W17 Wikipedia SOX-404 first paragraph",
                acceptance=(
                    "Response quotes content from the actual Section 404 "
                    "section (mentioning internal control + at least one "
                    "of management / auditor / annual / report / assess), "
                    "grounded in a successful Wikipedia fetch. Paraphrase "
                    "without retrieval is partial."
                ),
            ),
            test_id="W17",
            passing_labels=("grounded",),
            judge_state=judge_state,
            request=request,
        )

    async def test_scenario_18_fr_climate_links(
        self,
        runtime: KaosRuntime,
        judge_state: dict,
        request: pytest.FixtureRequest,
    ) -> None:
        """Scenario 18 — Link extraction from Federal Register search.

        Ground truth: response contains at least 3 distinct
        federalregister.gov/documents URLs, and the word 'climate'
        appears.

        What this guards: kaos-web-get-links / kaos-web-fetch-page on
        a search index page, or kaos-source-fr-search returning a
        list of result documents.
        """
        agent = _make_agent(runtime)
        prompt = (
            "Search the Federal Register for rules and notices about "
            "'climate disclosure'. List the first 5 result links — title "
            "and full federalregister.gov URL for each."
        )
        response = await agent.turn(prompt, session_id="scenario-18-fr-climate-links")
        text_lower = response.text.lower()
        assert "climate" in text_lower, f"no 'climate' in response. head: {response.text[:200]!r}"
        urls = re.findall(
            r"https?://(?:www\.)?federalregister\.gov/[^\s)\]'\"<>]+",
            response.text,
        )
        unique_urls = set(urls)
        assert len(unique_urls) >= 3, (
            f"fewer than 3 distinct federalregister.gov URLs in response — "
            f"link extraction failed. found={sorted(unique_urls)[:5]} "
            f"head: {response.text[:200]!r}"
        )
        _assert_cost_and_tools(response, families=("search", "fetch", "get-links", "fr-"))
        await _judge_and_record(
            response=response,
            user_prompt=prompt,
            rubric=_web_rubric(
                scenario="W18 FR climate-disclosure first 5 result links",
                acceptance=(
                    "Response lists at least 3 distinct real "
                    "federalregister.gov/documents URLs about climate "
                    "disclosure, all from a successful tool fetch / "
                    "search. The 2026-05-23 verifier run found this "
                    "test fabricating URLs after a 200-OK anti-bot "
                    "stub response — fabricated URLs fail the gate."
                ),
            ),
            test_id="W18",
            passing_labels=("grounded",),
            judge_state=judge_state,
            request=request,
        )

    async def test_scenario_19_sec_enforcement_table(
        self,
        runtime: KaosRuntime,
        judge_state: dict,
        request: pytest.FixtureRequest,
    ) -> None:
        """Scenario 19 — Table extraction from SEC press release.

        Ground truth: response contains a markdown table marker
        (a row with two or more pipe-delimited cells) AND a sec.gov
        URL AND the word 'enforcement' (case-insensitive).

        What this guards: kaos-web-get-tables / kaos-web-get-markdown
        on a regulator page that publishes structured data.
        """
        agent = _make_agent(runtime)
        prompt = (
            "Find a recent SEC press release or page that lists multiple "
            "enforcement actions in tabular form. Extract the list as a "
            "markdown table (headers + rows). Cite the SEC URL."
        )
        response = await agent.turn(prompt, session_id="scenario-19-sec-enforcement-table")
        text_lower = response.text.lower()
        assert "enforcement" in text_lower, (
            f"no 'enforcement' in response. head: {response.text[:200]!r}"
        )
        assert "sec.gov" in text_lower, f"no sec.gov URL. head: {response.text[:200]!r}"
        # Markdown table row: at least two cells separated by pipes, with
        # leading/trailing whitespace tolerated. We want >=2 such rows so
        # the response actually contains a table, not a one-line mention.
        table_rows = re.findall(r"^\s*\|.+\|.+\|\s*$", response.text, re.MULTILINE)
        assert len(table_rows) >= 2, (
            f"fewer than 2 markdown table rows — no real table extracted. "
            f"found {len(table_rows)} rows. head: {response.text[:200]!r}"
        )
        _assert_cost_and_tools(response, families=("get-tables", "get-markdown", "fetch", "search"))
        await _judge_and_record(
            response=response,
            user_prompt=prompt,
            rubric=_web_rubric(
                scenario="W19 SEC enforcement table extraction",
                acceptance=(
                    "Response contains a real markdown table extracted "
                    "from an SEC page about enforcement actions, "
                    "grounded in a successful fetch of a sec.gov URL. "
                    "Made-up table rows fail."
                ),
            ),
            test_id="W19",
            passing_labels=("grounded",),
            judge_state=judge_state,
            request=request,
        )

    async def test_scenario_20_legal_rss_feed(
        self,
        runtime: KaosRuntime,
        judge_state: dict,
        request: pytest.FixtureRequest,
    ) -> None:
        """Scenario 20 — RSS feed of legal news.

        Ground truth: response contains at least 3 headline-and-URL
        bullets; at least one http(s):// URL; word 'legal' or 'law'
        appears.

        What this guards: kaos-web-fetch-feed for RSS / Atom feeds.
        """
        agent = _make_agent(runtime)
        prompt = (
            "Fetch the most recent items from a major legal-news RSS feed "
            "(for example Reuters Legal, ABA Journal, or Law360 if "
            "available). Return the 5 most recent headlines with their "
            "links and publication dates."
        )
        response = await agent.turn(prompt, session_id="scenario-20-legal-rss")
        text_lower = response.text.lower()
        assert "legal" in text_lower or "law" in text_lower, (
            f"no 'legal' / 'law' in response. head: {response.text[:200]!r}"
        )
        urls = re.findall(r"https?://[^\s)\]'\"<>]+", response.text)
        assert len(urls) >= 3, (
            f"fewer than 3 URLs — feed didn't yield 3 headlines. found "
            f"{len(urls)} urls. head: {response.text[:200]!r}"
        )
        _assert_cost_and_tools(response, families=("fetch-feed", "fetch", "search", "get-"))
        await _judge_and_record(
            response=response,
            user_prompt=prompt,
            rubric=_web_rubric(
                scenario="W20 Legal-news RSS feed headlines",
                acceptance=(
                    "Response lists at least 3 real headlines with their "
                    "URLs and publication dates from an actual legal-news "
                    "feed the tool trace fetched successfully. Headlines "
                    "fabricated without a fetch fail."
                ),
            ),
            test_id="W20",
            passing_labels=("grounded",),
            judge_state=judge_state,
            request=request,
        )
