"""Corpus stress test suite — 15 increasingly-large live tests.

Closes the November-May "no pytest gate on full upload→VFS→agent→answer
chain at scale" gap documented in
``kaos-modules/docs/plans/2026-05-24-corpus-stress-suite.md``.

Tier layout (see design doc for the full matrix):

  TestMixedFileTypes  (S01-S05) — small N, format diversity
  TestNeedleInHaystack (S06-S10) — modest scale, targeting + content sniff
  TestPureScale (S11-S15)        — 200/500/1000-doc + multi-session

Every test:

  1. Builds an in-memory, GLOBAL-isolated runtime via
     ``KaosRuntime.test_mode()`` (per kaos-agents/CLAUDE.md "Isolation
     patterns for live tests"). Without this, session memory leaks
     across pytest invocations and tests false-green.
  2. Registers the full kaos-web + kaos-source + kaos-pdf + kaos-office
     + kaos-tabular + kaos-content tool surface (each wrapped in
     ``contextlib.suppress(ImportError)`` to match the SPA pattern).
  3. Generates a hermetic synthetic corpus via the helpers in
     ``_corpus_fixtures.py`` — no on-disk fixture files.
  4. Writes the corpus to the VFS at the same
     ``sessions/{sid}/files/{filename}`` layout the SPA uses, and
     hydrates the SessionMemory.DOCUMENTS section so the agent's
     ``triage_corpus`` / ``search_memory`` paths fire.
  5. Sends a question that requires the planted needle fact.
  6. Asserts: the planted fact appears, at least one search/retrieval
     tool was invoked, no tool call errored, the cost stays under the
     tier cap.

Model floor (per ``feedback_test_model_floor.md``): default model is
``anthropic:claude-sonnet-4-6``. Override with
``KAOS_TEST_MODEL=openai:gpt-5.4-mini`` (only legal alternative). Haiku
is rejected at env-read time — a "passing test on Haiku" gives
misleading signal at the production bar.

Run::

    KAOS_TEST_MODEL=anthropic:claude-sonnet-4-6 \\
        uv run pytest tests/integration/test_corpus_stress_suite.py \\
        -v -m live --no-cov --timeout=1200
"""

from __future__ import annotations

import contextlib
import os
import uuid
from typing import TYPE_CHECKING

import pytest

from kaos_agents.memory.store import SessionStore
from kaos_agents.patterns.chat import ChatAgent
from kaos_agents.settings import KaosAgentSettings
from tests.integration._corpus_fixtures import (
    SynthDoc,
    hydrate_corpus_into_memory,
    synth_corpus,
    synth_docx,
    synth_empty,
    synth_gibberish,
    synth_html,
    synth_json,
    synth_pdf_image,
    synth_pdf_text,
    synth_text,
    synth_xlsx,
    write_corpus_to_vfs,
    wrong_extension,
)
from tests.integration._judge import assert_judge_passes, judge_response
from tests.integration.conftest import record_tool_telemetry

if TYPE_CHECKING:
    from kaos_core.registry.container import KaosRuntime

    from kaos_agents.types.response import AgentResponse


# ---------------------------------------------------------------------------
# Model resolution (floor: gpt-5.4-mini or claude-sonnet-4-6)
# ---------------------------------------------------------------------------

_LEGAL_TEST_MODELS = frozenset(
    {
        "openai:gpt-5.4-mini",
        "openai:gpt-5.4",
        "openai:gpt-5.5",
        "anthropic:claude-sonnet-4-6",
        "anthropic:claude-opus-4-7",
    }
)
_DEFAULT_MODEL = "anthropic:claude-sonnet-4-6"


def _resolve_test_model() -> str:
    """Read ``KAOS_TEST_MODEL`` with a hard floor.

    Per ``feedback_test_model_floor.md``: Haiku is rejected here so
    nothing further down the stack can silently downgrade to a weaker
    reasoner and let the suite false-green on legal-research bar
    contracts.
    """
    requested = os.environ.get("KAOS_TEST_MODEL", _DEFAULT_MODEL)
    if requested not in _LEGAL_TEST_MODELS:
        # Refuse to run. The test should be loud, not silently wrong.
        raise RuntimeError(
            f"KAOS_TEST_MODEL={requested!r} is not on the legal-floor list. "
            f"Allowed: {sorted(_LEGAL_TEST_MODELS)}. Per "
            "feedback_test_model_floor.md, Haiku is forbidden as a fallback "
            "for the kaos-agents live test bar."
        )
    return requested


MODEL = _resolve_test_model()


def _requires_provider(model: str) -> pytest.MarkDecorator:
    prov = model.split(":", 1)[0] if ":" in model else "anthropic"
    env_var = {
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "google": "GOOGLE_API_KEY",
    }.get(prov, "ANTHROPIC_API_KEY")
    return pytest.mark.skipif(env_var not in os.environ, reason=f"{env_var} missing")


requires_provider = _requires_provider(MODEL)


# ---------------------------------------------------------------------------
# Cost ceilings (per design doc)
# ---------------------------------------------------------------------------

TIER1_CAP_USD = 0.20
TIER2_CAP_USD = 0.40
TIER3_CAP_USD = 1.00


# ---------------------------------------------------------------------------
# OCR optional gate (S03)
# ---------------------------------------------------------------------------


def _has_ocr() -> bool:
    """True iff pytesseract and the tesseract binary are both available."""
    try:
        import shutil

        import pytesseract  # noqa: F401  # ty: ignore[unresolved-import]
    except ImportError:
        return False
    return shutil.which("tesseract") is not None


# ---------------------------------------------------------------------------
# Runtime fixture: in-memory, GLOBAL-isolated, full tool surface
# ---------------------------------------------------------------------------


@pytest.fixture
def runtime() -> KaosRuntime:
    """Fresh per-test runtime with the full document + retrieval surface.

    See kaos-agents/CLAUDE.md "Isolation patterns for live tests":
    ``KaosRuntime.test_mode()`` installs a fresh in-memory
    ``VirtualFileSystem`` with ``IsolationMode.GLOBAL`` so cross-test
    session-memory leakage cannot false-green a contract.

    The optional tool registrations mirror
    ``kaos-ui/kaos_ui/agents.py::build_chat_runtime`` so the runtime
    catalog matches the SPA's production surface. Each is wrapped in
    ``contextlib.suppress(ImportError)`` — a slim install (e.g.
    kaos-tabular missing in a CI tier) still runs the suite; the agent
    just has a smaller toolbox.
    """
    from kaos_core.registry.container import KaosRuntime

    rt = KaosRuntime.test_mode()

    with contextlib.suppress(ImportError):
        from kaos_core.tools import register_core_tools

        register_core_tools(rt)
    with contextlib.suppress(ImportError):
        from kaos_pdf import register_pdf_tools

        register_pdf_tools(rt)
    with contextlib.suppress(ImportError):
        from kaos_office import register_office_tools

        register_office_tools(rt)
    with contextlib.suppress(ImportError):
        from kaos_tabular.tools import (
            register_tabular_tools,
        )

        register_tabular_tools(rt)
    with contextlib.suppress(ImportError):
        from kaos_content.tools import (
            register_content_tools,
        )

        register_content_tools(rt)
    with contextlib.suppress(ImportError):
        from kaos_web import register_web_all_tools

        register_web_all_tools(rt)
    with contextlib.suppress(ImportError):
        from kaos_source import register_source_tools

        register_source_tools(rt)
    with contextlib.suppress(ImportError):
        from kaos_agents.tools.registry import register_agent_tools

        register_agent_tools(rt)

    return rt


def _settings(*, max_cost: float, max_react: int = 12) -> KaosAgentSettings:
    """Per-test agent settings: cost-bounded planning, generous tool budget."""
    return KaosAgentSettings(
        default_llm_model=MODEL,
        planning_llm_model=MODEL,
        max_tools=120,
        max_react_iterations=max_react,
        tool_timeout_seconds=60.0,
        plan_max_steps=20,
        plan_max_replans=2,
        plan_max_cost_usd=max_cost,
        plan_max_wall_clock_seconds=300.0,
    )


def _make_agent(
    runtime: KaosRuntime,
    *,
    session_id: str,
    max_cost: float,
    max_react: int = 12,
) -> ChatAgent:
    """Construct a ChatAgent over the full runtime catalog.

    The context's ``default_vfs_namespace`` is set to
    ``sessions/{session_id}/files/`` so the agent's bridged tools
    (kaos-core-vfs-list, kaos-pdf-*, kaos-office-*, etc.) resolve bare
    filenames at the same on-disk location the corpus was written to.
    This mirrors the SPA's per-session-context factory
    (``kaos-ui/examples/single-user-chat/backend/app/main.py``).
    """
    from kaos_core.base.context import KaosContext

    namespace = f"sessions/{session_id}/files/"
    context = KaosContext(
        session_id=session_id,
        runtime=runtime,
        vfs=runtime.vfs,
        default_vfs_namespace=namespace,
    )
    return ChatAgent(
        runtime.vfs,
        runtime=runtime,
        context=context,
        model=MODEL,
        settings=_settings(max_cost=max_cost, max_react=max_react),
    )


# ---------------------------------------------------------------------------
# Assertion helpers
# ---------------------------------------------------------------------------


def _tool_names(response: AgentResponse) -> list[str]:
    return [tc.tool_name for tc in response.tool_calls]


def _assert_no_tool_errors(response: AgentResponse) -> None:
    """Zero tool calls or zero errors are both fine. Mixed-with-error is
    fine if at least one succeeded. ALL-error is the silent zero-result
    fingerprint we want to fail loudly on.
    """
    if not response.tool_calls:
        return
    if not any(not tc.is_error for tc in response.tool_calls):
        raise AssertionError(
            "every tool call errored — silent zero-result regression. "
            f"tools={_tool_names(response)}"
        )


def _assert_cost_cap(response: AgentResponse, cap: float) -> None:
    if response.cost_usd > cap:
        raise AssertionError(
            f"cost ${response.cost_usd:.4f} exceeded tier cap ${cap:.2f} — "
            f"investigate cost-guard regression before raising the cap. "
            f"tools={_tool_names(response)}"
        )


def _assert_retrieval_tool(response: AgentResponse) -> None:
    """At least one tool from the search / retrieval / document-read
    family must have fired. Empty tool list is allowed only when the
    answer is grounded directly off DOCUMENTS-section content the agent
    was already shown in context (legal for very small corpora).
    """
    names = _tool_names(response)
    if not names:
        # Tiny corpora can be answered without a tool call when the
        # body is already in context. We don't fail on empty tools —
        # the needle-substring check above already proved grounding.
        return
    families = (
        "search",
        "memory-search",
        "memory-query",
        "search-document",
        "search-memory",
        "triage",
        "corpus",
        "filter",
        "findings",
        "retriev",
        "parse",
        "extract",
        "get-text",
        "get-markdown",
        "vfs-read",
        "vfs-list",
        "read",
        "list",
        "summarize",
        "narrow",
        "filter-entity",
        "filter-sentences",
        "chat",
        "plan",
    )
    matched = [n for n in names if any(f in n for f in families)]
    if not matched:
        raise AssertionError(
            f"agent did not invoke any search / retrieval / document-read tool. Invoked: {names}"
        )


def _assert_needle_present(response: AgentResponse, needle_substr: str) -> None:
    if needle_substr.lower() not in response.text.lower():
        raise AssertionError(
            f"planted needle {needle_substr!r} missing from response. "
            f"response head: {response.text[:300]!r} "
            f"tools: {_tool_names(response)}"
        )


async def _hydrate_session(
    runtime: KaosRuntime,
    docs: list[SynthDoc],
    *,
    session_id: str,
) -> None:
    """Async helper: write to VFS + register artifacts + hydrate
    SessionMemory.DOCUMENTS. Persists the memory via the canonical
    ``SessionStore`` so the agent's per-turn ``load_or_create`` sees
    our planted documents.
    """
    await write_corpus_to_vfs(docs, runtime, session_id=session_id)
    store = SessionStore(runtime.vfs)
    memory = await store.load_or_create(session_id)
    hydrate_corpus_into_memory(docs, memory, session_id=session_id)
    await store.save(memory)


def _new_session_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


# ---------------------------------------------------------------------------
# Opus-as-judge wiring (P1.1)
# ---------------------------------------------------------------------------


def _render_tool_trace(response: AgentResponse, *, max_chars: int = 300) -> str:
    """Flatten ``response.tool_calls`` into a string the judge can read.

    One line per tool call: ``[N] <name> ok|ERR -> <result preview>``.
    The result is truncated so a long parse-docx dump doesn't blow the
    judge's context window — the judge is grounding "did this tool
    actually run and return something" rather than re-parsing the
    content.
    """
    lines: list[str] = []
    for i, tc in enumerate(response.tool_calls):
        flag = "ERR" if tc.is_error else "ok"
        preview = (tc.result_summary or "").strip().replace("\n", " ")
        if len(preview) > max_chars:
            preview = preview[:max_chars] + "..."
        lines.append(f"[{i + 1}] {tc.tool_name} {flag} -> {preview}")
    return "\n".join(lines) if lines else "(no tool calls)"


# Per-scenario rubrics. Each enumerates the labels the judge may emit
# verbatim (per JudgeSignature contract). No regex / keyword logic in
# the rubric — the LLM reads the response and judges semantically. The
# tool trace is in context so the judge can ground "did the agent
# actually fetch / parse the cited source" not just "does the text
# look right".

_GROUNDING_RUBRIC_TEMPLATE = """\
Evaluate whether the agent's response correctly identified the planted
fact for this scenario AND grounded its answer in actual tool output.

Labels (emit one verbatim):

- ``grounded``: the response contains the planted fact (or a clear
  near-verbatim paraphrase that preserves the specific value), AND
  the tool trace shows the agent ran at least one retrieval / read /
  parse tool that returned without error. This is the passing label.
- ``partial``: the response mentions the topic but is missing the
  specific planted value, OR cites the value without any successful
  retrieval in the tool trace (claim without evidence).
- ``fabricated``: the response confidently states a specific value
  that is plainly wrong, OR the tool trace shows zero successful
  retrievals yet the response confidently asserts the planted fact
  (citation without grounding).
- ``refused``: the response declines to answer / claims it could
  not find the fact. Acceptable ONLY if the tool trace also shows
  the agent made a reasonable effort.

For this scenario:

  Scenario: {scenario}
  Planted fact: {planted_fact}

Strict: only ``grounded`` is the passing label. Confident-partial and
confident-wrong both fail the gate — that's the failure mode this
judge layer exists to catch.
"""


def _grounding_rubric(*, scenario: str, planted_fact: str) -> str:
    """Build a scenario-specific grounding rubric from the template."""
    return _GROUNDING_RUBRIC_TEMPLATE.format(
        scenario=scenario,
        planted_fact=planted_fact,
    )


_MULTI_NEEDLE_RUBRIC_TEMPLATE = """\
Evaluate whether the agent's response found AND grounded ALL planted
needles for this scenario.

Labels (emit one verbatim):

- ``grounded``: response contains ALL of the planted facts (or clear
  near-verbatim paraphrases preserving each specific value), AND the
  tool trace shows the agent ran successful retrieval / read tools to
  surface them. This is the passing label.
- ``partial``: response contains SOME but not ALL planted facts (the
  confident-partial failure mode from S12 — N-1 of N needles found,
  declared victory without exhaustion proof).
- ``fabricated``: response confidently states values that are plainly
  wrong, OR cites needle facts without any successful retrieval.
- ``refused``: response declines to answer or claims it could not
  find the facts.

For this scenario:

  Scenario: {scenario}
  Planted facts (ALL must be grounded):
{planted_block}

Strict: only ``grounded`` is the passing label. ``partial`` is the
exact failure mode this judge layer was added to catch.
"""


def _multi_needle_rubric(*, scenario: str, planted_facts: list[str]) -> str:
    """Build a multi-needle grounding rubric from the template."""
    planted_block = "\n".join(f"  - {f}" for f in planted_facts)
    return _MULTI_NEEDLE_RUBRIC_TEMPLATE.format(
        scenario=scenario,
        planted_block=planted_block,
    )


_ISOLATION_RUBRIC_TEMPLATE = """\
Evaluate whether the agent's response correctly surfaced ONLY this
session's secret and did NOT leak any other session's secret.

Labels (emit one verbatim):

- ``grounded``: response cites the SESSION's secret verbatim AND does
  NOT mention any of the FORBIDDEN secrets from sibling sessions.
- ``partial``: response mentions the topic but is missing the specific
  session secret value, with no leakage.
- ``fabricated``: response cites the wrong secret OR a value not in
  any session (hallucinated isolation breach).
- ``leaked``: response surfaces one or more FORBIDDEN secrets from
  sibling sessions — the isolation contract has been violated.
- ``refused``: response declines to answer.

For this scenario:

  Scenario: {scenario}
  Session secret (MUST appear): {planted_fact}
  Forbidden secrets (MUST NOT appear):
{forbidden_block}

Strict: only ``grounded`` is the passing label. ``leaked`` is a
critical regression — surface it loudly.
"""


def _isolation_rubric(
    *,
    scenario: str,
    planted_fact: str,
    forbidden_facts: list[str],
) -> str:
    forbidden_block = "\n".join(f"  - {f}" for f in forbidden_facts)
    return _ISOLATION_RUBRIC_TEMPLATE.format(
        scenario=scenario,
        planted_fact=planted_fact,
        forbidden_block=forbidden_block,
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
    """Run the judge, record tool telemetry, hard-gate on verdict.

    Convenience wrapper shared across every test in the suite — runs
    ``judge_response``, stashes tool telemetry for the SUMMARY.jsonl
    writer, and calls ``assert_judge_passes`` so the test fails loudly
    on confident-partial / confident-wrong outcomes.
    """
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
# Tier 1 — Mixed file types (S01-S05)
# ===========================================================================


@pytest.mark.live
@pytest.mark.network
@requires_provider
class TestMixedFileTypes:
    """Small N, format diversity. Targets file loading + content-type detection."""

    async def test_scenario_01_nda_plus_amendment_cross_file(
        self,
        runtime: KaosRuntime,
        judge_state: dict,
        request: pytest.FixtureRequest,
    ) -> None:
        """S01 — NDA (PDF) + amendment (DOCX) cross-file synthesis.

        Ground truth: the amendment's effective date is the planted
        fact. The agent must read BOTH the original NDA and the
        amendment to identify it.
        """
        sid = _new_session_id("s01")
        nda_body = (
            "MUTUAL NON-DISCLOSURE AGREEMENT\n\n"
            "Parties: ACME LLC and Beta Holdings.\n"
            "Effective Date: 2026-01-15.\n"
            "Term: 3 years.\n"
            "Confidential Information includes business plans, pricing models, "
            "customer lists, and product roadmaps."
        )
        amendment_needle = "Amendment effective date: 2026-09-15"
        amendment_body = (
            "FIRST AMENDMENT TO MUTUAL NON-DISCLOSURE AGREEMENT\n\n"
            f"{amendment_needle}.\n"
            "This Amendment extends the Term by 2 years and adds source code "
            "to the definition of Confidential Information."
        )
        docs = [
            SynthDoc(
                filename="nda-acme-beta.pdf",
                bytes=synth_pdf_text(nda_body, title="MNDA"),
                mime="application/pdf",
                is_needle=False,
            ),
            SynthDoc(
                filename="amendment-1.docx",
                bytes=synth_docx(amendment_body, title="Amendment 1"),
                mime=("application/vnd.openxmlformats-officedocument.wordprocessingml.document"),
                is_needle=True,
                needle_fact=amendment_needle,
            ),
        ]
        await _hydrate_session(runtime, docs, session_id=sid)

        agent = _make_agent(runtime, session_id=sid, max_cost=TIER1_CAP_USD)
        prompt = (
            "The session has two attached files: an NDA and its first amendment. "
            "What is the effective date of the AMENDMENT? Cite the source filename."
        )
        response = await agent.turn(prompt, session_id=sid)
        _assert_needle_present(response, "2026-09-15")
        _assert_no_tool_errors(response)
        _assert_retrieval_tool(response)
        _assert_cost_cap(response, TIER1_CAP_USD)
        await _judge_and_record(
            response=response,
            user_prompt=prompt,
            rubric=_grounding_rubric(
                scenario="S01 NDA + amendment cross-file effective date",
                planted_fact=amendment_needle,
            ),
            test_id="S01",
            passing_labels=("grounded",),
            judge_state=judge_state,
            request=request,
        )

    async def test_scenario_02_wrong_extensions_content_sniff(
        self,
        runtime: KaosRuntime,
        judge_state: dict,
        request: pytest.FixtureRequest,
    ) -> None:
        """S02 — extension-spoof: PDF named .txt + DOCX named .pdf + real HTML.

        Ground truth: the planted phrase is in the spoofed PDF. The
        agent's tool stack must content-sniff (not trust the .txt
        extension) and route to a PDF-capable reader.
        """
        sid = _new_session_id("s02")
        spoofed_pdf_needle = "Spoof-test passphrase: KAOS-S02-SNIFF-OK"
        spoofed_pdf_body = (
            "REGULATORY ASSESSMENT\n\n"
            f"{spoofed_pdf_needle}.\n"
            "All compliance checkpoints passed for fiscal year 2026."
        )
        # PDF body named .txt
        pdf_bytes = synth_pdf_text(spoofed_pdf_body, title="Assessment")
        _, pdf_as_txt_name = wrong_extension(pdf_bytes, ".txt")
        # DOCX body named .pdf
        docx_bytes = synth_docx(
            "QUARTERLY REVIEW\n\nMargins held steady at 18.4%.",
            title="QR",
        )
        _, docx_as_pdf_name = wrong_extension(docx_bytes, ".pdf")
        # Real HTML
        html_bytes = synth_html(
            "Investor Letter\n\nThank you for your continued support.",
            title="Letter",
        )
        docs = [
            SynthDoc(
                filename=pdf_as_txt_name,
                bytes=pdf_bytes,
                mime="application/pdf",
                is_needle=True,
                needle_fact=spoofed_pdf_needle,
            ),
            SynthDoc(
                filename=docx_as_pdf_name,
                bytes=docx_bytes,
                mime=("application/vnd.openxmlformats-officedocument.wordprocessingml.document"),
                is_needle=False,
            ),
            SynthDoc(
                filename="letter.html",
                bytes=html_bytes,
                mime="text/html",
                is_needle=False,
            ),
        ]
        await _hydrate_session(runtime, docs, session_id=sid)

        agent = _make_agent(runtime, session_id=sid, max_cost=TIER1_CAP_USD)
        prompt = (
            "One of the attached files contains a 'Spoof-test passphrase'. "
            "Find it and quote it verbatim. Note that filenames may have "
            "misleading extensions — trust the file contents."
        )
        response = await agent.turn(prompt, session_id=sid)
        _assert_needle_present(response, "KAOS-S02-SNIFF-OK")
        _assert_no_tool_errors(response)
        _assert_retrieval_tool(response)
        _assert_cost_cap(response, TIER1_CAP_USD)
        await _judge_and_record(
            response=response,
            user_prompt=prompt,
            rubric=_grounding_rubric(
                scenario="S02 wrong-extension content sniff (PDF named .txt)",
                planted_fact=spoofed_pdf_needle,
            ),
            test_id="S02",
            passing_labels=("grounded",),
            judge_state=judge_state,
            request=request,
        )

    @pytest.mark.skipif(not _has_ocr(), reason="tesseract OCR not available")
    async def test_scenario_03_scanned_pdf_ocr_required(
        self,
        runtime: KaosRuntime,
        judge_state: dict,
        request: pytest.FixtureRequest,
    ) -> None:
        """S03 — scanned PDF (image-only) + native text PDF.

        Ground truth: the planted fact is *only* in the scanned PDF.
        Without OCR the agent has no way to recover it; the test
        skips if tesseract isn't installed.
        """
        sid = _new_session_id("s03")
        scanned_needle = "Project Oryx codename: 7K-FALCON-2026"
        scanned_body = f"INTERNAL MEMO\n{scanned_needle}.\nDistribution: leadership only."
        native_body = (
            "PUBLIC ANNOUNCEMENT\n\n"
            "We are pleased to announce updated company guidelines.\n"
            "Effective date: 2026-06-01."
        )
        docs = [
            SynthDoc(
                filename="memo-scanned.pdf",
                bytes=synth_pdf_image(scanned_body),
                mime="application/pdf",
                is_needle=True,
                needle_fact=scanned_needle,
            ),
            SynthDoc(
                filename="public-announcement.pdf",
                bytes=synth_pdf_text(native_body, title="Announcement"),
                mime="application/pdf",
                is_needle=False,
            ),
        ]
        await _hydrate_session(runtime, docs, session_id=sid)

        agent = _make_agent(runtime, session_id=sid, max_cost=TIER1_CAP_USD)
        prompt = (
            "One of the attached PDFs is a scanned image. It contains a "
            "'Project Oryx codename'. Extract and quote the codename verbatim."
        )
        response = await agent.turn(prompt, session_id=sid)
        _assert_needle_present(response, "7K-FALCON-2026")
        _assert_no_tool_errors(response)
        _assert_retrieval_tool(response)
        _assert_cost_cap(response, TIER1_CAP_USD)
        await _judge_and_record(
            response=response,
            user_prompt=prompt,
            rubric=_grounding_rubric(
                scenario="S03 scanned-PDF OCR Project Oryx codename",
                planted_fact=scanned_needle,
            ),
            test_id="S03",
            passing_labels=("grounded",),
            judge_state=judge_state,
            request=request,
        )

    async def test_scenario_04_xlsx_plus_docx_cross_reference(
        self,
        runtime: KaosRuntime,
        judge_state: dict,
        request: pytest.FixtureRequest,
    ) -> None:
        """S04 — XLSX line items + DOCX narrative referencing them.

        Ground truth: the narrative references a specific line item
        identifier from the workbook. The agent must read both.
        """
        sid = _new_session_id("s04")
        line_item_id = "LI-2026-0042"
        xlsx_rows = [
            ["line_item_id", "description", "amount_usd"],
            ["LI-2026-0040", "Office supplies", "1234.50"],
            ["LI-2026-0041", "Travel reimbursement", "987.25"],
            [line_item_id, "Critical vendor invoice", "55000.00"],
            ["LI-2026-0043", "Software license renewal", "4200.00"],
        ]
        narrative_needle = f"The disputed item is {line_item_id}; resolution is escalated."
        narrative_body = (
            "FINANCE NARRATIVE — Q2 2026\n\n"
            "Three line items were flagged during the quarterly review. "
            f"{narrative_needle}\n"
            "Counsel was engaged on 2026-05-19 to evaluate options."
        )
        docs = [
            SynthDoc(
                filename="line-items.xlsx",
                bytes=synth_xlsx(xlsx_rows, sheet_name="LineItems"),
                mime=("application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"),
                is_needle=False,
            ),
            SynthDoc(
                filename="narrative.docx",
                bytes=synth_docx(narrative_body, title="Q2 Narrative"),
                mime=("application/vnd.openxmlformats-officedocument.wordprocessingml.document"),
                is_needle=True,
                needle_fact=narrative_needle,
            ),
        ]
        await _hydrate_session(runtime, docs, session_id=sid)

        agent = _make_agent(runtime, session_id=sid, max_cost=TIER1_CAP_USD)
        prompt = (
            "The session has a line-items spreadsheet and a finance narrative. "
            "Which line item ID does the narrative flag as DISPUTED? "
            "Answer with the exact ID."
        )
        response = await agent.turn(prompt, session_id=sid)
        _assert_needle_present(response, line_item_id)
        _assert_no_tool_errors(response)
        _assert_retrieval_tool(response)
        _assert_cost_cap(response, TIER1_CAP_USD)
        await _judge_and_record(
            response=response,
            user_prompt=prompt,
            rubric=_grounding_rubric(
                scenario="S04 xlsx + docx cross-reference disputed line item",
                planted_fact=narrative_needle,
            ),
            test_id="S04",
            passing_labels=("grounded",),
            judge_state=judge_state,
            request=request,
        )

    async def test_scenario_05_wide_format_mix(
        self,
        runtime: KaosRuntime,
        judge_state: dict,
        request: pytest.FixtureRequest,
    ) -> None:
        """S05 — HTML + plain text + JSON in one turn.

        Ground truth: the JSON file contains the planted fact. Agent
        must handle all three formats in one turn.
        """
        sid = _new_session_id("s05")
        json_needle = "config_token: KAOS-S05-JSON-OK"
        docs = [
            SynthDoc(
                filename="release-notes.html",
                bytes=synth_html(
                    "Release 2.4.1\n\nBug fixes and performance improvements.",
                    title="Notes",
                ),
                mime="text/html",
                is_needle=False,
            ),
            SynthDoc(
                filename="readme.txt",
                bytes=synth_text(
                    "Run the installer with --quiet for unattended mode.\n"
                    "Logs land in /var/log/kaos/.\n"
                ),
                mime="text/plain",
                is_needle=False,
            ),
            SynthDoc(
                filename="config.json",
                bytes=synth_json(
                    {
                        "environment": "production",
                        "config_token": "KAOS-S05-JSON-OK",
                        "feature_flags": {"corpus_v2": True},
                    }
                ),
                mime="application/json",
                is_needle=True,
                needle_fact=json_needle,
            ),
        ]
        await _hydrate_session(runtime, docs, session_id=sid)

        agent = _make_agent(runtime, session_id=sid, max_cost=TIER1_CAP_USD)
        prompt = (
            "Three files are attached (HTML release notes, plain text README, "
            "and a JSON config). What is the value of 'config_token' in the "
            "JSON file? Answer with the exact token string."
        )
        response = await agent.turn(prompt, session_id=sid)
        _assert_needle_present(response, "KAOS-S05-JSON-OK")
        _assert_no_tool_errors(response)
        _assert_retrieval_tool(response)
        _assert_cost_cap(response, TIER1_CAP_USD)
        await _judge_and_record(
            response=response,
            user_prompt=prompt,
            rubric=_grounding_rubric(
                scenario="S05 wide format mix JSON config_token",
                planted_fact=json_needle,
            ),
            test_id="S05",
            passing_labels=("grounded",),
            judge_state=judge_state,
            request=request,
        )

    async def test_scenario_16_full_format_pile(
        self,
        runtime: KaosRuntime,
        judge_state: dict,
        request: pytest.FixtureRequest,
    ) -> None:
        """S16 — every supported format in one turn (PDF + DOCX + XLSX + HTML + JSON).

        Each format carries its OWN distinctive planted fact. The agent
        must reach into all five loaders in one turn — any single loader
        regression (e.g., the kaos-pdf-vs-DOCX fitness ranker miss) drops
        the count from 5/5 to 4/5 and the judge fails on ``partial``.
        """
        sid = _new_session_id("s16")
        facts = {
            "pdf": "Format-pile PDF fact: revenue plug-figure $93.14M.",
            "docx": "Format-pile DOCX fact: counsel-of-record is Hannah Brueggeman.",
            "xlsx": "LI-PILE-9001",  # cell value the narrative refers to
            "html": "Format-pile HTML fact: marketing launch on 2026-08-14.",
            "json": "PILE-JSON-TOKEN-Z7Q",
        }
        xlsx_rows = [
            ["row_id", "description", "amount_usd"],
            ["LI-PILE-9000", "Vendor onboarding", "2500.00"],
            [facts["xlsx"], "Pile-test critical line", "77777.77"],
            ["LI-PILE-9002", "Catering", "612.50"],
        ]
        docs = [
            SynthDoc(
                filename="pile-revenue.pdf",
                bytes=synth_pdf_text(
                    f"REVENUE BRIEF\n\n{facts['pdf']}\n\n"
                    "All other line items confirmed to roll forward.",
                    title="Revenue Brief",
                ),
                mime="application/pdf",
                is_needle=True,
                needle_fact=facts["pdf"],
            ),
            SynthDoc(
                filename="pile-counsel.docx",
                bytes=synth_docx(
                    f"COUNSEL OF RECORD\n\n{facts['docx']}\n\n"
                    "Engagement letter executed on 2026-03-01.",
                    title="Counsel Memo",
                ),
                mime=("application/vnd.openxmlformats-officedocument.wordprocessingml.document"),
                is_needle=True,
                needle_fact=facts["docx"],
            ),
            SynthDoc(
                filename="pile-lineitems.xlsx",
                bytes=synth_xlsx(xlsx_rows, sheet_name="LineItems"),
                mime=("application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"),
                is_needle=True,
                needle_fact=f"XLSX critical-line row id: {facts['xlsx']}",
            ),
            SynthDoc(
                filename="pile-launch.html",
                bytes=synth_html(
                    f"LAUNCH ANNOUNCEMENT\n\n{facts['html']}\n\nPre-orders open one week prior.",
                    title="Launch",
                ),
                mime="text/html",
                is_needle=True,
                needle_fact=facts["html"],
            ),
            SynthDoc(
                filename="pile-config.json",
                bytes=synth_json(
                    {
                        "environment": "production",
                        "release_token": facts["json"],
                        "feature_flags": {"format_pile": True},
                    }
                ),
                mime="application/json",
                is_needle=True,
                needle_fact=f"JSON release_token: {facts['json']}",
            ),
        ]
        await _hydrate_session(runtime, docs, session_id=sid)

        agent = _make_agent(runtime, session_id=sid, max_cost=TIER1_CAP_USD, max_react=14)
        prompt = (
            "Five files of different formats are attached (PDF, DOCX, "
            "XLSX, HTML, JSON). From each, surface one specific fact: "
            "(1) the PDF's revenue plug-figure, (2) the DOCX's counsel-"
            "of-record name, (3) the XLSX row_id labelled the 'critical "
            "line', (4) the HTML announcement's launch date, and "
            "(5) the JSON's release_token. List all five with their values."
        )
        response = await agent.turn(prompt, session_id=sid)
        _assert_needle_present(response, "$93.14M")
        _assert_needle_present(response, "Hannah Brueggeman")
        _assert_needle_present(response, "LI-PILE-9001")
        _assert_needle_present(response, "2026-08-14")
        _assert_needle_present(response, "PILE-JSON-TOKEN-Z7Q")
        _assert_no_tool_errors(response)
        _assert_retrieval_tool(response)
        _assert_cost_cap(response, TIER1_CAP_USD)
        await _judge_and_record(
            response=response,
            user_prompt=prompt,
            rubric=_multi_needle_rubric(
                scenario="S16 full-format pile (PDF + DOCX + XLSX + HTML + JSON)",
                planted_facts=list(facts.values()),
            ),
            test_id="S16",
            passing_labels=("grounded",),
            judge_state=judge_state,
            request=request,
        )

    async def test_scenario_17_wrong_mime_manifest_spoof(
        self,
        runtime: KaosRuntime,
        judge_state: dict,
        request: pytest.FixtureRequest,
    ) -> None:
        """S17 — artifact manifest mime LIES about the bytes (vs S02 which
        spoofs only the filename extension).

        Manifest declares ``text/plain`` but the bytes are a real PDF.
        Tools that trust the manifest mime will dispatch the wrong parser
        and either fail (loud) or fabricate (silent). The agent must
        either content-sniff itself or recover from the parser error and
        retry with the correct tool.
        """
        sid = _new_session_id("s17")
        needle_fact = "Manifest-spoof passphrase: KAOS-S17-MIME-OK"
        pdf_body = f"ARCHIVAL FILING\n\n{needle_fact}.\nConfirm receipt and route to compliance."
        pdf_bytes = synth_pdf_text(pdf_body, title="Archival")
        # Real DOCX distractor — no needle.
        docx_bytes = synth_docx(
            "QUARTERLY UPDATE\n\nNo material items to report.",
            title="QU",
        )
        docs = [
            SynthDoc(
                filename="filing.bin",
                bytes=pdf_bytes,
                # The LIE: the bytes are %PDF but we tell the artifact
                # store they're plain text. The on-disk filename also
                # gives no help (.bin). The agent's only way out is to
                # actually read the bytes or invoke a content-sniff
                # before trusting the manifest.
                mime="text/plain",
                is_needle=True,
                needle_fact=needle_fact,
            ),
            SynthDoc(
                filename="update.docx",
                bytes=docx_bytes,
                mime=("application/vnd.openxmlformats-officedocument.wordprocessingml.document"),
                is_needle=False,
            ),
        ]
        await _hydrate_session(runtime, docs, session_id=sid)

        agent = _make_agent(runtime, session_id=sid, max_cost=TIER1_CAP_USD)
        prompt = (
            "Two files are attached. One contains a 'Manifest-spoof "
            "passphrase' that I need verbatim. Note that file metadata "
            "may be misleading — trust the actual file contents."
        )
        response = await agent.turn(prompt, session_id=sid)
        _assert_needle_present(response, "KAOS-S17-MIME-OK")
        _assert_no_tool_errors(response)
        _assert_retrieval_tool(response)
        _assert_cost_cap(response, TIER1_CAP_USD)
        await _judge_and_record(
            response=response,
            user_prompt=prompt,
            rubric=_grounding_rubric(
                scenario="S17 manifest-mime spoof (PDF bytes, text/plain manifest)",
                planted_fact=needle_fact,
            ),
            test_id="S17",
            passing_labels=("grounded",),
            judge_state=judge_state,
            request=request,
        )

    async def test_scenario_18_corrupted_and_empty_files(
        self,
        runtime: KaosRuntime,
        judge_state: dict,
        request: pytest.FixtureRequest,
    ) -> None:
        """S18 — empty + gibberish + good file in one corpus.

        Two of three files are unreadable on purpose. The agent must
        (a) extract the planted fact from the ONE good file, and
        (b) not fabricate "from" the broken ones. The judge gates on
        grounded — if the agent claims content from the empty / gibberish
        files (or attributes the answer to them), the verdict drops.
        """
        sid = _new_session_id("s18")
        needle_fact = "Recovery passphrase: KAOS-S18-GOLDEN-PATH"
        good_body = (
            f"GOLDEN PATH MEMO\n\n{needle_fact}.\nUse only this memo for the recovery procedure."
        )
        docs = [
            SynthDoc(
                filename="empty-policy.pdf",
                bytes=synth_empty(),
                mime="application/pdf",
                is_needle=False,
            ),
            SynthDoc(
                filename="corrupted-archive.docx",
                bytes=synth_gibberish(512, seed=1818),
                mime=("application/vnd.openxmlformats-officedocument.wordprocessingml.document"),
                is_needle=False,
            ),
            SynthDoc(
                filename="golden-memo.pdf",
                bytes=synth_pdf_text(good_body, title="Golden Path"),
                mime="application/pdf",
                is_needle=True,
                needle_fact=needle_fact,
            ),
        ]
        await _hydrate_session(runtime, docs, session_id=sid)

        agent = _make_agent(runtime, session_id=sid, max_cost=TIER1_CAP_USD, max_react=14)
        prompt = (
            "Three files are attached. Find and quote the 'recovery "
            "passphrase'. If any of the files appear unreadable / "
            "empty / corrupted, list which ones so I know to re-upload."
        )
        response = await agent.turn(prompt, session_id=sid)
        _assert_needle_present(response, "KAOS-S18-GOLDEN-PATH")
        # NOT _assert_no_tool_errors — we EXPECT parser tool errors on
        # the empty / gibberish files. The helper already permits mixed
        # error/success as long as at least one tool call succeeded
        # (which it must here, to recover the needle).
        if response.tool_calls and not any(not tc.is_error for tc in response.tool_calls):
            raise AssertionError(
                "every tool call errored — agent could not recover the good file "
                f"despite the planted needle being present. tools={_tool_names(response)}"
            )
        _assert_retrieval_tool(response)
        _assert_cost_cap(response, TIER1_CAP_USD)
        await _judge_and_record(
            response=response,
            user_prompt=prompt,
            rubric=_grounding_rubric(
                scenario="S18 corrupted + empty + good file recovery",
                planted_fact=needle_fact,
            ),
            test_id="S18",
            passing_labels=("grounded",),
            judge_state=judge_state,
            request=request,
        )


# ===========================================================================
# Tier 2 — Needle in haystack (S06-S10)
# ===========================================================================


@pytest.mark.live
@pytest.mark.network
@requires_provider
class TestNeedleInHaystack:
    """Modest scale (25-50 docs). Targets retrieval + content-type fallback."""

    async def test_scenario_06_single_needle_in_25_distractors(
        self,
        runtime: KaosRuntime,
        judge_state: dict,
        request: pytest.FixtureRequest,
    ) -> None:
        """S06 — one needle PDF + 25 unrelated DOCX distractors."""
        sid = _new_session_id("s06")
        needle_fact = "Acquisition target codename: NIGHTHAWK-2026"
        # Distractors are plain DOCX in the legal cluster.
        distractors = synth_corpus(
            n_docs=25,
            n_needles=0,
            needle_facts=[],
            seed=606,
            cluster_topics=["legal"],
            file_types=("docx",),
        )
        needle_doc = SynthDoc(
            filename="needle-001.pdf",
            bytes=synth_pdf_text(
                f"CONFIDENTIAL\n\n{needle_fact}.\nDo not distribute.",
                title="Confidential",
            ),
            mime="application/pdf",
            is_needle=True,
            needle_fact=needle_fact,
        )
        # Place needle at the middle of the corpus.
        docs = [*distractors[:12], needle_doc, *distractors[12:]]
        await _hydrate_session(runtime, docs, session_id=sid)

        agent = _make_agent(runtime, session_id=sid, max_cost=TIER2_CAP_USD)
        prompt = (
            "Find the document that mentions an 'acquisition target codename'. "
            "Quote the codename verbatim."
        )
        response = await agent.turn(prompt, session_id=sid)
        _assert_needle_present(response, "NIGHTHAWK-2026")
        _assert_no_tool_errors(response)
        _assert_retrieval_tool(response)
        _assert_cost_cap(response, TIER2_CAP_USD)
        await _judge_and_record(
            response=response,
            user_prompt=prompt,
            rubric=_grounding_rubric(
                scenario="S06 single needle in 25 distractors",
                planted_fact=needle_fact,
            ),
            test_id="S06",
            passing_labels=("grounded",),
            judge_state=judge_state,
            request=request,
        )

    async def test_scenario_07_multi_needle_synthesis(
        self,
        runtime: KaosRuntime,
        judge_state: dict,
        request: pytest.FixtureRequest,
    ) -> None:
        """S07 — three needles across three file types in 50 distractors.

        The agent must find all three planted facts and synthesize.
        """
        sid = _new_session_id("s07")
        needles = [
            ("PDF needle: Project Alpha funding round closed at $42M.", "Alpha", "pdf"),
            ("DOCX needle: Project Bravo launch date is 2026-11-01.", "Bravo", "docx"),
            ("HTML needle: Project Charlie has 17 enrolled customers.", "Charlie", "html"),
        ]
        distractors = synth_corpus(
            n_docs=50,
            n_needles=0,
            needle_facts=[],
            seed=707,
            cluster_topics=["finance", "engineering"],
            file_types=("docx", "html", "text"),
        )
        needle_docs: list[SynthDoc] = []
        for needle_fact, project, ftype in needles:
            if ftype == "pdf":
                body = synth_pdf_text(needle_fact, title=project)
                mime = "application/pdf"
            elif ftype == "docx":
                body = synth_docx(needle_fact, title=project)
                mime = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            else:
                body = synth_html(needle_fact, title=project)
                mime = "text/html"
            needle_docs.append(
                SynthDoc(
                    filename=f"project-{project.lower()}.{ftype}",
                    bytes=body,
                    mime=mime,
                    is_needle=True,
                    needle_fact=needle_fact,
                )
            )
        # Sprinkle needles at three different positions.
        docs = [
            *distractors[:10],
            needle_docs[0],
            *distractors[10:25],
            needle_docs[1],
            *distractors[25:40],
            needle_docs[2],
            *distractors[40:],
        ]
        await _hydrate_session(runtime, docs, session_id=sid)

        agent = _make_agent(runtime, session_id=sid, max_cost=TIER2_CAP_USD, max_react=18)
        prompt = (
            "The corpus has documents about three projects: Alpha, Bravo, "
            "and Charlie. For each, find and report (1) Alpha's funding "
            "amount, (2) Bravo's launch date, (3) Charlie's enrolled "
            "customer count. Be specific."
        )
        response = await agent.turn(prompt, session_id=sid)
        # Needles checked one by one so the error message names the
        # exact one that's missing.
        _assert_needle_present(response, "$42M")
        _assert_needle_present(response, "2026-11-01")
        _assert_needle_present(response, "17")
        _assert_no_tool_errors(response)
        _assert_retrieval_tool(response)
        _assert_cost_cap(response, TIER2_CAP_USD)
        await _judge_and_record(
            response=response,
            user_prompt=prompt,
            rubric=_multi_needle_rubric(
                scenario="S07 multi-needle synthesis across PDF + DOCX + HTML",
                planted_facts=[n[0] for n in needles],
            ),
            test_id="S07",
            passing_labels=("grounded",),
            judge_state=judge_state,
            request=request,
        )

    async def test_scenario_08_topical_cluster_routing(
        self,
        runtime: KaosRuntime,
        judge_state: dict,
        request: pytest.FixtureRequest,
    ) -> None:
        """S08 — 50 files in 5 topical clusters of 10. Needle in cluster 3.

        The agent should not waste budget on clusters 1/2/4/5.
        """
        sid = _new_session_id("s08")
        needle_fact = "Engineering on-call escalation: page CTO at hour 4"
        clusters = ["finance", "legal", "engineering", "medical", "regulatory"]
        # Generate 10 docs per cluster (50 total), no needle planted by
        # the helper — we'll inject it manually into the "engineering"
        # cluster slice.
        base = synth_corpus(
            n_docs=50,
            n_needles=0,
            needle_facts=[],
            seed=808,
            cluster_topics=clusters,
            file_types=("docx", "html"),
        )
        # Replace one engineering doc with a needle doc.
        eng_indices = [i for i, _ in enumerate(base) if (i % 5) == 2]  # cluster idx 2
        needle_pos = eng_indices[len(eng_indices) // 2]
        needle_doc = SynthDoc(
            filename="oncall-runbook.docx",
            bytes=synth_docx(
                "ON-CALL RUNBOOK\n\n"
                f"{needle_fact}.\n"
                "Sev1 incidents trigger automatic Slack page within 60 seconds.",
                title="Runbook",
            ),
            mime=("application/vnd.openxmlformats-officedocument.wordprocessingml.document"),
            is_needle=True,
            needle_fact=needle_fact,
        )
        docs = list(base)
        docs[needle_pos] = needle_doc
        await _hydrate_session(runtime, docs, session_id=sid)

        agent = _make_agent(runtime, session_id=sid, max_cost=TIER2_CAP_USD)
        prompt = (
            "There are documents in this session about finance, legal, "
            "engineering, medical, and regulatory topics. Find the "
            "ENGINEERING on-call escalation policy: at what hour does "
            "an unresolved incident page the CTO?"
        )
        response = await agent.turn(prompt, session_id=sid)
        _assert_needle_present(response, "hour 4")
        _assert_no_tool_errors(response)
        _assert_retrieval_tool(response)
        _assert_cost_cap(response, TIER2_CAP_USD)
        await _judge_and_record(
            response=response,
            user_prompt=prompt,
            rubric=_grounding_rubric(
                scenario="S08 topical cluster routing — engineering escalation",
                planted_fact=needle_fact,
            ),
            test_id="S08",
            passing_labels=("grounded",),
            judge_state=judge_state,
            request=request,
        )

    async def test_scenario_09_adversarial_semantic_needle(
        self,
        runtime: KaosRuntime,
        judge_state: dict,
        request: pytest.FixtureRequest,
    ) -> None:
        """S09 — paraphrase needle vs distractors with literal keywords.

        The needle uses paraphrase ('compensation deferred') while
        distractors load the literal keyword ('salary'). Pure BM25 will
        favor the distractors; the agent must reach for semantic
        retrieval or content reading.
        """
        sid = _new_session_id("s09")
        needle_fact = (
            "Executive compensation deferred until 2027-Q1; bonus pool frozen pending board review."
        )
        # 30 distractors that all mention "salary" / "wages" but NOT
        # "compensation deferred" or "bonus pool".
        distractor_bodies = [
            "ANNUAL SALARY REVIEW\n\nThis year's salary adjustments range "
            "from 2% to 6%. The wage band for engineering remained unchanged."
            for _ in range(30)
        ]
        distractors: list[SynthDoc] = []
        for i, body in enumerate(distractor_bodies):
            distractors.append(
                SynthDoc(
                    filename=f"hr-salary-memo-{i + 1:02d}.docx",
                    bytes=synth_docx(body, title=f"HR Memo {i + 1}"),
                    mime=(
                        "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                    ),
                    is_needle=False,
                )
            )
        needle_doc = SynthDoc(
            filename="exec-comp-update.docx",
            bytes=synth_docx(
                f"EXECUTIVE COMPENSATION UPDATE\n\n{needle_fact}.\n"
                "Counsel notified Finance on the same day.",
                title="Exec Comp Update",
            ),
            mime=("application/vnd.openxmlformats-officedocument.wordprocessingml.document"),
            is_needle=True,
            needle_fact=needle_fact,
        )
        docs = [*distractors[:15], needle_doc, *distractors[15:]]
        await _hydrate_session(runtime, docs, session_id=sid)

        agent = _make_agent(runtime, session_id=sid, max_cost=TIER2_CAP_USD, max_react=16)
        prompt = (
            "When is executive compensation being deferred to, and what is "
            "the status of the bonus pool? The information is in one of "
            "the attached HR documents."
        )
        response = await agent.turn(prompt, session_id=sid)
        # Either the literal needle phrase or both anchors appearing
        # counts as a hit. We require the specific quarter token.
        _assert_needle_present(response, "2027")
        _assert_no_tool_errors(response)
        _assert_retrieval_tool(response)
        _assert_cost_cap(response, TIER2_CAP_USD)
        await _judge_and_record(
            response=response,
            user_prompt=prompt,
            rubric=_grounding_rubric(
                scenario="S09 adversarial semantic needle (exec comp deferred)",
                planted_fact=needle_fact,
            ),
            test_id="S09",
            passing_labels=("grounded",),
            judge_state=judge_state,
            request=request,
        )

    async def test_scenario_10_wrong_extension_needle_in_haystack(
        self,
        runtime: KaosRuntime,
        judge_state: dict,
        request: pytest.FixtureRequest,
    ) -> None:
        """S10 — needle has wrong extension; distractors correct.

        The agent's triage / content-read path must fall back to
        content-type detection rather than trusting the extension.
        """
        sid = _new_session_id("s10")
        needle_fact = "Authorization code for vault access: VAULT-X9-OMEGA"
        distractors = synth_corpus(
            n_docs=30,
            n_needles=0,
            needle_facts=[],
            seed=1010,
            cluster_topics=["legal", "regulatory"],
            file_types=("docx", "html"),
        )
        # Needle: real DOCX, advertised as .pdf.
        needle_bytes = synth_docx(
            f"SECURE OPERATIONS\n\n{needle_fact}.\nRotate quarterly; one-time-use per session.",
            title="Vault Auth",
        )
        _, spoof_name = wrong_extension(needle_bytes, ".pdf")
        needle_doc = SynthDoc(
            filename=spoof_name,
            bytes=needle_bytes,
            # The REAL mime — bytes are DOCX even though the filename
            # ends in .pdf.
            mime=("application/vnd.openxmlformats-officedocument.wordprocessingml.document"),
            is_needle=True,
            needle_fact=needle_fact,
        )
        docs = [*distractors[:15], needle_doc, *distractors[15:]]
        await _hydrate_session(runtime, docs, session_id=sid)

        agent = _make_agent(runtime, session_id=sid, max_cost=TIER2_CAP_USD)
        prompt = (
            "Find the document that contains an 'authorization code for "
            "vault access'. Quote the exact code. Note: some filenames "
            "may have wrong extensions; rely on the content."
        )
        response = await agent.turn(prompt, session_id=sid)
        _assert_needle_present(response, "VAULT-X9-OMEGA")
        _assert_no_tool_errors(response)
        _assert_retrieval_tool(response)
        _assert_cost_cap(response, TIER2_CAP_USD)
        await _judge_and_record(
            response=response,
            user_prompt=prompt,
            rubric=_grounding_rubric(
                scenario="S10 wrong-extension needle in 30 distractors",
                planted_fact=needle_fact,
            ),
            test_id="S10",
            passing_labels=("grounded",),
            judge_state=judge_state,
            request=request,
        )

    async def test_scenario_19_live_edgar_plus_uploaded_xlsx(
        self,
        runtime: KaosRuntime,
        judge_state: dict,
        request: pytest.FixtureRequest,
    ) -> None:
        """S19 — uploaded XLSX cross-referenced with a live SEC EDGAR fetch.

        The XLSX names Apple Inc + CIK 0000320193 as the issuer of interest.
        The agent must (a) read the XLSX, (b) call a kaos-source-edgar or
        kaos-web tool to retrieve the most recent 10-K accession, and (c)
        report both the CIK from the XLSX AND a 10-K accession number
        formatted ``NNNNNNNNNN-NN-NNNNNN``. Combines local file loading
        with live data freshness.
        """
        sid = _new_session_id("s19")
        xlsx_rows = [
            ["issuer_name", "cik", "form_type"],
            ["Apple Inc.", "0000320193", "10-K"],
            ["Microsoft Corporation", "0000789019", "10-Q"],
        ]
        docs = [
            SynthDoc(
                filename="watchlist.xlsx",
                bytes=synth_xlsx(xlsx_rows, sheet_name="Watchlist"),
                mime=("application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"),
                is_needle=True,
                needle_fact="Watchlist issuer Apple Inc CIK 0000320193 form 10-K",
            ),
        ]
        await _hydrate_session(runtime, docs, session_id=sid)

        agent = _make_agent(runtime, session_id=sid, max_cost=TIER2_CAP_USD, max_react=16)
        prompt = (
            "An XLSX 'watchlist' is attached. Read it: for the issuer "
            "marked as a 10-K filer, fetch from EDGAR (sec.gov) the "
            "accession number of its MOST RECENT 10-K filing. Report "
            "both the issuer's CIK from the watchlist AND the "
            "accession number in the EDGAR format "
            "(NNNNNNNNNN-NN-NNNNNN)."
        )
        response = await agent.turn(prompt, session_id=sid)
        # Both must appear: the CIK from the local XLSX AND a live
        # EDGAR-shaped accession number. The exact accession may move
        # filing-to-filing but the SHAPE is stable.
        _assert_needle_present(response, "0000320193")
        import re

        if not re.search(r"\d{10}-\d{2}-\d{6}", response.text):
            raise AssertionError(
                "no EDGAR-shaped accession number "
                "(NNNNNNNNNN-NN-NNNNNN) found in the response — agent "
                "did not bring the live-fetched filing identifier back. "
                f"response head: {response.text[:300]!r} "
                f"tools: {_tool_names(response)}"
            )
        _assert_no_tool_errors(response)
        _assert_retrieval_tool(response)
        _assert_cost_cap(response, TIER2_CAP_USD)
        await _judge_and_record(
            response=response,
            user_prompt=prompt,
            rubric=_grounding_rubric(
                scenario="S19 uploaded XLSX + live EDGAR 10-K accession",
                planted_fact="Apple Inc CIK 0000320193 most-recent 10-K accession",
            ),
            test_id="S19",
            passing_labels=("grounded",),
            judge_state=judge_state,
            request=request,
        )

    async def test_scenario_20_live_ecfr_plus_uploaded_docx(
        self,
        runtime: KaosRuntime,
        judge_state: dict,
        request: pytest.FixtureRequest,
    ) -> None:
        """S20 — uploaded legal memo references a CFR section; agent must
        fetch the live regulatory text and quote subsection (b).

        The memo says: "see 17 CFR § 240.10b-5 for the antifraud rule".
        The agent must read the memo to pick the citation, then use a
        kaos-source-ecfr (or kaos-web-fetch) tool to retrieve the text
        and quote subsection (b).
        """
        sid = _new_session_id("s20")
        memo_body = (
            "MEMORANDUM\n\n"
            "To: Litigation Team\n"
            "Re: Antifraud framework reference\n\n"
            "Counsel should review 17 CFR § 240.10b-5 for the SEC "
            "antifraud rule and quote subsection (b) verbatim when "
            "drafting the complaint."
        )
        docs = [
            SynthDoc(
                filename="memo-antifraud-ref.docx",
                bytes=synth_docx(memo_body, title="Antifraud Memo"),
                mime=("application/vnd.openxmlformats-officedocument.wordprocessingml.document"),
                is_needle=True,
                needle_fact="memo cites 17 CFR § 240.10b-5 subsection (b)",
            ),
        ]
        await _hydrate_session(runtime, docs, session_id=sid)

        agent = _make_agent(runtime, session_id=sid, max_cost=TIER2_CAP_USD, max_react=16)
        prompt = (
            "An internal memo (DOCX) is attached. It directs you to a "
            "CFR rule and asks for subsection (b) verbatim. (1) State "
            "which CFR section the memo cites, then (2) fetch the live "
            "rule text and quote subsection (b)."
        )
        response = await agent.turn(prompt, session_id=sid)
        _assert_needle_present(response, "240.10b-5")
        # Subsection (b)'s text starts with "To make any untrue
        # statement of a material fact". We check the distinctive phrase.
        if "untrue statement of a material fact" not in response.text.lower():
            raise AssertionError(
                "agent did not return the verbatim text of 17 CFR § "
                "240.10b-5(b). The phrase 'untrue statement of a "
                "material fact' is the canonical anchor and was not "
                f"present. response head: {response.text[:400]!r} "
                f"tools: {_tool_names(response)}"
            )
        _assert_no_tool_errors(response)
        _assert_retrieval_tool(response)
        _assert_cost_cap(response, TIER2_CAP_USD)
        await _judge_and_record(
            response=response,
            user_prompt=prompt,
            rubric=_grounding_rubric(
                scenario="S20 DOCX legal memo + live eCFR 17 CFR 240.10b-5(b)",
                planted_fact="17 CFR 240.10b-5(b) verbatim text",
            ),
            test_id="S20",
            passing_labels=("grounded",),
            judge_state=judge_state,
            request=request,
        )

    async def test_scenario_21_live_fr_search_plus_uploaded_pdf(
        self,
        runtime: KaosRuntime,
        judge_state: dict,
        request: pytest.FixtureRequest,
    ) -> None:
        """S21 — uploaded PDF policy references a SEC rule; agent must
        fetch the live Federal Register document number.

        The PDF policy memo says: "Our annual reporting follows the SEC
        climate disclosure final rule for large filers." The agent must
        read the PDF, search Federal Register for the rule, and bring
        back the FR document number ``2024-05137``.
        """
        sid = _new_session_id("s21")
        policy_body = (
            "INTERNAL POLICY — CLIMATE REPORTING\n\n"
            "Our annual reporting follows the SEC's climate disclosure "
            "final rule (published in the Federal Register in 2024). "
            "Compliance owners must surface the FR document number on "
            "every cycle-cap deck."
        )
        docs = [
            SynthDoc(
                filename="climate-policy.pdf",
                bytes=synth_pdf_text(policy_body, title="Climate Policy"),
                mime="application/pdf",
                is_needle=True,
                needle_fact="policy references SEC climate disclosure FR doc 2024-05137",
            ),
        ]
        await _hydrate_session(runtime, docs, session_id=sid)

        agent = _make_agent(runtime, session_id=sid, max_cost=TIER2_CAP_USD, max_react=16)
        prompt = (
            "A short policy PDF is attached. It references a SEC "
            "climate disclosure final rule published in the Federal "
            "Register in 2024. Search Federal Register and report the "
            "exact FR document number for that rule."
        )
        response = await agent.turn(prompt, session_id=sid)
        _assert_needle_present(response, "2024-05137")
        _assert_no_tool_errors(response)
        _assert_retrieval_tool(response)
        _assert_cost_cap(response, TIER2_CAP_USD)
        await _judge_and_record(
            response=response,
            user_prompt=prompt,
            rubric=_grounding_rubric(
                scenario="S21 uploaded PDF policy + live Federal Register doc 2024-05137",
                planted_fact="Federal Register doc 2024-05137 (SEC climate disclosure)",
            ),
            test_id="S21",
            passing_labels=("grounded",),
            judge_state=judge_state,
            request=request,
        )

    async def test_scenario_22_cluster_routing_with_wrong_ext(
        self,
        runtime: KaosRuntime,
        judge_state: dict,
        request: pytest.FixtureRequest,
    ) -> None:
        """S22 — 50 docs in 5 clusters + needle is in a wrong-extension file
        inside the right cluster.

        Combines cluster routing (S08) with extension spoofing (S10) so
        the agent has to (a) narrow to the right cluster AND (b) trust
        bytes over filename inside that cluster. Two distinct failure
        modes layered on one corpus.
        """
        sid = _new_session_id("s22")
        needle_fact = "Combined-stress fact: medical-trial cohort N=2486 enrolled."
        clusters = ["finance", "legal", "engineering", "medical", "regulatory"]
        base = synth_corpus(
            n_docs=50,
            n_needles=0,
            needle_facts=[],
            seed=2222,
            cluster_topics=clusters,
            file_types=("docx", "html"),
        )
        # Needle is a DOCX advertised as ``.pdf`` and placed inside the
        # medical cluster (round-robin index 3).
        needle_bytes = synth_docx(
            f"MEDICAL TRIAL UPDATE\n\n{needle_fact}.\n"
            "Pre-specified endpoints unchanged from prior protocol.",
            title="Trial Update",
        )
        _, spoof_name = wrong_extension(needle_bytes, ".pdf")
        needle_doc = SynthDoc(
            filename=spoof_name,
            bytes=needle_bytes,
            mime=("application/vnd.openxmlformats-officedocument.wordprocessingml.document"),
            is_needle=True,
            needle_fact=needle_fact,
        )
        # Find a medical-cluster slot (i % 5 == 3 in the round-robin).
        medical_indices = [i for i, _ in enumerate(base) if (i % 5) == 3]
        needle_pos = medical_indices[len(medical_indices) // 2]
        docs = list(base)
        docs[needle_pos] = needle_doc
        await _hydrate_session(runtime, docs, session_id=sid)

        agent = _make_agent(runtime, session_id=sid, max_cost=TIER2_CAP_USD, max_react=16)
        prompt = (
            "Documents in this session span five topics: finance, "
            "legal, engineering, medical, regulatory. Find the "
            "MEDICAL document that reports a trial cohort enrollment "
            "count and quote the exact N. Some filenames may have "
            "the wrong extension — rely on the file contents."
        )
        response = await agent.turn(prompt, session_id=sid)
        _assert_needle_present(response, "2486")
        _assert_no_tool_errors(response)
        _assert_retrieval_tool(response)
        _assert_cost_cap(response, TIER2_CAP_USD)
        await _judge_and_record(
            response=response,
            user_prompt=prompt,
            rubric=_grounding_rubric(
                scenario="S22 cluster routing + wrong-extension combined",
                planted_fact=needle_fact,
            ),
            test_id="S22",
            passing_labels=("grounded",),
            judge_state=judge_state,
            request=request,
        )

    async def test_scenario_23_conflicting_versions_authoritative(
        self,
        runtime: KaosRuntime,
        judge_state: dict,
        request: pytest.FixtureRequest,
    ) -> None:
        """S23 — same fact appears verbatim in 5 docs with different
        values; only ONE doc is marked AUTHORITATIVE.

        Real-world version-conflict pathology: the corpus shows five
        candidate values for "next board meeting date" because earlier
        drafts persisted. The agent must (a) recognize the conflict,
        (b) prefer the doc that says "AUTHORITATIVE", and (c) cite
        the source filename.
        """
        sid = _new_session_id("s23")
        candidate_dates = [
            "2026-07-10",
            "2026-08-21",
            "2026-09-04",
            "2026-10-15",
            "2026-11-12",  # this one is in the AUTHORITATIVE doc
        ]
        authoritative_date = candidate_dates[-1]
        docs: list[SynthDoc] = []
        for i, dt in enumerate(candidate_dates):
            is_auth = i == len(candidate_dates) - 1
            tag = (
                "AUTHORITATIVE — supersedes all prior drafts" if is_auth else "DRAFT v" + str(i + 1)
            )
            body = (
                f"BOARD MEETING NOTICE [{tag}]\n\n"
                f"The next board meeting will be held on {dt}.\n"
                "Logistics will be confirmed by the corporate secretary."
            )
            docs.append(
                SynthDoc(
                    filename=(
                        "board-notice-FINAL.docx" if is_auth else f"board-notice-draft-{i + 1}.docx"
                    ),
                    bytes=synth_docx(body, title="Board Notice"),
                    mime=(
                        "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                    ),
                    is_needle=is_auth,
                    needle_fact=(f"AUTHORITATIVE board meeting date: {dt}" if is_auth else None),
                )
            )
        await _hydrate_session(runtime, docs, session_id=sid)

        agent = _make_agent(runtime, session_id=sid, max_cost=TIER2_CAP_USD, max_react=14)
        prompt = (
            "Five board-meeting notices are attached. Some are drafts, "
            "one is the AUTHORITATIVE version. What is the date of the "
            "next board meeting, and which filename is the authoritative "
            "source?"
        )
        response = await agent.turn(prompt, session_id=sid)
        _assert_needle_present(response, authoritative_date)
        if "board-notice-FINAL" not in response.text:
            raise AssertionError(
                "agent did not cite the AUTHORITATIVE filename "
                "'board-notice-FINAL.docx'; this is the version-conflict "
                f"pathology. response head: {response.text[:400]!r} "
                f"tools: {_tool_names(response)}"
            )
        _assert_no_tool_errors(response)
        _assert_retrieval_tool(response)
        _assert_cost_cap(response, TIER2_CAP_USD)
        await _judge_and_record(
            response=response,
            user_prompt=prompt,
            rubric=_grounding_rubric(
                scenario="S23 conflicting versions — AUTHORITATIVE supersedes drafts",
                planted_fact=f"AUTHORITATIVE date {authoritative_date}",
            ),
            test_id="S23",
            passing_labels=("grounded",),
            judge_state=judge_state,
            request=request,
        )

    async def test_scenario_24_high_spoof_rate_mixed_corpus(
        self,
        runtime: KaosRuntime,
        judge_state: dict,
        request: pytest.FixtureRequest,
    ) -> None:
        """S24 — 40 docs, 50% wrong-extension spoof rate.

        High wrong-extension density stresses the content-type detector
        and the agent's willingness to retry with a different parser
        after the first one fails. ``synth_corpus`` handles the spoof
        distribution; we plant one needle and ride the random spoof
        roulette.
        """
        sid = _new_session_id("s24")
        needle_fact = "High-spoof corpus needle: VAULT-PHOENIX-2026-7Q"
        docs = synth_corpus(
            n_docs=40,
            n_needles=1,
            needle_facts=[needle_fact],
            seed=2424,
            cluster_topics=["legal", "regulatory", "finance"],
            file_types=("pdf", "docx", "html", "text"),
            # 50% spoof rate — half of all files will have their
            # filename extension lying about the real bytes.
            wrong_extension_rate=0.5,
        )
        await _hydrate_session(runtime, docs, session_id=sid)

        agent = _make_agent(runtime, session_id=sid, max_cost=TIER2_CAP_USD, max_react=16)
        prompt = (
            "Find the document that contains a 'high-spoof corpus "
            "needle' and quote it verbatim. Note: roughly half of the "
            "filenames in this corpus may have the wrong extension — "
            "rely on actual file contents."
        )
        response = await agent.turn(prompt, session_id=sid)
        _assert_needle_present(response, "VAULT-PHOENIX-2026-7Q")
        _assert_no_tool_errors(response)
        _assert_retrieval_tool(response)
        _assert_cost_cap(response, TIER2_CAP_USD)
        await _judge_and_record(
            response=response,
            user_prompt=prompt,
            rubric=_grounding_rubric(
                scenario="S24 high (50%) wrong-extension spoof rate, 40 docs",
                planted_fact=needle_fact,
            ),
            test_id="S24",
            passing_labels=("grounded",),
            judge_state=judge_state,
            request=request,
        )


# ===========================================================================
# Tier 3 — Pure scale (S11-S15)
# ===========================================================================


@pytest.mark.live
@pytest.mark.network
@pytest.mark.slow
@requires_provider
class TestPureScale:
    """200/500/1000-doc + multi-session VFS isolation."""

    async def test_scenario_11_200_doc_pure_scale(
        self,
        runtime: KaosRuntime,
        judge_state: dict,
        request: pytest.FixtureRequest,
    ) -> None:
        """S11 — 200 mixed-type docs, 1 needle. Tests triage_corpus."""
        sid = _new_session_id("s11")
        needle_fact = "Master kill switch combination: ALPHA-OMEGA-7-7-2"
        docs = synth_corpus(
            n_docs=200,
            n_needles=1,
            needle_facts=[needle_fact],
            seed=1111,
            file_types=("docx", "html", "text"),
        )
        await _hydrate_session(runtime, docs, session_id=sid)

        agent = _make_agent(runtime, session_id=sid, max_cost=TIER3_CAP_USD, max_react=20)
        prompt = (
            "Search the attached corpus (about 200 documents). Find the "
            "'master kill switch combination' and quote it verbatim."
        )
        response = await agent.turn(prompt, session_id=sid)
        _assert_needle_present(response, "ALPHA-OMEGA-7-7-2")
        _assert_no_tool_errors(response)
        _assert_retrieval_tool(response)
        _assert_cost_cap(response, TIER3_CAP_USD)
        await _judge_and_record(
            response=response,
            user_prompt=prompt,
            rubric=_grounding_rubric(
                scenario="S11 200-doc pure-scale single needle",
                planted_fact=needle_fact,
            ),
            test_id="S11",
            passing_labels=("grounded",),
            judge_state=judge_state,
            request=request,
        )

    async def test_scenario_12_500_doc_deal_room(
        self,
        runtime: KaosRuntime,
        judge_state: dict,
        request: pytest.FixtureRequest,
    ) -> None:
        """S12 — 500-doc deal room, 5 needles. Multi-needle synthesis."""
        sid = _new_session_id("s12")
        needle_facts = [
            "Deal Room Item 1: Indemnification cap is $25,000,000.",
            "Deal Room Item 2: Closing date is 2026-12-15.",
            "Deal Room Item 3: Earnout target is $180M revenue by 2027.",
            "Deal Room Item 4: Working capital peg is $42M.",
            "Deal Room Item 5: Escrow holdback is 8% of purchase price.",
        ]
        docs = synth_corpus(
            n_docs=500,
            n_needles=5,
            needle_facts=needle_facts,
            seed=1212,
            file_types=("docx", "html", "text"),
        )
        await _hydrate_session(runtime, docs, session_id=sid)

        agent = _make_agent(runtime, session_id=sid, max_cost=TIER3_CAP_USD, max_react=25)
        prompt = (
            "The attached corpus is a deal room with about 500 documents. "
            "Find and report the FIVE 'Deal Room Item' facts. List them "
            "1 through 5 with the specific values."
        )
        response = await agent.turn(prompt, session_id=sid)
        # Each needle's distinctive numeric anchor must show.
        _assert_needle_present(response, "$25,000,000")
        _assert_needle_present(response, "2026-12-15")
        _assert_needle_present(response, "$180M")
        _assert_needle_present(response, "$42M")
        _assert_needle_present(response, "8%")
        _assert_no_tool_errors(response)
        _assert_retrieval_tool(response)
        _assert_cost_cap(response, TIER3_CAP_USD)
        await _judge_and_record(
            response=response,
            user_prompt=prompt,
            rubric=_multi_needle_rubric(
                scenario="S12 500-doc deal room — five needles",
                planted_facts=needle_facts,
            ),
            test_id="S12",
            passing_labels=("grounded",),
            judge_state=judge_state,
            request=request,
        )

    async def test_scenario_13_1000_doc_legal_corpus(
        self,
        runtime: KaosRuntime,
        judge_state: dict,
        request: pytest.FixtureRequest,
    ) -> None:
        """S13 — 1000 docs, 3 needles. Largest single-session test."""
        sid = _new_session_id("s13")
        needle_facts = [
            "1000-corpus marker A: Brand-name PATENT-EAGLE-2026 active.",
            "1000-corpus marker B: Litigation reserve sits at $12.7M.",
            "1000-corpus marker C: Patent expiration date is 2028-03-31.",
        ]
        docs = synth_corpus(
            n_docs=1000,
            n_needles=3,
            needle_facts=needle_facts,
            seed=1313,
            cluster_topics=["legal", "finance", "regulatory"],
            file_types=("docx", "html", "text"),
            paragraphs_per_doc=4,
        )
        await _hydrate_session(runtime, docs, session_id=sid)

        agent = _make_agent(runtime, session_id=sid, max_cost=TIER3_CAP_USD, max_react=25)
        prompt = (
            "The attached corpus has roughly 1000 documents. Find the three "
            "'1000-corpus markers' (A, B, C). For each report the specific "
            "fact: marker A's brand-name, marker B's reserve amount, "
            "marker C's expiration date."
        )
        response = await agent.turn(prompt, session_id=sid)
        _assert_needle_present(response, "PATENT-EAGLE-2026")
        _assert_needle_present(response, "$12.7M")
        _assert_needle_present(response, "2028-03-31")
        _assert_no_tool_errors(response)
        _assert_retrieval_tool(response)
        _assert_cost_cap(response, TIER3_CAP_USD)
        await _judge_and_record(
            response=response,
            user_prompt=prompt,
            rubric=_multi_needle_rubric(
                scenario="S13 1000-doc legal corpus — three markers",
                planted_facts=needle_facts,
            ),
            test_id="S13",
            passing_labels=("grounded",),
            judge_state=judge_state,
            request=request,
        )

    async def test_scenario_14_multi_turn_refinement_500_doc(
        self,
        runtime: KaosRuntime,
        judge_state: dict,
        request: pytest.FixtureRequest,
    ) -> None:
        """S14 — two-turn drill on a 500-doc corpus.

        Turn 1: broad question; agent surfaces a top-level fact.
        Turn 2: anaphoric drill; agent must use retained session memory
        to drill into a related fact.
        """
        sid = _new_session_id("s14")
        broad_needle = "Multi-turn anchor: lead counsel is Patricia Vega."
        narrow_needle = "Patricia Vega bar number: NY-1998-44213."
        # Plant both needles deterministically.
        docs = synth_corpus(
            n_docs=500,
            n_needles=2,
            needle_facts=[broad_needle, narrow_needle],
            seed=1414,
            cluster_topics=["legal"],
            file_types=("docx", "html"),
        )
        await _hydrate_session(runtime, docs, session_id=sid)

        agent = _make_agent(runtime, session_id=sid, max_cost=TIER3_CAP_USD, max_react=20)

        # Turn 1
        prompt_t1 = "Find the name of the lead counsel mentioned in the corpus."
        response_t1 = await agent.turn(prompt_t1, session_id=sid)
        _assert_needle_present(response_t1, "Patricia Vega")
        _assert_no_tool_errors(response_t1)
        _assert_cost_cap(response_t1, TIER3_CAP_USD)

        # Turn 2 — anaphoric ("her bar number")
        prompt_t2 = "What is her bar number?"
        response_t2 = await agent.turn(prompt_t2, session_id=sid)
        _assert_needle_present(response_t2, "NY-1998-44213")
        _assert_no_tool_errors(response_t2)
        _assert_retrieval_tool(response_t2)
        # Both turns combined cost stays under tier cap (separate
        # per-turn enforcement above).
        # Judge gates on the FINAL (turn-2) answer — that's the
        # anaphoric drill that needs the retained-memory grounding.
        await _judge_and_record(
            response=response_t2,
            user_prompt=f"Turn 1: {prompt_t1}\nTurn 2: {prompt_t2}",
            rubric=_grounding_rubric(
                scenario="S14 multi-turn drill — anaphoric bar-number followup",
                planted_fact=narrow_needle,
            ),
            test_id="S14",
            passing_labels=("grounded",),
            judge_state=judge_state,
            request=request,
        )

    async def test_scenario_15_vfs_isolation_under_load(
        self,
        runtime: KaosRuntime,
        judge_state: dict,
        request: pytest.FixtureRequest,
    ) -> None:
        """S15 — three sessions x 100 files each. Cross-session query
        in session A must NOT surface session B's docs.

        This is the VFS / SessionMemory isolation contract under load.
        """
        sid_a = _new_session_id("s15-A")
        sid_b = _new_session_id("s15-B")
        sid_c = _new_session_id("s15-C")

        # Each session has its OWN distinctive needle. We then query
        # session A — the response must mention only session A's needle.
        needle_a = "Session A secret: WIDGET-CODE-AAAA-1111"
        needle_b = "Session B secret: GADGET-CODE-BBBB-2222"
        needle_c = "Session C secret: SPROCKET-CODE-CCCC-3333"

        docs_a = synth_corpus(
            n_docs=100,
            n_needles=1,
            needle_facts=[needle_a],
            seed=15001,
            file_types=("docx", "html"),
        )
        docs_b = synth_corpus(
            n_docs=100,
            n_needles=1,
            needle_facts=[needle_b],
            seed=15002,
            file_types=("docx", "html"),
        )
        docs_c = synth_corpus(
            n_docs=100,
            n_needles=1,
            needle_facts=[needle_c],
            seed=15003,
            file_types=("docx", "html"),
        )
        await _hydrate_session(runtime, docs_a, session_id=sid_a)
        await _hydrate_session(runtime, docs_b, session_id=sid_b)
        await _hydrate_session(runtime, docs_c, session_id=sid_c)

        # Query session A only.
        agent = _make_agent(runtime, session_id=sid_a, max_cost=TIER3_CAP_USD, max_react=18)
        prompt = (
            "What 'secret' code is mentioned in this session's documents? "
            "Quote the exact secret verbatim."
        )
        response = await agent.turn(prompt, session_id=sid_a)
        # MUST contain session A's secret...
        _assert_needle_present(response, "WIDGET-CODE-AAAA-1111")
        # ...and MUST NOT contain B's or C's (isolation contract).
        text = response.text
        for forbidden in ("GADGET-CODE-BBBB-2222", "SPROCKET-CODE-CCCC-3333"):
            if forbidden in text:
                raise AssertionError(
                    f"VFS / session isolation breach: session A's response "
                    f"surfaced {forbidden!r} which belongs to a different "
                    f"session. response head: {text[:300]!r}"
                )
        _assert_no_tool_errors(response)
        _assert_retrieval_tool(response)
        _assert_cost_cap(response, TIER3_CAP_USD)
        await _judge_and_record(
            response=response,
            user_prompt=prompt,
            rubric=_isolation_rubric(
                scenario="S15 VFS isolation under load — 3 sessions x 100 docs",
                planted_fact=needle_a,
                forbidden_facts=[needle_b, needle_c],
            ),
            test_id="S15",
            passing_labels=("grounded",),
            judge_state=judge_state,
            request=request,
        )

    async def test_scenario_25_250_doc_needle_in_wrong_ext(
        self,
        runtime: KaosRuntime,
        judge_state: dict,
        request: pytest.FixtureRequest,
    ) -> None:
        """S25 — 250 docs, needle is in a wrong-extension file.

        Scale (250) plus extension spoof on the one doc that matters.
        Stress retrieval + content-sniff together at modest scale.
        """
        sid = _new_session_id("s25")
        needle_fact = "Scale-spoof needle: GOLDEN-FALCON-PHASE-9"
        base = synth_corpus(
            n_docs=250,
            n_needles=0,
            needle_facts=[],
            seed=2525,
            file_types=("docx", "html", "text"),
            paragraphs_per_doc=4,
        )
        needle_bytes = synth_pdf_text(
            f"PHASE 9 BRIEF\n\n{needle_fact}.\nPhase-gate review scheduled next quarter.",
            title="Phase 9",
        )
        # PDF bytes named .txt.
        _, spoof_name = wrong_extension(needle_bytes, ".txt")
        needle_doc = SynthDoc(
            filename=spoof_name,
            bytes=needle_bytes,
            mime="application/pdf",
            is_needle=True,
            needle_fact=needle_fact,
        )
        # Place needle two-thirds in so a "first 50" cap won't reach it.
        insert_at = (len(base) * 2) // 3
        docs = [*base[:insert_at], needle_doc, *base[insert_at:]]
        await _hydrate_session(runtime, docs, session_id=sid)

        agent = _make_agent(runtime, session_id=sid, max_cost=TIER3_CAP_USD, max_react=20)
        prompt = (
            "About 250 documents are attached. One contains a "
            "'scale-spoof needle' — find and quote it. Some filenames "
            "may have the wrong extension; rely on contents."
        )
        response = await agent.turn(prompt, session_id=sid)
        _assert_needle_present(response, "GOLDEN-FALCON-PHASE-9")
        _assert_no_tool_errors(response)
        _assert_retrieval_tool(response)
        _assert_cost_cap(response, TIER3_CAP_USD)
        await _judge_and_record(
            response=response,
            user_prompt=prompt,
            rubric=_grounding_rubric(
                scenario="S25 250-doc scale + wrong-extension needle",
                planted_fact=needle_fact,
            ),
            test_id="S25",
            passing_labels=("grounded",),
            judge_state=judge_state,
            request=request,
        )

    async def test_scenario_26_500_doc_six_format_mix(
        self,
        runtime: KaosRuntime,
        judge_state: dict,
        request: pytest.FixtureRequest,
    ) -> None:
        """S26 — 500 docs in six formats (PDF/DOCX/HTML/XLSX/JSON/TXT)
        with 3 needles spread across three different formats.

        Exercises the full file-loader matrix at scale: 500 docs x 6
        round-robin types means each loader gets ~80 docs to process.
        The three needles are placed in three different formats so a
        single-loader regression also drops needle count from 3 to 2.
        """
        sid = _new_session_id("s26")
        needle_facts = [
            "Six-format mark 1: confidential project budget $87.4M.",
            "Six-format mark 2: launch readiness 92.5% per QA sign-off.",
            "Six-format mark 3: regulatory window closes 2027-01-31.",
        ]
        docs = synth_corpus(
            n_docs=500,
            n_needles=3,
            needle_facts=needle_facts,
            seed=2626,
            cluster_topics=["finance", "engineering", "regulatory"],
            file_types=("pdf", "docx", "html", "xlsx", "json", "text"),
            paragraphs_per_doc=4,
        )
        await _hydrate_session(runtime, docs, session_id=sid)

        agent = _make_agent(runtime, session_id=sid, max_cost=TIER3_CAP_USD, max_react=25)
        prompt = (
            "Roughly 500 documents in six different formats (PDF, DOCX, "
            "HTML, XLSX, JSON, plain text) are attached. Find all three "
            "'six-format mark' facts and report each with its specific "
            "value (budget figure, readiness percentage, regulatory date)."
        )
        response = await agent.turn(prompt, session_id=sid)
        _assert_needle_present(response, "$87.4M")
        _assert_needle_present(response, "92.5%")
        _assert_needle_present(response, "2027-01-31")
        _assert_no_tool_errors(response)
        _assert_retrieval_tool(response)
        _assert_cost_cap(response, TIER3_CAP_USD)
        await _judge_and_record(
            response=response,
            user_prompt=prompt,
            rubric=_multi_needle_rubric(
                scenario="S26 500-doc six-format mix — three needles",
                planted_facts=needle_facts,
            ),
            test_id="S26",
            passing_labels=("grounded",),
            judge_state=judge_state,
            request=request,
        )

    async def test_scenario_27_1000_doc_five_cluster_five_format(
        self,
        runtime: KaosRuntime,
        judge_state: dict,
        request: pytest.FixtureRequest,
    ) -> None:
        """S27 — 1000 docs across 5 clusters, 5 needles each in a different
        cluster AND a different file type.

        Largest planted-cardinality scenario. Combines pure scale,
        cluster routing, and format diversity. A regression in any of
        triage / loader / retrieval drops the needle count.
        """
        sid = _new_session_id("s27")
        needle_facts = [
            "27-needle finance: cash reserves at $412M held in T-bills.",
            "27-needle legal: governing-law clause amended to Delaware.",
            "27-needle engineering: API rate limit raised to 10k QPS.",
            "27-needle medical: protocol enrolled 312 patients in cohort B.",
            "27-needle regulatory: FDA submission cleared on 2026-04-22.",
        ]
        docs = synth_corpus(
            n_docs=1000,
            n_needles=5,
            needle_facts=needle_facts,
            seed=2727,
            cluster_topics=["finance", "legal", "engineering", "medical", "regulatory"],
            file_types=("pdf", "docx", "html", "xlsx", "text"),
            paragraphs_per_doc=4,
        )
        await _hydrate_session(runtime, docs, session_id=sid)

        agent = _make_agent(runtime, session_id=sid, max_cost=TIER3_CAP_USD, max_react=25)
        prompt = (
            "Roughly 1000 documents span five topical clusters "
            "(finance, legal, engineering, medical, regulatory) and "
            "five file formats. Find the five '27-needle' facts — one "
            "per cluster. For each, report the specific value: cash "
            "reserves figure, governing-law jurisdiction, API rate "
            "limit, patient enrollment count, FDA clearance date."
        )
        response = await agent.turn(prompt, session_id=sid)
        _assert_needle_present(response, "$412M")
        _assert_needle_present(response, "Delaware")
        _assert_needle_present(response, "10k QPS")
        _assert_needle_present(response, "312")
        _assert_needle_present(response, "2026-04-22")
        _assert_no_tool_errors(response)
        _assert_retrieval_tool(response)
        _assert_cost_cap(response, TIER3_CAP_USD)
        await _judge_and_record(
            response=response,
            user_prompt=prompt,
            rubric=_multi_needle_rubric(
                scenario="S27 1000-doc, 5 clusters x 5 formats x 5 needles",
                planted_facts=needle_facts,
            ),
            test_id="S27",
            passing_labels=("grounded",),
            judge_state=judge_state,
            request=request,
        )

    async def test_scenario_28_multi_turn_corpus_growth(
        self,
        runtime: KaosRuntime,
        judge_state: dict,
        request: pytest.FixtureRequest,
    ) -> None:
        """S28 — corpus GROWS between turns.

        Turn 1: 100 docs hydrated; agent answers question A.
        Turn 2: ANOTHER 50 docs hydrated mid-session; agent must answer
        question B that can ONLY be satisfied by the newly-added docs.
        Exercises SessionMemory hydration after a save/reload cycle —
        the F4 path (per-session run_state scoping) is on the critical
        line for this.
        """
        sid = _new_session_id("s28")
        first_needle = "Turn-1 anchor: legal-hold reference number is LH-2026-0901."
        second_needle = "Turn-2 anchor: privilege log row count is 4,128."
        # Round 1 corpus.
        first = synth_corpus(
            n_docs=100,
            n_needles=1,
            needle_facts=[first_needle],
            seed=28001,
            cluster_topics=["legal"],
            file_types=("docx", "html"),
        )
        await _hydrate_session(runtime, first, session_id=sid)

        agent = _make_agent(runtime, session_id=sid, max_cost=TIER3_CAP_USD, max_react=18)
        prompt_t1 = (
            "About 100 legal documents are attached. What is the "
            "'legal-hold reference number'? Quote the exact ID."
        )
        response_t1 = await agent.turn(prompt_t1, session_id=sid)
        _assert_needle_present(response_t1, "LH-2026-0901")
        _assert_no_tool_errors(response_t1)
        _assert_cost_cap(response_t1, TIER3_CAP_USD)

        # Round 2: grow the corpus, then answer a question that only the
        # newly-added docs can satisfy.
        second = synth_corpus(
            n_docs=50,
            n_needles=1,
            needle_facts=[second_needle],
            seed=28002,
            cluster_topics=["legal"],
            file_types=("docx", "html"),
        )
        await _hydrate_session(runtime, second, session_id=sid)
        prompt_t2 = (
            "Additional documents have been added since the last "
            "answer. What is the privilege-log row count? Answer with "
            "the exact number."
        )
        response_t2 = await agent.turn(prompt_t2, session_id=sid)
        _assert_needle_present(response_t2, "4,128")
        _assert_no_tool_errors(response_t2)
        _assert_retrieval_tool(response_t2)
        _assert_cost_cap(response_t2, TIER3_CAP_USD)
        await _judge_and_record(
            response=response_t2,
            user_prompt=f"Turn 1: {prompt_t1}\nTurn 2: {prompt_t2}",
            rubric=_grounding_rubric(
                scenario="S28 multi-turn corpus growth (100 → 150 docs)",
                planted_fact=second_needle,
            ),
            test_id="S28",
            passing_labels=("grounded",),
            judge_state=judge_state,
            request=request,
        )

    async def test_scenario_29_live_web_search_plus_uploaded_xlsx(
        self,
        runtime: KaosRuntime,
        judge_state: dict,
        request: pytest.FixtureRequest,
    ) -> None:
        """S29 — live web search + fetch + uploaded XLSX cross-reference.

        Upload an XLSX with a 'reference rule citation' column. The
        agent must (a) read the XLSX to discover the citation, (b)
        run a web search / fetch to retrieve the live regulation's
        title, and (c) report both. Combines kaos-web tools with
        local file loading at Tier 3 scope.
        """
        sid = _new_session_id("s29")
        xlsx_rows = [
            ["case_number", "rule_citation", "status"],
            ["DOJ-2026-0001", "15 U.S.C. § 1", "open"],
            ["DOJ-2026-0002", "15 U.S.C. § 78j", "review"],
        ]
        docs = [
            SynthDoc(
                filename="caseload.xlsx",
                bytes=synth_xlsx(xlsx_rows, sheet_name="Cases"),
                mime=("application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"),
                is_needle=True,
                needle_fact="caseload references 15 U.S.C. § 1 (Sherman Act)",
            ),
        ]
        await _hydrate_session(runtime, docs, session_id=sid)

        agent = _make_agent(runtime, session_id=sid, max_cost=TIER3_CAP_USD, max_react=18)
        prompt = (
            "An XLSX caseload is attached. For the OPEN case (status = "
            "'open'), the rule_citation column names a US Code section. "
            "(1) Identify the citation. (2) Search the web and report "
            "the common name of the statute that codifies that section. "
            "(3) Cite the source URL you used."
        )
        response = await agent.turn(prompt, session_id=sid)
        _assert_needle_present(response, "15 U.S.C.")
        if "sherman" not in response.text.lower():
            raise AssertionError(
                "agent did not name 'Sherman' (Antitrust Act) as the "
                "statute codifying 15 U.S.C. § 1, even though the open "
                f"case row points to it. response head: {response.text[:300]!r} "
                f"tools: {_tool_names(response)}"
            )
        _assert_no_tool_errors(response)
        _assert_retrieval_tool(response)
        _assert_cost_cap(response, TIER3_CAP_USD)
        await _judge_and_record(
            response=response,
            user_prompt=prompt,
            rubric=_grounding_rubric(
                scenario="S29 uploaded XLSX + live web search Sherman Act name",
                planted_fact="15 U.S.C. § 1 = Sherman Antitrust Act",
            ),
            test_id="S29",
            passing_labels=("grounded",),
            judge_state=judge_state,
            request=request,
        )

    async def test_scenario_30_mega_isolation_five_sessions(
        self,
        runtime: KaosRuntime,
        judge_state: dict,
        request: pytest.FixtureRequest,
    ) -> None:
        """S30 — five sessions x 200 docs each, format-diverse, query each
        in turn and verify zero leakage in any direction.

        Extends S15 (3 x 100, single query) to 5 x 200 + queries of two
        different sessions. The isolation contract must hold under both
        scale AND query-direction variation; an F4 regression that
        leaks the second session's secret into the first would surface
        here even when S15 passes.
        """
        sids = [_new_session_id(f"s30-{i}") for i in range(5)]
        secrets = {
            sids[0]: "Mega-A secret: PALADIN-AAAA-0001",
            sids[1]: "Mega-B secret: KOBOLD-BBBB-0002",
            sids[2]: "Mega-C secret: WIZARD-CCCC-0003",
            sids[3]: "Mega-D secret: RANGER-DDDD-0004",
            sids[4]: "Mega-E secret: CLERIC-EEEE-0005",
        }
        for idx, sid in enumerate(sids):
            docs = synth_corpus(
                n_docs=200,
                n_needles=1,
                needle_facts=[secrets[sid]],
                seed=30000 + idx,
                file_types=("docx", "html", "text"),
                paragraphs_per_doc=3,
            )
            await _hydrate_session(runtime, docs, session_id=sid)

        # Query session 0 and session 2 — proves isolation holds across
        # different querying positions, not just a single arbitrary one.
        for query_sid in (sids[0], sids[2]):
            agent = _make_agent(runtime, session_id=query_sid, max_cost=TIER3_CAP_USD, max_react=18)
            prompt = (
                "What 'secret' code is mentioned in this session's "
                "documents? Quote the exact secret verbatim."
            )
            response = await agent.turn(prompt, session_id=query_sid)
            expected = secrets[query_sid]
            expected_token = expected.rsplit(": ", 1)[-1]
            _assert_needle_present(response, expected_token)
            text = response.text
            for other_sid, other_secret in secrets.items():
                if other_sid == query_sid:
                    continue
                other_token = other_secret.rsplit(": ", 1)[-1]
                if other_token in text:
                    raise AssertionError(
                        f"MEGA isolation breach: session {query_sid} "
                        f"surfaced {other_token!r} which belongs to "
                        f"session {other_sid}. response head: {text[:300]!r}"
                    )
            _assert_no_tool_errors(response)
            _assert_retrieval_tool(response)
            _assert_cost_cap(response, TIER3_CAP_USD)
            # Judge gates on the final query of the pair (sids[2]).
            if query_sid == sids[2]:
                await _judge_and_record(
                    response=response,
                    user_prompt=prompt,
                    rubric=_isolation_rubric(
                        scenario=(
                            "S30 mega isolation — 5 sessions x 200 docs, "
                            "query session C; sibling secrets MUST NOT leak"
                        ),
                        planted_fact=expected,
                        forbidden_facts=[s for k, s in secrets.items() if k != query_sid],
                    ),
                    test_id="S30",
                    passing_labels=("grounded",),
                    judge_state=judge_state,
                    request=request,
                )
