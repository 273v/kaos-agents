"""ToolFitnessSignature — pre-ReAct tool-catalog narrowing.

M1 from `kaos-modules/docs/plans/2026-05-19-agentic-failure-taxonomy.md`.

Context. The 2026-05-19 kaos-agents-bench baseline showed
``gpt-5.4-mini`` correctly classifies factual questions as
``tool_use`` but then fires zero tool spans when the registered
catalog is large (~80 tools): the worker LLM short-circuits into a
text response instead of dispatching anything. Claude Sonnet 4.6
handles the same prompt — so the failure is model-dependent and
catalog-size-dependent, not a prompt bug.

The mitigation is a **pre-ReAct fitness ranker** that consumes the
full registered catalog plus the user's query and returns a small
top-K of fit candidates. The downstream ReAct call then sees only
the narrowed catalog, which empirically pushes weaker reasoners
into the tool-call branch.

This Signature is deliberately **catalog-agnostic**: it carries no
hardcoded tool names, no per-domain rules, no per-vertical
heuristics. The classifier reads the catalog at call time and
ranks. Same primitive serves senator-questions, regulatory
questions, document Q&A, and any future domain — because the
catalog itself is the only domain-specific signal.

Mirrors the docstring discipline of
:class:`~kaos_agents.intent.signature.IntentSignature` and
:class:`~kaos_agents.context.classify.ClassifyIntentSignature`:
inputs are typed, the class docstring is the prompt, the prompt
references the dynamic catalog field rather than baking tool
names in.

Integration plan (deferred — this module ships the primitive only):

* Called from :func:`kaos_agents.runtime.agent.BaseAgent.turn` after
  :class:`ClassifyIntentSignature` returns ``tool_use`` /
  ``research`` / ``plan`` and BEFORE the ReAct/Plan dispatcher
  bridges tools into kaos-llm-core.
* Narrows the worker's ``KaosTool[]`` bridge to ``picks`` plus the
  ``mandatory`` set (memory / corpus / handoff tools the worker
  always needs).
* Bypassed entirely when the catalog is already small
  (``len(catalog) <= settings.tool_fitness_bypass_threshold``,
  default 10) — the ranker is overhead for tiny catalogs.
* The picks list is also emitted as a
  :class:`~kaos_agents.events.lifecycle.IntentClassified`-adjacent
  span event for observability so the bench harness can assert
  "ranker chose web tools for senator query" without parsing
  assistant prose.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from kaos_llm_core.signatures.fields import InputField, OutputField
from kaos_llm_core.signatures.signature import Signature

logger = logging.getLogger("kaos.agents.planning.tool_fitness")


@dataclass(frozen=True, slots=True)
class ToolFitnessResult:
    """Outcome of one :func:`rank_tools_for_query` call.

    Frozen-slotted to match the rest of kaos-agents' value-type
    convention. ``fell_back`` is true when the ranker errored or
    produced unusable output and the caller should use the
    unranked catalog. ``valid_picks`` filters ``picks`` against
    the catalog names supplied at call time so the integration
    layer cannot accidentally trust a hallucinated tool name.
    """

    picks: tuple[str, ...]
    valid_picks: tuple[str, ...]
    rationale: str
    confidence: float
    cost_usd: float
    latency_ms: float
    fell_back: bool


def _render_catalog(items: Sequence[tuple[str, str]]) -> str:
    """Render a list of ``(name, description)`` pairs into the
    newline-shape :class:`ToolFitnessSignature` expects."""
    return "\n".join(f"{name}: {desc}" for name, desc in items if name)


async def rank_tools_for_query(
    *,
    query: str,
    catalog: Sequence[tuple[str, str]],
    model: str,
    recent_messages: str = "",
    top_k: int = 5,
    timeout_s: float = 8.0,
) -> ToolFitnessResult:
    """Run :class:`ToolFitnessSignature` and return a typed result.

    Lazy-imports kaos-llm-core so a kaos-agents install without the
    ``[llm]`` extra keeps loading — this matches the policy.py
    pattern. Any exception (provider error, schema parse failure,
    invalid pick names) collapses to ``fell_back=True`` with the
    full unranked catalog implied for the caller.
    """
    t_start = time.monotonic()
    try:
        from kaos_llm_core.programs.call import Call
    except ImportError as exc:
        logger.warning(
            "ToolFitness: kaos-llm-core not installed; falling back. err=%s",
            exc,
        )
        return ToolFitnessResult(
            picks=(),
            valid_picks=(),
            rationale="kaos-llm-core not installed",
            confidence=0.0,
            cost_usd=0.0,
            latency_ms=0.0,
            fell_back=True,
        )

    catalog_text = _render_catalog(catalog)
    valid_names = {name for name, _ in catalog if name}
    call = Call(ToolFitnessSignature, model=model)
    try:
        invocation = await asyncio.wait_for(
            call.invoke(
                query=query,
                tool_catalog=catalog_text,
                recent_messages=recent_messages,
                top_k=top_k,
            ),
            timeout=timeout_s,
        )
    except TimeoutError as exc:
        logger.warning(
            "ToolFitness: invocation timed out after %.1fs model=%s err=%s",
            timeout_s,
            model,
            exc,
        )
        return ToolFitnessResult(
            picks=(),
            valid_picks=(),
            rationale=f"invocation timed out after {timeout_s:.1f}s",
            confidence=0.0,
            cost_usd=0.0,
            latency_ms=(time.monotonic() - t_start) * 1000,
            fell_back=True,
        )
    except Exception as exc:
        logger.warning("ToolFitness: invocation failed model=%s err=%s", model, exc)
        return ToolFitnessResult(
            picks=(),
            valid_picks=(),
            rationale=f"invocation error: {type(exc).__name__}",
            confidence=0.0,
            cost_usd=0.0,
            latency_ms=(time.monotonic() - t_start) * 1000,
            fell_back=True,
        )

    output: Any = invocation.output
    raw_picks_seq: Any = getattr(output, "picks", ())
    rationale: str = str(getattr(output, "rationale", "") or "")
    try:
        confidence = float(getattr(output, "confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    usage = getattr(invocation, "usage", None)
    cost_usd = float(getattr(usage, "cost_usd", 0.0) or 0.0) if usage is not None else 0.0
    latency_ms = (time.monotonic() - t_start) * 1000

    # Coerce iterable → tuple[str, ...].
    picks: tuple[str, ...] = tuple(str(p) for p in (raw_picks_seq or ()))
    valid_picks: tuple[str, ...] = tuple(p for p in picks if p in valid_names)
    fell_back = bool(picks) and not valid_picks
    if fell_back:
        logger.warning(
            "ToolFitness: model %s emitted %d picks, all unknown to catalog",
            model,
            len(picks),
        )
    return ToolFitnessResult(
        picks=picks,
        valid_picks=valid_picks,
        rationale=rationale,
        confidence=confidence,
        cost_usd=cost_usd,
        latency_ms=latency_ms,
        fell_back=fell_back,
    )


class ToolFitnessSignature(Signature):
    """Rank tools in the catalog by fit to the user's query.

    You are a tool-selection assistant for an agentic system that
    has a large catalog of typed tools registered THIS turn. Your
    job is to read the user's query plus the catalog and return a
    ranked top-K of the tools that best fit the query's domain
    and intent. A downstream worker LLM will then call one or more
    of these tools.

    Decision rules:

    1. Read ``query`` and ``tool_catalog`` together. Each non-empty
       line of ``tool_catalog`` is one tool in the shape
       ``"<name>: <one-sentence description>"``. The first token
       before ``:`` is the tool's canonical name and is the only
       value you may emit in ``picks``.

    2. Rank by **domain fit, not name keyword overlap**. A tool
       whose description mentions the user's domain (e.g. a
       web-search tool for a "current officeholder" question, a
       PDF-parser for an "attached document" question) is a fit;
       a tool whose name contains a keyword from the query but
       whose description targets a different domain is not.

    3. The user's query may target a *current real-world entity*
       (a current officeholder, a current price, a current
       version of a rule, the current status of a public filing).
       For such queries, web-search-class tools are almost always
       a fit; specialized-source tools (SEC filings, federal regs,
       case law, entity-identifier registries) are a fit only when
       the entity's domain matches the source. Do NOT pick a
       specialized-source tool just because the query mentions a
       legal-adjacent noun if the question is really about a
       current officeholder.

    4. Emit ``top_k`` tools at most. When the catalog has multiple
       tools that legitimately fit (e.g. several web-search
       backends), include them — the worker picks the actual
       one. When fewer tools fit than ``top_k`` requests, return
       a shorter list rather than padding with poor fits.

    5. Every name in ``picks`` MUST appear verbatim in
       ``tool_catalog`` as the leading token of some non-empty
       line. Do NOT invent tool names. Do NOT abbreviate. Do NOT
       guess at tool names that "should" exist — the bench
       harness rejects unknown names and the integration layer
       falls back to the unranked catalog when the ranker
       produces invalid output.

    6. The catalog may be empty (no tools registered this turn).
       In that case return ``picks=()`` — the worker has nothing
       to dispatch and the AgenticLoop will fall through to a
       text response or a refusal.

    7. When the query is conversational small-talk that no tool
       can usefully address (greeting, acknowledgement, opinion
       request), return ``picks=()`` even when the catalog is
       large. The ``ClassifyIntentSignature`` should have caught
       this upstream by routing to ``respond``, but the ranker
       is a safety net.

    8. ``rationale`` is one short sentence (≤ 30 words) that names
       the DOMAIN you matched against (e.g. "web search for
       current US officeholder", "federal-regulations source for
       emissions-rule lookup", "office-document parser for the
       attached DOCX"). Do NOT enumerate individual tool names in
       the rationale — that information is already in ``picks``.

    9. ``confidence`` is your self-rating of the ranking, 0.0 to
       1.0. Reserve ``>=0.9`` for queries where the domain is
       unambiguous AND the catalog clearly contains fitting
       tools. Reserve ``<=0.3`` for queries where the catalog
       contains nothing that obviously fits — the worker will
       then proceed cautiously or refuse.

    Anti-patterns (do NOT do these — they are the documented
    failure modes from the 2026-05-19 incident review):

    * Picking ``kaos-source-edgar`` for a "current US senator"
      question because both touch federal-government nouns.
      EDGAR is SEC corporate filings; senator questions need
      web search.
    * Picking ``kaos-source-gleif`` for a question about a
      person's role. GLEIF is legal-entity identifiers for
      companies; it does not cover natural persons.
    * Picking ``kaos-source-ecfr`` for an emission-regulation
      question UNLESS the catalog has no Federal-Register tool
      that better matches "current rule" semantics; eCFR is the
      *consolidated* rule, FR is the *latest* rule.
    * Returning an empty ``picks`` just because the catalog is
      large; "I don't know which to use" is a worker-side
      decision, not a ranker-side decision.

    **0.1.1 (#549.A) — atomic-over-composite preference for tool
    families with a "profile" / "summary" / "domain-intel" rollup.**
    Some tool families ship a *composite* rollup tool (e.g.
    ``kaos-web-domain-profile`` bundles DNS + WHOIS + TLS + HTTP
    headers + DKIM/DMARC/SPF under one call) alongside the
    *atomic* tools the rollup wraps (``kaos-web-dns-enumerate``,
    ``kaos-web-whois-lookup``, ``kaos-web-service-detect``,
    ``kaos-web-dns-security``, ...). When the user's query targets
    a single atomic axis (e.g. "what is the IP address of
    example.com" → DNS only; "who registered example.com" → WHOIS
    only), PREFER the atomic tool over the composite. Worked
    example (WU-K v2 Case E6, the documented failure mode):

      Query: "what is the IP address of example.com"
      Catalog includes ``kaos-web-domain-profile`` AND
      ``kaos-web-dns-enumerate``.
      RIGHT: ``picks=["kaos-web-dns-enumerate", "kaos-web-domain-profile"]``
      WRONG: ``picks=["kaos-web-domain-profile", "kaos-web-dns-enumerate"]``

      Reason: ``kaos-web-domain-profile`` is a heavy composite
      with multiple sub-failure modes (any of DNS / WHOIS / TLS
      can raise and the whole call surfaces as
      ``BaseExceptionGroup``). When the query needs only DNS, the
      atomic tool is the cheaper, lower-risk fit. The composite
      is appropriate ONLY when the query genuinely needs ≥2 of
      its axes (e.g. "give me a security snapshot of example.com",
      "is example.com legitimate", "show me everything about
      example.com").

    Identify composite tools by the words ``profile``, ``summary``,
    ``snapshot``, ``intel``, or ``overview`` in the description, OR
    when the description enumerates ≥3 axes ("DNS, WHOIS, TLS, ..."
    -shape). Atomic tools have one-axis descriptions.
    """

    # inputs
    query: str = InputField(
        description=(
            "The user's message verbatim. The classifier has already "
            "decided this turn requires tools; your job is which tools."
        ),
    )
    tool_catalog: str = InputField(
        description=(
            "Newline-separated catalogue of every tool registered on "
            "the runtime THIS turn, in the shape "
            '``"<name>: <one-sentence description>"`` per line. '
            "The catalogue is rendered by "
            ":func:`kaos_agents.context.tool_catalog.render_tool_catalog` "
            "in ``flat`` mode at ``FieldSet.MEDIUM``. Empty string "
            "means no tools are registered."
        ),
    )
    recent_messages: str = InputField(
        default="",
        description=(
            "Flattened recent-conversation context (same shape as "
            ":class:`IntentSignature.recent_messages`). Used to "
            "resolve anaphoric tool-selection: a follow-up "
            "'summarize that one' inherits the corpus context of "
            "the prior turn so the ranker can prefer document-"
            "parsing tools. Empty string keeps behavior unchanged."
        ),
    )
    top_k: int = InputField(
        default=5,
        ge=1,
        le=20,
        description=(
            "Upper bound on the number of tools to return. Caller "
            "tunes per pattern: 3 for cheap models on simple "
            "questions, up to 10 when the worker is expected to "
            "compose multiple tools."
        ),
    )

    # outputs
    picks: tuple[str, ...] = OutputField(
        default=(),
        description=(
            "Ranked top-K tool names, most-fit first. Each value "
            "MUST appear verbatim as the leading token of some "
            "non-empty line in ``tool_catalog``. Length is between "
            "0 and ``top_k``; empty means 'no tool in the catalog "
            "fits this query'."
        ),
    )
    rationale: str = OutputField(
        default="",
        description=(
            "One short sentence (≤ 30 words) naming the DOMAIN "
            "matched. See rule 8 — the rationale is for human/audit "
            "review, not for the downstream worker; the worker "
            "only sees the narrowed catalog itself."
        ),
    )
    confidence: float = OutputField(
        default=1.0,
        ge=0.0,
        le=1.0,
        description=(
            "Self-rating of the ranking, 0.0 to 1.0. See rule 9 for the calibration scale."
        ),
    )
