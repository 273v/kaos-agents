"""LLM-as-judge scorer for Harvey LAB-style rubric criteria.

Sister of :mod:`kaos_agents.benchmarks.llm_judge`. The Q&A judge in
``llm_judge.py`` evaluates "did the agent answer the question correctly?".
This module evaluates "does the agent's deliverable satisfy this PASS/FAIL
rubric criterion?" — the shape that ships in
[Harvey LAB](https://github.com/harveyai/harvey-labs) ``task.json`` files.

Built on the typed kaos-llm-core stack
----------------------------------------
The judge is a :class:`~kaos_llm_core.programs.call.Call` over a typed
:class:`~kaos_llm_core.signatures.signature.Signature`. That gives us:

* **Pydantic-validated I/O.** No hand-built JSON Schema dicts; the
  Signature is the schema. Outputs come back as a typed Pydantic
  instance with ``passed: bool``, ``confidence: float``, ``reasoning:
  str``.
* **Provider-native structured output.** ``Call`` picks the best-fit
  codec for the target model (Anthropic ``output_config.format`` GA
  April 2026, OpenAI ``response_format``, Google native tool, etc.).
* **Trace + cost attribution.** Every judge call appears in the
  benchmark's :class:`Invocation` tree with usage and ``cost_usd``,
  same path as every other Call/Program in the system.
* **Validation-retry built in.** Decode/parse failures retry through
  ``Call._run_pipeline`` instead of crashing the whole rubric fanout.

A rubric criterion looks like::

    {
      "id": "C-001",
      "title": "Identifies anti-assignment clause in Section 14.2",
      "deliverables": ["change-of-control-extraction-report.docx"],
      "match_criteria": "PASS if the report identifies the anti-assignment "
                        "clause in Section 14.2 of the Northland Refining MSA "
                        "and references its prohibition on assignment 'whether "
                        "by operation of law or otherwise' without prior "
                        "written consent. FAIL if the report does not mention "
                        "Section 14.2 or the anti-assignment provision in the "
                        "Northland agreement."
    }

The judge gets the criterion id + title + ``match_criteria`` text + the
agent's verbatim deliverable, and renders a :class:`RubricVerdict`.

Failure mode contract
---------------------
The judge is best-effort scoring infrastructure, not a hard gate.
``CallError`` (provider 5xx, auth), ``ValidationRetryExhaustedError``
(unparseable response), ``ImportError`` (kaos-llm-client not installed),
and any other exception all map to ``passed=False, confidence=0.0,
reasoning="judge unavailable: <why>"``. The benchmark harness can fall
back to a coarser signal (deliverable-exists, length-floor) without
crashing the whole run.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from kaos_core.logging import get_logger
from kaos_llm_core import InputField, OutputField, Signature
from kaos_llm_core.programs.call import Call

from kaos_agents.usage import InvocationUsage

logger = get_logger("benchmarks.rubric_judge")


# Per-criterion deliverable budget. The harness handles truncation
# below; this constant lives here so consumers can override at the
# Signature input level (the InputField itself accepts arbitrary str).
# Sized for ~15K tokens — fits comfortably under any frontier-model
# context window once the system prompt + other inputs are accounted
# for.
_DELIVERABLE_TRUNCATION = 60_000


class RubricVerdictSignature(Signature):
    """Decide whether an agent's deliverable PASSES a single rubric criterion.

    You are an impartial grader for a legal-work benchmark. Your job is
    to decide whether the deliverable satisfies the PASS condition in
    the criterion text below.

    Decision rules:

    1. The criterion text is the ground truth. Do not introduce your
       own criteria. If it says "identifies Section 14.2", the
       deliverable passes iff it identifies Section 14.2 — not "14"
       or "a section about assignment".
    2. Paraphrase is fine. The deliverable does not need to use the
       exact phrase from the criterion as long as the substance is
       present and unambiguous. "no change of control without consent"
       = "requires consent for change of control".
    3. Be strict on specific facts. If the criterion calls for a
       specific section number, party name, dollar amount, percentage,
       or date, those must appear (or be unambiguously identifiable)
       in the deliverable. A deliverable that says "this contract has
       an anti-assignment clause" fails a criterion that requires
       identifying Section 14.2 specifically.
    4. Be lenient on framing. The criterion may say "identifies X" —
       the deliverable passes whether X appears in a bullet list, a
       paragraph, a table cell, or an inline citation. Output format
       does not affect the verdict.
    5. If the criterion has multiple conjunctive requirements ("PASS
       if the report identifies X and references Y"), all must be
       present to pass. If it has disjunctive requirements ("PASS if X
       or Y"), one is enough.
    6. If the deliverable is empty or clearly off-topic (the agent
       errored, refused the task, or wrote about a different matter),
       the criterion fails.
    """

    criterion_id: str = InputField(description="Stable criterion identifier (e.g., 'C-001').")
    criterion_title: str = InputField(description="Short human-readable title for the criterion.")
    match_criteria: str = InputField(
        description=(
            "Full PASS/FAIL rubric text from the task author. Format: "
            "'PASS if <condition>. FAIL if <condition>.'"
        )
    )
    deliverable: str = InputField(
        description=(
            "The agent's verbatim deliverable text to evaluate. "
            "May be truncated by the harness if very long; the judgment "
            "should be made on what's present."
        )
    )

    passed: bool = OutputField(
        description=("Whether the deliverable PASSES the criterion under the decision rules above.")
    )
    confidence: float = OutputField(
        description=(
            "Verdict confidence in [0.0, 1.0]. 1.0 = unambiguous match "
            "or unambiguous miss; 0.5-0.7 = relies on judgment about "
            "what counts as 'identifies' or 'references'."
        ),
        ge=0.0,
        le=1.0,
    )
    reasoning: str = OutputField(
        description=(
            "One or two sentences explaining the verdict. Quote the "
            "specific part of the deliverable that satisfied or failed "
            "the criterion. If the criterion failed, name what was "
            "missing."
        )
    )


@dataclass(frozen=True, slots=True)
class RubricVerdict:
    """Observability wrapper around a typed :class:`RubricVerdictSignature` verdict.

    The verdict itself (``passed`` / ``confidence`` / ``reasoning``) is
    the typed Pydantic output produced by the judge :class:`Call`; we
    don't duplicate those fields here. This dataclass carries:

    * ``criterion_id`` — echoes the input so the harness can map back
      to its rubric without an explicit lookup table.
    * ``verdict`` — the typed Signature output (Pydantic instance).
    * ``judge_*`` — observability fields (model, cost, tokens) for
      cost rollups in the benchmark JSON.

    The convenience properties ``passed`` / ``confidence`` /
    ``reasoning`` delegate to ``verdict`` so callers can read them
    flatly without walking the wrapper. ``to_dict`` flattens the whole
    thing into the legacy shape the benchmark harnesses still consume
    via ``verdict.get("passed", False)`` etc.
    """

    criterion_id: str
    verdict: Any  # Pydantic instance — typed at the call site, dynamic here
    judge_model: str = ""
    judge_cost_usd: float = 0.0
    judge_input_tokens: int = 0
    judge_output_tokens: int = 0

    @property
    def passed(self) -> bool:
        return bool(self.verdict.passed)

    @property
    def confidence(self) -> float:
        return float(self.verdict.confidence)

    @property
    def reasoning(self) -> str:
        return str(self.verdict.reasoning)

    def to_dict(self) -> dict[str, object]:
        """Flatten to the legacy harness shape: dict with verdict +
        observability fields side-by-side. Avoid mutating across calls."""
        return {
            "criterion_id": self.criterion_id,
            "passed": self.passed,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "judge_model": self.judge_model,
            "judge_cost_usd": self.judge_cost_usd,
            "judge_input_tokens": self.judge_input_tokens,
            "judge_output_tokens": self.judge_output_tokens,
        }


def _truncate_deliverable(deliverable: str) -> str:
    """Bound the deliverable to ~15K tokens with a head/tail split."""
    text = deliverable.strip() or "(empty deliverable)"
    if len(text) <= _DELIVERABLE_TRUNCATION:
        return text
    head = text[: _DELIVERABLE_TRUNCATION // 2]
    tail = text[-_DELIVERABLE_TRUNCATION // 2 :]
    elided = len(deliverable) - _DELIVERABLE_TRUNCATION
    return f"{head}\n\n...[deliverable truncated — middle {elided} chars elided]...\n\n{tail}"


async def rubric_judge(
    *,
    criterion_id: str,
    criterion_title: str,
    match_criteria: str,
    deliverable: str,
    model: str = "anthropic:claude-haiku-4-5",
) -> dict:
    """Score one rubric criterion against an agent's deliverable.

    Constructs a typed :class:`Call` over :class:`RubricVerdictSignature`
    and invokes it. Errors map to a benign "judge unavailable" verdict
    so the benchmark fanout never aborts on transient infra failures.

    Parameters
    ----------
    criterion_id, criterion_title, match_criteria, deliverable:
        Rubric criterion + agent's deliverable. See
        :class:`RubricVerdictSignature` for shape.
    model:
        kaos-llm-client model identifier. Defaults to the cheapest
        current Anthropic model (``claude-haiku-4-5``).

    Returns
    -------
    dict
        :meth:`RubricVerdict.to_dict` payload. On failure, a benign
        verdict with ``passed=False, confidence=0.0, reasoning="judge
        unavailable: ..."``.
    """
    try:
        judge_call = Call(RubricVerdictSignature, model=model, max_retries=2)
    except ImportError as exc:
        return _unavailable_verdict(
            criterion_id,
            f"judge unavailable: kaos-llm-client not installed ({exc})",
            model,
        ).to_dict()

    try:
        invocation = await judge_call.invoke(
            criterion_id=criterion_id,
            criterion_title=criterion_title,
            match_criteria=match_criteria,
            deliverable=_truncate_deliverable(deliverable),
        )
    except Exception as exc:  # broad: provider 5xx, auth, validation, etc.
        logger.warning("Rubric judge failed (%s): %s", type(exc).__name__, exc)
        return _unavailable_verdict(
            criterion_id,
            f"judge unavailable: {type(exc).__name__}: {exc}",
            model,
        ).to_dict()

    usage = InvocationUsage.from_invocation(invocation)
    verdict = RubricVerdict(
        criterion_id=criterion_id,
        verdict=invocation.output,  # typed RubricVerdictSignature output
        judge_model=model,
        judge_cost_usd=usage.cost_usd,
        judge_input_tokens=usage.input_tokens,
        judge_output_tokens=usage.output_tokens,
    )
    return verdict.to_dict()


def _unavailable_verdict(
    criterion_id: str,
    reason: str,
    judge_model: str,
) -> RubricVerdict:
    """Build a 'judge unavailable' :class:`RubricVerdict`.

    Constructs a synthetic typed Signature output using the Pydantic
    constructor so the wrapper invariant holds (``verdict`` is always
    a typed instance with ``passed``/``confidence``/``reasoning``).
    """
    synthetic_output = RubricVerdictSignature.model_construct(
        criterion_id=criterion_id,
        criterion_title="",
        match_criteria="",
        deliverable="",
        passed=False,
        confidence=0.0,
        reasoning=reason,
    )
    return RubricVerdict(
        criterion_id=criterion_id,
        verdict=synthetic_output,
        judge_model=judge_model,
    )
