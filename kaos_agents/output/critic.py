"""Deliverable-layer critic — Protocol + rubric-grounded implementation.

Lifts the rubric-judge primitive (per-criterion PASS/FAIL) from the
benchmark layer to the deliverable layer. Three pieces:

* :class:`DeliverableCritic` Protocol — minimal interface every
  critic implements: ``score(deliverable, rubric) ->
  DeliverableVerdict``.
* :class:`DeliverableVerdict` — aggregate verdict + per-criterion
  breakdown + total cost.
* :class:`RubricDeliverableCritic` — implementation that runs
  :func:`rubric_judge` over each criterion in fan-out
  (asyncio.gather, bounded concurrency).

The rubric input is a sequence of :class:`RubricCriterion` (``id``,
``description``, ``match_criteria``, ``weight``); the critic renders
the deliverable to markdown and submits each criterion to
``rubric_judge`` against that markdown.

Cross-family invariant
----------------------
Per the architecture-audit synthesis (and the academic review):
same-family critics inflate apparent pass rates via self-preference
bias. The critic accepts an optional ``producer_model`` argument and
warns at construction time if ``critic_model`` and ``producer_model``
share a model family. Hard-fail is a future option; warning-only
keeps the API forgiving for benchmarks where the producer/critic
split isn't a hard production constraint.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from kaos_core.logging import get_logger

from kaos_agents.benchmarks.rubric_judge import rubric_judge

if TYPE_CHECKING:
    from kaos_agents.output.types import Deliverable

logger = get_logger(__name__)


# Default concurrency for the rubric fan-out. Providers rate-limit
# JSON-mode aggressively; 5 finishes a 55-criterion rubric in ~10s
# while staying within Anthropic's per-minute caps for haiku-4-5.
_DEFAULT_RUBRIC_CONCURRENCY = 5


@dataclass(frozen=True, slots=True)
class RubricCriterion:
    """One PASS/FAIL criterion the deliverable must satisfy.

    Mirrors the Harvey LAB ``task.json`` shape but lives at the
    kaos-agents layer so callers can construct rubrics
    programmatically (without going through the Harvey fixture).

    Attributes:
        id: Stable identifier (e.g., ``"C-001"``).
        description: Short title shown to the judge.
        match_criteria: PASS/FAIL rubric text. Format:
            ``"PASS if <condition>. FAIL if <condition>."``.
        weight: Aggregation weight. Default 1.0; use 0.0 to make a
            criterion observability-only (won't affect pass rate).
    """

    id: str
    description: str
    match_criteria: str
    weight: float = 1.0


@dataclass(frozen=True, slots=True)
class CriterionResult:
    """Verdict for one criterion.

    Composes the typed rubric_judge output (``passed`` / ``confidence`` /
    ``reasoning``) with the criterion id and per-call cost. This is
    the minimal record callers need to attribute pass/fail to specific
    criteria + show cost in benchmark dashboards.
    """

    criterion_id: str
    passed: bool
    confidence: float
    reasoning: str
    cost_usd: float = 0.0


@dataclass(frozen=True, slots=True)
class DeliverableVerdict:
    """Aggregate verdict from a critic over a full rubric.

    ``per_criterion`` carries the typed per-criterion record;
    ``passed`` / ``failed`` accessors filter that list.
    ``weighted_pass_rate`` is the sum of weights of passed criteria
    divided by total weight (Harvey BigLaw Bench shape, generalised
    to weighted sums).
    """

    per_criterion: tuple[CriterionResult, ...]
    weighted_pass_rate: float
    pooled_pass_rate: float
    total_cost_usd: float
    n_passed: int
    n_failed: int
    n_judge_unavailable: int

    @property
    def passed_criteria(self) -> tuple[CriterionResult, ...]:
        """Subset of per_criterion where the verdict was PASS."""
        return tuple(c for c in self.per_criterion if c.passed)

    @property
    def failed_criteria(self) -> tuple[CriterionResult, ...]:
        """Subset where verdict was FAIL (excludes judge-unavailable)."""
        return tuple(
            c for c in self.per_criterion if not c.passed and "judge unavailable" not in c.reasoning
        )

    @property
    def unavailable_criteria(self) -> tuple[CriterionResult, ...]:
        """Subset where the judge failed to return a verdict."""
        return tuple(c for c in self.per_criterion if "judge unavailable" in c.reasoning)


@runtime_checkable
class DeliverableCritic(Protocol):
    """Minimal contract every deliverable critic implements.

    Implementations decide HOW to score; the Protocol fixes only the
    shape. :class:`RubricDeliverableCritic` is the rubric-grounded
    implementation; future critics could ground in regex extraction
    + structural checks (no-LLM tier) or in a stronger
    cross-family judge.
    """

    async def score(
        self,
        deliverable: Deliverable,
        rubric: Sequence[RubricCriterion],
    ) -> DeliverableVerdict:
        """Score a deliverable against a rubric — return the typed verdict."""


class RubricDeliverableCritic:
    """Rubric-grounded critic — runs :func:`rubric_judge` per criterion.

    Implementation: render the deliverable to markdown, fan out each
    criterion through ``rubric_judge`` under bounded concurrency,
    aggregate verdicts.

    Cross-family invariant: optional warn-only check at construction
    time. Set ``producer_model`` to enable; pass ``strict=True`` to
    raise instead of warn when the families collide.
    """

    def __init__(
        self,
        *,
        critic_model: str = "anthropic:claude-haiku-4-5",
        producer_model: str | None = None,
        strict: bool = False,
        concurrency: int = _DEFAULT_RUBRIC_CONCURRENCY,
    ) -> None:
        self._model = critic_model
        self._concurrency = max(1, concurrency)
        if producer_model is not None:
            self._enforce_family_separation(critic_model, producer_model, strict=strict)

    @staticmethod
    def _enforce_family_separation(critic_model: str, producer_model: str, *, strict: bool) -> None:
        """Warn (or raise) when critic + producer share a model family.

        Family is the prefix before the colon in our model id format
        (``anthropic:claude-haiku-4-5`` → ``anthropic``). The
        academic review flagged this as the configuration the field
        most consistently shows fails (Huang ICLR 2024, Stechly 2023,
        Preference Leakage 2025).
        """
        critic_family = critic_model.split(":", 1)[0].split("-", 1)[0]
        producer_family = producer_model.split(":", 1)[0].split("-", 1)[0]
        if critic_family == producer_family:
            msg = (
                f"RubricDeliverableCritic: critic_model='{critic_model}' shares the "
                f"'{critic_family}' family with producer_model='{producer_model}'. "
                "Same-family critics inflate apparent pass rates via self-preference "
                "bias (Panickssery 2024, Li 2025 Preference Leakage). "
                "Pass a different family critic, or set strict=False to suppress."
            )
            if strict:
                raise ValueError(msg)
            logger.warning(msg)

    async def score(
        self,
        deliverable: Deliverable,
        rubric: Sequence[RubricCriterion],
    ) -> DeliverableVerdict:
        """Score a deliverable against a rubric.

        Renders the deliverable via :meth:`Deliverable.to_markdown`
        once, then fans out :func:`rubric_judge` per criterion under
        a semaphore-bounded gather. Verdicts are sorted back to the
        rubric's input order.
        """
        deliverable_md = deliverable.to_markdown()
        sem = asyncio.Semaphore(self._concurrency)

        async def _one(c: RubricCriterion) -> CriterionResult:
            async with sem:
                verdict = await rubric_judge(
                    criterion_id=c.id,
                    criterion_title=c.description,
                    match_criteria=c.match_criteria,
                    deliverable=deliverable_md,
                    model=self._model,
                )
            return CriterionResult(
                criterion_id=str(verdict.get("criterion_id", c.id)),
                passed=bool(verdict.get("passed", False)),
                confidence=float(verdict.get("confidence", 0.0) or 0.0),
                reasoning=str(verdict.get("reasoning", "") or ""),
                cost_usd=float(verdict.get("judge_cost_usd", 0.0) or 0.0),
            )

        # Preserve rubric order in the returned tuple. asyncio.gather
        # returns results in the same order as awaitables, so a
        # straightforward gather works.
        results = await asyncio.gather(*(_one(c) for c in rubric))

        # Per-rubric weight map for the weighted pass rate.
        weights = {c.id: c.weight for c in rubric}

        n_passed = sum(1 for r in results if r.passed)
        n_unavailable = sum(1 for r in results if "judge unavailable" in r.reasoning)
        n_failed = len(results) - n_passed - n_unavailable
        total_cost = sum(r.cost_usd for r in results)

        total_weight = sum(weights.values()) or 1.0
        weighted_pass = sum(weights.get(r.criterion_id, 1.0) for r in results if r.passed)
        weighted_pass_rate = weighted_pass / total_weight
        pooled = (n_passed / len(results)) if results else 0.0

        return DeliverableVerdict(
            per_criterion=tuple(results),
            weighted_pass_rate=weighted_pass_rate,
            pooled_pass_rate=pooled,
            total_cost_usd=total_cost,
            n_passed=n_passed,
            n_failed=n_failed,
            n_judge_unavailable=n_unavailable,
        )


__all__ = [
    "CriterionResult",
    "DeliverableCritic",
    "DeliverableVerdict",
    "RubricCriterion",
    "RubricDeliverableCritic",
]
