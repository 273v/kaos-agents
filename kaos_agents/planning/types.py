"""Planning type system — the data contracts between primitives.

Every planning primitive produces and consumes these types. They are the
stable interface that strategies compose over. Adding a new strategy never
requires changing these types.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import StrEnum, unique
from typing import Any

# ---------------------------------------------------------------------------
# Step — a single unit of work in a plan
# ---------------------------------------------------------------------------


@unique
class StepType(StrEnum):
    """What kind of action a step represents."""

    TOOL = "tool"  # Call a registered KaosTool
    LLM = "llm"  # Make an LLM call (Call/ReAct/RAG)
    SUBPLAN = "subplan"  # Recursive: expand into a sub-graph


@unique
class StepStatus(StrEnum):
    """Execution lifecycle of a plan step."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass(frozen=True, slots=True)
class Step:
    """A single step in a plan graph.

    Immutable after creation. Status updates happen on the PlanGraph
    node properties, not by mutating Step instances.
    """

    id: str
    step_type: StepType
    description: str
    tool_name: str | None = None  # Required for TOOL type
    input_spec: dict[str, Any] = field(default_factory=dict)
    expected_output: str = ""
    depends_on: tuple[str, ...] = ()  # Step IDs this depends on
    estimated_latency_ms: int = 0


# ---------------------------------------------------------------------------
# Judgment — the output of Evaluate
# ---------------------------------------------------------------------------


@unique
class EvalMode(StrEnum):
    """How a judgment was produced."""

    STRUCTURAL = "structural"  # Deterministic check, no LLM
    SEMANTIC = "semantic"  # LLM-based judgment


@dataclass(frozen=True, slots=True)
class Judgment:
    """The result of an Evaluate primitive call.

    Produced by both structural checks (tool exists? schema matches?)
    and semantic evaluation (is this result good enough?).
    """

    matched: bool  # Did the result meet expectations?
    confidence: float  # 0.0 to 1.0
    reasoning: str  # Why this judgment was made
    mode: EvalMode  # How the judgment was produced
    evidence: tuple[str, ...] = ()  # Specific inputs that drove the judgment
    new_facts: tuple[str, ...] = ()  # Facts extracted from the result
    surprises: tuple[str, ...] = ()  # Unexpected observations


# ---------------------------------------------------------------------------
# Decision — the output of Route
# ---------------------------------------------------------------------------


@unique
class Decision(StrEnum):
    """Control flow decisions from the Route primitive."""

    CONTINUE = "continue"  # Proceed to next ready step
    REPLAN = "replan"  # Current plan is invalid, re-expand
    DEEPEN = "deepen"  # Current step too complex, expand into substeps
    ESCALATE = "escalate"  # Ask user for input
    STOP_SUCCESS = "stop_success"  # Goal achieved
    STOP_BUDGET = "stop_budget"  # Budget exhausted
    STOP_FAILURE = "stop_failure"  # Unrecoverable after max replans


@dataclass(frozen=True, slots=True)
class RouteResult:
    """A routing decision with its rationale."""

    decision: Decision
    reason: str
    step_id: str | None = None  # Which step triggered this decision


# ---------------------------------------------------------------------------
# PlanBudget — tracks resource consumption during plan execution
# ---------------------------------------------------------------------------


@unique
class StopReason(StrEnum):
    """Why plan execution stopped."""

    SUCCESS = "success"
    MAX_COST = "max_cost"
    MAX_TOKENS = "max_tokens"
    MAX_STEPS = "max_steps"
    MAX_REPLANS = "max_replans"
    MAX_WALL_CLOCK = "max_wall_clock"
    FAILURE = "failure"
    NEEDS_REPLAN = "needs_replan"  # Route decided REPLAN — strategy should re-expand
    ESCALATED = "escalated"


@dataclass(slots=True)
class PlanBudget:
    """Tracks and enforces resource limits during plan execution.

    Mutable — accumulated counters are updated by Act and Compose.
    This is one of the few mutable types in the planning system.
    """

    # Limits (set at creation, immutable by convention)
    max_cost_usd: float = 1.0
    max_tokens: int = 100_000
    max_steps: int = 20
    max_replans: int = 3
    max_wall_clock_seconds: float = 120.0

    # Accumulated (updated during execution)
    cost_usd: float = 0.0
    tokens_used: int = 0
    steps_executed: int = 0
    replans: int = 0
    start_time: float = field(default_factory=time.time)

    def should_stop(self) -> StopReason | None:
        """Check if any budget limit has been exceeded.

        Returns the StopReason if a limit is hit, None otherwise.
        """
        if self.cost_usd >= self.max_cost_usd:
            return StopReason.MAX_COST
        if self.tokens_used >= self.max_tokens:
            return StopReason.MAX_TOKENS
        if self.steps_executed >= self.max_steps:
            return StopReason.MAX_STEPS
        if self.replans >= self.max_replans:
            return StopReason.MAX_REPLANS
        if time.time() - self.start_time >= self.max_wall_clock_seconds:
            return StopReason.MAX_WALL_CLOCK
        return None

    def record_step(self, cost_usd: float = 0.0, tokens: int = 0) -> None:
        """Record a completed step's resource usage."""
        self.steps_executed += 1
        self.cost_usd += cost_usd
        self.tokens_used += tokens

    def record_replan(self) -> None:
        """Record a replan event."""
        self.replans += 1


# ---------------------------------------------------------------------------
# PrimitiveTrace — observability record for each primitive invocation
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PrimitiveTrace:
    """Structured trace record for a single primitive invocation.

    Collected by Compose for production observability. Feeds dashboards,
    not just narrative logs.
    """

    primitive: str  # "evaluate", "expand", "act", "route", etc.
    step_id: str | None = None
    wall_clock_ms: float = 0.0
    token_count: int = 0
    cost_usd: float = 0.0
    success: bool = True
    details: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# ComposeResult — the final output of plan execution
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ComposeResult:
    """The result of Compose executing a plan graph."""

    plan_json: str  # Serialized plan graph (for persistence/debugging)
    traces: tuple[PrimitiveTrace, ...] = ()
    stop_reason: StopReason = StopReason.SUCCESS
    total_cost_usd: float = 0.0
    total_tokens: int = 0
    steps_executed: int = 0
    replans: int = 0
    wall_clock_ms: float = 0.0
    step_results: dict[str, Any] = field(default_factory=dict)  # step_id -> result
