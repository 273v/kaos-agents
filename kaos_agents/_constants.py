"""Shared constants for kaos-agents.

Module-level constants used across multiple files. Centralizes values that
would otherwise be duplicated as magic numbers in function bodies.

These are implementation constants, not user-configurable settings.
For user-configurable values, see ``KaosAgentSettings`` in ``settings.py``.
"""

from __future__ import annotations

# --- Truncation limits for LLM context and log display ---

# Stage B4: RESULT_SUMMARY_TRUNCATE + EVAL_RESULT_MAX_CHARS promoted to
# KaosAgentSettings.result_summary_max_chars / .eval_result_max_chars
# (env override: KAOS_AGENT_*) per the no-hardcoded-caps plan principle 4.

REFLECTION_GOAL_TRUNCATE = 100
"""Max chars for goal text in reflection memory entries."""

SYNTHESIS_RESULT_TRUNCATE = 300
"""Max chars for per-step results in plan synthesis."""

EVAL_CONTEXT_MAX_CHARS = 1000
"""Max chars for additional context passed to the semantic evaluator LLM."""

EXPAND_CONTEXT_MAX_CHARS = 3000
"""Max chars for context passed to the plan-expansion LLM."""

ROLLING_RESULT_PREVIEW = 100
"""Max chars for step-result preview in rolling-horizon accumulated context."""

# --- Fallback/heuristic values ---

FALLBACK_RECENT_MESSAGES = 10
"""Number of recent messages to use when structured context_items is unavailable."""

PLAN_PRIOR_REFLECTION_COUNT = 3
"""Number of recent reflection items to include in plan prior-failure context."""

DEFAULT_MEMORY_QUERY_LIMIT = 20
"""Default item limit for memory-query MCP tool."""

# --- Complexity assessment thresholds ---

COMPLEX_MIN_PERIODS = 2
"""Minimum period count (sentences) to trigger complexity deduction."""

COMPLEX_MIN_COMMAS = 3
"""Minimum comma count to trigger complexity deduction."""

FEW_TOOLS_THRESHOLD = 2
"""Tool count at or below which the 'few tools' bonus applies."""

# --- Heuristic intent classification confidence scores ---

HEURISTIC_CONFIDENCE_GREETING = 0.8
HEURISTIC_CONFIDENCE_RESEARCH = 0.6
HEURISTIC_CONFIDENCE_PLAN = 0.5
HEURISTIC_CONFIDENCE_TOOL_USE = 0.6
HEURISTIC_CONFIDENCE_DEFAULT = 0.4

# --- Summarization ---

SUMMARIZE_MIN_ITEMS = 2
"""Minimum items in a section before summarization is attempted."""

SUMMARIZE_MIN_TARGET_TOKENS = 50
"""Floor for summary target token count."""

SUMMARIZE_BUDGET_DIVISOR = 3
"""Divisor to compute target summary tokens from section budget (budget // N)."""

SUMMARIZE_FALLBACK_TOKENS = 200
"""Target summary tokens when section has unlimited budget."""

FALLBACK_SUMMARY_LINE_MAX = 100
"""Max chars per line in mechanical fallback summarization."""

FALLBACK_LINE_OVERHEAD = 4
"""Token overhead per line in fallback summarization ('- ' prefix + newline + padding)."""

# --- Reranking ---

RERANK_CANDIDATE_K = 100
"""Widen initial retrieval to this many candidates per round before reranking."""

RERANK_FINAL_K = 50
"""After cross-encoder reranking, return this many top results."""
