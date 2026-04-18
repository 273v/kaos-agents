"""Agent runtime settings."""

from __future__ import annotations

from kaos_core.config import ModuleSettings
from pydantic import Field
from pydantic_settings import SettingsConfigDict


class KaosAgentSettings(ModuleSettings):
    """Configuration for the kaos-agents runtime.

    Resolved via the standard KAOS hierarchy:
    explicit overrides > context config > KAOS_AGENT_ env vars > .env > defaults.
    """

    # Context budget
    default_context_budget_tokens: int = Field(
        default=16_000,
        gt=0,
        description="Default total token budget for context assembly across all sections.",
    )

    # Persistence
    snapshot_interval_turns: int = Field(
        default=1,
        ge=1,
        description="Persist SNAPSHOT sections every N turns. 1 = every turn.",
    )

    # Session
    max_session_age_hours: int = Field(
        default=168,
        ge=1,
        description="Maximum session age before automatic cleanup (default: 7 days).",
    )

    # Token estimation
    chars_per_token: float = Field(
        default=4.0,
        gt=0,
        description="Characters per token estimate for budget calculations. "
        "Conservative default; adjust per model family.",
    )

    # LLM model defaults
    default_llm_model: str = Field(
        default="anthropic:claude-haiku-4-5",
        description="Default LLM model for agent operations (classify, respond, evaluate). "
        "Use the cheapest current-generation model for routine operations.",
    )
    planning_llm_model: str = Field(
        default="anthropic:claude-haiku-4-5",
        description="LLM model for plan expansion. Same as default unless stronger "
        "reasoning is needed for complex decomposition.",
    )

    # Tool execution
    max_tools: int = Field(
        default=30,
        ge=1,
        description="Maximum tools bridged for ReAct. Performance degrades above ~30.",
    )
    max_react_iterations: int = Field(
        default=10,
        ge=1,
        description="Maximum iterations in a ReAct tool-calling loop.",
    )
    tool_timeout_seconds: float = Field(
        default=60.0,
        gt=0,
        description="Timeout for individual tool invocations.",
    )

    # Planning: route thresholds
    confidence_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Below this confidence, Route triggers REPLAN.",
    )
    deepen_threshold: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Below this confidence, Route triggers DEEPEN instead of REPLAN.",
    )

    # Planning: budget defaults
    plan_max_steps: int = Field(
        default=20,
        ge=1,
        description="Maximum steps in a single plan execution.",
    )
    plan_max_replans: int = Field(
        default=3,
        ge=0,
        description="Maximum replan attempts before STOP_FAILURE (circuit breaker).",
    )
    plan_max_tokens: int = Field(
        default=100_000,
        ge=1,
        description="Maximum tokens across all steps in a plan execution.",
    )
    plan_max_cost_usd: float = Field(
        default=1.0,
        gt=0,
        description="Maximum cost in USD for a single plan execution.",
    )
    plan_max_wall_clock_seconds: float = Field(
        default=120.0,
        gt=0,
        description="Maximum wall-clock time for a single plan execution.",
    )

    # Context retrieval
    retrieval_threshold: int = Field(
        default=20,
        ge=1,
        description="When a memory section has >= this many items, use BM25 "
        "retrieval instead of FIFO for context assembly.",
    )

    # Retrieval: adaptive multi-round
    retrieval_top_k: int = Field(
        default=50,
        ge=1,
        description="Maximum items retrieved per BM25/hybrid round.",
    )
    retrieval_max_rounds: int = Field(
        default=3,
        ge=1,
        description="Maximum retrieval rounds (raw + lexicon + PRF + LLM).",
    )
    retrieval_confidence: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Retrieval judge confidence threshold for stopping.",
    )
    retrieval_prf_top_terms: int = Field(
        default=15,
        ge=1,
        description="Number of top TF terms to extract for PRF/Rocchio expansion.",
    )
    retrieval_prf_docs: int = Field(
        default=10,
        ge=1,
        description="Number of top documents to use for PRF term extraction.",
    )
    retrieval_max_synonym_queries: int = Field(
        default=3,
        ge=1,
        description="Maximum alternative queries from Lexicon synonym expansion.",
    )
    retrieval_llm_timeout_seconds: float = Field(
        default=15.0,
        gt=0,
        description="Timeout for LLM calls in retrieval (HyDE, query expansion).",
    )

    # Doc2Query (document expansion at index time)
    doc2query_enabled: bool = Field(
        default=False,
        description="When true, documents loaded into memory are expanded "
        "with LLM-predicted queries for better BM25 recall. "
        "One-time cost per document (~$0.001 on Haiku).",
    )

    # RAG (Research pattern)
    rag_top_k: int = Field(
        default=25,
        ge=1,
        description="Number of passages retrieved by RAG for document Q&A. "
        "Higher values improve recall on large corpora (60K+ passages) at the cost of "
        "more context tokens. Validated: top_k=25 scored 75% vs top_k=10 at 67% on "
        "116-doc multiformat benchmark.",
    )
    rag_max_retries: int = Field(
        default=2,
        ge=0,
        description="Maximum RAG verification retries before accepting best result.",
    )

    # Planning: adaptive strategy
    complexity_threshold: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="ADaPT complexity threshold. Above → direct execution, below → decompose.",
    )
    simple_goal_word_threshold: int = Field(
        default=15,
        ge=1,
        description="Goals with fewer words than this are assessed as simple.",
    )

    model_config = SettingsConfigDict(
        env_prefix="KAOS_AGENT_",
        env_file=".env",
        extra="ignore",
    )


# Default model string for use as function parameter defaults.
# Derived from KaosAgentSettings so there's a single source of truth.
DEFAULT_MODEL: str = KaosAgentSettings.model_fields["default_llm_model"].default
