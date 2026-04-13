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
