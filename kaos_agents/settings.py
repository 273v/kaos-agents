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
        description="Default total token budget for context assembly across all sections.",
    )

    # Persistence
    snapshot_interval_turns: int = Field(
        default=1,
        description="Persist SNAPSHOT sections every N turns. 1 = every turn.",
    )

    # Session
    max_session_age_hours: int = Field(
        default=168,
        description="Maximum session age before automatic cleanup (default: 7 days).",
    )

    # Token estimation
    chars_per_token: float = Field(
        default=4.0,
        description="Characters per token estimate for budget calculations. "
        "Conservative default; adjust per model family.",
    )

    model_config = SettingsConfigDict(
        env_prefix="KAOS_AGENT_",
        env_file=".env",
        extra="ignore",
    )
