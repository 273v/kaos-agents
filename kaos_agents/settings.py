"""Agent runtime settings."""

from __future__ import annotations

from typing import Literal

from kaos_core.config import ModuleSettings
from pydantic import Field
from pydantic_settings import SettingsConfigDict

# KC17-P2-1 — "strict" research profile preset values.
#
# These are the thresholds the field-level docstrings already
# recommend "for legal deployments". The profile is a single
# opt-in switch that flips all three at once so a partner can
# set ``KAOS_AGENT_RESEARCH_PROFILE=strict`` (or
# ``research_profile="strict"``) and not have to also know to
# tune ``bm25_score_floor`` + ``verifier_min_confidence`` +
# ``refuse_unverified_answers`` independently.
#
# The constants live at module scope so they are explicit, testable,
# and visible to consumers / audit reviewers without needing to
# instantiate a settings object.
STRICT_PROFILE_DEFAULTS: dict[str, float | bool] = {
    # 1.5 sits at the lower end of the field docstring's
    # "1.0-3.0 for legal corpora" recommendation. Empirically
    # filters spurious lexical-overlap hits on adversarial
    # (out-of-corpus) queries without erasing legitimate weak
    # matches on the right corpus.
    "bm25_score_floor": 1.5,
    # 0.6 sits at the lower end of the docstring's "0.5-0.7
    # recommended floor for legal Q&A" range. Above this the
    # verifier collapses answers to InsufficientEvidence rather
    # than ship a low-confidence claim.
    "verifier_min_confidence": 0.6,
    # When verification fails entirely (citation spans don't
    # resolve in source corpus) strict mode refuses outright
    # rather than ship the answer with a "could not be verified"
    # warning. This is the legal-deployment posture.
    "refuse_unverified_answers": True,
}


class KaosAgentSettings(ModuleSettings):
    """Configuration for the kaos-agents runtime.

    Resolved via the standard KAOS hierarchy:
    explicit overrides > context config > KAOS_AGENT_ env vars > .env > defaults.
    """

    # Context budget — sized for 2026 frontier models (Claude 4.x: 200K
    # input, Gemini 2.5/3: 1M input, GPT-5.x: 200K-400K input). 16K was
    # the GPT-3.5 / Claude 2 era default and silently capped agents that
    # could have used 10x more context. Set to 200K which fits every
    # frontier model's input window with comfortable headroom for the
    # output budget.
    default_context_budget_tokens: int = Field(
        default=200_000,
        gt=0,
        description=(
            "Default total token budget for context assembly across all "
            "sections. Sized for 2026 frontier models (200K context "
            "windows). Override down for cheap/local models."
        ),
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
    research_llm_model: str = Field(
        default="anthropic:claude-sonnet-4-6",
        description=(
            "LLM model for the research pattern (RAG + grounded Q&A). "
            "Defaults to sonnet-grade because retrieval-query quality "
            "scales materially with model strength: gpt-5.4-mini/sonnet-4-6 "
            "write better BM25 queries than haiku-4-5 (observed: top_score "
            "6.4 → 16.9 on identical legal-corpus queries) and aggregate "
            "across passages more reliably. Haiku-grade is still fine for "
            "the chat-pattern smoke-test path."
        ),
    )

    # Tool execution
    max_tools: int = Field(
        default=100,
        ge=1,
        description=(
            "Maximum tools bridged for ReAct. Frontier models in 2026 "
            "handle large tool inventories well; 100 covers a full KAOS "
            "deployment (206 tools across 14 servers) with room for "
            "agents to delegate to most of them. Was 30 in the GPT-4 era."
        ),
    )
    max_react_iterations: int = Field(
        default=50,
        ge=1,
        description=(
            "Maximum iterations in a ReAct tool-calling loop. Real "
            "agentic legal/research workflows often need 20-40 tool "
            "calls (search → read → search → read → cite → verify). "
            "10 was a 2023-era cap that truncated long-horizon tasks."
        ),
    )
    tool_timeout_seconds: float = Field(
        default=120.0,
        gt=0,
        description=(
            "Timeout for individual tool invocations. Bumped from 60s — "
            "playwright loads and large-corpus extractions routinely "
            "take 60-90s on first call."
        ),
    )

    # Planning: route thresholds
    confidence_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Below this confidence, Route triggers REPLAN.",
    )

    # Planning: budget defaults — sized for realistic agentic tasks in 2026.
    # A real M&A deal-room review or regulatory research task can easily
    # need 50+ steps, 10+ replans, 5M tokens across the whole plan. Old
    # defaults (20 steps, 100K tokens, $1, 120s) were sized for toy
    # demos and silently capped real workloads.
    plan_max_steps: int = Field(
        default=100,
        ge=1,
        description=(
            "Maximum steps in a single plan execution. Real legal/"
            "research workflows need 50-100 steps when iterating across "
            "many documents."
        ),
    )
    plan_max_replans: int = Field(
        default=10,
        ge=0,
        description=(
            "Maximum replan attempts before STOP_FAILURE (circuit "
            "breaker). Long-horizon tasks routinely hit failed steps "
            "that need replanning; 3 was too tight."
        ),
    )
    plan_max_tokens: int = Field(
        default=5_000_000,
        ge=1,
        description=(
            "Maximum tokens across all steps in a plan execution. With "
            "200K-context models making 50+ calls, 100K was orders of "
            "magnitude too small. 5M = ~$15-25 of cost on Sonnet, "
            "matches plan_max_cost_usd."
        ),
    )
    plan_max_cost_usd: float = Field(
        default=10.0,
        gt=0,
        description=(
            "Maximum cost in USD for a single plan execution. $1 was "
            "the toy-demo budget; legal-deal review easily justifies "
            "$10/task. Hard ceiling — agent stops, never silently "
            "exceeds."
        ),
    )
    plan_max_wall_clock_seconds: float = Field(
        default=900.0,
        gt=0,
        description=(
            "Maximum wall-clock time for a single plan execution. 15 "
            "minutes covers realistic agentic workloads (50+ tool "
            "calls at 5-15s each). 120s was for toy demos."
        ),
    )

    # Context retrieval
    retrieval_threshold: int = Field(
        default=5,
        ge=1,
        description="When a memory section has >= this many items, use BM25 "
        "retrieval instead of FIFO for context assembly. Default 5 — for "
        "legal corpora, even a small deal room (5-20 docs) benefits from "
        "BM25 selection over FIFO. Was 20 historically but that meant "
        "small corpora silently bypassed retrieval.",
    )

    # Track 3 chunk B4 — graph-aware context expansion.
    graph_context_enabled: bool = Field(
        default=True,
        description=(
            "When True (default), assemble_context walks the per-session "
            "knowledge graph 1 hop from each retrieved FINDING and injects "
            "synthetic MemoryType.GRAPH items describing the provenance "
            "(cited documents, supporting findings, producing tool calls). "
            "Adds typically 50-200 tokens per finding to context. Disable "
            "to save tokens or when the graph is known to be empty/noisy."
        ),
    )
    graph_context_max_neighbors: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Max graph edges followed per finding during expansion.",
    )
    graph_context_max_findings: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Max findings to expand per turn (cap on graph_expand cost).",
    )

    # Refusal-robustness knobs (N4). Sit alongside the existing
    # ``Agent.refusal_policy`` (kaos-llm-core ``RefusalPolicy``) — these
    # are the system-wide defaults that the agent layer applies when the
    # caller doesn't supply an explicit policy. The mf09 Voyager
    # false-positive (chunked retrieval lured the agent into answering
    # an out-of-corpus question) drove these defaults — they're a hard-
    # to-misuse second line of defense even when retrieval finds garbage.
    bm25_score_floor: float = Field(
        default=0.0,
        ge=0.0,
        description=(
            "Absolute BM25 score floor. Passages scoring below this are "
            "filtered out before the LLM ever sees them. 0.0 disables — "
            "default for backward compat. Set 1.0-3.0 for legal corpora "
            "to filter spurious lexical-overlap hits in adversarial "
            "(out-of-corpus) queries. Sits in front of the relative "
            "top-K cutoff: if no passages clear the floor, retrieval "
            "returns empty and the agent gets InsufficientEvidence."
        ),
    )
    verifier_min_confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description=(
            "Verifier confidence threshold for grounded answers. When > 0, "
            "answers whose confidence falls below this are collapsed to "
            "InsufficientEvidence (via kaos-llm-core RefusalPolicy). "
            "Default 0.0 keeps the legacy permissive behavior; 0.5-0.7 is "
            "the recommended floor for legal Q&A where hedging on a low-"
            "confidence claim is worse than refusing. Applied only when "
            "the agent itself doesn't carry an explicit refusal_policy."
        ),
    )
    refuse_unverified_answers: bool = Field(
        default=False,
        description=(
            "When True, the research pattern collapses any RAG answer "
            "whose ``result.is_verified`` is False to InsufficientEvidence "
            "before returning. Catches the mf09 failure mode: low-quality "
            "lexical retrieval produces a fluent answer the LLM is "
            "confident about, but the citation spans don't actually verify "
            "in the source corpus. Off by default (legacy permissive — "
            "matches the historical 'present an unverified answer with a "
            "warning' shape). Recommended on for legal deployment."
        ),
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

    # KC17-P2-1 — opt-in legal/regulated-deployment profile.
    # When ``"strict"``, ``resolved_bm25_score_floor()`` /
    # ``resolved_verifier_min_confidence()`` /
    # ``resolved_refuse_unverified_answers()`` return the values in
    # ``STRICT_PROFILE_DEFAULTS`` (above), so the research agent
    # filters low-score BM25 hits, refuses below the verifier
    # confidence floor, AND refuses unverified answers instead of
    # warn-and-return. The underlying field values themselves are
    # left untouched (for back-compat in alpha) — the profile is a
    # resolution-time override only.
    research_profile: Literal["default", "strict"] = Field(
        default="default",
        description=(
            'Research-pattern safety profile. ``"default"`` keeps '
            "the historical permissive behavior (bm25 floor / verifier "
            "threshold / unverified-refusal all disabled unless set "
            'explicitly). ``"strict"`` is the recommended preset '
            "for legal / regulated deployments: enables the BM25 score "
            "floor, the verifier confidence threshold, and refusal of "
            "unverified answers. Configure via "
            "``KAOS_AGENT_RESEARCH_PROFILE=strict``. The three "
            "underlying knobs (bm25_score_floor, verifier_min_"
            "confidence, refuse_unverified_answers) remain available "
            "for fine-grained override."
        ),
    )

    # Observability — emit assembled-prompt previews on Span attributes
    # so a misbehaving turn can be diagnosed without rebuilding the
    # exact stack trace from logs. Off by default because the previews
    # add ~1-3 KB per Span and bloat the SSE stream when enabled
    # everywhere. See ``kaos-modules/docs/plans/
    # kaos-agents-autonomy-improvement-1.md`` §3.3 for the design.
    debug_prompts: bool = Field(
        default=False,
        description=(
            "When True, every actor / planner / judge / goal-check Call "
            "extends its Span attributes with "
            "{assembled_prompt_preview (<=500 chars), tool_catalog_size, "
            "tool_names_sample (<=10 names)}. Diagnostic surface for "
            "the 0.1.0a10 PLAN→tools dispatch fix and for any future "
            "case where the agent silently degrades without tools. "
            "Resolves via KAOS_AGENT_DEBUG_PROMPTS env var or per-request "
            "_meta.kaos_config override; loose os.environ reads at Call "
            "sites are forbidden (see Configuration Hierarchy in "
            "kaos-modules/CLAUDE.md)."
        ),
    )

    model_config = SettingsConfigDict(
        env_prefix="KAOS_AGENT_",
        env_file=".env",
        extra="ignore",
    )

    # KC17-P2-1 — resolved accessors. These compose ``research_profile``
    # with the underlying field value: when the profile is ``"strict"``
    # AND the user hasn't explicitly raised the underlying field above
    # the strict default, the strict default wins. When the user
    # explicitly raises the floor (e.g. to a stricter 2.0), that
    # explicit value wins — the profile is a floor, not a ceiling.
    # When the profile is ``"default"``, the underlying field value is
    # returned verbatim (legacy behavior).
    def resolved_bm25_score_floor(self) -> float:
        """Effective BM25 score floor honoring ``research_profile``.

        Strict profile: max(field, STRICT_PROFILE_DEFAULTS["bm25_score_floor"]).
        Default profile: the raw field value (unchanged for back-compat).
        """
        if self.research_profile == "strict":
            strict_value = float(STRICT_PROFILE_DEFAULTS["bm25_score_floor"])
            return max(self.bm25_score_floor, strict_value)
        return self.bm25_score_floor

    def resolved_verifier_min_confidence(self) -> float:
        """Effective verifier confidence floor honoring ``research_profile``.

        Strict profile: max(field, STRICT_PROFILE_DEFAULTS["verifier_min_confidence"]).
        Default profile: the raw field value (unchanged for back-compat).
        """
        if self.research_profile == "strict":
            strict_value = float(STRICT_PROFILE_DEFAULTS["verifier_min_confidence"])
            return max(self.verifier_min_confidence, strict_value)
        return self.verifier_min_confidence

    def resolved_refuse_unverified_answers(self) -> bool:
        """Effective unverified-answer refusal flag honoring ``research_profile``.

        Strict profile: True (the strict preset forces refusal on).
        Default profile: the raw field value (unchanged for back-compat).
        """
        if self.research_profile == "strict":
            return True
        return self.refuse_unverified_answers


# Default model string for use as function parameter defaults.
# Derived from KaosAgentSettings so there's a single source of truth.
DEFAULT_MODEL: str = KaosAgentSettings.model_fields["default_llm_model"].default
