"""Memory type system — sections, items, policies, and configuration."""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from enum import StrEnum, unique
from typing import Any

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


@unique
class MemoryType(StrEnum):
    """Section identifiers for SessionMemory.

    Each section serves a distinct role in context assembly.
    Sections are grouped by mutability:
    - Explicit: loaded from config, immutable during session
    - Learned: written by the agent during the session
    - Ephemeral: cleared between turns
    """

    # Explicit (loaded from config, immutable)
    ROLE = "role"
    PLAYBOOKS = "playbooks"
    PLAN_EXAMPLES = "plan_examples"

    # Learned (agent-written, persistent)
    MESSAGES = "messages"
    ACTIONS = "actions"
    DOCUMENTS = "documents"
    FINDINGS = "findings"
    PLAN_HISTORY = "plan_history"
    REFLECTION = "reflection"
    LAST_INTENT = "last_intent"
    # Distilled cross-session lessons. Where REFLECTION is per-turn
    # (what did this turn observe?), LESSONS is the persistent
    # compound — write-once, recalled in future sessions when the
    # situation matches. Populated by the ReflexionHook (or manually
    # via write_lesson). Persistent across sessions via streaming JSONL.
    LESSONS = "lessons"

    # Ephemeral (cleared between turns)
    WORKING = "working"
    PLANNING_CONTEXT = "planning_context"

    # Derived (computed from other sections, not stored)
    LAST_USER_MESSAGE = "last_user_message"
    RECENT_ACTIONS = "recent_actions"

    # System (internal, always persisted)
    AUDIT = "audit"

    # Knowledge graph (per-session RDF triple store; PROV-O + CiTO + kaos: vocab).
    # Track 3 chunk B1 — backed by a single ``kaos_graph.Graph`` instance per
    # session. Triples are emitted by the chunk-B2 emit_from_event hook from
    # KaosEvents (CitationFound, ToolExecution end, IntentClassified, ...)
    # and persisted to VFS as Turtle on session save. Queryable via SPARQL
    # (chunk B3 graph MCP tools) and consulted by graph-aware context
    # assembly (chunk B4).
    GRAPH = "graph"


@unique
class EvictionPolicy(StrEnum):
    """How a section sheds items when over budget."""

    NONE = "none"  # Never evict; section bounded by design
    FIFO = "fifo"  # Drop oldest items first
    LIFO = "lifo"  # Drop newest items (keeps only most recent)
    LRU = "lru"  # Drop least-recently-accessed
    LFU = "lfu"  # Drop least-frequently-accessed
    PRIORITY = "priority"  # Drop lowest priority value first
    REFUSE = "refuse"  # Reject new writes when full


@unique
class SummarizationPolicy(StrEnum):
    """When a section's content is summarized to save tokens."""

    NEVER = "never"  # Never summarize
    ON_OVERFLOW = "on_overflow"  # When eviction alone is insufficient
    ON_TURN = "on_turn"  # At end of each turn, unconditionally
    MANUAL = "manual"  # Only via explicit MCP call
    AUTO = "auto"  # Heuristic: overflow OR unused for N turns


@unique
class PersistenceMode(StrEnum):
    """How a section's data is persisted to VFS."""

    NONE = "none"  # Session-only, lost on restart (WORKING)
    SNAPSHOT = "snapshot"  # Full JSON at checkpoints (ROLE, PLAYBOOKS)
    STREAMING = "streaming"  # JSONL append on every write (MESSAGES, ACTIONS)


# ---------------------------------------------------------------------------
# Section Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SectionConfig:
    """Configuration for a single memory section.

    Immutable — defined at session creation and never changed.
    """

    memory_type: MemoryType
    budget_tokens: int
    eviction_policy: EvictionPolicy = EvictionPolicy.FIFO
    summarization_policy: SummarizationPolicy = SummarizationPolicy.NEVER
    persistence_mode: PersistenceMode = PersistenceMode.SNAPSHOT
    explicit_only: bool = False  # True for ROLE, PLAYBOOKS, PLAN_EXAMPLES
    searchable: bool = False  # True for sections indexed by BM25
    max_items: int | None = None  # Optional hard item count limit


# ---------------------------------------------------------------------------
# Memory Items
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class MemoryItem:
    """A single item in a memory section.

    Immutable after creation. Items are never mutated — they are replaced
    or evicted. The id is a deterministic hash of content + section + timestamp
    for deduplication.
    """

    id: str
    section: MemoryType
    content: str  # Serialized content (text, JSON, etc.)
    token_count: int  # Pre-computed token estimate
    added_at: float  # time.time() timestamp
    metadata: dict[str, Any] = field(default_factory=dict)
    priority: int = 0  # Higher = more important (for PRIORITY eviction)
    tags: tuple[str, ...] = ()

    @staticmethod
    def make_id(content: str, section: MemoryType, timestamp: float) -> str:
        """Deterministic content-addressed ID."""
        raw = f"{section.value}:{timestamp}:{content}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]


def estimate_tokens(text: str, chars_per_token: float = 4.0) -> int:
    """Estimate token count from text length.

    Uses character-based heuristic. Conservative default of 4 chars/token
    works well for English text with Claude/GPT models.
    """
    if not text:
        return 0
    return max(1, int(len(text) / chars_per_token))


def create_item(
    section: MemoryType,
    content: str,
    *,
    chars_per_token: float = 4.0,
    priority: int = 0,
    tags: tuple[str, ...] = (),
    metadata: dict[str, Any] | None = None,
) -> MemoryItem:
    """Create a MemoryItem with automatic ID and token estimation."""
    now = time.time()
    return MemoryItem(
        id=MemoryItem.make_id(content, section, now),
        section=section,
        content=content,
        token_count=estimate_tokens(content, chars_per_token),
        added_at=now,
        metadata=metadata or {},
        priority=priority,
        tags=tags,
    )


# ---------------------------------------------------------------------------
# Default Profile
# ---------------------------------------------------------------------------

DEFAULT_SECTIONS: tuple[SectionConfig, ...] = (
    # Section budgets (2026 sizing). Old values were calibrated for the
    # 8K-context era (Claude 2 / GPT-3.5). Frontier models in 2026 have
    # 200K+ contexts; allocating only 200-3000 tokens per section meant
    # the agent threw away most of what it had remembered.
    #
    # New sizing fits comfortably within the 200K context budget:
    # - Explicit (role/playbooks/examples): 2K-16K each — load once,
    #   the agent keeps it for the whole session.
    # - Learned (messages/actions/findings): 16K-64K — these are
    #   what the agent actually uses turn-to-turn.
    # - Documents/Audit: unlimited (already correct).
    # Total committed budget: ~200K (matches default_context_budget_tokens).
    #
    # Explicit (immutable, loaded from config)
    SectionConfig(
        memory_type=MemoryType.ROLE,
        budget_tokens=2_000,
        eviction_policy=EvictionPolicy.NONE,
        persistence_mode=PersistenceMode.SNAPSHOT,
        explicit_only=True,
    ),
    SectionConfig(
        memory_type=MemoryType.PLAYBOOKS,
        budget_tokens=8_000,
        eviction_policy=EvictionPolicy.NONE,
        persistence_mode=PersistenceMode.SNAPSHOT,
        explicit_only=True,
    ),
    SectionConfig(
        memory_type=MemoryType.PLAN_EXAMPLES,
        budget_tokens=8_000,
        eviction_policy=EvictionPolicy.NONE,
        persistence_mode=PersistenceMode.SNAPSHOT,
        explicit_only=True,
    ),
    # Learned (agent-written, persistent)
    SectionConfig(
        memory_type=MemoryType.MESSAGES,
        budget_tokens=32_000,
        eviction_policy=EvictionPolicy.FIFO,
        summarization_policy=SummarizationPolicy.ON_OVERFLOW,
        persistence_mode=PersistenceMode.STREAMING,
        searchable=True,
    ),
    SectionConfig(
        memory_type=MemoryType.ACTIONS,
        budget_tokens=16_000,
        eviction_policy=EvictionPolicy.LRU,
        summarization_policy=SummarizationPolicy.ON_OVERFLOW,
        persistence_mode=PersistenceMode.STREAMING,
        searchable=True,
    ),
    SectionConfig(
        memory_type=MemoryType.DOCUMENTS,
        budget_tokens=0,
        eviction_policy=EvictionPolicy.NONE,
        persistence_mode=PersistenceMode.SNAPSHOT,
        searchable=True,
    ),
    SectionConfig(
        memory_type=MemoryType.FINDINGS,
        budget_tokens=16_000,
        eviction_policy=EvictionPolicy.NONE,
        summarization_policy=SummarizationPolicy.MANUAL,
        persistence_mode=PersistenceMode.STREAMING,
        searchable=True,
    ),
    SectionConfig(
        memory_type=MemoryType.PLAN_HISTORY,
        budget_tokens=8_000,
        eviction_policy=EvictionPolicy.LRU,
        summarization_policy=SummarizationPolicy.ON_OVERFLOW,
        persistence_mode=PersistenceMode.SNAPSHOT,
    ),
    SectionConfig(
        memory_type=MemoryType.REFLECTION,
        budget_tokens=8_000,
        eviction_policy=EvictionPolicy.FIFO,
        summarization_policy=SummarizationPolicy.ON_TURN,
        persistence_mode=PersistenceMode.STREAMING,
    ),
    SectionConfig(
        # Distilled cross-session lessons. Larger budget than REFLECTION
        # because lessons compound across sessions and the value is
        # exactly in being able to recall many of them via BM25.
        # FIFO eviction so the oldest lessons drop when full —
        # callers who care about specific lessons can pin them via
        # priority.
        memory_type=MemoryType.LESSONS,
        budget_tokens=16_000,
        eviction_policy=EvictionPolicy.FIFO,
        summarization_policy=SummarizationPolicy.MANUAL,
        persistence_mode=PersistenceMode.STREAMING,
        searchable=True,
    ),
    SectionConfig(
        memory_type=MemoryType.LAST_INTENT,
        budget_tokens=2_000,
        eviction_policy=EvictionPolicy.LIFO,
        persistence_mode=PersistenceMode.STREAMING,
        max_items=1,
    ),
    # Ephemeral (cleared between turns)
    SectionConfig(
        memory_type=MemoryType.WORKING,
        budget_tokens=16_000,
        eviction_policy=EvictionPolicy.FIFO,
        persistence_mode=PersistenceMode.NONE,
    ),
    SectionConfig(
        memory_type=MemoryType.PLANNING_CONTEXT,
        budget_tokens=16_000,
        eviction_policy=EvictionPolicy.LIFO,
        persistence_mode=PersistenceMode.NONE,
    ),
    # System (always persisted, unbounded)
    SectionConfig(
        memory_type=MemoryType.AUDIT,
        budget_tokens=0,  # 0 = unlimited
        eviction_policy=EvictionPolicy.NONE,
        persistence_mode=PersistenceMode.STREAMING,
    ),
    # Knowledge graph — per-session RDF triple store.
    #
    # Track 3 chunk B1 ships this as a section so the graph participates
    # in the existing eviction / persistence / hydration plumbing the
    # rest of memory uses. The graph itself is *not* a list of MemoryItems
    # — SessionMemory holds a parallel ``.graph`` attribute (chunk B1
    # adds it). The section here is the persistence + budget surface
    # only; ``budget_tokens=0`` keeps it unbounded (graphs grow with
    # the session) and ``EvictionPolicy.NONE`` reflects that we don't
    # evict triples (the graph is the durable knowledge representation).
    SectionConfig(
        memory_type=MemoryType.GRAPH,
        budget_tokens=0,
        eviction_policy=EvictionPolicy.NONE,
        persistence_mode=PersistenceMode.SNAPSHOT,
    ),
)
