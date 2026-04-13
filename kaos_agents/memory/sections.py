"""Section — a budgeted, eviction-aware container of MemoryItems."""

from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING

from kaos_core.logging import get_logger

from kaos_agents.errors import EvictionError, MemoryBudgetExceededError
from kaos_agents.memory.types import (
    EvictionPolicy,
    MemoryItem,
    MemoryType,
    SectionConfig,
    create_item,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

logger = get_logger(__name__)


class Section:
    """A single memory section — holds items within a token budget.

    Mutable container (not frozen). Items are added, evicted, and accessed.
    The section tracks access patterns for LRU/LFU eviction policies.

    Thread-safety: not thread-safe. Designed for single-threaded async use
    within one agent turn. The agent loop serializes access.
    """

    __slots__ = (
        "_access_counts",
        "_config",
        "_items",
        "_last_access",
        "_token_count",
    )

    def __init__(self, config: SectionConfig) -> None:
        self._config = config
        self._items: deque[MemoryItem] = deque()
        self._token_count: int = 0
        self._access_counts: dict[str, int] = {}  # item_id -> access count (LFU)
        self._last_access: dict[str, float] = {}  # item_id -> last access time (LRU)

    # -- Properties ----------------------------------------------------------

    @property
    def config(self) -> SectionConfig:
        return self._config

    @property
    def memory_type(self) -> MemoryType:
        return self._config.memory_type

    @property
    def token_count(self) -> int:
        """Current total tokens in this section."""
        return self._token_count

    @property
    def budget_tokens(self) -> int:
        """Maximum token budget for this section. 0 = unlimited."""
        return self._config.budget_tokens

    @property
    def is_over_budget(self) -> bool:
        """True if current tokens exceed the budget (0 = never over)."""
        if self._config.budget_tokens == 0:
            return False
        return self._token_count > self._config.budget_tokens

    @property
    def available_tokens(self) -> int:
        """Tokens remaining before budget is reached. -1 if unlimited."""
        if self._config.budget_tokens == 0:
            return -1
        return max(0, self._config.budget_tokens - self._token_count)

    @property
    def item_count(self) -> int:
        return len(self._items)

    # -- Core operations -----------------------------------------------------

    def add(
        self,
        content: str,
        *,
        chars_per_token: float = 4.0,
        priority: int = 0,
        tags: tuple[str, ...] = (),
        metadata: dict | None = None,
    ) -> MemoryItem:
        """Add content to this section.

        If adding would exceed the budget, eviction is attempted first.
        If eviction cannot free enough space, raises MemoryBudgetExceededError
        (for REFUSE policy) or EvictionError (if eviction fails to free enough).

        Returns the created MemoryItem.
        """
        if self._config.explicit_only:
            raise MemoryBudgetExceededError(
                f"Section {self.memory_type.value} is explicit-only "
                f"and cannot be written to during the session. "
                f"Load explicit sections from config at session creation.",
            )

        item = create_item(
            self.memory_type,
            content,
            chars_per_token=chars_per_token,
            priority=priority,
            tags=tags,
            metadata=metadata,
        )

        # Check max_items limit
        if self._config.max_items is not None and len(self._items) >= self._config.max_items:
            self._evict_one()

        # Check token budget
        budget = self._config.budget_tokens
        if budget > 0 and self._token_count + item.token_count > budget:
            if self._config.eviction_policy == EvictionPolicy.REFUSE:
                raise MemoryBudgetExceededError(
                    f"Section {self.memory_type.value} is full "
                    f"({self._token_count}/{budget} tokens) and uses REFUSE eviction. "
                    f"Free space with evict() or increase the budget.",
                )
            # Evict until we have room
            needed = self._token_count + item.token_count - budget
            freed = self._evict_tokens(needed)
            if freed < needed:
                raise EvictionError(
                    f"Section {self.memory_type.value}: eviction freed {freed} tokens "
                    f"but {needed} were needed. {len(self._items)} items remain.",
                )

        self._items.append(item)
        self._token_count += item.token_count
        logger.debug(
            "memory.add: section=%s tokens=%d/%s items=%d",
            self.memory_type.value,
            self._token_count,
            budget or "∞",
            len(self._items),
        )
        return item

    def add_item(self, item: MemoryItem) -> None:
        """Add a pre-built MemoryItem (used during restore from persistence)."""
        self._items.append(item)
        self._token_count += item.token_count

    def get(self, *, max_tokens: int | None = None) -> list[MemoryItem]:
        """Get items from this section, up to max_tokens.

        Updates access tracking for LRU/LFU policies.
        Returns items in insertion order (oldest first).
        """
        import time as _time

        now = _time.time()
        result: list[MemoryItem] = []
        tokens_used = 0

        for item in self._items:
            if max_tokens is not None and tokens_used + item.token_count > max_tokens:
                break
            result.append(item)
            tokens_used += item.token_count
            # Track access for LRU/LFU
            self._access_counts[item.id] = self._access_counts.get(item.id, 0) + 1
            self._last_access[item.id] = now

        return result

    def get_recent(self, n: int) -> list[MemoryItem]:
        """Get the N most recent items (newest first)."""
        items = list(self._items)
        items.reverse()
        return items[:n]

    def clear(self) -> int:
        """Remove all items. Returns tokens freed."""
        freed = self._token_count
        self._items.clear()
        self._token_count = 0
        self._access_counts.clear()
        self._last_access.clear()
        return freed

    def items(self) -> Iterator[MemoryItem]:
        """Iterate over items in insertion order."""
        return iter(self._items)

    # -- Eviction ------------------------------------------------------------

    def _evict_one(self) -> MemoryItem | None:
        """Evict a single item per the section's eviction policy."""
        if not self._items:
            return None

        policy = self._config.eviction_policy

        if policy == EvictionPolicy.NONE:
            return None

        if policy == EvictionPolicy.REFUSE:
            return None

        if policy == EvictionPolicy.FIFO:
            item = self._items.popleft()
        elif policy == EvictionPolicy.LIFO:
            item = self._items.pop()
        elif policy == EvictionPolicy.LRU:
            item = self._evict_by_key(self._last_access, least=True)
        elif policy == EvictionPolicy.LFU:
            item = self._evict_by_key(self._access_counts, least=True)
        elif policy == EvictionPolicy.PRIORITY:
            item = self._evict_lowest_priority()
        else:
            return None

        self._token_count -= item.token_count
        self._access_counts.pop(item.id, None)
        self._last_access.pop(item.id, None)
        logger.debug(
            "memory.evict: section=%s policy=%s freed=%d tokens",
            self.memory_type.value,
            policy.value,
            item.token_count,
        )
        return item

    def _evict_tokens(self, needed: int) -> int:
        """Evict items until at least `needed` tokens are freed."""
        freed = 0
        while freed < needed and self._items:
            item = self._evict_one()
            if item is None:
                break
            freed += item.token_count
        return freed

    def _evict_by_key(
        self, tracking: dict[str, int] | dict[str, float], *, least: bool
    ) -> MemoryItem:
        """Evict the item with the min (least=True) or max (least=False) tracking value."""
        # Find the item with the target tracking value
        target_id: str | None = None
        target_val: float | int | None = None
        for item in self._items:
            val = tracking.get(item.id, 0)
            if (
                target_val is None
                or (least and val < target_val)
                or (not least and val > target_val)
            ):
                target_val = val
                target_id = item.id

        # Remove it from the deque (O(n) but sections are small)
        for i, item in enumerate(self._items):
            if item.id == target_id:
                del self._items[i]
                return item

        # Fallback: remove oldest
        return self._items.popleft()

    def _evict_lowest_priority(self) -> MemoryItem:
        """Evict the item with the lowest priority value."""
        min_idx = 0
        min_priority = self._items[0].priority
        for i, item in enumerate(self._items):
            if item.priority < min_priority:
                min_priority = item.priority
                min_idx = i
        item = self._items[min_idx]
        del self._items[min_idx]
        return item

    # -- Serialization -------------------------------------------------------

    def to_dict(self) -> dict:
        """Serialize section state for persistence."""
        return {
            "memory_type": self.memory_type.value,
            "items": [
                {
                    "id": item.id,
                    "content": item.content,
                    "token_count": item.token_count,
                    "added_at": item.added_at,
                    "metadata": item.metadata,
                    "priority": item.priority,
                    "tags": list(item.tags),
                }
                for item in self._items
            ],
        }

    @classmethod
    def from_dict(cls, data: dict, config: SectionConfig) -> Section:
        """Restore section from persisted data."""
        section = cls(config)
        for item_data in data.get("items", []):
            item = MemoryItem(
                id=item_data["id"],
                section=config.memory_type,
                content=item_data["content"],
                token_count=item_data["token_count"],
                added_at=item_data["added_at"],
                metadata=item_data.get("metadata", {}),
                priority=item_data.get("priority", 0),
                tags=tuple(item_data.get("tags", ())),
            )
            section.add_item(item)
        return section

    def __repr__(self) -> str:
        budget = self._config.budget_tokens or "∞"
        return (
            f"Section({self.memory_type.value}, "
            f"items={len(self._items)}, "
            f"tokens={self._token_count}/{budget})"
        )
