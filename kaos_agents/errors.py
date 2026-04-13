"""Exception hierarchy for kaos-agents."""

from __future__ import annotations

from kaos_core.exceptions import KaosCoreError


class KaosAgentError(KaosCoreError):
    """Base exception for all kaos-agents errors."""


class SessionNotFoundError(KaosAgentError):
    """Raised when a session cannot be found in the store."""


class SessionCorruptedError(KaosAgentError):
    """Raised when a persisted session cannot be deserialized."""


class MemoryBudgetExceededError(KaosAgentError):
    """Raised when a memory operation would exceed the section's token budget."""


class EvictionError(KaosAgentError):
    """Raised when eviction fails to free sufficient space."""


class SectionNotConfiguredError(KaosAgentError, KeyError):
    """Raised when accessing a section not in the memory profile.

    Inherits KeyError for stdlib compatibility (dict-like interface).
    """
