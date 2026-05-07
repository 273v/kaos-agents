"""SessionToolSet — declarative per-session tool-allow-list. Track 4 chunk T4-5.

The agent runtime registers many tools; not every session needs all of
them. A "research" session may use the extraction + graph + agent
tools but want to block the destructive memory-clear; a "billing"
session may want only the chat tool. SessionToolSet captures the
allow / deny rules so the bridge layer can filter the tool surface
to exactly what the session is permitted to call.

Design:

- ``allowed_tools`` — explicit allow-list of tool names. When empty,
  the *default* is "all tools allowed" (subject to denied_tools).
- ``allowed_groups`` — allow-list of group names; resolved through
  the :class:`ToolGroupRegistry` at filter time.
- ``denied_tools`` — explicit deny-list. Overrides every allow.
- All three are frozensets — order doesn't matter, comparison is by
  membership, and the value type is hashable.

A session that declares neither allowed_tools nor allowed_groups (and
no deny list) is unrestricted. This default-allow choice mirrors how
existing kaos-agents code reads bridges today — silent restriction
would be surprising.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SessionToolSet:
    """Per-session tool allow / deny configuration.

    Resolution semantics (apply in order):

    1. If ``allowed_tools`` is non-empty, only tools whose name is in
       that set pass the allow check.
    2. If ``allowed_groups`` is non-empty (and the tool didn't pass
       step 1's allow), the tool passes if any of its groups is in
       this set. (Tools in no group fail this check.)
    3. If both ``allowed_tools`` and ``allowed_groups`` are empty,
       the default is **allow** (no restriction yet).
    4. ``denied_tools`` always wins — a tool present here is blocked
       regardless of any allow list.

    All collections are frozensets so the value type is hashable and
    comparison-safe.
    """

    allowed_tools: frozenset[str] = frozenset()
    allowed_groups: frozenset[str] = frozenset()
    denied_tools: frozenset[str] = frozenset()

    @property
    def is_unrestricted(self) -> bool:
        """True when no allow / deny list constrains the catalogue."""
        return not self.allowed_tools and not self.allowed_groups and not self.denied_tools

    @classmethod
    def unrestricted(cls) -> SessionToolSet:
        """Identity-stable empty config (allow-all, no denies)."""
        return _UNRESTRICTED


_UNRESTRICTED = SessionToolSet()


__all__ = ["SessionToolSet"]
