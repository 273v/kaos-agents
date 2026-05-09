"""RateLimiter — a :class:`KaosHook` that enforces per-tool QPS limits.

Reads tool calls from :meth:`KaosHook.on_tool_call_start` and decides
whether to allow or refuse via :attr:`HookAction.SKIP`. Uses a token
bucket algorithm per tool name. Default is unlimited unless configured.

Phase 1.C of the rewrite. Phase 2 wires the Runner to instantiate
RateLimiters from settings; Phase 1.C just ships the hook.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from fnmatch import fnmatch
from typing import TYPE_CHECKING

from kaos_agents.hooks.base import HookAction, KaosHook

if TYPE_CHECKING:
    from kaos_agents.events import Span


@dataclass
class _TokenBucket:
    """Mutable token-bucket accumulator for one tool name pattern.

    Capacity is the maximum burst. Refill is steady-state QPS — the
    bucket gains ``refill_per_sec`` tokens per second up to ``capacity``.
    """

    capacity: int
    refill_per_sec: float
    tokens: float = field(init=False)
    last_refill: float = field(init=False)

    def __post_init__(self) -> None:
        self.tokens = float(self.capacity)
        self.last_refill = time.monotonic()

    def consume(self, n: int = 1) -> bool:
        """Try to consume ``n`` tokens. Return True on success."""
        now = time.monotonic()
        elapsed = max(0.0, now - self.last_refill)
        self.last_refill = now
        # Refill, clamped to capacity.
        self.tokens = min(float(self.capacity), self.tokens + elapsed * self.refill_per_sec)
        if self.tokens >= float(n):
            self.tokens -= float(n)
            return True
        return False


class RateLimiter(KaosHook):
    """Per-tool token-bucket rate limiter hook.

    Constructed with a mapping ``{tool_name_glob: (capacity, refill_per_sec)}``.
    Tools matching no glob get the default rate
    (``default_rate=None`` means unlimited).

    Globs are matched in insertion order via :func:`fnmatch.fnmatch`;
    the first matching pattern wins. Each unique pattern owns its own
    bucket — multiple tools matching the same pattern share that bucket.

    The hook returns :attr:`HookAction.SKIP` when a call exceeds the
    bucket; the Runner translates SKIP into a refusal at the
    higher-level Actor surface (Phase 2 wiring).
    """

    def __init__(
        self,
        rates: dict[str, tuple[int, float]] | None = None,
        default_rate: tuple[int, float] | None = None,
    ) -> None:
        # Keep insertion order so the first matching glob wins
        # deterministically.
        self._rates: dict[str, tuple[int, float]] = dict(rates or {})
        self._default_rate: tuple[int, float] | None = default_rate
        # One bucket per pattern (or the synthetic "*" key for default).
        self._buckets: dict[str, _TokenBucket] = {}

    def _bucket_for(self, tool_name: str) -> _TokenBucket | None:
        """Return the matching bucket (creating it lazily) or None for unlimited."""
        for pattern, (capacity, refill) in self._rates.items():
            if fnmatch(tool_name, pattern):
                bucket = self._buckets.get(pattern)
                if bucket is None:
                    bucket = _TokenBucket(capacity=capacity, refill_per_sec=refill)
                    self._buckets[pattern] = bucket
                return bucket

        if self._default_rate is None:
            return None  # Unlimited.
        bucket = self._buckets.get("__default__")
        if bucket is None:
            capacity, refill = self._default_rate
            bucket = _TokenBucket(capacity=capacity, refill_per_sec=refill)
            self._buckets["__default__"] = bucket
        return bucket

    def allow(self, tool_name: str) -> bool:
        """Synchronous predicate — True iff a call to ``tool_name`` is allowed.

        Useful for the Actor's classify_plan path which gates without
        the Runner's hook dispatcher.
        """
        bucket = self._bucket_for(tool_name)
        if bucket is None:
            return True
        return bucket.consume(1)

    async def on_tool_call_start(self, event: Span) -> HookAction:
        """Allow or skip the tool call based on the matching token bucket."""
        tool_name = ""
        attrs = getattr(event, "attributes", None)
        if isinstance(attrs, dict):
            tool_name = str(attrs.get("tool_name", "")) or ""
        if not tool_name:
            # No tool name → can't gate; default to CONTINUE (the
            # Runner's caller is responsible for naming spans).
            return HookAction.CONTINUE
        return HookAction.CONTINUE if self.allow(tool_name) else HookAction.SKIP


__all__ = ["RateLimiter"]
