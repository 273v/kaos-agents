"""KaosPersona — typed, named bundle of tool-group + model + budget policy.

Plan §12 of
``kaos-modules/docs/plans/2026-05-19-lateral-redesign-capability-layer.md``.

The SPA today hard-codes a "Research / Drafting / Forensics" persona
trio inside ``single-user-chat``. Anywhere else that wants to honor
the same persona definition (a CLI, an external MCP host, a future
TUI) has had to re-implement the policy by hand. ``KaosPersona``
makes the persona a first-class runtime primitive on top of the
existing tool-group taxonomy: any consumer of the Runner reads the
same shape, so the policy stops drifting between deployments.

A persona carries four things:

- ``allowed_groups`` — the working set of tool groups the agent
  is permitted to dispatch by default. This is the persona's
  "comfortable lane."
- ``ceiling_groups`` — the hard upper bound. Even with
  auto-elevation enabled, a Runner never grants the agent access to
  a group outside ``ceiling_groups``. The session policy may elevate
  inside ``ceiling_groups`` but never beyond it. Validated to be a
  superset of ``allowed_groups`` at construction time so a typo
  can't silently demote the persona to "nothing allowed."
- ``model_role_preferences`` — a ``{role -> model_id}`` map (e.g.
  ``{"respond": "anthropic:claude-sonnet-4-6"}``) the Runner consults
  before falling back to its global default for that role. The
  persona doesn't enforce that any specific role is present; the
  Runner is the authority on which roles it understands.
- ``default_budgets`` — a ``{key -> float}`` map of soft per-turn /
  per-loop budget knobs (cost ceilings, max tool calls). Free-form
  so new budget primitives don't require a persona type bump.

The value type is intentionally narrow: persona names are NOT
hardcoded anywhere in business logic — they are registry entries.
The seeded "Research", "Drafting", and "Forensics" personas live in
:mod:`kaos_agents.personas.builtin` and register themselves into the
default :class:`PersonaRegistry`, but a downstream deployment may
register its own personas in addition to or instead of those.

Hashability + frozen-dataclass equality are preserved by storing the
``{role -> model}`` and ``{key -> value}`` overrides as sorted tuples
of ``(key, value)`` pairs. Convenience accessors
(:meth:`KaosPersona.model_for_role`, :meth:`KaosPersona.budget_for`)
hide the tuple shape from typical callers.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from operator import itemgetter


def _sort_str_pairs(
    pairs: tuple[tuple[str, str], ...],
) -> tuple[tuple[str, str], ...]:
    """Return ``pairs`` sorted lexicographically by key.

    Sorting yields a canonical storage order so two personas with the
    same content but different insertion orders hash + compare equal.
    """
    return tuple(sorted(pairs, key=itemgetter(0)))


def _sort_float_pairs(
    pairs: tuple[tuple[str, float], ...],
) -> tuple[tuple[str, float], ...]:
    """Return ``pairs`` sorted lexicographically by key."""
    return tuple(sorted(pairs, key=itemgetter(0)))


def _mapping_to_str_pairs(items: Mapping[str, str]) -> tuple[tuple[str, str], ...]:
    """Convert a ``Mapping[str, str]`` to a sorted pair-tuple."""
    return _sort_str_pairs(tuple(items.items()))


def _mapping_to_float_pairs(items: Mapping[str, float]) -> tuple[tuple[str, float], ...]:
    """Convert a ``Mapping[str, float]`` to a sorted pair-tuple."""
    return _sort_float_pairs(tuple(items.items()))


@dataclass(frozen=True, slots=True)
class KaosPersona:
    """Named tool-group + model + budget policy bundle.

    Identity is the ``name``. Two personas with the same name are equal
    iff every other field matches (frozen dataclass default equality).

    Args:
        name: Stable kebab-case identifier ("research", "drafting",
            "forensics", "custom-diligence-2026"). Non-empty and
            non-whitespace.
        description: Human-readable summary surfaced in UI selectors
            and audit events. Free-form.
        allowed_groups: Tool-group names the persona enables by default.
            Order is preserved but not semantically significant — the
            tuple is treated as a set for ceiling / elevation checks.
        ceiling_groups: Hard upper bound on auto-elevation. Must be a
            (non-strict) superset of ``allowed_groups``. Equal to
            ``allowed_groups`` when the persona has no headroom for
            auto-elevation (e.g. "Forensics").
        model_role_preferences: ``{role_name -> model_id}`` overrides.
            Accepted as either a ``Mapping`` (typical Python-side
            construction) or a pre-normalized ``tuple[tuple[str, str],
            ...]``. Stored canonically as a sorted pair-tuple so the
            dataclass stays hashable and equality is order-insensitive.
        default_budgets: ``{budget_key -> float}`` soft caps. Conventional
            keys: ``max_loop_cost_usd``, ``max_tool_calls``,
            ``max_loop_wall_clock_seconds``. Same shape contract as
            ``model_role_preferences``.

    Raises:
        ValueError: If ``name`` is empty / whitespace-only, or if any
            entry in ``allowed_groups`` is absent from
            ``ceiling_groups``.
    """

    name: str
    description: str
    allowed_groups: tuple[str, ...] = ()
    ceiling_groups: tuple[str, ...] = ()
    model_role_preferences: tuple[tuple[str, str], ...] = field(default_factory=tuple)
    default_budgets: tuple[tuple[str, float], ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("KaosPersona.name must be a non-empty string")
        if not self.name.strip():
            raise ValueError("KaosPersona.name must not be whitespace-only")

        allowed_set = frozenset(self.allowed_groups)
        ceiling_set = frozenset(self.ceiling_groups)
        if not allowed_set.issubset(ceiling_set):
            missing = sorted(allowed_set - ceiling_set)
            raise ValueError(
                "KaosPersona.ceiling_groups must be a superset of "
                f"allowed_groups; missing from ceiling: {missing}. "
                "Fix: extend ceiling_groups, or remove the extra "
                "entries from allowed_groups."
            )

        # Re-sort the pair-tuple fields so equality is order-insensitive
        # and the dataclass stays hashable regardless of how the
        # constructor was called. ``object.__setattr__`` is the
        # documented escape hatch for frozen-dataclass field rewrites
        # in ``__post_init__``.
        object.__setattr__(
            self,
            "model_role_preferences",
            _sort_str_pairs(self.model_role_preferences),
        )
        object.__setattr__(
            self,
            "default_budgets",
            _sort_float_pairs(self.default_budgets),
        )

    # ─── Construction helpers ────────────────────────────────────────

    @classmethod
    def build(
        cls,
        *,
        name: str,
        description: str,
        allowed_groups: tuple[str, ...] = (),
        ceiling_groups: tuple[str, ...] | None = None,
        model_role_preferences: Mapping[str, str] | None = None,
        default_budgets: Mapping[str, float] | None = None,
    ) -> KaosPersona:
        """Ergonomic builder that accepts plain ``dict`` overrides.

        The canonical constructor stores ``model_role_preferences`` /
        ``default_budgets`` as sorted pair-tuples for hashability. Most
        code paths prefer a plain ``dict`` literal — this builder
        normalizes the inputs and defaults the ceiling to
        ``allowed_groups`` when omitted.
        """
        return cls(
            name=name,
            description=description,
            allowed_groups=allowed_groups,
            ceiling_groups=ceiling_groups if ceiling_groups is not None else allowed_groups,
            model_role_preferences=_mapping_to_str_pairs(model_role_preferences or {}),
            default_budgets=_mapping_to_float_pairs(default_budgets or {}),
        )

    # ─── Set-flavored ergonomics ─────────────────────────────────────

    def allows(self, group: str) -> bool:
        """Whether ``group`` is in the persona's default working set."""
        return group in self.allowed_groups

    def within_ceiling(self, group: str) -> bool:
        """Whether ``group`` is within the persona's auto-elevation ceiling."""
        return group in self.ceiling_groups

    # ─── Mapping-flavored accessors ──────────────────────────────────

    def model_for_role(self, role: str) -> str | None:
        """Return the model override for ``role``, or ``None`` when absent."""
        for key, value in self.model_role_preferences:
            if key == role:
                return value
        return None

    def budget_for(self, key: str) -> float | None:
        """Return the budget override for ``key``, or ``None`` when absent."""
        for k, value in self.default_budgets:
            if k == key:
                return value
        return None

    def model_role_preferences_as_dict(self) -> dict[str, str]:
        """Materialize the role-preference overrides as a plain ``dict``."""
        return dict(self.model_role_preferences)

    def default_budgets_as_dict(self) -> dict[str, float]:
        """Materialize the budget overrides as a plain ``dict``."""
        return dict(self.default_budgets)


__all__ = ["KaosPersona"]
