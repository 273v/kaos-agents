"""Reversibility — paper §5.3 four-tier framework.

Today, KaosTool annotations declare ``readOnlyHint`` and
``destructiveHint`` as booleans — too coarse. Reversibility is the
principal axis for action gating: ``REVERSIBLE`` actions auto-allow,
``RECOVERABLE`` actions log, ``EXTERNALLY_VISIBLE`` actions ask, and
``IRREVERSIBLE`` actions always require dual-key approval.

Phase 1.C introduces the enum and the helper that infers a default tier
from the existing legacy hints (since most tools haven't been
re-annotated yet). New tools should declare reversibility explicitly via
the ``reversibility`` annotation key (read by
:func:`infer_reversibility`).

The default for a tool with NO annotations is ``IRREVERSIBLE`` — fail
safe.
"""

from __future__ import annotations

from enum import StrEnum, unique
from typing import Any


@unique
class Reversibility(StrEnum):
    """Four-tier reversibility classification for tool actions.

    Ordered loosely from "safest" to "most dangerous":

    - :attr:`REVERSIBLE` — the action can be undone with no residual
      side effects (e.g. filesystem write to a scratch dir we own).
    - :attr:`RECOVERABLE` — the action can be undone by a known reversal
      strategy (e.g. DB insert with a rollback transaction or DELETE).
    - :attr:`EXTERNALLY_VISIBLE` — the action causes a visible side
      effect outside the agent's own state (email send, API call) but
      doesn't permanently mutate critical state.
    - :attr:`IRREVERSIBLE` — the action causes a permanent change that
      cannot be undone (legal filing, payment, public release).
    """

    REVERSIBLE = "reversible"
    RECOVERABLE = "recoverable"
    EXTERNALLY_VISIBLE = "externally_visible"
    IRREVERSIBLE = "irreversible"


def _get(annotations: Any, key: str, default: Any = None) -> Any:
    """Read ``key`` from ``annotations`` whether it's a dict or attr-shaped."""
    if annotations is None:
        return default
    if isinstance(annotations, dict):
        return annotations.get(key, default)
    return getattr(annotations, key, default)


def infer_reversibility(annotations: Any) -> Reversibility:
    """Infer a :class:`Reversibility` tier from a tool's annotations.

    Priority:

    1. Explicit ``reversibility`` field (if present in annotations)
       wins. Accepts a :class:`Reversibility` instance or any string
       parseable as one.
    2. ``readOnlyHint=True`` → :attr:`Reversibility.REVERSIBLE`
       (read-only is reversible by definition).
    3. ``destructiveHint=True`` → :attr:`Reversibility.IRREVERSIBLE`
       (worst-case fallback under coarse legacy hints).
    4. Neither hint set, no explicit reversibility →
       :attr:`Reversibility.IRREVERSIBLE` (fail safe).

    Accepts dict-shaped or attribute-shaped annotations (duck-typed)
    and ``None``.
    """
    explicit = _get(annotations, "reversibility")
    if explicit is not None:
        if isinstance(explicit, Reversibility):
            return explicit
        try:
            return Reversibility(explicit)
        except ValueError:
            # Unknown string; fall through to legacy-hint inference.
            pass

    if bool(_get(annotations, "readOnlyHint", False)):
        return Reversibility.REVERSIBLE

    if bool(_get(annotations, "destructiveHint", False)):
        return Reversibility.IRREVERSIBLE

    return Reversibility.IRREVERSIBLE


__all__ = ["Reversibility", "infer_reversibility"]
