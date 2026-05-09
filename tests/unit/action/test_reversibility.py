"""Reversibility enum + infer_reversibility truth table."""

from __future__ import annotations

from dataclasses import dataclass

from kaos_agents.action.reversibility import Reversibility, infer_reversibility


class TestReversibilityEnum:
    def test_members(self) -> None:
        assert Reversibility.REVERSIBLE.value == "reversible"
        assert Reversibility.RECOVERABLE.value == "recoverable"
        assert Reversibility.EXTERNALLY_VISIBLE.value == "externally_visible"
        assert Reversibility.IRREVERSIBLE.value == "irreversible"

    def test_str_round_trip(self) -> None:
        for tier in Reversibility:
            assert Reversibility(tier.value) is tier

    def test_strenum(self) -> None:
        # StrEnum values compare equal to their string.
        assert Reversibility.REVERSIBLE == "reversible"


class TestInferFromDict:
    def test_explicit_field_wins_enum(self) -> None:
        annotations = {
            "reversibility": Reversibility.RECOVERABLE,
            "readOnlyHint": True,
            "destructiveHint": True,
        }
        # Explicit wins over both legacy hints.
        assert infer_reversibility(annotations) == Reversibility.RECOVERABLE

    def test_explicit_field_wins_string(self) -> None:
        annotations = {"reversibility": "externally_visible", "readOnlyHint": True}
        assert infer_reversibility(annotations) == Reversibility.EXTERNALLY_VISIBLE

    def test_explicit_invalid_string_falls_back(self) -> None:
        # Unknown reversibility string falls through to legacy hints.
        annotations = {"reversibility": "nonsense", "readOnlyHint": True}
        assert infer_reversibility(annotations) == Reversibility.REVERSIBLE

    def test_read_only_implies_reversible(self) -> None:
        annotations = {"readOnlyHint": True}
        assert infer_reversibility(annotations) == Reversibility.REVERSIBLE

    def test_destructive_implies_irreversible(self) -> None:
        annotations = {"destructiveHint": True}
        assert infer_reversibility(annotations) == Reversibility.IRREVERSIBLE

    def test_read_only_takes_priority_over_destructive(self) -> None:
        # Pathological case — readOnlyHint and destructiveHint both
        # set; the helper returns REVERSIBLE because read-only is
        # checked first (a tool that is read-only cannot be
        # meaningfully destructive).
        annotations = {"readOnlyHint": True, "destructiveHint": True}
        assert infer_reversibility(annotations) == Reversibility.REVERSIBLE

    def test_neither_hint_set_is_irreversible(self) -> None:
        annotations: dict[str, bool] = {}
        assert infer_reversibility(annotations) == Reversibility.IRREVERSIBLE

    def test_none_annotations_irreversible(self) -> None:
        assert infer_reversibility(None) == Reversibility.IRREVERSIBLE


@dataclass
class _AttrAnnotations:
    """Attribute-shaped annotations used to verify duck-typing."""

    readOnlyHint: bool = False
    destructiveHint: bool = False
    reversibility: Reversibility | str | None = None


class TestInferFromAttribute:
    def test_explicit_field_wins(self) -> None:
        annotations = _AttrAnnotations(
            reversibility=Reversibility.EXTERNALLY_VISIBLE,
            destructiveHint=True,
        )
        assert infer_reversibility(annotations) == Reversibility.EXTERNALLY_VISIBLE

    def test_read_only_implies_reversible(self) -> None:
        annotations = _AttrAnnotations(readOnlyHint=True)
        assert infer_reversibility(annotations) == Reversibility.REVERSIBLE

    def test_destructive_implies_irreversible(self) -> None:
        annotations = _AttrAnnotations(destructiveHint=True)
        assert infer_reversibility(annotations) == Reversibility.IRREVERSIBLE

    def test_neither_hint_set_is_irreversible(self) -> None:
        annotations = _AttrAnnotations()
        assert infer_reversibility(annotations) == Reversibility.IRREVERSIBLE
