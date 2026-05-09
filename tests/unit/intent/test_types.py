"""Tests for the Intent subsystem value types.

Construction defaults, frozen-mutation rejection,
``IntentResult.has_blocking_ambiguity()`` truth table, and JSON
round-trip via Pydantic ``model_dump_json`` / ``model_validate_json``.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from kaos_agents.config import AgentPattern
from kaos_agents.intent.types import (
    Ambiguity,
    AmbiguityKind,
    Constraint,
    ConstraintKind,
    Goal,
    IntentResult,
)
from kaos_agents.types.intents import IntentType


def _build_goal() -> Goal:
    return Goal(
        statement="Summarize the contract.",
        intent_type=IntentType.RESEARCH,
        success_criteria=("identify all defined terms", "list every covenant"),
        target_format="markdown",
        domain="legal",
        matter_client=("M-100", "ACME"),
    )


class TestGoal:
    def test_construction_defaults(self):
        g = Goal(statement="Hello", intent_type=IntentType.RESPOND)
        assert g.statement == "Hello"
        assert g.intent_type is IntentType.RESPOND
        assert g.success_criteria == ()
        assert g.target_format is None
        assert g.domain is None
        assert g.matter_client is None

    def test_full_construction(self):
        g = _build_goal()
        assert g.success_criteria == (
            "identify all defined terms",
            "list every covenant",
        )
        assert g.target_format == "markdown"
        assert g.matter_client == ("M-100", "ACME")

    def test_frozen_mutation_rejected(self):
        g = Goal(statement="x", intent_type=IntentType.RESPOND)
        with pytest.raises(ValidationError):
            g.statement = "y"  # type: ignore[misc]

    def test_extra_fields_forbidden(self):
        # Use model_validate + a dict so ty does not flag the unknown
        # kwarg statically — extra="forbid" rejects it at runtime.
        with pytest.raises(ValidationError):
            Goal.model_validate(
                {
                    "statement": "x",
                    "intent_type": IntentType.RESPOND.value,
                    "extra_field": "boom",
                }
            )

    def test_json_round_trip(self):
        g = _build_goal()
        revived = Goal.model_validate_json(g.model_dump_json())
        assert revived == g


class TestConstraint:
    def test_construction_defaults(self):
        c = Constraint(kind=ConstraintKind.DEADLINE, value="by Friday")
        assert c.kind is ConstraintKind.DEADLINE
        assert c.value == "by Friday"
        assert c.mandatory is True

    def test_optional_constraint(self):
        c = Constraint(kind=ConstraintKind.STYLE, value="bullet points", mandatory=False)
        assert c.mandatory is False

    def test_frozen_mutation_rejected(self):
        c = Constraint(kind=ConstraintKind.BUDGET, value="$10")
        with pytest.raises(ValidationError):
            c.value = "$100"  # type: ignore[misc]

    def test_kind_string_value(self):
        # ConstraintKind is a StrEnum so its value is the string itself.
        assert ConstraintKind.JURISDICTION.value == "jurisdiction"
        c = Constraint(kind=ConstraintKind("scope"), value="EDGAR only")
        assert c.kind is ConstraintKind.SCOPE

    def test_json_round_trip(self):
        c = Constraint(kind=ConstraintKind.FORMAT, value="json", mandatory=False)
        revived = Constraint.model_validate_json(c.model_dump_json())
        assert revived == c


class TestAmbiguity:
    def test_construction_defaults(self):
        a = Ambiguity(kind=AmbiguityKind.UNKNOWN_REFERENCE, span=(0, 12), excerpt="the contract")
        assert a.kind is AmbiguityKind.UNKNOWN_REFERENCE
        assert a.span == (0, 12)
        assert a.excerpt == "the contract"
        assert a.candidate_interpretations == ()
        assert a.preferred_clarification == ""

    def test_full_construction(self):
        a = Ambiguity(
            kind=AmbiguityKind.AMBIGUOUS_PRONOUN,
            span=(15, 19),
            excerpt="they",
            candidate_interpretations=("the parties", "the courts"),
            preferred_clarification="Who does 'they' refer to?",
        )
        assert a.candidate_interpretations == ("the parties", "the courts")
        assert a.preferred_clarification == "Who does 'they' refer to?"

    def test_frozen_mutation_rejected(self):
        a = Ambiguity(kind=AmbiguityKind.MISSING_CONTEXT, span=(0, 0), excerpt="")
        with pytest.raises(ValidationError):
            a.excerpt = "boom"  # type: ignore[misc]

    def test_json_round_trip(self):
        a = Ambiguity(
            kind=AmbiguityKind.CONFLICTING_REQUIREMENTS,
            span=(40, 52),
            excerpt="cheap and best",
            candidate_interpretations=("optimize cost", "optimize quality"),
            preferred_clarification="Which should I prioritize?",
        )
        revived = Ambiguity.model_validate_json(a.model_dump_json())
        assert revived == a


class TestIntentResult:
    def test_construction_minimal(self):
        ir = IntentResult(goal=_build_goal())
        assert ir.constraints == ()
        assert ir.ambiguities == ()
        assert ir.requires_clarification is False
        assert ir.pattern is AgentPattern.CHAT
        assert ir.confidence == 1.0
        assert ir.raw_input == ""

    def test_full_construction(self):
        ir = IntentResult(
            goal=_build_goal(),
            constraints=(Constraint(kind=ConstraintKind.DEADLINE, value="EOD"),),
            ambiguities=(
                Ambiguity(
                    kind=AmbiguityKind.UNKNOWN_REFERENCE,
                    span=(0, 3),
                    excerpt="the",
                ),
            ),
            requires_clarification=False,
            pattern=AgentPattern.RESEARCH,
            confidence=0.83,
            raw_input="Summarize the contract by EOD.",
        )
        assert ir.pattern is AgentPattern.RESEARCH
        assert ir.confidence == 0.83
        assert ir.raw_input.startswith("Summarize")

    def test_confidence_bounds_enforced(self):
        with pytest.raises(ValidationError):
            IntentResult(goal=_build_goal(), confidence=1.5)
        with pytest.raises(ValidationError):
            IntentResult(goal=_build_goal(), confidence=-0.1)

    def test_extra_fields_forbidden(self):
        # Use model_validate + a dict so ty does not flag the unknown
        # kwarg statically — extra="forbid" rejects it at runtime.
        with pytest.raises(ValidationError):
            IntentResult.model_validate({"goal": _build_goal().model_dump(), "bonus": "boom"})

    def test_frozen_mutation_rejected(self):
        ir = IntentResult(goal=_build_goal())
        with pytest.raises(ValidationError):
            ir.requires_clarification = True  # type: ignore[misc]

    def test_has_blocking_ambiguity_empty(self):
        ir = IntentResult(goal=_build_goal())
        assert ir.has_blocking_ambiguity() is False

    def test_has_blocking_ambiguity_only_non_blocking(self):
        ir = IntentResult(
            goal=_build_goal(),
            ambiguities=(
                Ambiguity(
                    kind=AmbiguityKind.UNKNOWN_REFERENCE,
                    span=(0, 3),
                    excerpt="the",
                ),
                Ambiguity(
                    kind=AmbiguityKind.AMBIGUOUS_PRONOUN,
                    span=(4, 8),
                    excerpt="them",
                ),
                Ambiguity(
                    kind=AmbiguityKind.CONFLICTING_REQUIREMENTS,
                    span=(9, 20),
                    excerpt="cheap fast",
                ),
            ),
        )
        assert ir.has_blocking_ambiguity() is False

    def test_has_blocking_ambiguity_missing_context(self):
        ir = IntentResult(
            goal=_build_goal(),
            ambiguities=(
                Ambiguity(
                    kind=AmbiguityKind.MISSING_CONTEXT,
                    span=(0, 0),
                    excerpt="",
                ),
            ),
        )
        assert ir.has_blocking_ambiguity() is True

    def test_has_blocking_ambiguity_out_of_domain(self):
        ir = IntentResult(
            goal=_build_goal(),
            ambiguities=(
                Ambiguity(
                    kind=AmbiguityKind.OUT_OF_DOMAIN,
                    span=(0, 5),
                    excerpt="hello",
                ),
            ),
        )
        assert ir.has_blocking_ambiguity() is True

    def test_has_blocking_ambiguity_mixed(self):
        # As soon as one blocking kind appears, has_blocking_ambiguity is True.
        ir = IntentResult(
            goal=_build_goal(),
            ambiguities=(
                Ambiguity(
                    kind=AmbiguityKind.UNKNOWN_REFERENCE,
                    span=(0, 3),
                    excerpt="the",
                ),
                Ambiguity(
                    kind=AmbiguityKind.MISSING_CONTEXT,
                    span=(10, 18),
                    excerpt="contract",
                ),
            ),
        )
        assert ir.has_blocking_ambiguity() is True

    def test_json_round_trip_minimal(self):
        ir = IntentResult(goal=_build_goal())
        revived = IntentResult.model_validate_json(ir.model_dump_json())
        assert revived == ir

    def test_json_round_trip_full(self):
        ir = IntentResult(
            goal=_build_goal(),
            constraints=(
                Constraint(kind=ConstraintKind.DEADLINE, value="EOD"),
                Constraint(kind=ConstraintKind.FORMAT, value="markdown", mandatory=False),
            ),
            ambiguities=(
                Ambiguity(
                    kind=AmbiguityKind.UNKNOWN_REFERENCE,
                    span=(0, 3),
                    excerpt="the",
                    candidate_interpretations=("contract A", "contract B"),
                    preferred_clarification="Which contract?",
                ),
            ),
            requires_clarification=True,
            pattern=AgentPattern.PLAN,
            confidence=0.42,
            raw_input="Summarize the contract.",
        )
        revived = IntentResult.model_validate_json(ir.model_dump_json())
        assert revived == ir
        # Defensive: spot-check a couple of nested fields survived the round-trip.
        assert revived.ambiguities[0].candidate_interpretations == ("contract A", "contract B")
        assert revived.constraints[1].mandatory is False
