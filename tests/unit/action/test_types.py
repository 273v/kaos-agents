"""ActionPlan / ActionResult / ActionRefusal shape + JSON round-trip."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from kaos_agents.action.reversibility import Reversibility
from kaos_agents.action.types import ActionPlan, ActionRefusal, ActionResult


class TestActionPlan:
    def test_defaults(self) -> None:
        plan = ActionPlan(tool_name="kaos-source-pacer-fetch")
        assert plan.tool_name == "kaos-source-pacer-fetch"
        assert plan.args == {}
        assert plan.reversibility == Reversibility.IRREVERSIBLE
        assert plan.approval_required is True
        assert plan.reversal_strategy is None
        assert plan.rationale == ""

    def test_explicit_construction(self) -> None:
        plan = ActionPlan(
            tool_name="kaos-pdf-render",
            args={"path": "/tmp/x.pdf", "page": 1},
            reversibility=Reversibility.REVERSIBLE,
            approval_required=False,
            reversal_strategy="rm /tmp/x.pdf",
            rationale="render page 1 for inspection",
        )
        assert plan.reversibility == Reversibility.REVERSIBLE
        assert plan.approval_required is False
        assert plan.reversal_strategy == "rm /tmp/x.pdf"

    def test_frozen(self) -> None:
        plan = ActionPlan(tool_name="x")
        with pytest.raises(ValidationError):
            plan.tool_name = "y"  # type: ignore[misc]

    def test_extra_forbidden(self) -> None:
        with pytest.raises(ValidationError):
            ActionPlan(tool_name="x", unknown_field=42)  # ty: ignore[unknown-argument]

    def test_json_round_trip(self) -> None:
        plan = ActionPlan(
            tool_name="kaos-source-pacer-fetch",
            args={"docket": "1234"},
            reversibility=Reversibility.EXTERNALLY_VISIBLE,
            approval_required=True,
            rationale="user asked",
        )
        as_json = plan.model_dump_json()
        restored = ActionPlan.model_validate_json(as_json)
        assert restored == plan


class TestActionResult:
    def test_defaults(self) -> None:
        r = ActionResult(tool_name="x")
        assert r.success is True
        assert r.output is None
        assert r.side_effects == ()
        assert r.reversibility_actually == Reversibility.IRREVERSIBLE
        assert r.duration_ms is None
        assert r.cost_usd is None

    def test_construction(self) -> None:
        r = ActionResult(
            tool_name="kaos-pdf-render",
            output={"page_image": "data:..."},
            side_effects=("wrote /tmp/render.png",),
            reversibility_actually=Reversibility.REVERSIBLE,
            duration_ms=42.5,
            cost_usd=0.0,
        )
        assert r.side_effects == ("wrote /tmp/render.png",)
        assert r.duration_ms == 42.5

    def test_frozen(self) -> None:
        r = ActionResult(tool_name="x")
        with pytest.raises(ValidationError):
            r.success = False  # type: ignore[misc]

    def test_json_round_trip(self) -> None:
        r = ActionResult(
            tool_name="x",
            output=[1, 2, 3],
            side_effects=("a", "b"),
            duration_ms=10.0,
            cost_usd=0.001,
        )
        restored = ActionResult.model_validate_json(r.model_dump_json())
        assert restored == r


class TestActionRefusal:
    def test_defaults(self) -> None:
        r = ActionRefusal(tool_name="x", reason="why", kind="permission_denied")
        assert r.recommended_alternatives == ()
        assert r.retry_after_seconds is None

    def test_frozen(self) -> None:
        r = ActionRefusal(tool_name="x", reason="r", kind="rate_limit")
        with pytest.raises(ValidationError):
            r.kind = "circuit_open"  # type: ignore[misc]

    def test_extra_forbidden(self) -> None:
        with pytest.raises(ValidationError):
            ActionRefusal(
                tool_name="x",
                reason="r",
                kind="rate_limit",
                bogus=1,  # ty: ignore[unknown-argument]
            )

    def test_json_round_trip(self) -> None:
        r = ActionRefusal(
            tool_name="kaos-source-pacer-fetch",
            reason="rate limit hit",
            kind="rate_limit",
            recommended_alternatives=("kaos-source-pacer-cache-read",),
            retry_after_seconds=2.5,
        )
        restored = ActionRefusal.model_validate_json(r.model_dump_json())
        assert restored == r
