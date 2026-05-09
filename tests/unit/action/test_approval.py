"""ApprovalWorkflow predicate + event builder."""

from __future__ import annotations

from kaos_agents.action.approval import ApprovalWorkflow
from kaos_agents.action.reversibility import Reversibility
from kaos_agents.action.types import ActionPlan
from kaos_agents.events.tools import ToolCallApprovalRequired


class TestRequiredFor:
    def test_reversible_never_required(self) -> None:
        wf = ApprovalWorkflow()
        plan = ActionPlan(
            tool_name="kaos-pdf-render",
            reversibility=Reversibility.REVERSIBLE,
            approval_required=True,  # ignored for REVERSIBLE
        )
        assert wf.required_for(plan) is False

    def test_recoverable_never_required(self) -> None:
        wf = ApprovalWorkflow()
        plan = ActionPlan(
            tool_name="kaos-tabular-insert",
            reversibility=Reversibility.RECOVERABLE,
            approval_required=True,  # ignored
        )
        assert wf.required_for(plan) is False

    def test_externally_visible_follows_plan_flag(self) -> None:
        wf = ApprovalWorkflow()
        with_flag = ActionPlan(
            tool_name="kaos-source-pacer-fetch",
            reversibility=Reversibility.EXTERNALLY_VISIBLE,
            approval_required=True,
        )
        without_flag = with_flag.model_copy(update={"approval_required": False})
        assert wf.required_for(with_flag) is True
        assert wf.required_for(without_flag) is False

    def test_irreversible_always_required(self) -> None:
        wf = ApprovalWorkflow()
        plan_with = ActionPlan(
            tool_name="kaos-source-pacer-file",
            reversibility=Reversibility.IRREVERSIBLE,
            approval_required=True,
        )
        plan_without = plan_with.model_copy(update={"approval_required": False})
        assert wf.required_for(plan_with) is True
        # Even with the flag flipped off, IRREVERSIBLE always requires.
        assert wf.required_for(plan_without) is True


class TestBuildEvent:
    def test_returns_tool_call_approval_required(self) -> None:
        wf = ApprovalWorkflow()
        plan = ActionPlan(
            tool_name="kaos-source-pacer-file",
            args={"docket": "1234", "page_count": 3},
            reversibility=Reversibility.IRREVERSIBLE,
        )
        cls, kwargs = wf.build_event(plan, reason="user-initiated")
        assert cls is ToolCallApprovalRequired
        assert kwargs["tool_name"] == "kaos-source-pacer-file"
        assert kwargs["reason"] == "user-initiated"
        # Arguments encoded as a tuple-of-pairs of stringified values.
        args_tuple = kwargs["arguments"]
        assert isinstance(args_tuple, tuple)
        assert ("docket", repr("1234")) in args_tuple
        assert ("page_count", repr(3)) in args_tuple

    def test_default_reason_includes_reversibility(self) -> None:
        wf = ApprovalWorkflow()
        plan = ActionPlan(
            tool_name="kaos-source-pacer-file",
            reversibility=Reversibility.IRREVERSIBLE,
        )
        _, kwargs = wf.build_event(plan)
        assert "reversibility=irreversible" in kwargs["reason"]

    def test_event_constructible(self) -> None:
        wf = ApprovalWorkflow()
        plan = ActionPlan(
            tool_name="kaos-source-pacer-file",
            args={"x": 1},
            reversibility=Reversibility.IRREVERSIBLE,
        )
        cls, kwargs = wf.build_event(plan)
        # Verify the kwargs match the actual ToolCallApprovalRequired
        # constructor — guards against drift in the event signature.
        # build_event() returns the action-specific kwargs only; the
        # KaosEvent base fields are filled in by the EventEmitter at
        # runtime, so the test supplies them directly here.
        event = cls(timestamp=0.0, sequence=0, session_id="s1", run_id="r1", **kwargs)
        assert isinstance(event, ToolCallApprovalRequired)
        assert event.tool_name == "kaos-source-pacer-file"
