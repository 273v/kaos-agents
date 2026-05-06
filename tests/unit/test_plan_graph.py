"""Tests for PlanGraph — typed graph wrapper for plan execution."""

from __future__ import annotations

from kaos_agents.planning.graph import PlanGraph
from kaos_agents.types.plan import (
    EvalMode,
    Judgment,
    Step,
    StepStatus,
    StepType,
)


def _tool_step(id: str, name: str = "test-tool", deps: tuple[str, ...] = ()) -> Step:
    return Step(
        id=id,
        step_type=StepType.TOOL,
        description=f"Step {id}",
        tool_name=name,
        depends_on=deps,
    )


def _three_step_plan() -> PlanGraph:
    """s1, s2 (parallel) → s3 (depends on both)."""
    pg = PlanGraph(name="test")
    pg.add_step(_tool_step("s1", "tool-a"))
    pg.add_step(_tool_step("s2", "tool-b"))
    pg.add_step(_tool_step("s3", "tool-c", deps=("s1", "s2")))
    return pg


class TestPlanGraphAdd:
    def test_add_single_step(self):
        pg = PlanGraph()
        pg.add_step(_tool_step("s1"))
        assert pg.n_steps == 1
        assert "s1" in pg

    def test_add_with_dependencies(self):
        pg = _three_step_plan()
        assert pg.n_steps == 3

    def test_get_step(self):
        pg = PlanGraph()
        pg.add_step(_tool_step("s1", "my-tool"))
        props = pg.get_step("s1")
        assert props is not None
        assert props["tool_name"] == "my-tool"
        assert props["status"] == StepStatus.PENDING.value

    def test_get_missing_step(self):
        pg = PlanGraph()
        assert pg.get_step("nonexistent") is None


class TestPlanGraphReadySteps:
    def test_no_deps_all_ready(self):
        pg = PlanGraph()
        pg.add_step(_tool_step("s1"))
        pg.add_step(_tool_step("s2"))
        ready = pg.get_ready_steps()
        assert set(ready) == {"s1", "s2"}

    def test_deps_block_downstream(self):
        pg = _three_step_plan()
        ready = pg.get_ready_steps()
        assert set(ready) == {"s1", "s2"}
        assert "s3" not in ready

    def test_completing_deps_unblocks(self):
        pg = _three_step_plan()
        pg.mark_complete("s1", "done")
        assert "s3" not in pg.get_ready_steps()  # s2 still pending

        pg.mark_complete("s2", "done")
        assert "s3" in pg.get_ready_steps()

    def test_failed_dep_does_not_unblock(self):
        pg = _three_step_plan()
        pg.mark_complete("s1", "done")
        pg.mark_failed("s2", "timeout")
        assert "s3" not in pg.get_ready_steps()


class TestPlanGraphStatusUpdates:
    def test_mark_running(self):
        pg = PlanGraph()
        pg.add_step(_tool_step("s1"))
        pg.mark_running("s1")
        props = pg.get_step("s1")
        assert props is not None
        assert props["status"] == StepStatus.RUNNING.value

    def test_mark_complete_with_judgment(self):
        pg = PlanGraph()
        pg.add_step(_tool_step("s1"))
        j = Judgment(matched=True, confidence=0.95, reasoning="OK", mode=EvalMode.STRUCTURAL)
        pg.mark_complete("s1", "result data", j)
        props = pg.get_step("s1")
        assert props is not None
        assert props["status"] == StepStatus.COMPLETED.value
        assert props["result"] == "result data"
        assert props["judgment"]["confidence"] == 0.95

    def test_mark_failed(self):
        pg = PlanGraph()
        pg.add_step(_tool_step("s1"))
        pg.mark_failed("s1", "Connection timeout")
        props = pg.get_step("s1")
        assert props is not None
        assert props["status"] == StepStatus.FAILED.value
        assert "Connection timeout" in props["result"]

    def test_mark_skipped(self):
        pg = PlanGraph()
        pg.add_step(_tool_step("s1"))
        pg.mark_skipped("s1", "Dependency failed")
        props = pg.get_step("s1")
        assert props is not None
        assert props["status"] == StepStatus.SKIPPED.value


class TestPlanGraphExecutionLevels:
    def test_single_level(self):
        pg = PlanGraph()
        pg.add_step(_tool_step("s1"))
        pg.add_step(_tool_step("s2"))
        levels = pg.get_execution_levels()
        assert len(levels) == 1
        assert set(levels[0]) == {"s1", "s2"}

    def test_two_levels(self):
        pg = _three_step_plan()
        levels = pg.get_execution_levels()
        assert len(levels) == 2
        assert set(levels[0]) == {"s1", "s2"}
        assert levels[1] == ["s3"]

    def test_three_levels(self):
        pg = PlanGraph()
        pg.add_step(_tool_step("s1"))
        pg.add_step(_tool_step("s2", deps=("s1",)))
        pg.add_step(_tool_step("s3", deps=("s2",)))
        levels = pg.get_execution_levels()
        assert len(levels) == 3


class TestPlanGraphCompletion:
    def test_not_complete_initially(self):
        pg = _three_step_plan()
        assert not pg.is_complete()

    def test_complete_after_all_done(self):
        pg = _three_step_plan()
        pg.mark_complete("s1", "done")
        pg.mark_complete("s2", "done")
        pg.mark_complete("s3", "done")
        assert pg.is_complete()

    def test_skipped_counts_as_complete(self):
        pg = PlanGraph()
        pg.add_step(_tool_step("s1"))
        pg.mark_skipped("s1", "not needed")
        assert pg.is_complete()

    def test_has_failures(self):
        pg = _three_step_plan()
        assert not pg.has_failures()
        pg.mark_failed("s1", "error")
        assert pg.has_failures()


class TestPlanGraphValidation:
    def test_valid_plan(self):
        pg = _three_step_plan()
        assert pg.validate() == []

    def test_missing_tool_name(self):
        pg = PlanGraph()
        pg.add_step(Step(id="s1", step_type=StepType.TOOL, description="test"))
        issues = pg.validate()
        assert any("no tool_name" in i for i in issues)

    def test_orphan_detection(self):
        pg = PlanGraph()
        pg.add_step(_tool_step("s1"))
        pg.add_step(_tool_step("s2"))
        pg.add_step(_tool_step("s3", deps=("s1",)))
        issues = pg.validate()
        assert any("orphan" in i for i in issues)  # s2 is orphan


class TestPlanGraphSubplanInsertion:
    def test_insert_subplan(self):
        # Main: s1 → s2 → s3
        main = PlanGraph()
        main.add_step(_tool_step("s1"))
        main.add_step(_tool_step("s2", deps=("s1",)))
        main.add_step(_tool_step("s3", deps=("s2",)))

        # Subplan for s2: sub-a → sub-b
        sub = PlanGraph()
        sub.add_step(_tool_step("sub-a"))
        sub.add_step(_tool_step("sub-b", deps=("sub-a",)))

        main.insert_subplan("s2", sub)

        # s1 → sub-a → sub-b → s3
        assert "s2" not in main
        assert "sub-a" in main
        assert "sub-b" in main
        assert main.n_steps == 4  # s1 + sub-a + sub-b + s3

    def test_insert_subplan_preserves_edges(self):
        main = PlanGraph()
        main.add_step(_tool_step("s1"))
        main.add_step(_tool_step("s2", deps=("s1",)))
        main.add_step(_tool_step("s3", deps=("s2",)))

        sub = PlanGraph()
        sub.add_step(_tool_step("sub-a"))

        main.insert_subplan("s2", sub)

        # sub-a should have s1 as predecessor and s3 as successor
        main.mark_complete("s1", "done")
        assert "sub-a" in main.get_ready_steps()

        main.mark_complete("sub-a", "done")
        assert "s3" in main.get_ready_steps()


class TestPlanGraphSerialization:
    def test_json_round_trip(self):
        pg = _three_step_plan()
        pg.mark_complete("s1", "result-1")

        data = pg.to_json()
        pg2 = PlanGraph.from_json(data)

        assert pg2.n_steps == 3
        props = pg2.get_step("s1")
        assert props is not None
        assert props["status"] == StepStatus.COMPLETED.value
        assert props["result"] == "result-1"

    def test_mermaid_export(self):
        pg = _three_step_plan()
        pg.mark_complete("s1", "done")
        pg.mark_failed("s2", "error")

        mermaid = pg.to_mermaid()
        assert "graph TD" in mermaid
        assert "s1" in mermaid
        assert "s2" in mermaid
        assert "fill:#90EE90" in mermaid  # completed = green
        assert "fill:#FFB6C1" in mermaid  # failed = red


class TestPlanGraphCriticalPath:
    def test_critical_path(self):
        pg = PlanGraph()
        pg.add_step(
            Step(
                id="s1",
                step_type=StepType.TOOL,
                description="fast",
                tool_name="t",
                estimated_latency_ms=100,
            )
        )
        pg.add_step(
            Step(
                id="s2",
                step_type=StepType.TOOL,
                description="slow",
                tool_name="t",
                estimated_latency_ms=500,
            )
        )
        pg.add_step(
            Step(
                id="s3",
                step_type=StepType.TOOL,
                description="merge",
                tool_name="t",
                depends_on=("s1", "s2"),
                estimated_latency_ms=50,
            )
        )

        path, cost = pg.get_critical_path()
        assert "s2" in path  # slow path is critical
        assert cost > 0
