"""PlanGraph — typed wrapper around kaos-graph for plan execution.

The plan is a directed graph where nodes are Steps and edges are
dependencies. PlanGraph adds step-typed operations (add_step,
get_ready_steps, mark_complete) on top of the raw graph API.

Supports:
- DAG validation and topological execution levels
- Mutable node properties (status, result, judgment) via update_node
- Subplan insertion (replace a node with a subgraph)
- Mermaid export for visualization
- JSON round-trip for VFS persistence
"""

from __future__ import annotations

from typing import Any

from kaos_core.logging import get_logger
from kaos_graph import Graph
from kaos_graph.algorithms import critical_path, find_cycles, topological_sort

from kaos_agents.planning.types import (
    Judgment,
    Step,
    StepStatus,
    StepType,
)

logger = get_logger(__name__)


class PlanGraph:
    """A plan represented as a directed graph of Steps.

    Wraps `kaos_graph.Graph` with step-typed node/edge operations and
    execution-lifecycle tracking.
    """

    __slots__ = ("_graph",)

    def __init__(self, name: str = "plan") -> None:
        self._graph = Graph(directed=True, name=name)

    # -- Step operations -----------------------------------------------------

    def add_step(self, step: Step) -> None:
        """Add a step to the plan graph.

        Creates the node with step properties and edges for declared
        dependencies. Dependency targets must already exist.
        """
        self._graph.add_node(
            step.id,
            step_type=step.step_type.value,
            description=step.description,
            tool_name=step.tool_name or "",
            input_spec=step.input_spec,
            expected_output=step.expected_output,
            status=StepStatus.PENDING.value,
            result=None,
            judgment=None,
            latency_ms=step.estimated_latency_ms,
        )
        for dep_id in step.depends_on:
            if dep_id in self._graph:
                self._graph.add_edge(dep_id, step.id, type="dependency")
            else:
                logger.warning(
                    "plan_graph.add_step: dependency %s not found for step %s",
                    dep_id,
                    step.id,
                )

    def get_step(self, step_id: str) -> dict[str, Any] | None:
        """Get step properties by ID. Returns None if not found."""
        node = self._graph.node(step_id)
        if node is None:
            return None
        return {
            k: node.get(k)
            for k in (
                "step_type",
                "description",
                "tool_name",
                "input_spec",
                "expected_output",
                "status",
                "result",
                "judgment",
                "latency_ms",
            )
        }

    def step_ids(self) -> list[str]:
        """All step IDs in the graph."""
        return self._graph.node_ids()

    @property
    def n_steps(self) -> int:
        return self._graph.n_nodes

    # -- Status updates ------------------------------------------------------

    def mark_running(self, step_id: str) -> None:
        """Mark a step as running."""
        self._graph.set_node_property(step_id, "status", StepStatus.RUNNING.value)

    def mark_complete(
        self,
        step_id: str,
        result: Any,
        judgment: Judgment | None = None,
    ) -> None:
        """Mark a step as completed with its result and optional judgment."""
        self._graph.update_node(
            step_id,
            status=StepStatus.COMPLETED.value,
            result=str(result) if result is not None else "",
        )
        if judgment is not None:
            self._graph.set_node_property(
                step_id,
                "judgment",
                {
                    "matched": judgment.matched,
                    "confidence": judgment.confidence,
                    "reasoning": judgment.reasoning,
                    "mode": judgment.mode.value,
                },
            )

    def mark_failed(self, step_id: str, error: str) -> None:
        """Mark a step as failed."""
        self._graph.update_node(
            step_id,
            status=StepStatus.FAILED.value,
            result=f"ERROR: {error}",
        )

    def mark_skipped(self, step_id: str, reason: str) -> None:
        """Mark a step as skipped."""
        self._graph.update_node(
            step_id,
            status=StepStatus.SKIPPED.value,
            result=f"SKIPPED: {reason}",
        )

    # -- Execution queries ---------------------------------------------------

    def get_ready_steps(self) -> list[str]:
        """Get step IDs that are ready to execute.

        A step is ready if:
        - Its status is PENDING
        - All predecessor steps are COMPLETED
        """
        ready = []
        for step_id in self._graph.node_ids():
            node = self._graph.node(step_id)
            if node is None or node.get("status") != StepStatus.PENDING.value:
                continue
            preds = self._graph.predecessors(step_id)
            all_deps_done = True
            for p in preds:
                pred_node = self._graph.node(p)
                if pred_node is None or pred_node.get("status") != StepStatus.COMPLETED.value:
                    all_deps_done = False
                    break
            if all_deps_done:
                ready.append(step_id)
        return ready

    def get_execution_levels(self) -> list[list[str]]:
        """Get steps grouped by topological level for parallel execution.

        Level 0: steps with no dependencies
        Level 1: steps depending only on level-0 steps
        ...and so on.
        """
        if not self._graph.is_dag():
            logger.warning("plan_graph: graph has cycles, falling back to flat ordering")
            return [topological_sort(self._graph)]

        order = topological_sort(self._graph)
        if not order:
            return []

        # Assign levels by maximum predecessor level + 1
        levels: dict[str, int] = {}
        for step_id in order:
            preds = self._graph.predecessors(step_id)
            if not preds:
                levels[step_id] = 0
            else:
                levels[step_id] = max(levels.get(p, 0) for p in preds) + 1

        # Group by level
        max_level = max(levels.values()) if levels else 0
        result: list[list[str]] = [[] for _ in range(max_level + 1)]
        for step_id, level in levels.items():
            result[level].append(step_id)

        return result

    def is_complete(self) -> bool:
        """True if all steps are COMPLETED or SKIPPED."""
        for step_id in self._graph.node_ids():
            node = self._graph.node(step_id)
            status = node.get("status") if node else None
            if status not in (StepStatus.COMPLETED.value, StepStatus.SKIPPED.value):
                return False
        return True

    def has_failures(self) -> bool:
        """True if any step is FAILED."""
        for step_id in self._graph.node_ids():
            node = self._graph.node(step_id)
            if node and node.get("status") == StepStatus.FAILED.value:
                return True
        return False

    def get_results(self) -> dict[str, Any]:
        """Get results from all completed steps."""
        results = {}
        for step_id in self._graph.node_ids():
            node = self._graph.node(step_id)
            if node and node.get("status") == StepStatus.COMPLETED.value:
                results[step_id] = node.get("result")
        return results

    # -- Validation ----------------------------------------------------------

    def validate(self) -> list[str]:
        """Validate the plan graph. Returns a list of issues (empty = valid)."""
        issues = []

        # Check for cycles
        cycles = find_cycles(self._graph)
        if cycles:
            issues.append(f"Plan graph has {len(cycles)} cycle(s): {cycles}")

        # Check for missing tool names on TOOL steps
        for step_id in self._graph.node_ids():
            node = self._graph.node(step_id)
            if node and node.get("step_type") == StepType.TOOL.value and not node.get("tool_name"):
                issues.append(f"Step {step_id}: TOOL type but no tool_name")

        # Check for orphan nodes (no edges at all, not a root)
        for step_id in self._graph.node_ids():
            preds = self._graph.predecessors(step_id)
            succs = self._graph.successors(step_id)
            if not preds and not succs and self._graph.n_nodes > 1:
                issues.append(f"Step {step_id}: orphan node (no dependencies)")

        return issues

    # -- Subplan insertion ---------------------------------------------------

    def insert_subplan(self, node_id: str, subplan: PlanGraph) -> None:
        """Replace a node with a subplan graph.

        The node's incoming edges are connected to the subplan's roots
        (steps with no predecessors). The node's outgoing edges are
        connected to the subplan's leaves (steps with no successors).
        """
        if node_id not in self._graph:
            raise KeyError(f"Node {node_id} not found in plan graph")

        # Capture edges before removing
        predecessors = self._graph.predecessors(node_id)
        successors = self._graph.successors(node_id)

        # Find subplan roots and leaves
        sub_roots = [
            sid for sid in subplan._graph.node_ids() if not subplan._graph.predecessors(sid)
        ]
        sub_leaves = [
            sid for sid in subplan._graph.node_ids() if not subplan._graph.successors(sid)
        ]

        # Remove the original node (and its edges)
        self._graph.remove_node(node_id)

        # Add all subplan nodes and edges
        for sid in subplan._graph.node_ids():
            snode = subplan._graph.node(sid)
            if snode:
                props = {
                    k: snode.get(k)
                    for k in (
                        "step_type",
                        "description",
                        "tool_name",
                        "input_spec",
                        "expected_output",
                        "status",
                        "result",
                        "judgment",
                        "latency_ms",
                    )
                    if snode.get(k) is not None
                }
                self._graph.add_node(sid, **props)

        for edge in subplan._graph.edges():
            self._graph.add_edge(edge.source, edge.target, type="dependency")

        # Connect predecessors → subplan roots
        for pred in predecessors:
            for root in sub_roots:
                if not self._graph.has_edge(pred, root):
                    self._graph.add_edge(pred, root, type="dependency")

        # Connect subplan leaves → successors
        for leaf in sub_leaves:
            for succ in successors:
                if succ in self._graph and not self._graph.has_edge(leaf, succ):
                    self._graph.add_edge(leaf, succ, type="dependency")

        # Validate after insertion
        issues = self.validate()
        if issues:
            logger.warning("plan_graph.insert_subplan: validation issues: %s", issues)

    # -- Critical path -------------------------------------------------------

    def get_critical_path(self) -> tuple[list[str], float]:
        """Get the critical path (longest path) and its estimated total latency."""
        cp = critical_path(self._graph)
        return cp.path, cp.cost

    # -- Serialization -------------------------------------------------------

    def to_json(self) -> str:
        """Serialize the plan graph to JSON."""
        return self._graph.to_json()

    @classmethod
    def from_json(cls, data: str) -> PlanGraph:
        """Restore a plan graph from JSON."""
        pg = cls.__new__(cls)
        pg._graph = Graph.from_json(data)
        return pg

    def to_mermaid(self) -> str:
        """Export the plan graph as a Mermaid flowchart."""
        lines = ["graph TD"]
        for step_id in self._graph.node_ids():
            node = self._graph.node(step_id)
            if node is None:
                continue
            desc = node.get("description", step_id)
            status = node.get("status", "pending")
            # Truncate long descriptions
            label = desc[:40] + "..." if len(str(desc)) > 40 else desc
            safe_label = str(label).replace('"', "'")
            lines.append(f'    {step_id}["{safe_label}"]')
            # Style by status
            if status == "completed":
                lines.append(f"    style {step_id} fill:#90EE90")
            elif status == "failed":
                lines.append(f"    style {step_id} fill:#FFB6C1")
            elif status == "running":
                lines.append(f"    style {step_id} fill:#87CEEB")

        for edge in self._graph.edges():
            lines.append(f"    {edge.source} --> {edge.target}")

        return "\n".join(lines)

    # -- Dunder --------------------------------------------------------------

    def __len__(self) -> int:
        return self._graph.n_nodes

    def __contains__(self, step_id: str) -> bool:
        return step_id in self._graph

    def __repr__(self) -> str:
        ready = len(self.get_ready_steps())
        return f"PlanGraph(steps={self.n_steps}, ready={ready})"
