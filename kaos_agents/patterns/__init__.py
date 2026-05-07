"""Agent patterns — ChatAgent, PlanExecuteAgent, ResearchAgent.

These three pattern classes auto-register into
:data:`kaos_agents.registry.pattern_registry.default_pattern_registry`
under :class:`AgentPattern` enum values when this module is imported.
:class:`Runner._build_internal_agent` resolves the right pattern class
via the registry; replacing the historical if/elif switch.
"""

from __future__ import annotations

from kaos_agents.config import AgentPattern
from kaos_agents.patterns.chat import ChatAgent
from kaos_agents.patterns.plan_execute import PlanExecuteAgent
from kaos_agents.patterns.research import ResearchAgent
from kaos_agents.registry.pattern_registry import default_pattern_registry

# Auto-register the three built-in patterns under their canonical
# :class:`AgentPattern` enum values. Out-of-tree custom patterns can
# register themselves explicitly:
#
#     from kaos_agents.registry import default_pattern_registry
#     default_pattern_registry.register("my-pattern", MyCustomAgent)
default_pattern_registry.register(AgentPattern.CHAT.value, ChatAgent, force=True)
default_pattern_registry.register(AgentPattern.PLAN.value, PlanExecuteAgent, force=True)
default_pattern_registry.register(AgentPattern.RESEARCH.value, ResearchAgent, force=True)

__all__ = [
    "ChatAgent",
    "PlanExecuteAgent",
    "ResearchAgent",
]
