"""ReActPlanner — wraps kaos-llm-core's :class:`~kaos_llm_core.programs.react.ReAct` Program.

Pattern: paper §7.1 (Reasoning + Acting). The most fundamental
planner — interleaves reasoning with tool calls.

The "plan" step is trivial: ReAct does its planning *inside* the loop,
so :meth:`ReActPlanner.plan` returns a thin :class:`ReActPlan` object
that just captures the goal text and any constraint hints.
:meth:`ReActPlanner.execute` is the real work — it constructs (or
reuses) a kaos-llm-core :class:`ReAct`, threads tools from the
``perceiver`` + ``actor`` (or an explicit ``tools=`` override), invokes,
and projects the result into a :class:`PlanResult`.

Tools sourced from (in priority order):

1. The explicit ``tools=`` constructor kwarg, when not ``None`` — this
   is the override used by tests and bespoke wiring.
2. The ``perceiver.tools`` attribute (when present) — read-only tools.
3. The ``actor.tools`` attribute (when present) — mutating tools.

Phase 3.A: tools fan-out from ``perceiver`` / ``actor`` is intentionally
duck-typed. The Phase 1 ``Perceiver`` / ``Actor`` types are typed
:class:`Any` here; their ``.tools`` attribute (when present) is read at
:meth:`execute` time. Phase 4 will formalise the tool-fan-out contract
(typed tool sets, deduplication, permission gating).

Resolved Decision §13 #3 — *classifier picks at runtime*: Phase 3.A
ships the planner; Phase 3.D wires
:class:`~kaos_agents.loop.agent_loop.AgentLoop` to construct a default
:class:`ReActPlanner` when
:attr:`~kaos_agents.intent.types.IntentResult.pattern` is
:attr:`~kaos_agents.config.AgentPattern.CHAT`.

Notable deviations from the spec sketch:

* kaos-llm-core's :class:`ReAct` requires an explicit
  :class:`~kaos_llm_core.signatures.signature.Signature` class — there
  is no built-in ``ToolTaskSignature``. We define a default
  :class:`_DefaultReActSignature` with input field ``task: str`` and
  output field ``answer: str`` so callers who don't bring their own
  Signature get a sensible default. Custom Signatures can be passed
  via the ``signature=`` kwarg or by handing a pre-constructed
  :class:`ReAct` via ``react_program=``.
* The ``instructions=`` kwarg is forwarded to the
  :class:`ReAct` constructor (which threads it through the inner
  :class:`~kaos_llm_core.programs.call.Call`'s system prompt). It is
  *not* a separate Signature attribute.
* :meth:`ReAct.invoke` accepts the Signature's input fields as kwargs
  — for the default Signature that means ``task=...``. When a custom
  Signature is supplied, :meth:`ReActPlanner.execute` still passes the
  goal under the ``task`` keyword; custom Signatures are expected to
  expose a ``task: str`` input field. (Phase 4 will generalise this.)
"""

from __future__ import annotations

from typing import Any

from kaos_llm_core import InputField, OutputField, Signature
from kaos_llm_core.programs.react import ReAct

from kaos_agents.intent.types import IntentResult
from kaos_agents.planning.planner import Plan, PlanResult
from kaos_agents.types.usage import ZERO_USAGE, InvocationUsage


class _DefaultReActSignature(Signature):
    """Default Signature for :class:`ReActPlanner` when none is supplied.

    A ReAct loop with a single ``task`` input and a single ``answer``
    output. Most agentic chat interactions fit this shape; callers who
    need richer typed I/O bring their own Signature via
    ``ReActPlanner(signature=MySig)`` or ``react_program=ReAct(MySig, ...)``.
    """

    task: str = InputField(description="The goal or question for the ReAct agent to solve.")
    answer: str = OutputField(description="The final answer to the task.")


class ReActPlan(Plan):
    """Concrete :class:`Plan` for ReAct: just the goal text + tool-name hints.

    The ``goal_statement`` is the natural-language objective the agent
    will pursue inside the ReAct loop. ``tool_hints`` is a tuple of tool
    names the planner suggests the agent prefer; Phase 3.A always emits
    an empty tuple here (Phase 4 wires intent-driven tool selection).
    """

    goal_statement: str = ""
    tool_hints: tuple[str, ...] = ()


class ReActPlanner:
    """Planner that uses kaos-llm-core :class:`ReAct` as the inner loop.

    Args:
        model: Provider-prefixed model id (e.g.
            ``"anthropic:claude-haiku-4-5"``). Forwarded to the inner
            :class:`ReAct` constructor.
        max_iterations: ReAct iteration cap (default 10). Forwarded to
            :class:`ReAct`.
        instructions: System instructions for the ReAct loop. Forwarded
            verbatim to the inner :class:`Call`'s system prompt.
        signature: Optional custom :class:`Signature` class. Defaults
            to :class:`_DefaultReActSignature` (``task: str`` →
            ``answer: str``). Custom Signatures must expose a
            ``task: str`` input field for :meth:`execute` to drive the
            loop with the goal.
        tools: Optional explicit tool tuple. When provided, overrides
            the ``perceiver`` / ``actor`` fan-out at :meth:`execute`
            time.
        react_program: Optionally pre-constructed :class:`ReAct`. When
            provided, ``model`` / ``max_iterations`` / ``instructions``
            / ``signature`` / ``tools`` are ignored — the caller has
            already configured the program. Useful for advanced wiring
            and tests.

    Phase 3.A: tools fan-out from ``perceiver`` / ``actor`` is a
    placeholder — the perceiver and actor types are typed :class:`Any`;
    their ``.tools`` attribute (when present) is read at :meth:`execute`
    time. Phase 4 formalises the tool-fan-out contract.
    """

    def __init__(
        self,
        *,
        model: str = "anthropic:claude-haiku-4-5",
        max_iterations: int = 10,
        instructions: str = (
            "You are a helpful assistant. Reason step by step and call tools when needed."
        ),
        signature: type[Signature] | None = None,
        tools: tuple[Any, ...] | None = None,
        react_program: ReAct | None = None,
        hooks: tuple[Any, ...] = (),
    ) -> None:
        self._model = model
        self._max_iterations = max_iterations
        self._instructions = instructions
        self._signature: type[Signature] = signature or _DefaultReActSignature
        self._tools = tools
        self._react = react_program
        # Phase 5.D: tuple of :class:`KaosHook` instances to bridge into
        # the inner :class:`ReAct` Program's lifecycle. Adapted via
        # :func:`kaos_agents.hooks.adapter.adapt_hooks` at build time so
        # a single OTelHook can observe both the agent's outer events
        # (Span(TURN/STEP/...)) and the inner kaos-llm-core CallHooks /
        # ProgramHooks events. Empty tuple (default) skips the bridge —
        # back-compat for tests that don't exercise hook flow.
        self._hooks = hooks

    async def plan(
        self,
        intent: IntentResult,
        memory: Any | None = None,
    ) -> ReActPlan:
        """Return a thin :class:`ReActPlan` capturing the goal + tool hints.

        ``memory`` is accepted for protocol compliance but ignored in
        Phase 3.A — Phase 4 wires memory-aware tool selection and
        success-criteria propagation.
        """
        del memory  # Phase 4 wires memory-aware planning.
        return ReActPlan(
            pattern="react",
            goal_statement=intent.goal.statement,
            tool_hints=(),  # Phase 4 wires tool selection from intent
            metadata={"intent_type": intent.goal.intent_type.value},
        )

    async def execute(
        self,
        plan: Plan,
        *,
        perceiver: Any | None = None,
        actor: Any | None = None,
    ) -> PlanResult:
        """Drive ReAct over the goal in :attr:`ReActPlan.goal_statement`.

        Falls back to ``plan.metadata["goal_statement"]`` when the
        passed plan is not a :class:`ReActPlan` (defensive — keeps the
        Protocol open to non-ReAct plans coerced through this planner).
        """
        # Resolve tools: explicit > perceiver+actor fan-out > empty.
        tools = self._resolve_tools(perceiver, actor)

        # Reuse pre-constructed ReAct or build one for this run.
        react = self._react or self._build_react(tools)

        # Read goal text from a ReActPlan if that's what we got, else
        # fall back to plan.metadata.
        goal_statement = (
            getattr(plan, "goal_statement", "")
            or plan.metadata.get("goal_statement", "")
            or "Respond to the user's request."
        )

        # Drive ReAct via .invoke() so we get the full Invocation back.
        # The default Signature uses ``task`` as its input field name;
        # custom Signatures are expected to follow the same convention
        # (Phase 4 will generalise this via field introspection).
        invocation = await react.invoke(task=goal_statement)

        # Project the kaos-llm-core ReActResult / Invocation into a PlanResult.
        text = self._extract_text(invocation)
        usage = self._extract_usage(invocation)
        iterations_used = self._extract_iterations(invocation)
        tool_executions = self._extract_tool_executions(invocation)
        return PlanResult(
            text=text,
            output=text,  # alias — Phase 2.B AgentLoop reads either
            tool_executions=tool_executions,
            usage=usage,
            metadata={"react_iterations": iterations_used},
        )

    @staticmethod
    def _extract_tool_executions(invocation: Any) -> tuple[Any, ...]:
        """Walk the ReActResult trajectory into ToolExecution records.

        Without this, AgentLoop's TOOL_CALL span emission sees an
        empty list (PlanResult.tool_executions used to be hard-coded
        to `()`) and the v2 audit log carries zero tool_call
        telemetry — bug #3 in the workflow audit.

        Source-of-truth: kaos-llm-core's ReActResult.trajectory is a
        list of `Iteration`. Each Iteration has:
          * `tool_calls: list[ToolCall]` — what the model requested
          * `tool_results: list[ToolObservation]` — what the tool
            executor returned (tool_name, arguments, result, is_error,
            tool_call_id). This is the field carrying executed work.
        Read from `tool_results` (NOT `observations` — that field
        does not exist on Iteration).
        """
        from kaos_agents.types.tool_call import ToolExecution

        result = getattr(invocation, "output", None) or invocation
        trajectory = getattr(result, "trajectory", None) or ()
        execs: list[ToolExecution] = []
        for iteration in trajectory:
            for obs in getattr(iteration, "tool_results", None) or ():
                tool_name = str(getattr(obs, "tool_name", "") or "")
                if not tool_name:
                    continue
                arguments = getattr(obs, "arguments", {}) or {}
                args_tuple = tuple(sorted(arguments.items())) if isinstance(arguments, dict) else ()
                obs_result = getattr(obs, "result", None)
                execs.append(
                    ToolExecution(
                        tool_name=tool_name,
                        call_id=str(getattr(obs, "tool_call_id", "") or tool_name),
                        arguments=args_tuple,
                        result_summary=(str(obs_result)[:200]) if obs_result else "",
                        is_error=bool(getattr(obs, "is_error", False)),
                    )
                )
        return tuple(execs)

    # ---- internals -------------------------------------------------

    def _resolve_tools(self, perceiver: Any | None, actor: Any | None) -> tuple[Any, ...]:
        """Resolve the tool list for this :meth:`execute` call.

        Explicit ``tools=`` (constructor kwarg) wins outright. Otherwise
        the union of ``perceiver.tools`` + ``actor.tools`` is returned
        (each defaults to empty when the attribute is missing or
        falsy). Order is preserved: perceiver tools first, then actor
        tools — read before write.
        """
        if self._tools is not None:
            return self._tools
        collected: list[Any] = []
        if perceiver is not None:
            collected.extend(getattr(perceiver, "tools", ()) or ())
        if actor is not None:
            collected.extend(getattr(actor, "tools", ()) or ())
        return tuple(collected)

    def _build_react(self, tools: tuple[Any, ...]) -> ReAct:
        """Construct a fresh :class:`ReAct` for this run.

        kaos-llm-core's :class:`ReAct` constructor signature
        (``kaos_llm_core/programs/react.py:266``) is::

            ReAct(
                signature: type[Signature],
                tools: list[Tool],
                *,
                model: str | None = None,
                max_iterations: int = 50,
                instructions: str | None = None,
                ...
            )

        We pass ``signature`` positionally, ``tools`` as a list (kaos-llm-core
        uses :class:`list` internally and warns on the empty case rather
        than raising), and forward ``model`` / ``max_iterations`` /
        ``instructions`` as kwargs.
        """
        # Phase 5.D: bridge KaosHook tree into the inner ReAct via the
        # Phase 0.C adapter so a single OTelHook (or any other KaosHook)
        # observes both the agent layer and the inner kaos-llm-core
        # CallHooks/ProgramHooks events. Empty hooks skip the adapter
        # and the inner ReAct gets the kaos-llm-core defaults.
        call_hooks = None
        program_hooks = None
        if self._hooks:
            from kaos_agents.hooks.adapter import adapt_hooks

            call_hooks, program_hooks = adapt_hooks(self._hooks)
        kwargs: dict[str, Any] = {
            "tools": list(tools),
            "model": self._model,
            "max_iterations": self._max_iterations,
            "instructions": self._instructions,
        }
        if call_hooks is not None:
            kwargs["hooks"] = call_hooks
        if program_hooks is not None:
            kwargs["program_hooks"] = program_hooks
        return ReAct(self._signature, **kwargs)

    @staticmethod
    def _extract_text(invocation: Any) -> str:
        """Pull the final-answer text out of a kaos-llm-core
        :class:`Invocation` returned by :meth:`ReAct.invoke`.

        :class:`Invocation.output` is a :class:`ReActResult` whose
        ``outputs`` is a dict (or ``None`` on overrun). The conventional
        final field is ``answer`` (per :class:`_DefaultReActSignature`);
        fall back to ``output``, then to the dict's first value, then
        the result's text-ish ``str()``. This keeps the projection
        robust to custom Signatures that name their output field
        something other than ``answer``.
        """
        out = getattr(invocation, "output", invocation)
        # ReActResult: try .outputs[answer], .answer (forwarded), then str(out).
        outputs = getattr(out, "outputs", None)
        if isinstance(outputs, dict):
            for key in ("answer", "output", "text", "response"):
                if key in outputs and outputs[key] is not None:
                    return str(outputs[key])
            # Fall back to first value if any
            for value in outputs.values():
                if value is not None:
                    return str(value)
        # Bare string output (degenerate / mocked case)
        text = getattr(out, "text", None)
        if text is not None:
            return str(text)
        return str(out) if out is not None else ""

    @staticmethod
    def _extract_usage(invocation: Any) -> InvocationUsage:
        """Roll up token + cost usage from a kaos-llm-core
        :class:`Invocation`. Returns :data:`ZERO_USAGE` when the
        invocation lacks a usage record.
        """
        usage = getattr(invocation, "usage", None)
        if usage is None:
            return ZERO_USAGE
        return InvocationUsage.from_llm_usage(usage)

    @staticmethod
    def _extract_iterations(invocation: Any) -> int:
        """Pull ``iterations_used`` off the :class:`ReActResult`.

        :class:`ReActResult` exposes ``iterations_used: int``. Defensive
        getattr chain handles both the real result type and synthetic
        test stubs that may not carry the field.
        """
        out = getattr(invocation, "output", invocation)
        return int(getattr(out, "iterations_used", 0) or 0)


__all__ = ["ReActPlan", "ReActPlanner"]
