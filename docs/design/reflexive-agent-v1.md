# ReflexiveAgent — closing the agent's outer loop

**Status:** v1 draft, 2026-05-06. Pending external review.

## Why

Today's agent runs **open-loop**. It has no thermostat — no comparison
between "what I was asked to produce" and "what I actually produced",
and therefore no error signal to drive iteration. Every Harvey LAB
criterion the agent missed (45/55 on the 2026-05-06 CoC baseline
=> 18.2% pooled pass rate) is a symptom of this single gap. Pointing
at any other benchmark would produce the same shape of failure.

A thermostat needs three things, and today's agent has none of them:

| Need | Today |
|---|---|
| **Target state** — what counts as done | Implicit. "Comprehensive" is undefined. |
| **Measurement of current state** — what did I actually produce | Agent emits text and stops. It never reads its own output. |
| **Iteration on the gap** — act, re-measure, repeat | Single pass. No second look. |

Adding more retrieval/planning/extraction options didn't help because
the agent has no signal driving *which* option to pick or *whether*
the chosen option worked. The fix is the loop, not more capability.

## What we already have (and what's missing)

The thermostat is almost entirely built — but at the wrong level. We
have iteration loops for individual LLM calls (Programs); we don't
have one for full agent tasks (Agents).

### Existing primitives we reuse

| Component | Module | What it gives us |
|---|---|---|
| `LoopRunner` | `kaos-llm-core/programs/loop_runner.py` | Generic iteration kernel — state, step, build_result, on_step_error, stop-reasons. Used by Refine + ReAct. |
| `Refine` | `kaos-llm-core/programs/refine.py` | Producer→Judge→Re-invoke loop. Returns best-of-iterations (so iterating never makes things worse). Concurrency-safe via per-iteration producer cloning. |
| `Judge` | `kaos-llm-core/programs/judge.py` | LLM-as-evaluator with `criteria: str` → `quality_score`, `reasoning`. |
| `BestOfN` | `kaos-llm-core/programs/best_of_n.py` | Parallel sampling with metric/judge selection. |
| `MemoryType.REFLECTION` | `kaos-agents/memory/types.py` | Dedicated section for self-reflection text. |
| `recall(memory, [REFLECTION])` | `kaos-agents/planning/recall.py` | Pulls reflections back into agent context. |
| `_reflect_on_coverage` | `kaos-agents/context/retrieval.py:595` | Gap-finding for retrieval ("which queries have weak coverage?"). |
| Two-round retrieval reflection | `context/retrieval.py:279` | `for reflect_round in range(2):` — proof-of-concept for the loop pattern. |
| `evaluate_structural` + `evaluate_semantic` | `planning/evaluate.py` | Two-mode judgment. Structural first (no LLM cost), semantic when needed. |
| `route()` | `planning/route.py` | CONTINUE/REPLAN/DEEPEN/STOP_BUDGET/STOP_FAILURE control flow. |
| `rubric_judge` | `kaos-agents/benchmarks/rubric_judge.py` | Per-criterion PASS/FAIL judge (just shipped for Harvey LAB benchmark). |
| `ReflectiveOptimizer` | `kaos-llm-core/optimization/reflective.py` | Same critique-revise pattern, but runs offline at optimization time. |

### What's missing

**A loop over `Agent` with rubric criteria.** The codebase already has:

1. `LoopRunner` — generic kernel
2. `Refine` — loop over `Program` (single LLM call) ✅
3. `ReAct` — loop over `Program` with tool calls ✅
4. `_reflect_on_coverage` — loop over retrieval queries ✅
5. **(missing) ReflexiveAgent — loop over `BaseAgent` with rubric criteria**

## Proposal: ReflexiveAgent

Wrap any `BaseAgent` (`ChatAgent`, `ResearchAgent`, `PlanExecuteAgent`)
in a `Refine`-shaped outer loop. Same kernel, same trace tree, same
best-of-iterations safety net.

### New types

```python
@dataclass(frozen=True, slots=True)
class RubricCriterion:
    """A single PASS/FAIL judgment on a deliverable."""
    id: str
    description: str        # human-readable title
    match: str              # PASS-if/FAIL-if text (Harvey LAB shape)
    weight: float = 1.0     # for weighted pass rates; default 1.0
    deliverables: tuple[str, ...] = ()  # which output(s) this criterion scopes to


@dataclass(frozen=True, slots=True)
class Rubric:
    """The target — what counts as done."""
    criteria: tuple[RubricCriterion, ...]
    min_pass_rate: float = 1.0     # all-pass by default (Harvey methodology)
    description: str = ""          # optional preamble for the producer prompt


@dataclass(frozen=True, slots=True)
class RubricVerdict:
    """One pass through the critic — per-criterion verdicts + aggregate."""
    criteria: tuple[CriterionVerdict, ...]
    pass_rate: float
    weighted_pass_rate: float
    failed: tuple[CriterionVerdict, ...]      # convenience accessor
    judge_cost_usd: float
    judge_latency_s: float


@dataclass(frozen=True, slots=True)
class CriterionVerdict:
    criterion_id: str
    passed: bool
    confidence: float
    reasoning: str
```

### New components

```python
# 1. RubricDeriver — turns a prose task into a Rubric when one isn't supplied.
class RubricDeriver(Program):
    """Signature: (task: str, examples: list[Rubric] | None) -> Rubric.

    Built on a Refine'd Call so the derived rubric goes through its own
    critique pass before being used. Avoids "the rubric is wrong because
    the model was sloppy" failure mode.
    """


# 2. RubricCritic — multi-criterion judge. Composes rubric_judge with map-reduce.
class RubricCritic:
    """Score a deliverable against every criterion in a Rubric.

    Implementation: asyncio.gather over rubric.criteria with bounded
    concurrency (same machinery as harvey_coc_benchmark._judge_all).
    Structural pre-checks first (deliverable empty? error prefix?) so
    obvious failures don't cost LLM tokens.
    """
    async def __call__(self, task: str, deliverable: str, rubric: Rubric) -> RubricVerdict: ...


# 3. ReflexiveAgent — wraps a BaseAgent in the outer loop.
class ReflexiveAgent(BaseAgent):
    def __init__(
        self,
        producer: BaseAgent,                # the agent that produces the deliverable
        rubric: Rubric | RubricDeriver,     # target — explicit or derived
        critic: RubricCritic | None = None, # default: RubricCritic with claude-haiku-4-5
        max_iterations: int = 3,
        min_pass_rate: float = 0.8,         # below 1.0 because perfect is rare
        budget: PlanBudget | None = None,   # cost/wall-clock ceiling
        feedback_strategy: FeedbackStrategy = FeedbackStrategy.GAP_LIST,
    ): ...

    async def turn(self, task: str, *, session_id: str) -> AgentResponse:
        """Produce → critique → iterate. Returns best-of-iterations."""
        rubric = await self._materialize_rubric(task)
        deliverable = await self._producer.turn(task, session_id=session_id)
        history: list[tuple[AgentResponse, RubricVerdict]] = []

        for iteration in range(self._max_iterations):
            verdict = await self._critic(task, deliverable.text, rubric)
            history.append((deliverable, verdict))

            if verdict.weighted_pass_rate >= self._min_pass_rate:
                return self._best(history)

            if self._budget.should_stop():
                return self._best(history)

            # Write gap analysis into REFLECTION memory; the producer's
            # existing recall() wiring picks it up on next turn.
            self._inject_gap_feedback(verdict, session_id)

            deliverable = await self._producer.turn(task, session_id=session_id)

        return self._best(history)  # best-of-iterations by pass_rate
```

### Feedback strategies

The "how do gaps become next-iteration prompts?" question is the only
non-trivial design choice. Three modes, evaluated empirically:

| Strategy | What gets injected into REFLECTION | When to use |
|---|---|---|
| `GAP_LIST` | Bulleted list of failed criteria + judge reasoning | Default — clear and structured |
| `NARRATIVE` | LLM-rewritten paragraph explaining the gap | When criteria are technical and need translation |
| `HYBRID` | List + narrative summary | Highest information density, highest token cost |

The strategy is settable per-call so we can A/B them on the same task.

### Reuse of existing wiring

- **Iteration kernel**: `LoopRunner` from `kaos-llm-core`. We supply `make_state`, `step`, `build_result`, `on_step_error`. ~30 LoC of glue.
- **Memory channel**: existing `MemoryType.REFLECTION` section. `plan_execute.py` already pulls it via `recall(memory, [..., REFLECTION])` and injects it as `prior_failures` into planning context.
- **Stop conditions**: existing `PlanBudget` + `route()` give us `STOP_BUDGET` and `STOP_FAILURE` (max replans).
- **Best-of-iterations**: existing pattern from `Refine` — `max(history, key=lambda h: h[1].weighted_pass_rate)`. Means iterating never makes things worse.
- **Trace tree**: `LoopRunner`'s trace collector composes naturally with the agent's existing event stream.

### What's genuinely new

| Component | LoC estimate | Net new logic |
|---|---|---|
| `Rubric` / `RubricCriterion` / `CriterionVerdict` / `RubricVerdict` types | 80 | Frozen dataclasses, no logic |
| `RubricCritic` | 120 | Wraps `rubric_judge` with bounded `asyncio.gather`, structural pre-checks |
| `RubricDeriver` | 100 | Signature + Call wrapped in Refine; mostly composition |
| `ReflexiveAgent` | 200 | LoopRunner glue + memory injection + best-of selection |
| Tests + integration with Harvey CoC benchmark | 200 | |
| **Total new code** | **~700 LoC** | All composition, no novel architecture |

## Loop properties

The loop must satisfy these invariants for it to be safe:

1. **Termination guaranteed**: budget + max_iterations + min_pass_rate; at least one must trip.
2. **Best-of-iterations selection**: never returns a worse deliverable than was already produced.
3. **Bounded cost**: per-iteration critic cost is `O(|criteria|)`, total cost is `O(max_iter × |criteria|)`. Budget ceiling is hard.
4. **Trace-complete**: every iteration's deliverable, verdict, and gap feedback is auditable from the run's trace tree.
5. **Stateless wrap**: composing `ReflexiveAgent(producer=existing_agent)` doesn't change `existing_agent`'s behavior outside the wrap.
6. **Composable**: `ReflexiveAgent(producer=ReflexiveAgent(...))` works (though probably not useful).
7. **Producer-agnostic**: works with `ChatAgent`, `ResearchAgent`, `PlanExecuteAgent`, future patterns.

## Failure modes to design against

These come up in the literature on Reflexion / self-refinement agents
and are worth flagging up-front:

1. **Judge bias / score drift**: LLM-as-judge scores swing 20-40 points
   across reruns of the same input. Refine already documents this
   (`refine.py:Score stability caveat`). Mitigation: critic uses
   `temperature=0` for final scoring; iteration uses normal sampling.
2. **Over-refinement**: model "fixes" things that weren't broken,
   regressing on prior-iteration passes. Mitigation: best-of-iterations
   selection by `weighted_pass_rate`. Prior-iteration deliverable wins
   if no better one is produced.
3. **Infinite-loop on impossible criteria**: producer can never satisfy
   a criterion (corpus doesn't contain the answer). Mitigation:
   `max_iterations` ceiling + `min_pass_rate` < 1.0 + per-criterion
   stagnation detection (criterion failed last 2 iterations identically
   → mark as "blocked", excluded from min_pass_rate computation).
4. **Critic hallucinates pass**: judge says PASS when deliverable is
   actually wrong. Mitigation: `rubric_judge` system prompt is strict
   about specific facts; structural pre-checks catch obviously empty
   deliverables; future work — a verifier program that grounds the
   critic's reasoning in source spans.
5. **Cost runaway**: 10 iterations × 50 criteria × $0.001/judgment =
   $0.50 per task. Mitigation: explicit `PlanBudget` with `STOP_BUDGET`
   route decision. Default budget caps the loop at 3 iterations and $0.30.
6. **Rubric drift on derived rubrics**: if the same task produces a
   different rubric on rerun, scores aren't comparable. Mitigation:
   `RubricDeriver` is itself wrapped in `Refine` (judges its own output)
   AND uses `temperature=0`. If determinism still isn't enough, the
   rubric is cached against the task hash for the session.
7. **Producer ignores REFLECTION feedback**: agent re-runs, re-produces
   the same output. Mitigation: feedback is injected with explicit
   instruction prefix ("Address these gaps from your prior attempt: ...");
   also, the producer prompt mutation pattern from `Refine` works (clone
   producer per iteration, mutate the cloned instructions).

## Validation plan

The Harvey LAB CoC benchmark we just shipped is the natural test bed.

| Run | Configuration | Expected pass rate | What it measures |
|---|---|---|---|
| **Baseline** | Producer alone, no loop | 18.2% (measured 2026-05-06) | The current ceiling |
| **Loop-1** | ReflexiveAgent(max_iter=1) — judge once, no re-iteration | 18.2% (sanity check) | Critic doesn't change the answer alone |
| **Loop-2** | ReflexiveAgent(max_iter=2) | ? | First real signal of lift |
| **Loop-3** | ReflexiveAgent(max_iter=3) | ? | Plateau or continued lift |
| **Loop-3-strong-critic** | ReflexiveAgent with claude-sonnet-4-6 critic | ? | Does a stronger critic find more gaps? |
| **Loop-3-derived-rubric** | ReflexiveAgent with `RubricDeriver` instead of supplied rubric | ? | Self-derivation cost — is the agent willing to grade itself? |
| **Loop-3-narrative-feedback** | `FeedbackStrategy.NARRATIVE` | ? | Does paragraph feedback beat bullet lists? |

If lift is real and sustained across 3+ datasets (Harvey CoC, our own
multiformat suite, our cross-doc EDGAR benchmark), `ReflexiveAgent`
becomes the default wrap for production agents. If lift is marginal
(<5%) or unstable across reruns, we'd revisit the feedback strategy
or critic strength before shipping it as default.

## What this enables downstream

Once the closed loop exists, choices that today are config knobs
become **moves the loop selects from based on which criteria are
still failing**:

- "Section 14.2 anti-assignment" criterion fails → loop picks vocabulary
  expansion (`kaos-retrieval-synonyms`) for the next iteration's retrieval.
- "Quantification: $36.2M revenue concentration" criterion fails → loop
  picks structured extraction (`ExtractionSchema` with a numeric field)
  instead of free-form prose.
- "Cross-contract timing analysis" criterion fails → loop switches the
  producer pattern from `research` to `plan-execute` with explicit
  cross-document synthesis steps.

The dispatcher is downstream of the loop; the loop is upstream.

## Open questions for v2

These are the calls I'm least confident on:

1. **Should ReflexiveAgent be a `BaseAgent` subclass or a `Runner`-level
   wrapper?** Subclass means it composes through `agent_as_tool()`;
   wrapper means it sits at a higher layer.
2. **Per-criterion vs whole-deliverable feedback**: do we tell the
   producer "you missed C-001, C-003, C-005" or "you missed the
   anti-assignment language and the reverse-triangular-merger
   analysis"? Empirical question.
3. **Does the producer need to know about the rubric during the FIRST
   pass, or only on iterations?** First-pass injection might bias the
   producer toward writing-to-the-test; no injection means the first
   pass is "fair" but might miss criteria the producer would have
   addressed if asked.
4. **Stagnation detection**: when does "this iteration produced the
   same set of failures as last iteration" mean "stop" vs "switch
   strategies"?
5. **Rubric reuse across tasks**: if I derive a rubric from "review
   contracts for change-of-control", can the same rubric apply to a
   different M&A target? Or is every rubric task-specific?
6. **Interaction with delegation**: can a delegated sub-agent be
   wrapped in `ReflexiveAgent`? Should it be? (Probably yes —
   subagent failures are exactly the thing we'd want to iterate on.)

## References

- Shinn et al., *Reflexion: Language Agents with Verbal Reinforcement
  Learning*, NeurIPS 2023 — the canonical critique-revise pattern.
- Madaan et al., *Self-Refine: Iterative Refinement with Self-Feedback*,
  NeurIPS 2023 — closer to our `Refine` shape.
- DSPy `BootstrapFewShot` and `MIPROv2` optimizers — show the
  generality of "produce → score → revise" loops, validate that
  this isn't a Harvey-specific pattern.
- Harvey LAB methodology docs (`docs/eval-strategies.md`) — all-pass
  rubric scoring is the methodology this design directly supports.

## Status

This is v1, written from the inside without external review. v2 will
synthesize feedback from sub-agent reviews of the literature, the
production agent ecosystem (LangGraph, AutoGen, Swarm, Claude Computer
Use, Computer-Using Agents), and a critical/skeptical pass on the
proposal itself.
