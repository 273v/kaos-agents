# ReflexiveAgent v1 — Production-Systems Review

**Reviewer angle:** how do shipping agent frameworks (LangGraph, AutoGen, OpenAI Agents SDK, Devin, Harvey, Pydantic AI, CrewAI, Mastra, Vercel AI SDK) actually solve the "did I solve it?" question, and where does this proposal match or diverge?
**Date:** 2026-05-06.

---

## What production systems are actually doing

The 2025-2026 landscape has converged on a small number of patterns for closing an agent's outer loop. They are not all the same shape, and the differences matter.

### 1. "Critic agent" pair — the dominant cheap pattern

Two agents, sequenced. Producer writes; critic reviews; loop terminates when critic returns "no critique" or a budget trips.

- **`langgraph-reflection`** (https://github.com/langchain-ai/langgraph-reflection) is the cleanest reference implementation. The factory `create_reflection_graph(assistant_graph, judge_graph)` wraps two arbitrary LangGraph subgraphs. Termination contract is brutally simple: "if the critique agent returns a user message, run main again; if it returns no messages, finish." There's no rubric, no per-criterion verdict, no weighted score — pure binary "still complaining? / done."
- **AutoGen `Reflection` design pattern** (https://microsoft.github.io/autogen/stable/user-guide/core-user-guide/design-patterns/reflection.html) is the same shape: a `coder` agent and a `reviewer` agent exchange messages until the reviewer signals approval or `max_turns` trips. AutoGen 0.4 (Jan 2025) and the merged "Microsoft Agent Framework" release (Oct 2025) keep this as a *pattern*, not a built-in primitive — you compose it from `RoutedAgent` plus the standard message-handler decorators.
- **Cognition Devin** (https://cognition.ai/blog/closing-the-agent-loop-devin-autofixes-review-comments) ships this in production but the critic is *external infra*: GitHub bots, linters, CI failures, and security scanners. Devin doesn't try to grade itself; it lets the existing review pipeline emit signals and auto-fixes against those. Quote: "One agent writes, the other pressure-tests, and this continues in a loop." The 2025 performance review (https://cognition.ai/blog/devin-annual-performance-review-2025) is explicit that Devin struggles with *ambiguous* requirements and that *clear upfront scoping is prerequisite* — the loop only closes when the target is concretely defined.

### 2. Schema/validator-driven retry — the "type-checked" pattern

Producer emits structured output; validator rejects → retry with the validation error prepended.

- **Pydantic AI** (https://ai.pydantic.dev/api/agent/) calls this "reflection." The agent has a `retries` field; on schema validation failure, the framework synthesizes a `ModelRetry` message containing the pydantic error and re-invokes. Default `retries=1`. Per-tool and per-output overrides exist. This is not "rubric grading" — it is "schema rejection feedback as natural-language hint."
- **Vercel AI SDK 5** (https://ai-sdk.dev/docs/ai-sdk-core/generating-structured-data, https://vercel.com/blog/ai-sdk-6) does the same with Zod/Valibot. Streaming structured output (`streamObject()`) cannot validate mid-stream; final-state validation triggers retry through the agentic loop introduced in 5.0.
- **Mastra** (https://mastra.ai/docs/workflows/overview) uses Zod input/output schemas at every workflow step boundary. Validation failure at `run.start()` aborts the workflow; there is no automatic retry — that's a deliberate choice (workflows are "state machines you control," not loops).
- **OpenAI Agents SDK guardrails** (https://openai.github.io/openai-agents-python/guardrails/) are *tripwires*, not retries. `InputGuardrailTripwireTriggered` raises and stops the run. Iteration is the developer's job, not the framework's.

### 3. Tree search with reflection — the "expensive but powerful" pattern

LATS, MCTS, BestOfN with critic.

- **LangGraph LATS tutorial** (https://langchain-ai.github.io/langgraph/tutorials/lats/lats/) implements the Zhou et al. ICML 2024 paper. Six operations: select, expand, simulate (parallel), reflect+evaluate, backpropagate, terminate. Branching factor 5, MCTS UCB1 selection. LangChain's own blog (https://www.langchain.com/blog/reflection-agents) explicitly says LATS "can be sensitive to the reward scores" — the brittleness of LLM-as-judge propagates upward through the tree.
- **AG2** (the AutoGen fork, formerly ag2.ai) ships an LATS notebook (https://docs.ag2.ai/0.8.7/docs/use-cases/notebooks/notebooks/lats_search/), but it's filed under "use-cases/notebooks" — not core. Same status in LangGraph: tutorial, not primitive.
- **Nobody runs LATS in production for legal review.** The cost is `O(branching^depth × judge_calls)` and the lift over Reflexion + best-of-N is task-dependent. It survives in benchmark papers and reasoning-heavy code agents.

### 4. RL-style verifier loops — the new direction (2025-2026)

Process Reward Models (PRMs), AgentPRM, RLVR (Reinforcement Learning from Verifiable Rewards). Used by DeepSeek R1, Tülu 3, the Qwen-Agent line.

- **AgentPRM** (https://arxiv.org/html/2502.10325v1, Feb 2025) trains an actor-critic pair where the critic is a per-step process reward model. Sample-efficient because it doesn't wait for outcome rewards.
- **RLVR** (https://labelstud.io/blog/reinforcement-learning-from-verifiable-rewards/) is the dominant *training* signal in 2025; verifiable rewards are binary 0/1 ground-truth functions, not LLM-judges. Used during fine-tuning, not at inference.
- **Gaia2** (https://openreview.net/forum?id=9gw03JpKK4) pairs each scenario with a write-action verifier "directly usable for reinforcement learning from verifiable rewards," not just inference-time grading.

This direction is **out of scope for v1** — it's a training-time concern, not a runtime wrap. But it tells us where the field is going: away from LLM-judges-at-inference, toward verifiable-reward functions.

### 5. The Harvey approach — explicit, but not what the proposal assumes

The proposal cites Harvey LAB / BigLaw Bench as the validation target. Harvey's *actual* production pattern, per the ZenML LLMOps case study (https://www.zenml.io/llmops-database/scaling-agent-based-architecture-for-legal-ai-assistant) and the BigLaw Bench Workflows post (https://www.harvey.ai/blog/biglaw-bench-workflows-spa-deal-points), is:

- **Tool Bundles, not loops.** Every feature is a Tool Bundle with its own dataset and evaluator. There is no published runtime self-critique loop.
- **Leave-one-out validation gating.** Any system change must pass tests confirming existing capabilities maintain performance — this is *offline* eval, not inference-time iteration.
- **Human ground truth.** SPA Deal Points hits 98.47% recall vs GPT-4o at 66.04% / Gemini at 72.27% (https://www.harvey.ai/blog/biglaw-bench-workflows-spa-deal-points). The Harvey blog explicitly attributes this to *human-traced reasoning chains* the legal research team built into the agent, not to a critique loop. Quote: "trace the correct way to reason about these deal points."
- **Rubrics are human-written.** Per the BigLaw Bench launch post (https://www.harvey.ai/blog/introducing-biglaw-bench): "Harvey's research team developed bespoke rubrics to evaluate each task." Bespoke = attorney-authored, per practice area. Scoring is `(positive points + negative points) / total positive points` — continuous, not pass/fail. Hallucinations get *negative* scores.
- **Workflow UX, not a closed loop.** The Workflows help doc (https://help.harvey.ai/articles/assistant-workflows) describes "Check Steps," "Review Citations," and "Draft Editor" — all *human-in-the-loop verification*, not agent self-critique. The lawyer is the critic.

The 96-99% recall floors the proposal cites are **not produced by a self-critique loop**. They are produced by (a) carefully scoped Tool Bundles, (b) human-traced reasoning, and (c) lawyer review at the UI layer. This matters for the proposal's framing.

---

## Where the proposal matches current best-practice

1. **Best-of-iterations safety net.** The proposal's invariant 2 ("never returns a worse deliverable than was already produced") is correct and consistent with Self-Refine's documented failure mode where 33% of unsuccessful Self-Refine attempts are due to feedback inaccurately pinpointing errors and 61% are due to inappropriate fixes (https://reflectedintelligence.com/2025/05/20/self-refine/). Without best-of-N selection, iteration regresses on average. LangGraph's reflection notebooks explicitly do not have this guard, which is a known weakness.
2. **Temperature-0 critic with normal-temperature producer.** Standard practice. Echoed in eugeneyan's LLM-evaluators post (https://eugeneyan.com/writing/llm-evaluators/) and the Justice or Prejudice paper (https://openreview.net/forum?id=3GTtZFiajM).
3. **Explicit budget ceiling.** Matches OpenAI Agents SDK's `max_turns`, AutoGen's stop-condition pattern, and Devin's hard wall-clock cap. Without it, the Vending-Bench failure mode (https://andonlabs.com/evals/vending-bench) appears: agents enter "tangential meltdown loops" at long horizons.
4. **Per-criterion judging via `asyncio.gather`.** Correct. Independent criteria are embarrassingly parallel; map-reduce over criteria is what the BigLaw Bench scoring formula naturally requires (per-criterion positive/negative points, summed).
5. **Memory-channel injection (REFLECTION section).** Matches the canonical Reflexion paper (Shinn et al., NeurIPS 2023) — verbal RL via persistent memory across trials. Note: the langgraph-reflection package skips this and just re-invokes with the critique appended, so the proposal is closer to the original Reflexion paper than to LangChain's own production package.
6. **`max_iterations=3` default.** Calibrated correctly. The Comparing Self-Refine and Reflexion analysis (https://reflectedintelligence.com/2025/05/20/self-refine/) shows multi-agent Reflexion at 300-400 API calls per task, ~3× single-agent — i.e., production pressure is to *cap* iteration, not extend it.
7. **Recognizing judge bias / score drift.** Cited but the proposal could go harder here. The 2024 "Justice or Prejudice" paper and the 2025 unreliability work (https://arxiv.org/html/2412.12509v2) both show LLM-judge scores swinging *more* than 20-40 points across reruns of identical inputs in some setups, and that explicit bias-mitigation prompts can *increase* bias paradoxically. `temperature=0` is not enough.

---

## Where the proposal diverges from production patterns and why

### 1. Wrap-as-`BaseAgent` is a layering smell

Every shipping framework I checked treats reflection as either (a) a *Runner-level* concern or (b) a graph composition primitive — never as a `BaseAgent` subclass.

- LangGraph: `create_reflection_graph(assistant_graph, judge_graph)` returns a `Pregel` graph, not an agent.
- AutoGen: it's a *design pattern* assembled from `RoutedAgent` instances at the runtime level.
- OpenAI Agents SDK: the `Runner` runs the loop; `Agent` is config.
- Mastra: workflows wrap agents, not the other way around.
- The proposal's invariant 6 ("`ReflexiveAgent(producer=ReflexiveAgent(...))` works (though probably not useful)") concedes the smell — when composition produces "works but useless," that's the type system telling you the abstraction is wrong.

The *kaos-agents* internal split is `Agent` (frozen config) vs `Runner` (execution engine). Wrapping at the `BaseAgent` level forces the loop config (max_iterations, min_pass_rate, budget) into "frozen agent config," which mixes layers. **Recommendation: ship it as a Runner-level wrapper** (`Runner.run_with_reflection(agent, rubric, ...)` or a `ReflexiveRunner`). This matches what every reference framework does and makes the open question 1 in the proposal answer itself.

### 2. RubricDeriver is a pattern Harvey explicitly rejected

The proposal's `RubricDeriver` turns a prose task into a Rubric when one isn't supplied. This is **not** how production legal AI works.

- Harvey: "bespoke rubrics… developed by Harvey's research team" — attorney-written per practice area (https://www.harvey.ai/blog/introducing-biglaw-bench).
- Auto-Rubric (https://arxiv.org/html/2510.17314v1, Oct 2025) is a *training-time* method to compress expert-defined rubrics, not a runtime auto-derivation.
- "Agentic Rubrics as Contextual Verifiers for SWE Agents" (https://arxiv.org/html/2601.04171v1) ground rubrics in *concrete repository entities* (files, classes, methods) via tool calls — not free-form derivation from a prose task.
- The empirical result from the rubric-based code evaluation paper (https://arxiv.org/html/2503.23989v1) is that **question-specific rubrics substantially outperform question-agnostic ones**. A model-derived rubric from a prose task is closer to a question-agnostic rubric than a question-specific one — it lacks the domain ground truth.

The failure mode here is well-documented in the Reflexion literature: the Reflected Intelligence post on Reflexion (http://reflectedintelligence.com/2025/05/19/reflexion/) flagged that "Reflexion sometimes hallucinated a new task specification and confidently steered the agent away from the true objective." A self-derived rubric is exactly that hallucinated task spec, then graded against itself. Both errors will be correlated. **Recommendation: drop `RubricDeriver` from v1.** Ship rubrics-as-input only. If a caller doesn't supply one, fail closed. Add `RubricDeriver` later as an *advisory* tool that surfaces a candidate rubric for human approval, not as an auto-pilot.

### 3. The shared-context-correlated-errors problem isn't addressed

The proposal's failure mode 4 ("Critic hallucinates pass") notes the issue but the mitigation is weak ("structural pre-checks catch obviously empty deliverables"). The 2025 production literature is sharper: "shared context creates correlated errors. When the same agent that wrote the code also judges its correctness, it evaluates its own output using the same reasoning patterns and the same context window that produced the errors in the first place" (Augment Code multi-agent guide).

Devin's solution: **isolated worktrees** for the implementor; verifier reads results from a *separate* context (https://cognition.ai/blog/devin-annual-performance-review-2025).

The proposal currently has the producer and critic share neither context window nor model family by default (it cites `claude-haiku-4-5` for the critic in `__init__`), but the rubric, the deliverable, and the prior REFLECTION memory all go into the critic's prompt, which is the same channel that fed the producer. Mitigation: critic should not see the producer's reasoning trace, only the *final* deliverable and the rubric. Append a unit test in v1 that ensures `RubricCritic.__call__` does not receive any of the producer's intermediate state.

### 4. Pass-rate threshold collapses to brittle binary at the edge

The proposal sets `min_pass_rate=0.8` as default. With Harvey's all-pass methodology (`min_pass_rate=1.0`) on a 50-criterion rubric, one judge false-negative permanently blocks the loop until budget trips. With score drift documented at 20-40 points per criterion, the all-pass mode is unreliable on rubrics > ~10 criteria.

- BigLaw Bench scoring is `(positive_points + negative_points) / total_positive_points`, **continuous**, not pass/fail. The proposal's `RubricCriterion.weight` field supports this, but the default is "all-pass," which is wrong as a default for a system that will use LLM judges.
- The `failed: tuple[CriterionVerdict, ...]` accessor is the right shape for feedback-injection; per-criterion threshold-on-confidence (only feed back criteria where `confidence > 0.7`) would help.

### 5. Cost model is light by ~3-5×

The proposal estimates "$0.30 per task with 3 iterations on the Harvey CoC benchmark." Math check: 3 iterations × 50-criterion rubric × 1 judge call/criterion × ~$0.001/Haiku-judgment = $0.15 just on the critic, plus 3× the producer cost. If the producer is `claude-sonnet-4-6`-class on a 30k-token deal-room corpus, **producer cost dominates by ~10×**. Realistic estimate per task with prompt caching: $1.50-$3.00 for a real M&A CoC review.

For comparison:
- Harvey's per-seat pricing is $1,000-$1,200/lawyer/month with 20-seat minimums — i.e., a $288k/year floor (https://www.eesel.ai/blog/harvey-ai-pricing). At ~50 matters per lawyer per year × 20 seats = ~1000 matters / year for $288k → **$288/matter implicit ceiling**. $1.50-$3.00 for an automated CoC review is well inside that, but ten of those per matter starts to bite.
- **Prompt caching is non-negotiable.** Anthropic's prompt caching (https://platform.claude.com/docs/en/build-with-claude/prompt-caching) cuts cached-prefix cost to 10% of base. The cache *write* costs 25% extra, breakeven at two API calls — and the producer/critic loop is ≥ 2 calls by definition. The proposal does not mention prompt caching at all. **Add it as an explicit invariant.** With Sonnet 4.6 + caching, multi-iteration cost falls 5-10× per the Anthropic guidance.
- Bigger miss: the proposal critic re-judges *all* criteria every iteration. After iteration 1, criteria that passed should be *frozen* (cached verdict + small re-check, or skipped) and only failed criteria re-judged. This is the equivalent of incremental compilation — `O(failed_criteria)` per iteration, not `O(|criteria|)`. Easy ~3× cost win.

### 6. Stagnation detection is described but not designed

Failure mode 3 mentions "criterion failed last 2 iterations identically → mark as 'blocked'." This is the right instinct — Cognition's 2025 review (https://cognition.ai/blog/devin-annual-performance-review-2025) explicitly calls out that Devin "performs poorly with mid-task scope changes" and that detecting when an agent is *stuck vs. progressing* is hard. The proposal needs a concrete `StagnationDetector` interface — diff between iteration N and N-1 deliverables, plus criterion-set-equality on the failed list. This is ~20 LoC and prevents the budget from being burned on identical retries.

### 7. The proposal doesn't describe how the rubric is *given* to the producer

Open question 3 in the proposal ("does the producer need to know about the rubric during the FIRST pass") is the most important unanswered design question. Empirical answer from rubric-based code evaluation (https://arxiv.org/html/2503.23989v1): **question-specific rubrics in the prompt substantially outperform** the alternative. Harvey's Workflows expose "25+ deal points" to the user *before* the agent runs (https://www.harvey.ai/products/workflows) — the agent has been trained against a known schema. Production answer: yes, give the producer the rubric on first pass. The "writing-to-the-test" concern the proposal raises is real but smaller than the lift from grounding.

---

## Specific framework citations

| Framework | Version | URL | Key behavior |
|---|---|---|---|
| `langgraph-reflection` | (latest, no semver) | https://github.com/langchain-ai/langgraph-reflection | Two-agent factory: `create_reflection_graph(assistant, judge)`. Termination = judge returns no message. No rubric. |
| LangGraph LATS tutorial | LangGraph 0.x | https://langchain-ai.github.io/langgraph/tutorials/lats/lats/ | MCTS + reflection over agent trajectories. Branching=5. Sensitive to reward score noise. |
| AutoGen Reflection | AutoGen 0.4 (Jan 2025) | https://microsoft.github.io/autogen/stable/user-guide/core-user-guide/design-patterns/reflection.html | Coder/reviewer pair. Pattern, not primitive. |
| Microsoft Agent Framework | Oct 2025 | https://visualstudiomagazine.com/articles/2025/10/01/semantic-kernel-autogen--open-source-microsoft-agent-framework.aspx | AutoGen + Semantic Kernel merger. Reflection promoted to enterprise-durable pattern. |
| OpenAI Agents SDK | (current) | https://openai.github.io/openai-agents-python/guardrails/ | Guardrails are *tripwires* (raise + stop). Iteration is the dev's responsibility. Replaces Swarm. |
| Pydantic AI | 0.x (current) | https://ai.pydantic.dev/api/agent/ | `retries=1` default; `ModelRetry` triggers schema-error feedback. Per-tool overrides. |
| Vercel AI SDK | 5.0 (Jul 2025), 6.0 (Q4 2025) | https://vercel.com/blog/ai-sdk-6 | `streamObject` validates final state; agentic loops are 5.0-and-later primitive. |
| Mastra | (current) | https://mastra.ai/docs/workflows/overview | Zod-validated step boundaries; no automatic retry. Workflows over agents. |
| CrewAI hierarchical | 2025 | https://docs.crewai.com/en/learn/hierarchical-process | Manager agent delegates and *validates* outcomes; 6-9 manager calls per 3-worker crew. |
| Cognition Devin | 2.2 (2025) | https://cognition.ai/blog/devin-2 | Critic is *external* (CI, linters, security scanners). Isolated worktrees for implementor. |
| Devin annual review | 2025 | https://cognition.ai/blog/devin-annual-performance-review-2025 | "Verification and human review remain essential for subjective decisions." |
| Browser Use | 2025 | https://github.com/browser-use/browser-use | Screenshot verification; ~15k tokens/screenshot is a real cost. |
| Tau-bench / τ²-bench | 2024 / Jun 2025 | https://arxiv.org/abs/2506.07982 | DB-state-equality at end-of-conversation. `pass^k` reliability metric — matters for self-critique. |
| BigLaw Bench Workflows | 2025 (SPA), 2026 expansions | https://github.com/harveyai/biglaw-bench, https://www.harvey.ai/blog/biglaw-bench-workflows-spa-deal-points | 98.47% on SPA deal points; rubrics human-written; scoring is positive-minus-negative-over-total. |
| Vending-Bench 2 | Feb 2025 | https://arxiv.org/abs/2502.15840 | Long-horizon agent failure isn't context-window-bound. Loops "meltdown" without external reset. |
| BFCL V3 | 2024-2025 | https://gorilla.cs.berkeley.edu/blogs/13_bfcl_v3_multi_turn.html | Multi-turn = state-equality on the API system, not LLM judging. |
| GAIA / Gaia2 | 2024 / 2025 | https://openreview.net/forum?id=9gw03JpKK4 | GPT-5 (high) 42% pass@1. Each scenario has a write-action verifier (RLVR-ready). |
| Reflexion | NeurIPS 2023 | (canonical) | Verbal RL + persistent memory across trials. ~30% latency increase, 10-30% quality lift on failure-mode subset. |
| Self-Refine | NeurIPS 2023 | https://arxiv.org/abs/2303.17651 | 33% of failures = bad error localization; 61% = inappropriate fix. Mitigation: best-of-N. |
| AgentPRM | Feb 2025 | https://arxiv.org/html/2502.10325v1 | Process reward models replace LLM-judges for training-time critic. |

---

## Cost / latency reality check

For a real M&A CoC review across a deal room of ~500 contracts, with the proposal's defaults:

| Item | Estimate |
|---|---|
| Producer (Sonnet-4.6, 30k input + 5k output × 3 iterations) | ~$0.90 uncached, ~$0.18 with 5-min cache |
| Critic (Haiku-4.5, 50 criteria × 3 iterations × 1.5k tokens each) | ~$0.15 |
| Memory hydration / dehydration (VFS reads) | trivial |
| **Per-CoC-task total (cached)** | **$0.33-$0.50** |
| **Per-CoC-task total (uncached)** | **$1.05-$1.50** |

The proposal's $0.30 estimate is achievable **only with prompt caching**. Without it, costs run 3-5× higher. Production numbers from Anthropic's prompt caching guide (https://platform.claude.com/docs/en/build-with-claude/prompt-caching) show 5-10× cost reduction on multi-turn loops with a 10k+ token system prompt — exactly this shape.

Latency: Haiku-4.5 critic at 50 criteria with concurrency=10 is ~5s wall-clock per iteration. Producer is the long pole at ~30s/iteration on Sonnet for a 30k-token corpus. **3 iterations ≈ 90-120s wall-clock**, which is acceptable for a non-interactive workflow but will not pass real-time interactive UX.

Compare to Harvey's per-matter implicit cost: at $1,200/lawyer/month and ~10-20 active matters/lawyer-month, the per-matter cost ceiling is *$60-$120 for the entire AI assist*, not per task. ReflexiveAgent at $0.50/task fits comfortably even with 50-100 tasks per matter.

The dominant cost lever the proposal misses: **incremental criterion re-judging**. After iteration 1, ~80% of criteria typically pass and don't need re-judgment. Skip them. This is `O(failed_criteria)` per subsequent iteration, not `O(|criteria|)` — a 2-4× cost win on iterations 2 and 3.

---

## Bottom line

The architectural shape is right, but it should be a `Runner`-level wrapper, not a `BaseAgent` subclass. Drop `RubricDeriver` from v1 — production legal AI uses human-written rubrics and the empirical case for auto-derivation is weak; it's the same hallucinated-task-spec failure mode Reflexion is known for. Mandate prompt caching and incremental criterion re-judging as cost invariants. Tighten the critic-context-isolation story (the producer's reasoning must not flow into the critic's prompt). Replace "all-pass" defaults with weighted continuous pass-rate matching BigLaw Bench's `(pos+neg)/total_pos` formula. Build stagnation detection in v1, not v2.
