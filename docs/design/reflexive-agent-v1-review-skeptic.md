# ReflexiveAgent v1 — Skeptical Review

**Reviewer stance:** principled skeptic. Argues against building `ReflexiveAgent` as
proposed, or for building something materially different. Author of the proposal
is treated as a sharp, well-meaning practitioner who has talked themselves into
the wrong fix.

## TL;DR

The 18.2% Harvey CoC pass rate is **not** evidence the agent's outer loop is
missing. It is evidence that (a) the producer is `claude-haiku-4-5`, (b) the
deliverable was *truncated mid-sentence* in the Technology License section
(see `docs/benchmarks/harvey-coc-2026-05-06.json:23`, deliverable string ends
mid-paragraph), and (c) the failures cluster on **specific factual recall** the
agent never retrieved (Section 14.2 anti-assignment, $36.2M revenue
concentration, Section 8.01(j)). None of those gaps are fixed by adding a
critic loop on top of the same producer with the same retrieval. The proposal
is solving a *thermostat* problem when the actual failure is a *furnace*
problem (the producer never read the right pages and ran out of output budget).

The right v1 is not `ReflexiveAgent`. It is, in order:

1. Switch the M&A-review preset to `claude-sonnet-4-6` or `claude-opus-4-7`
   for the producer (the architecture already supports per-role models — see
   `kaos_agents/providers.py:56`, the `BALANCED` preset already routes
   `PLAN`/`RESEARCH` to Sonnet).
2. Fix the truncation. The deliverable cut off at ~15.7 KB mid-section. That
   is a max-tokens / streaming-cap bug, not a loop bug.
3. Verify the existing `RetrievalAgent` is actually being delegated to (it
   exposes `kaos-retrieval-synonyms` and `kaos-retrieval-hyde`, see
   `kaos_agents/retrieval_agent.py:42-46`). It is plausible the haiku producer
   never invoked it.
4. Turn the existing `_reflect_on_coverage` knob from 2 reflective rounds
   (`kaos_agents/context/retrieval.py:279`) up to 4–5 and re-measure.
5. *Then*, if the residual gap is large and clearly diagnostic-of-loop, build
   the smallest possible critique step — and even then, build it as a
   one-shot `kaos-agent-rubric-critique` MCP tool, not as a `BaseAgent`
   subclass with seven invariants.

Below: the strongest objection, what's being conflated, the cheaper
alternative, the decisive experiment, and what to build instead.

---

## 1. The strongest objection

**The proposal commits to building Reflexion-shaped self-correction at the
exact moment the literature has converged on the conclusion that intrinsic
self-correction does not work.**

Two papers settle this:

- **Huang et al., "Large Language Models Cannot Self-Correct Reasoning Yet,"
  ICLR 2024.** Direct quote from the paper: when LLMs attempt to correct
  their initial responses based solely on their inherent capabilities,
  *without external feedback*, performance often **degrades**. Not "fails to
  improve" — degrades.
- **Stechly, Marquez, Kambhampati, "GPT-4 Doesn't Know It's Wrong," 2023
  (later expanded in Stechly et al. 2024 NeurIPS).** Tested iterative
  self-critique on Game of 24, Graph Coloring, STRIPS planning. Self-critique
  produced a *significant performance collapse*. External sound verifiers
  recovered the gains. Re-prompting with a sound verifier captured most of
  the benefit of more elaborate setups — meaning the loop wasn't doing the
  work, the verifier was.

The proposal's response to this (`reflexive-agent-v1.md:230-234`) is one
bullet: "Critic hallucinates pass: judge says PASS when deliverable is
actually wrong. Mitigation: `rubric_judge` system prompt is strict about
specific facts; structural pre-checks; future work — a verifier program."
That is exactly the wrong direction. The literature says: *the judge is the
load-bearing component, not the loop.* If the judge is `claude-haiku-4-5`
grading a `claude-haiku-4-5` deliverable (proposal default at line 137,
matching the Harvey baseline at `harvey-coc-2026-05-06.json:14`), the loop
adds **no information**. It is a same-distribution sampler dressed up as a
control system.

The single experiment in the validation plan that would actually test this
("Loop-3-strong-critic" with a Sonnet critic, line 259) is presented as
*one of seven* runs. It should be the **first** run, because if it doesn't
clear the existing baseline, none of the others matter.

Concrete prediction: with same-model critic, lift will be 0–4 percentage
points (within rerun noise — the proposal itself documents 20-40-point judge
score swings at line 218). With Sonnet critic on Haiku producer, lift may
hit 10–15 points but at that point the cheaper experiment was: just use
Sonnet as the producer.

## 2. What the proposal is conflating

The proposal silently merges three orthogonal capabilities under one
abstraction. They have different cost curves, different failure modes, and
different evidence bases.

| Capability | What it actually is | Evidence base |
|---|---|---|
| **Stronger producer** | More compute / better model on the producer call | Strong (Snell et al. 2024 "Scaling LLM Test-Time Compute" — but with the asterisk that test-time scaling helps *math/coded reasoning*, not free-form retrieval-bound legal drafting) |
| **External verifier** | A judge that can detect errors the producer cannot | Strong (Stechly 2024) — but only if the verifier is **stronger or independent** of the producer |
| **Self-critique loop** | Same model grades its own work, iterates | Weak-to-negative (Huang 2024, Stechly 2024) |

`ReflexiveAgent` is positioned as #3 but argued for using evidence for #1
and #2. The "thermostat" metaphor in the opening hides this. A thermostat
works because the thermometer is *physically independent* of the furnace.
A same-model critic is a thermometer made out of the furnace.

There's a second conflation: **planning vs. critique**. The proposal at
lines 14-21 says the agent has no "target state, measurement of current
state, or iteration on the gap." But `plan_execute.py:184-186` already
assembles `prior_failures` from `MemoryType.REFLECTION`, and
`route()` (referenced at line 45 of the proposal, lives in
`kaos_agents/planning/route.py`) already has CONTINUE/REPLAN/DEEPEN/STOP
control flow. The thermostat *exists*. The proposal is not adding the
thermostat — it is adding a *second* thermostat at a higher level. That may
be fine, or it may be a duplicate-control-loop bug waiting to happen
(plan-execute decides to REPLAN, ReflexiveAgent decides to re-iterate, the
agent burns 6× the budget producing 2× the work).

## 3. What the cheaper / simpler alternative is

The Harvey CoC failure modes, as enumerated in
`docs/benchmarks/harvey-coc-2026-05-06.json:24-100`, are **retrieval
failures and output-budget failures**, not metacognition failures. Look at
the criteria the agent missed:

- C-001: "Identifies Section 14.2." Reasoning: "deliverable does not
  identify or reference Section 14.2." Retrieval miss.
- C-004: "Northland revenue concentration ~19.3% / $36.2M." Reasoning:
  "report notes Northland accounts for material portion of revenues but
  provides no specific percentage or dollar figure." Retrieval miss
  *and* extraction miss (numbers were in the document — agent didn't
  extract them).
- C-005: "Section 1.01 'Change of Control' definition >50% voting stock."
  Reasoning: "deliverable explicitly states 'Not explicitly defined as
  Change of Control'." Retrieval/reading miss.
- C-006: "Section 8.01(j)." Reasoning: "no section 8.01(j) reference
  appears anywhere in the deliverable." Retrieval miss.

And critically: the deliverable is **truncated mid-sentence** in the
Technology License section (`docs/benchmarks/harvey-coc-2026-05-06.json:23`,
last characters: `"the hardware, firmware, and non-FlowLogic software
components"` — sentence does not finish). Multiple criteria explicitly note
this in their reasoning (C-002: "deliverable cuts off mid-sentence in the
Technology License section"). This is not a loop problem. This is an output
token cap or streaming truncation bug.

The cheaper alternative stack:

```
Step 1 (cost: ~0): Fix the output truncation. Audit max_tokens in the
producer call path. If deliverable_chars cap is at 15.7 KB on a task that
needs 50+ KB of analysis, no loop will save it.

Step 2 (cost: ~$0.10/task): Switch the M&A preset producer to
claude-sonnet-4-6 via the existing ProviderConfig.BALANCED machinery.
Re-run Harvey CoC. The 5x cost increase ($0.028 -> $0.14) is well under
the proposed $0.30/task ceiling. Predicted lift: 15-30 percentage points
based on Anthropic's own model-card retrieval gains haiku→sonnet.

Step 3 (cost: ~$0.05/task): Verify the RetrievalAgent's
kaos-retrieval-synonyms and kaos-retrieval-hyde tools were actually
called during the baseline run. If they weren't, that's the bug. The
retrieval tools already cover "consumer-language vs. technical-language
gap" (retrieval_agent.py:44-46) — exactly the C-001 / Section-14.2
failure mode.

Step 4 (cost: marginal): Bump _reflect_on_coverage from 2 rounds to 4
(retrieval.py:279). It already exists. It already produces "missing
aspects" gap queries. It's the thermostat the proposal claims doesn't
exist, just at retrieval level instead of deliverable level.
```

If after all four steps the pass rate is still below 60%, *then* and only
then build a critique step. And build it as a single-call `kaos-extract-
verify`-shaped tool that the agent invokes when its plan finishes —
**not** as a `BaseAgent` subclass with composition guarantees.

## 4. The cost ceiling is wrong

The proposal targets `$0.30 / task with 3 iterations / 50 criteria`
(line 235). The existing baseline already costs **$0.27 just for judging**
($0.024 producer + $0.24 judge, see
`docs/benchmarks/harvey-coc-2026-05-06.json:17-18`). With three loop
iterations re-judging on each round, the floor is ~$0.72/task, not $0.30.
The proposal's math at line 235 ("10 iterations × 50 criteria × $0.001
= $0.50") is plausibly off by 5-10× because:

- It uses 10 iterations, not the 3 the architecture defaults to.
- It assumes $0.001/judgment when the measured rate is $0.0044/judgment
  (haiku, $0.24 / 55 criteria) — already 4.4× higher than the assumption.
- It does not account for the producer cost on each re-iteration (the
  producer rewrites the deliverable; that is the dominant cost on
  Sonnet/Opus).

For a realistic deal-room (100+ contracts, 200+ criteria) with a
Sonnet producer at $3/MTok input, and 3 iterations:
- Producer per iteration: ~30 KB output × 3 iter = ~25k output tokens
  at $15/MTok = $0.38 producer
- Judge per iteration: 200 criteria × $0.005 = $1.00 × 3 = $3.00 judge
- Total: ~$3.40/task

Multiply by 50–100 tasks in a real deal review: **$170–$340 per
deal**. Not $0.30. That may still be defensible vs. associate hours, but
the proposal's stated cost ceiling is for the toy benchmark, not for
production scale.

## 5. The decisive experiment that would disprove the proposal

The proposal's validation plan (lines 252-262) is structured to almost
guarantee a positive result: it varies *iteration count* with the same
producer and same critic, so any iteration > 1 will find *something*
to change, and best-of-iterations selection ensures the loop is never
worse than baseline. That isn't a falsification design.

A clean falsifier:

> **Experiment**: Run the Harvey CoC benchmark four ways, holding
> total inference cost constant at $X (say $0.30):
>
> 1. **Strong producer, no loop**: `claude-sonnet-4-6` producer, single
>    pass. No critic. Spend the entire $0.30 on the producer.
> 2. **Same-model loop**: `claude-haiku-4-5` producer + same-model
>    critic, 3 iterations. Proposal default.
> 3. **Cross-model loop**: `claude-haiku-4-5` producer +
>    `claude-sonnet-4-6` critic, however many iterations fit in $0.30.
> 4. **BoN sampling**: `claude-haiku-4-5` producer, 6 parallel samples,
>    pick best by `claude-sonnet-4-6` judge (BestOfN already exists in
>    `kaos-llm-core/programs/best_of_n.py`).
>
> **Prediction**: ranking will be 1 ≈ 4 > 3 >> 2. That is, a
> stronger single pass and BoN-with-strong-judge tie or beat
> the loop, and the same-model loop barely moves vs. baseline.
>
> If 2 wins by >10 points over 1 and 4, the proposal is right and I
> withdraw the objection. If 2 is within 5 points of baseline (likely
> outcome), the proposal is wrong and the team has spent ~700 LoC to
> learn a lesson Huang 2024 already published.

The proposal can be built **after** this experiment, scoped to whatever
shape actually wins. If cross-model loop (option 3) wins, build that —
and `ReflexiveAgent` then reduces to "make sure the critic is a different,
stronger model than the producer" plus 100 LoC of glue, not 700.

## 6. Rubric eval as a production methodology — secondary concern

The proposal commits to all-pass rubric scoring with `min_pass_rate=0.8`
default (line 138) as the production methodology. Two issues, both
manageable but worth flagging:

- **All-pass scoring tracks Harvey's benchmark methodology, not user
  utility.** A deal memo that captures 90% of risks but misses one
  tertiary criterion fails Harvey's all-pass; a partner reviewing it
  would call it excellent. The proposal is optimizing for a benchmark
  artifact.
- **Criteria-as-spec drift.** The literature (Auto-rubric 2026 / Zheng
  et al. 2024 LLM-as-judge surveys) repeatedly finds that rubrics
  designed by the same team that builds the agent suffer from
  "criteria-the-agent-can-pass" selection bias. Harvey's published
  criteria sidestep this for Harvey's tasks but transfer poorly to
  open-ended drafting, negotiation, or multi-stakeholder synthesis —
  exactly the M&A-review surface area the proposal targets at line 281.

Neither is a reason to not build the loop. Both are reasons to scope it
to *checklist-shaped* tasks (extraction, due diligence, compliance) and
explicitly *not* to drafting/synthesis — and the proposal does not draw
that line.

## 7. If skeptic is right, here's what to build instead

In strict order, with concrete acceptance criteria:

### Build #1 (1–2 days): Fix the producer

- Audit the producer's `max_tokens` ceiling. Confirm with a unit test
  that a 50KB deliverable round-trips intact.
- Add a benchmark target: `Harvey CoC, sonnet-4-6 producer, no other
  changes`. If pass rate jumps from 18% → 45%+, the entire ReflexiveAgent
  motivation evaporates. **Run this before writing any new code.**

### Build #2 (1 day): Verify retrieval is wired

- Add an event-trace assertion to `harvey_coc_benchmark.py` that
  `kaos-retrieval-synonyms` or `kaos-retrieval-hyde` was invoked at
  least once when the corpus exceeds 100 chunks. If it wasn't invoked
  in the 18%-baseline run, the bug is there, not in the outer loop.
- Bump `_reflect_on_coverage` rounds from 2 to 4 and re-measure. Free
  experiment.

### Build #3 (2 days, only if #1 and #2 don't clear 60%): One-shot rubric critique tool

- Add a single MCP tool `kaos-agent-rubric-critique` that takes a
  deliverable + rubric and returns a `RubricVerdict`. **No agent
  subclass. No outer loop component. No invariants 1–7.**
- Wire it into the existing `plan_execute` REPLAN path: if any
  criterion fails, REPLAN with the gap as `prior_failures`. The
  thermostat is already there; you're just giving it a better
  thermometer.
- Default critic model: at least one tier above the producer.
  Hard-code this; do not let users configure same-model critic.

### Build #4 (only if #3 clears another 15+ points): The actual loop

- Now you've earned the right to build `ReflexiveAgent`. By this
  point you'll know which iteration count, which feedback strategy,
  and which critic strength actually matter. The 700 LoC will
  collapse to ~300 because half the questions in the "Open
  questions for v2" section (lines 287-310) will have answered
  themselves.

The total path saves ~2 weeks of engineering, avoids committing to an
abstraction that the literature warns against, and produces an
auditable improvement curve instead of a single "we shipped the loop"
event whose lift cannot be cleanly attributed.

## Summary

The proposal is well-engineered, well-cited, and wrong about the diagnosis.
The 18.2% Harvey CoC baseline is a producer + retrieval + truncation
failure dressed up as a metacognition gap. The fix Huang 2024 and Stechly
2024 both point to is *external verification* (different model, different
information), not *self-iteration*. Before writing 700 LoC of outer-loop
machinery, the team should run a constant-cost head-to-head between a
stronger single-pass producer, a same-model loop, and a cross-model loop
on the same benchmark. My prediction is the stronger producer wins or
ties at lower complexity, and the same-model loop adds noise. If the
prediction is wrong, the proposal as designed is the right call. If the
prediction is right, the team has a much smaller, much cheaper
intervention — make the critic strictly stronger than the producer, wire
it into the existing `plan_execute` REPLAN path, and ship 300 LoC
instead of 700.
