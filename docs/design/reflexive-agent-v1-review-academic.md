# ReflexiveAgent v1 — Academic / Literature Review

**Reviewer mandate:** evaluate the v1 design proposal for `ReflexiveAgent`
against the published literature on critique-revise loops, self-correction,
LLM-as-judge reliability, and rubric-based agent evaluation. Honest
disagreement is requested.

**TL;DR.** The proposal is well-engineered as software but rests on a
much weaker empirical foundation than its citation list (Reflexion,
Self-Refine, MIPROv2) implies. Three independent lines of post-2023
literature — Huang et al. ICLR 2024, Stechly et al. 2023, Valmeekam &
Kambhampati NeurIPS 2023, Kamoi et al. TACL 2024 — converge on the
finding that **intrinsic self-critique without an externally grounded
verifier is at best a wash and frequently degrades performance on
reasoning-heavy tasks**. The proposal's most load-bearing assumption —
that an LLM critic of comparable strength to the producer can supply
the missing error signal — is the assumption that field most strongly
challenges. Several of the seven "loop invariants" are correct but
weaker than they sound; one of them (best-of-iterations) is
operationally unsafe in the way the proposal currently writes it
because the *score itself* is the noisy output of the same critic that
already has known biases. The fix is not to abandon the design but to
reframe it: ReflexiveAgent is a useful scaffold *if and only if* its
critic is grounded in something cheaper and more reliable than another
LLM call against the same model family.

---

## 1. Where the proposal lines up with strong evidence

**1.1 Best-of-iterations selection is the right safety net.** The
proposal's invariant #2 (never return a worse deliverable than was
already produced) is exactly the lesson AlphaCode (Li et al., *Science*
2022) institutionalised: **filter, don't refine in place**. AlphaCode's
contribution was not "a better generator" — it was a million-sample
filter pipeline that submitted only 10 candidates. The same
filter-rather-than-rewrite pattern shows up in Brown et al. *Large
Language Monkeys* (arXiv:2407.21787, 2024) and Snell et al. (arXiv:
2408.03314, 2024), which found that compute spent on more candidates +
a verifier often beats compute spent on a larger model. Keeping the
best-scoring iteration rather than the last is genuinely defensible.

**1.2 Producer-cloning per iteration is correct.** Madaan et al.
(*Self-Refine*, NeurIPS 2023) and the DSPy `Refine`/`BootstrapFewShot`
optimizers (Khattab et al., ICLR 2024) both take this approach. The
proposal's reuse of `Refine`'s per-iteration clone pattern is well
grounded in the optimizer literature.

**1.3 Multi-criterion structured rubrics beat holistic scoring.** Kim
et al. *Prometheus 2* (EMNLP 2024) showed that fine-grained, criterion-
separated rubrics produce dramatically higher correlation with human
judgments than holistic Likert-style judging. The proposal's
`RubricCriterion`/`Rubric` shape (id, description, match, weight, scoped
deliverables) is exactly the analytic-rubric form that the literature
endorses, and is consistent with the *Rubrics as Rewards* (RaR) line
of work that has shown smaller models trained with structured rubrics
can outperform GPT-4 on domain-specific legal tasks.

**1.4 Structural pre-checks before LLM judgment are correct.** Empty
deliverable / error-prefix gating before invoking the critic is
consistent with the *CRITIC* paper's (Gou et al., 2023) central
finding: external/structural feedback is qualitatively more reliable
than self-prompted feedback. Even a regex-level structural check is
"external" in the sense that matters.

**1.5 Termination guarantees are well-engineered.** The combination of
`max_iterations` + `PlanBudget` + `min_pass_rate` matches the
"defense-in-depth" termination pattern that the production-systems
literature (LangGraph, AutoGen v0.4, Microsoft Agent Framework) has
converged on. Modexa, AgentPatterns, and the public AutoGen
infinite-loop bug threads all confirm: budgets must live outside the
agent, not be self-policed.

---

## 2. Where the proposal contradicts published findings

**2.1 The single most important load-bearing claim is the one the field
most consistently rejects.** The proposal frames `RubricCritic` as the
"thermostat" — the missing measurement device. That framing only works
if the critic produces a reliable error signal. The dominant empirical
finding 2023-2026 is that **same-family LLM critics do not reliably
produce that signal on reasoning-rich tasks**:

- Huang et al., *Large Language Models Cannot Self-Correct Reasoning
  Yet*, ICLR 2024 (arXiv:2310.01798): intrinsic self-correction
  *degrades* GSM8K and CommonSenseQA performance under fair conditions
  (no oracle). The paper attributes prior positive results to leaked
  oracle signals (knowing when to stop is itself external info).

- Stechly, Marquez, Kambhampati, *GPT-4 Doesn't Know It's Wrong*
  (arXiv:2310.12397, 2023): on graph coloring, "the correctness and
  content of the criticisms — whether by LLMs or external solvers —
  seems largely irrelevant to the performance." Apparent gains came
  from the correct answer happening to be in the top-k completions,
  not from critique driving correction.

- Valmeekam, Marquez, Kambhampati (NeurIPS 2023 FMDM workshop): on
  Blocksworld planning, *self-critique diminishes plan generation
  performance* relative to no-critique baselines, and LLM verifiers
  generated a "notable number of false positives."

- Kamoi et al., *When Can LLMs Actually Correct Their Own Mistakes?*,
  TACL 2024: meta-survey concludes "no prior work demonstrates
  successful self-correction with feedback from prompted LLMs, except
  for studies in tasks that are exceptionally suited for self-
  correction" — i.e., tasks with cheap external verification baked in.

The proposal cites Reflexion and Self-Refine as if those papers settled
the question. They did not. Subsequent reproductions (Kamoi 2024;
Huang 2024) re-graded those results under fair conditions and found
the gains substantially smaller or negative. **The 91% HumanEval pass@1
in Reflexion is leaked-oracle territory** — it relies on unit-test
execution as the verifier, which is exactly the externally grounded
signal the proposal does *not* have for legal-deliverable rubrics.

**2.2 "Score drift" is acknowledged but the mitigation is wishful.**
The proposal cites "20-40 point swings" anecdotally and proposes
`temperature=0` as the mitigation. Two papers say this is insufficient:

- Stureborg et al., *Rating Roulette: Self-Inconsistency in
  LLM-As-A-Judge* (Findings of EMNLP 2025): even at temperature 0,
  modern API providers introduce nondeterminism (kernel-level batching,
  speculative decoding). Test-retest agreement on the same input
  routinely falls below 0.7 Spearman.

- Wei et al., *Rubrics as an Attack Surface: Stealthy Preference Drift
  in LLM Judges* (arXiv:2602.13576, 2026): identifies *Rubric-Induced
  Preference Drift* — small rubric edits that pass benchmark sanity
  checks can shift judge preferences systematically, without being
  detectable by aggregate metrics. The proposal's `RubricDeriver`
  (a *generated* rubric) is the canonical example of the attack
  surface this paper identifies.

`temperature=0` is necessary but not sufficient. The proposal needs
test-retest measurement of the critic's own reliability before using
its scores as the loss function for iteration.

**2.3 Same-model critic is the worst-case configuration the bias
literature warns about.** The proposal's default critic is
`claude-haiku-4-5` against a producer that is also Anthropic-family
in this codebase. The literature has named this:

- Panickssery, Bowman, Feng, *Self-Preference Bias in LLM-as-a-Judge*
  (arXiv:2410.21819, 2024): LLMs prefer outputs that look like their
  own training distribution; the bias correlates with perplexity, not
  task quality. Within-family bias is measurable and consistent.

- Li et al., *Preference Leakage: A Contamination Problem in
  LLM-as-a-Judge* (arXiv:2502.01534, 2025): when generator and judge
  share family/lineage, reported gains can be "evaluator preference
  rather than real improvement." They demonstrate this empirically on
  Llama→Llama, GPT→GPT, and inheritance-based pairings.

- Park et al., *Justice or Prejudice* (arXiv:2410.02736, 2024):
  catalogues 12 LLM-judge biases, including style bias (0.76-0.92
  effect sizes — far exceeding position bias).

The proposal's "default critic = same provider as producer" is the
cleanest possible setup for self-preference bias to inflate apparent
pass rates. The Harvey LAB rubric won't shield against this because
the bias attaches to phrasing, not to whether the deliverable actually
satisfies the criterion.

**2.4 The "infinite loop on impossible criteria" mitigation is
under-specified relative to the published failure modes.** The
proposal's per-criterion stagnation detection ("if the same criterion
fails 2 iterations identically, mark blocked") is operationally fine
but assumes the critic produces *deterministic* failure signatures.
Stureborg 2025 (*Rating Roulette*) shows that LLM judges produce
*different* reasoning text across reruns of identical inputs — so a
naive equality check on `verdict.reasoning` will not fire even when
the criterion is genuinely blocked. Dedup needs to be by `criterion_id
+ passed`, not by reasoning content, or this mitigation is theatre.

**2.5 The "producer ignores REFLECTION feedback" failure is more severe
than the proposal admits.** The proposal mitigates it by prepending
"Address these gaps from your prior attempt" to the reflection. The
RL-based self-correction literature (Kumar et al., *SCoRe*,
arXiv:2409.12917, 2024) found that **prompted models trained with
standard SFT actively ignore correction prompts** — they have a
distribution-shift incentive to reproduce their first attempt. SCoRe
needed a two-stage RL recipe to overcome it. A pure prompting
mitigation, with no fine-tuning, is documented in SCoRe as the failure
case that motivated the paper. Producer non-responsiveness to
REFLECTION is therefore a *predicted* failure mode, not a corner case.

---

## 3. What the proposal is missing that the literature considers essential

**3.1 No external grounding signal.** Every paper that reports
*positive* self-correction results either (a) has unit tests / a
solver / a calculator (CRITIC, Reflexion-on-HumanEval, AlphaCode), or
(b) trains the model with verifier-grounded RL (SCoRe, ReST^EM, o1-
style RL on verifiable rewards). The proposal has neither. For Harvey
LAB-style legal deliverables, the literature's prescription would be:
mandatory verifier hooks per criterion-class.

  - Numeric criteria (e.g. "$36.2M revenue concentration"): regex
    extraction + structural compare against source. This is
    Dhuliawala et al. *Chain-of-Verification* (arXiv:2309.11495, ACL
    Findings 2024) applied at the criterion level: each verification
    question is answered *independently of the draft*, so error
    propagation from the draft is broken.
  - Citation/quote criteria: substring match against source spans (the
    `[nlp]` BM25 + `kaos-citations` substring verifier is already in
    the codebase — wire it in).
  - Coverage criteria ("all change-of-control clauses identified"):
    enumerate from corpus, then ask the critic to *deny* coverage
    (asymmetric verification), not affirm it.

  Without these grounded sub-verifiers, `RubricCritic` is a same-model
  judge dressed up as a thermostat.

**3.2 No measurement plan for the critic itself.** The validation plan
proposes 7 runs of the *agent* but no runs of the *critic*. The
*Prometheus 2*, *JudgeBench*, and *RewardBench* lines of work all
consider judge reliability a prerequisite to using a judge at all. At
minimum the proposal should add:

  - **Test-retest reliability** of `RubricCritic` on a held-out frozen
    set of (deliverable, rubric) pairs. Report Krippendorff's α or
    Gwet's AC2. The legal-domain literature (LeMAJ, Chen et al. 2025
    on legal RAG) finds α ≥ 0.8 is the practical bar; agreement rates
    below ~0.65 mean the critic is hallucinating verdicts.
  - **Inter-judge agreement** between the critic and a held-out
    different-family judge (e.g. GPT-5.4-mini judging deliverables that
    Claude judged). If they correlate at <0.5, neither is a usable
    error signal and the loop will optimise noise.
  - **Calibration against human gold** on a small (20-50 deliverable)
    Harvey-style frozen set. Per Yu et al. (*When AIs Judge AIs*,
    arXiv:2508.02994, 2025), domain agreement gaps in legal often
    drop to 64-68% vs. 72-75% inter-expert baseline; without
    measuring this gap, "pass_rate=0.8" means nothing.

**3.3 No story for criterion correlation / weighting.** The rubric
treats criteria as independent and uses an arithmetic
`weighted_pass_rate`. The legal-evals literature is explicit about
this: rubric criteria are typically *highly correlated* (the same
clause analysis often satisfies 4-5 criteria simultaneously), and
equal-weighting inflates apparent diversity of measurement. *Rulers*
(arXiv:2601.08654, 2025) and the *Rubrics as Rewards* line both
prescribe a hard-fail / soft-fail split (essential vs.
important/optional/pitfall) before averaging. The proposal's flat
`weight: float = 1.0` is below the published bar.

**3.4 No diminishing-returns budget shape.** The proposal sets
`max_iterations=3` by fiat. The empirical literature on reflection
budgets is consistent that **gains plateau after 2-3 iterations on
most tasks** (AI Agent Prompt Engineering studies; *The Cost of
Dynamic Reasoning*, arXiv:2506.04301, 2025; *Efficient Agents*,
arXiv:2508.02694, 2025). The proposal's budget should not be "3 by
default"; it should be "stop when the marginal `weighted_pass_rate`
delta < ε for one iteration", which is the *patience* idiom from
classical early stopping literature applied to test-time compute.
Otherwise the loop pays full LLM cost for what is statistically
indistinguishable from noise.

**3.5 No coverage of the asymmetric-verifier insight.** Brown et al.
(*Large Language Monkeys*, 2024) and Snell et al. 2024 are explicit
that **best-of-N + verifier scales test-time compute roughly as well
as a 14× larger model**, but only when the verifier is a *different
artifact* from the generator (PRM, ORM, executable test). The
proposal's critic is the same architecture and (likely) the same
provider as the producer. The literature predicts this gives most of
the cost of best-of-N with little of the benefit.

**3.6 No mention of the `RubricDeriver` failure mode the literature
specifically names.** The proposal acknowledges "rubric drift on
derived rubrics" and proposes caching by task hash. This misses the
deeper problem: a *generated* rubric is downstream of the same model
that will produce the deliverable, so the rubric will systematically
under-cover the producer's blind spots (Stechly 2023 generalised:
the model cannot reliably enumerate what it doesn't know). The
literature's recipe is *human-curated rubrics from gold-standard
exemplars*, not model-derived ones. The proposal's `RubricDeriver`
is convenient but the literature flags it as an anti-pattern unless
externally validated.

---

## 4. Critique of the seven loop invariants

| # | Invariant | Verdict | Notes |
|---|-----------|---------|-------|
| 1 | Termination guaranteed | **Strong**. Triple-redundant termination matches state of art. |
| 2 | Best-of-iterations selection | **Strong as stated, fragile in practice**. The selection key is `weighted_pass_rate` from the same critic across iterations — but Stureborg 2025 shows test-retest variance on identical inputs. Iteration N's score is *not* directly comparable to iteration N-1's if they're scored in separate critic calls. Need either (a) re-score all iterations against the same critic call, or (b) treat scores as a CI not a point estimate. |
| 3 | Bounded cost | **Correct but incomplete**. The bound is per-iteration; the proposal needs a *per-task* bound that includes the critic. With 50 criteria and 3 iterations, you've made 150 LLM judge calls before any producer call. |
| 4 | Trace-complete | **Strong**. Auditable iteration is required by the literature on agent eval (Yu et al., 2025). |
| 5 | Stateless wrap | **Strong**. Matches DSPy module purity. |
| 6 | Composable | **Probably wrong as stated**. `ReflexiveAgent(producer=ReflexiveAgent(...))` will have the *same critic at both levels*. Stacking same-family critics is exactly the configuration that compounds self-preference bias. The literature would say this should be explicitly forbidden, not "probably not useful." |
| 7 | Producer-agnostic | **Strong**. Pattern-agnostic wrappers are the DSPy/LangGraph default. |

**Missing invariants the literature would insist on:**

- **Critic-generator independence.** No invariant says the critic must
  be a different model family than the producer. The
  preference-leakage and self-preference-bias literature say this is
  a precondition for the loop to measure what it claims to measure.
- **Test-retest reliability floor.** No invariant says the critic must
  achieve some minimum test-retest agreement before its scores are
  used as the iteration signal. Without this, the loop optimises
  noise.
- **Monotonic-improvement check.** No invariant says
  `weighted_pass_rate` must be measured *consistently across
  iterations* (same critic call or paired re-scoring). Without this,
  best-of-iterations selection on a noisy estimator can pick the
  highest-variance iteration rather than the highest-quality one
  (regression to mean in reverse).
- **Externally grounded floor.** No invariant says some non-zero
  fraction of criteria must be verifiable without an LLM call. This
  is the single biggest gap relative to the literature.

---

## 5. Open questions: what the literature has already answered

**Q1: BaseAgent subclass vs. Runner-level wrapper.** *Open in the
field; both architectures exist.* DSPy uses module-as-program (subclass
analogue); LangGraph uses graph-level wrapper. Both work. This is a
codebase-ergonomics call, not a literature call.

**Q2: Per-criterion vs whole-deliverable feedback.** *Substantially
answered.* Madaan 2023 (*Self-Refine*) and *Prometheus 2* both find
**criterion-level feedback strictly dominates holistic feedback** on
correlation with human judgments and downstream refinement gains. Use
GAP_LIST as default.

**Q3: First-pass rubric injection.** *Substantially answered: yes,
inject.* DSPy `BootstrapFewShot` (Khattab 2024) and the *Rubrics as
Rewards* line both find producers given the rubric upfront converge
faster and to higher pass rates than producers shown the rubric only
after failure. The "writing to the test" concern is real but is
empirically dominated by the upfront-clarity gain. The mitigation in
the literature is *using a held-out rubric for evaluation* (so
training rubric ≠ eval rubric), not withholding the rubric from the
producer.

**Q4: Stagnation detection.** *Open with consensus heuristic.* The
field uses "no improvement for `patience=1` iterations" as the default
(early-stopping idiom). What the proposal labels "switch strategies"
is what *Reflexion* calls a *mode shift* — and the empirical evidence
is that mode shifts tend to *not* help unless the mode change is
externally grounded (e.g., switching from BM25 to a structured
extraction tool because a numeric criterion failed). Free-form mode
switches under same-critic guidance are the failure mode CRITIC names.

**Q5: Rubric reuse across tasks.** *Open and genuinely an empirical
question.* The Harvey workflow library findings suggest *deliverable-
type-level rubrics* (e.g., merger-agreement-v2 across many merger
agreements) are reusable; *task-level rubrics* (this specific
contract's specific concerns) are not. This matches the *Rubrics as
Rewards* finding that domain-level rubrics generalise better than
instance-level ones.

**Q6: Delegation under ReflexiveAgent.** *Open.* No published evidence
either way. Worth flagging: each delegation level multiplies the
cost ceiling, so the budget invariant must be per-root not per-agent.

---

## 6. Specific paper citations (year, venue, key finding)

- **Shinn et al., *Reflexion: Language Agents with Verbal
  Reinforcement Learning*, NeurIPS 2023.** Critique-revise loop with
  episodic memory; 91% HumanEval. **Caveat:** HumanEval has unit-test
  oracle — externally grounded verifier, not pure self-critique.

- **Madaan et al., *Self-Refine: Iterative Refinement with
  Self-Feedback*, NeurIPS 2023.** Same-LLM produce-feedback-refine
  pattern; gains on dialogue/sentiment, weak on math. The original
  paper itself acknowledges limited efficacy on Math Reasoning.

- **Gou et al., *CRITIC: LLMs Can Self-Correct with Tool-Interactive
  Critiquing*, ICLR 2024.** **Self-critique without external tools is
  significantly weaker; sometimes degrades performance.** External
  tools (search, calculator) are what make critique work. Single most
  relevant paper for this proposal — and it directly contradicts the
  "same-model RubricCritic is sufficient" implicit assumption.

- **Huang et al., *Large Language Models Cannot Self-Correct
  Reasoning Yet*, ICLR 2024 (arXiv:2310.01798).** Intrinsic self-
  correction *degrades* GSM8K, CommonSenseQA performance under fair
  conditions (no oracle stopping signal). Reasoning self-correction
  without external feedback is not a solved problem.

- **Stechly, Marquez, Kambhampati, *GPT-4 Doesn't Know It's Wrong*,
  arXiv:2310.12397, 2023.** On graph coloring, the *content* of self-
  critique is irrelevant to performance; gains attribute to top-k
  sampling, not critique. Direct empirical refutation of "the critic
  finds gaps" framing.

- **Valmeekam, Marquez, Kambhampati, *Can Large Language Models Really
  Improve by Self-Critiquing Their Own Plans?*, NeurIPS 2023 FMDM
  workshop.** Self-critique *diminishes* plan-generation performance
  on Blocksworld; LLM verifiers produce many false positives.
  Especially relevant since `PlanExecuteAgent` is one of the producers
  the proposal targets.

- **Kamoi, Zhang, Zhang, Han, Zhang, *When Can LLMs Actually Correct
  Their Own Mistakes? A Critical Survey*, TACL 2024.** Comprehensive
  meta-survey: no demonstrated success for prompted-LLM self-
  correction outside tasks "exceptionally suited" to it; fine-tuning
  + external feedback are the conditions that work.

- **Kumar et al., *SCoRe: Training Language Models to Self-Correct via
  RL*, arXiv:2409.12917, 2024 (DeepMind).** Two-stage RL is needed to
  overcome the failure mode where SFT-tuned models actively ignore
  correction prompts. Documents the "producer ignores REFLECTION"
  failure as a *predicted* outcome of pure prompting.

- **Singh et al., *Beyond Human Data: Scaling Self-Training with
  ReST^EM*, arXiv:2312.06585, 2024 (DeepMind).** EM-style self-
  training works on MATH/APPS *because the filter is a binary verifier
  on correctness* — externally grounded. Same theme.

- **Bai et al., *Constitutional AI: Harmlessness from AI Feedback*,
  Anthropic 2022 (arXiv:2212.08073).** Critique-revise works for
  alignment-style targets; **harmlessness improves monotonically with
  revisions, but pure helpfulness *decreases* with more revisions** —
  the proposal's "more iterations = better" intuition is contradicted
  for any helpfulness-shaped criterion.

- **Khattab et al., *DSPy: Compiling Declarative LM Calls into Self-
  Improving Pipelines*, ICLR 2024 + Opsahl-Ong et al. *Optimizing
  Instructions and Demonstrations for Multi-Stage LM Programs*
  (MIPROv2), 2024.** Validates that produce→score→revise loops
  generalise. **But:** MIPROv2 reports a 27-point improvement on
  HotPotQA (24% → 51%) only when the *score function is a held-out
  metric on a labelled set* — externally grounded.

- **Kim et al., *Prometheus 2*, EMNLP 2024.** Open-source 7B/8x7B
  evaluator, fine-grained rubrics, supports both absolute and pairwise
  modes. The proposal's `RubricCritic` should consider Prometheus-2
  rather than a same-family Claude judge as the default critic — it's
  cheaper, externally trained against human judgments, and cuts
  preference-leakage.

- **Zhuge et al., *Agent-as-a-Judge: Evaluate Agents with Agents*,
  arXiv:2410.10934, 2024 (Meta).** On DevAI, multi-step agent-judge
  outperforms single-shot LLM-judge and approaches human agreement —
  but only when the judge has *tool access* (file reads, tests,
  searches). Same lesson again: judges need to be more than a single
  LLM call.

- **Panickssery et al., *Self-Preference Bias in LLM-as-a-Judge*,
  arXiv:2410.21819, 2024.** Self-preference is a measurable, reliable
  effect; correlates with perplexity (model "recognises" its own
  family's distribution). Direct strike against same-family
  generator/judge.

- **Li et al., *Preference Leakage*, arXiv:2502.01534, 2025.** Empirical
  demonstration of bias in three relatedness regimes (same model,
  inheritance, same family). **Default critic of haiku-4-5 against
  haiku-4-5 producer = textbook same-model contamination.**

- **Park et al., *Justice or Prejudice: Quantifying Biases in
  LLM-as-a-Judge*, arXiv:2410.02736, 2024.** 12-bias taxonomy.
  **Style bias (0.76-0.92 effect) far exceeds position bias.** The
  proposal addresses none of these explicitly.

- **Stureborg et al., *Rating Roulette: Self-Inconsistency in
  LLM-As-A-Judge*, Findings of EMNLP 2025.** Test-retest agreement
  often < 0.7 even at temperature 0. The proposal's `temperature=0`
  mitigation is necessary but not sufficient.

- **Wei et al., *Rubrics as an Attack Surface: Stealthy Preference
  Drift in LLM Judges*, arXiv:2602.13576, 2026.** Rubric-Induced
  Preference Drift: small rubric edits compliant with benchmark
  validation can shift judge preferences. Direct strike against
  `RubricDeriver` regenerating rubrics across runs.

- **Dhuliawala et al., *Chain-of-Verification*, ACL Findings 2024
  (arXiv:2309.11495).** Decoupling verification questions from the
  draft (so the draft doesn't bias verification answers) measurably
  reduces hallucination. The proposal's `RubricCritic` should answer
  each criterion *without seeing the deliverable's reasoning chain*,
  only the final claim — the "factored verification" pattern.

- **Brown et al., *Large Language Monkeys: Scaling Inference Compute
  with Repeated Sampling*, arXiv:2407.21787, 2024.** Best-of-N with a
  verifier scales test-time compute as well as a much larger model —
  but only with a *separate* verifier.

- **Snell et al., *Scaling LLM Test-Time Compute Optimally*,
  arXiv:2408.03314, 2024 / ICLR 2025.** Compute-optimal frontier
  shows verifier-guided BoN beats sequential refinement at fixed
  compute on most tasks. **Sequential refinement (the proposal's
  shape) is not the compute-optimal pattern.** Parallel BoN with
  verifier-pick is.

- **Yu et al., *When AIs Judge AIs: The Rise of Agent-as-a-Judge
  Evaluation*, arXiv:2508.02994, 2025.** Multi-LLM judge frameworks
  (CourtEval, DEBATE) achieve closer-to-human consensus than single-
  judge frameworks. Legal-domain agreement gap: 64-68% LLM-vs-human
  vs. 72-75% inter-expert.

- **Kambhampati et al., *Position: LLMs Can't Plan*, ICML 2024.**
  LLMs verify plans poorly; verification false-positive rate alone
  defeats iterative correction. Argues for LLM-Modulo: external sound
  verifier in the loop.

- **Li et al., *AlphaCode*, *Science* 2022 (arXiv:2203.07814).**
  Filter-don't-refine pattern — million-sample BoN with executable
  test verifiers. The "best-of-iterations" intuition is directly from
  this work, but their verifier is the test suite, not another LLM.

---

## 7. Bottom-line recommendations

1. **Re-frame `RubricCritic` as a verifier hierarchy, not an LLM call.**
   Criteria with structural verifiers (regex, substring, schema match,
   citation lookup) skip LLM judging. Only criteria that *cannot* be
   structurally verified go to the LLM judge — and those are flagged
   as "lower-confidence judgment" in the trace. Without this, the
   proposal builds the configuration the literature most reliably says
   doesn't work.

2. **Default critic must be a different model family from the
   producer.** If producer is haiku-4-5, critic is gpt-5.4-mini or
   gemini-2.5-flash, or both with disagreement triggering escalation.
   The cost differential is small; the bias-mitigation gain is
   measurable per *Preference Leakage* 2025.

3. **Add test-retest measurement of the critic to the validation
   plan.** Before the seven `Loop-N` runs, include a `Critic-1`
   sanity run: 50 frozen (deliverable, rubric) pairs, 5 reruns each,
   measure Krippendorff's α. Block the launch on α ≥ 0.8.

4. **Reframe `max_iterations=3` as a *patience-bounded* early stop.**
   Stop when iteration N's `weighted_pass_rate` improvement over the
   prior best is < ε (e.g. 0.05) *and* the critic test-retest CI
   overlaps. Pay for the iteration only if the gain is bigger than
   the noise.

5. **Drop or heavily caveat `RubricDeriver` for v1.** The literature's
   verdict is unambiguous: model-derived rubrics under-cover the
   producer's blind spots. Ship v1 with hand-curated rubrics from
   exemplars (Harvey methodology already supplies these). Defer
   self-derivation to v2 with explicit human review of derived
   rubrics.

6. **Add a "criterion verifiable without LLM" coverage metric.** The
   proposal should report what fraction of rubric criteria the
   `RubricCritic` evaluated structurally vs. via LLM call. The
   literature would say below 30% structural coverage, the loop is
   running on judge-noise.

7. **Forbid same-family ReflexiveAgent stacking.** Invariant 6 should
   read "Composable, but only with critic-family separation between
   layers." Same-critic stacking compounds bias multiplicatively.

If those changes go in, the proposal is well-founded. If they don't,
the most likely outcome on the Harvey CoC benchmark is a 5-10 point
apparent lift that fails to replicate on a held-out 2nd benchmark
because the apparent lift is the same-family critic preferring
deliverables that look like its own prior outputs.
