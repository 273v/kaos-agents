# ReflexiveAgent v2 — verify the diagnosis, then build the right shape

**Status:** v2 plan, 2026-05-06. Synthesises three independent reviews of
v1: academic / literature, production / industry, principled-skeptic.
Each is in `reflexive-agent-v1-review-{academic,production,skeptic}.md`.

## What changed from v1

v1 proposed `ReflexiveAgent` as a `BaseAgent` subclass that wrapped an
existing producer agent, ran a same-model critic against a rubric, and
iterated based on gap feedback. It motivated this from the 18.2% pass
rate on the Harvey LAB CoC benchmark.

**All three reviewers independently rejected major load-bearing claims of v1.** The
verdicts converge so tightly that the synthesis is not "incorporate
feedback" — it is "rewrite the proposal." Specifically:

1. **The diagnosis was wrong.** The 18.2% baseline is not a metacognition
   gap. The deliverable was *truncated mid-sentence* in the Technology
   License section (multiple criteria explicitly cite the truncation
   in their reasoning); the remaining failures are pure retrieval misses
   (Section 14.2, $36.2M concentration, Section 8.01(j)). No outer loop
   would fix any of those.
2. **Same-family critic is the textbook anti-pattern.** Four post-2023
   papers converge: Huang ICLR 2024 ("LLMs Cannot Self-Correct Reasoning
   Yet" — performance *degrades*), Stechly 2023 ("GPT-4 Doesn't Know
   It's Wrong" — critique content irrelevant), Valmeekam NeurIPS 2023
   (self-critique *diminishes* plan generation), Kamoi TACL 2024
   (no demonstrated success outside externally-grounded tasks). Plus
   Self-Preference Bias 2024 + Preference Leakage 2025 on style bias
   between same-family models. v1's default of `claude-haiku-4-5`
   critic on `claude-haiku-4-5` producer is exactly the configuration
   the literature most strongly contradicts.
3. **Sequential refinement is not compute-optimal.** Snell 2024 + Brown
   2024 ("Large Language Monkeys") show **best-of-N + separate verifier
   beats sequential refinement at fixed compute** on most tasks. v1's
   shape is the wrong one.
4. **`RubricDeriver` is a named anti-pattern.** Wei 2026 calls this
   "Rubric-Induced Preference Drift"; Reflexion's documented failure is
   exactly this ("hallucinated task spec, then graded against itself").
   Harvey explicitly does *not* auto-derive — bespoke attorney-authored
   rubrics per practice area.
5. **`BaseAgent` subclass is the wrong layer.** Every shipping framework
   (LangGraph, AutoGen, OpenAI Agents SDK, Mastra, Pydantic AI, Devin)
   puts reflection at the Runner / graph layer. v1's invariant 6
   (`ReflexiveAgent(producer=ReflexiveAgent(...))` works but useless)
   is the type system telling us the abstraction is wrong.
6. **Cost ceiling math was wrong.** v1 stated $0.30/task at 3 iterations.
   The measured baseline alone is $0.27 (mostly judge cost). With three
   loop iterations, floor is ~$0.72/task on the toy benchmark and
   $3-$5/task at realistic scale. Prompt caching and incremental
   re-judging were not in v1; they are non-negotiable.

What v1 got right (kept in v2): best-of-iterations safety net,
producer-cloning per iteration, structured rubrics > holistic scoring
(Prometheus-2 confirms), structural pre-checks before LLM judgment,
multi-redundant termination, REFLECTION memory channel matching the
canonical Reflexion paper, criterion-level (not holistic) feedback.

## The three-phase plan

The synthesis principle: **verify the diagnosis before adding
architecture, then build the smallest shape that the empirical evidence
actually supports.**

### Phase 0 — Verify the diagnosis (~2 days, must run first)

Three checks. None require new architecture. All three must complete
before any new code lands.

#### 0.1 Fix the output truncation

The Harvey CoC deliverable (`docs/benchmarks/harvey-coc-2026-05-06.json:23`)
ends mid-sentence at 15.7 KB. Multiple criteria explicitly fail because
of the truncation. This is a `max_tokens` ceiling or a streaming-cap
bug, not a loop bug.

- Audit `max_tokens` in the producer call path (
  `kaos_agents/patterns/research.py`,
  `kaos_llm_core/programs/call.py`, `kaos_llm_core/programs/react.py`).
- Add a unit test: a 50KB-target deliverable round-trips intact.
- Re-run Harvey CoC after the fix. Expected lift from this alone:
  several percentage points, no architecture work.

#### 0.2 Verify retrieval tooling actually fired

The producer's failures on Section 14.2, Section 1.01, Section 8.01(j),
$36.2M concentration are the exact failure modes that
`kaos-retrieval-synonyms` and `kaos-retrieval-hyde` are designed to
catch. The `RetrievalAgent` is auto-injected for RESEARCH-pattern
agents (`kaos_agents/retrieval_agent.py:42-46`). Did the haiku producer
actually delegate to it?

- Add a trace assertion to `harvey_coc_benchmark.py`: when the corpus
  exceeds N chunks, at least one of `kaos-retrieval-synonyms` or
  `kaos-retrieval-hyde` was invoked. If the assertion fails on the
  baseline run, the bug is producer-not-invoking-tools, not
  agent-not-self-grading.
- Bump `_reflect_on_coverage` rounds from 2 to 4 in
  `kaos_agents/context/retrieval.py:279`. Free experiment.
- Re-run Harvey CoC. The thermostat already exists at retrieval level;
  this just turns its knob up.

#### 0.3 The decisive constant-cost experiment

Before any new architecture, settle the literature's prediction
empirically on our actual benchmark. Hold inference cost constant at
~$0.30/task across four configurations:

| Config | Producer | Critic | Iterations | Total cost target |
|---|---|---|---|---|
| **A. Strong-single** | `claude-sonnet-4-6` | none | 1 (no loop) | $0.30 |
| **B. Same-model loop** | `claude-haiku-4-5` | `claude-haiku-4-5` | 3 | $0.30 |
| **C. Cross-model loop** | `claude-haiku-4-5` | `claude-sonnet-4-6` | 2 | $0.30 |
| **D. BestOfN-with-verifier** | `claude-haiku-4-5` × 6 parallel samples | `claude-sonnet-4-6` picks best | 1 | $0.30 |

Run each configuration **3 times** on the full Harvey CoC rubric (55
criteria, all-pass scoring) so we have variance bars. Use the
existing `BestOfN` from `kaos-llm-core/programs/best_of_n.py` — no new
code needed for D.

**Pre-registered prediction** (academic + skeptic both made this):
- A ≥ D > C >> B
- Specifically: B will be within 5 points of baseline (the same-model
  loop literature predicts no lift), A will gain 15-30 points (model
  scaling literature), D will tie or beat A (BoN-with-verifier
  literature), C will land between A and B.

**What each outcome means:**
- If A wins outright: ship strong-producer preset, do not build the
  loop. Save 700+ LoC. Reframe Reflexive work as "model presets per
  task type."
- If D wins outright: build `BestOfNRunner`, not `ReflexiveAgent`.
  This is a 200-LoC composition, not a 700-LoC subsystem.
- If C wins outright: build the cross-model loop, ~400 LoC, with
  hard invariant that critic family ≠ producer family.
- If B wins outright (literature is wrong on this benchmark):
  build full v2 below.

### Phase 1 — Build only what Phase 0 justifies (~3-7 days)

Phase 1 is **conditional on Phase 0 results**. The v2 doc commits to
each branch ahead of time so we cannot rationalise after the fact.

#### Phase 1A: if Strong-single (A) wins

No new architecture. Add `kaos-agent --preset task-type` plumbing that
maps task-shape signals (corpus size, instruction length, deliverable
type) to model tiers:

```
task = "M&A contract review across deal room"  → producer = sonnet-4-6
task = "What is the filing fee?"               → producer = haiku-4-5
task = "Write the brief"                       → producer = opus-4-7
```

This is a routing table + provider config — uses the existing
`ProviderConfig.BALANCED`/`STRONG` machinery. ~100 LoC.

#### Phase 1B: if BestOfN-with-verifier (D) wins

Build `BestOfNRunner` (Runner-level, not Agent-level):

```python
class BestOfNRunner:
    """Run an agent N times in parallel, pick the best by an external verifier.
    
    This is the literature's compute-optimal shape (Snell 2024, Brown 2024,
    AlphaCode 2022). No sequential refinement, no self-critique. Sample
    diversity comes from temperature variation; quality comes from a
    cross-family verifier.
    """
    def __init__(self,
                 producer: BaseAgent,
                 verifier: VerifierHierarchy,
                 n_samples: int = 6,
                 cross_family_invariant: bool = True): ...
    
    async def run(self, task: str, *, session_id: str) -> AgentResponse:
        # asyncio.gather N samples with diverse temperatures
        # verifier scores each (verifier hierarchy: structural → grounded → LLM)
        # return highest-scoring sample
```

Reuses `kaos-llm-core/programs/best_of_n.py` shape, lifted to the agent
layer. ~250 LoC of glue + 200 LoC of `VerifierHierarchy`. Total ~450 LoC.

#### Phase 1C: if Cross-model loop (C) wins

Build `ReflexiveRunner` (Runner-level, NOT BaseAgent subclass):

```python
class ReflexiveRunner:
    """Iterate producer with cross-family verifier feedback.
    
    Hard invariants:
    - critic.model_family != producer.model_family (assert at __init__)
    - critic sees only (rubric, final_deliverable), NOT producer's trace
    - structural verifiers run before any LLM critic call
    - prompt caching enabled for both producer and critic
    - patience-bounded early stop (not fixed max_iterations)
    """
    def __init__(self,
                 producer: BaseAgent,
                 rubric: Rubric,                    # NOT a Deriver — explicit only
                 verifier: VerifierHierarchy,      # not just an LLM judge
                 max_iterations: int = 3,
                 patience: int = 1,                 # stop after N iter w/o improvement
                 min_iteration_gain: float = 0.05): ...
```

~600 LoC including verifier hierarchy. Detailed below.

### Phase 2 — Verifier-grounded improvements (only after Phase 1 lands)

The literature's actual recipe for what works at inference time:

**`VerifierHierarchy`** — a stack of cheap-to-expensive checks per
criterion. Each criterion declares the verifier it uses; criteria
without a structural verifier flag as "lower-confidence verdict" in
the trace.

```python
class VerifierHierarchy:
    """Per-criterion verification cascade.
    
    Levels (cheapest first):
    1. STRUCTURAL — regex, substring match, schema match, citation lookup,
       numeric extraction. Sub-millisecond, zero LLM cost.
    2. GROUNDED — substring/quote verification against source spans
       (kaos-citations substring verifier, BM25 grounding test).
       Microseconds, zero LLM cost.
    3. LLM — last-resort rubric_judge call for criteria not amenable
       to structural verification. Cost: ~$0.001-$0.005 per criterion.
    
    Reported metric: % of criteria evaluated structurally vs. via LLM.
    Per academic review: below 30% structural, the loop is judge-noise.
    """
```

Per-criterion-class verifiers:
- **Numeric criteria** ("$36.2M revenue concentration"): regex
  extraction + structural compare against source. Chain-of-Verification
  pattern (Dhuliawala ACL 2024) — verification answer independent of
  draft.
- **Citation/quote criteria** ("Section 14.2 anti-assignment"):
  substring match against source spans via `kaos-citations` substring
  verifier (already in codebase).
- **Coverage criteria** ("all change-of-control clauses identified"):
  enumerate from corpus first, then ask critic to *deny* coverage
  (asymmetric verification — academic review §3.1).
- **Free-form criteria**: LLM judge as fallback only. Trace flags
  these as low-confidence.

## Architecture changes from v1

### Drop entirely

| v1 Component | Why dropped |
|---|---|
| `ReflexiveAgent(BaseAgent)` subclass | Wrong layer. Build `ReflexiveRunner` instead. Production-review §1; academic-review Q1. |
| `RubricDeriver` | Named anti-pattern (Wei 2026). Harvey doesn't do this. Production-review §2; academic-review §3.6. |
| Same-family default critic | Textbook anti-pattern (Huang 2024, Stechly 2023, Preference Leakage 2025). Hard-invariant: critic.family ≠ producer.family. |
| `min_pass_rate=1.0` (all-pass) default | Wrong on a 50-criterion rubric with score drift. Use BigLaw Bench's `(positive + negative) / total_positive` continuous scoring. Production-review §4. |
| Fixed `max_iterations=3` | Replace with patience-bounded early stop. Academic-review §3.4. |
| One-shot rubric reuse | Move per-criterion `confidence > 0.7` threshold for feedback eligibility. |

### Add (new invariants)

| Invariant | Source | Why |
|---|---|---|
| **Critic-family separation** | Academic §2.3, §4 | Hard-fail at `__init__` if critic.model_family == producer.model_family. Tested. |
| **Critic-context isolation** | Production §3 | Critic sees `(rubric, deliverable)` only — not producer's reasoning chain, intermediate state, or REFLECTION items. Tested. |
| **Test-retest reliability gate** | Academic §3.2 | Before launch, run 50 frozen `(deliverable, rubric)` pairs through the critic, 5× each. Block if Krippendorff's α < 0.8. |
| **Structural-verifier coverage floor** | Academic §3.1, §7 | Report % of criteria evaluated structurally. Below 30%, flag the run as "judge-noise dominated." |
| **Prompt caching** | Production §5 | Producer system prompt + corpus chunks must use Anthropic prompt caching. Cache write costs 25% extra; breakeven at 2 calls; loop is ≥2 calls by definition. |
| **Incremental re-judging** | Production §5 | After iteration 1, freeze passed criteria. Re-judge only `failed` set. `O(failed_criteria)` per iteration, not `O(\|criteria\|)`. |
| **Patience-bounded early stop** | Academic §3.4 | Stop when iteration `weighted_pass_rate` improvement over best-so-far < ε *and* CI overlaps. Don't pay LLM cost for noise. |
| **Stagnation by criterion-set equality** | Academic §2.4 | Detect by `set(failed_criterion_ids)`, not by reasoning text. Reasoning text is non-deterministic per Stureborg 2025. |
| **Asymmetric verification for coverage** | Academic §3.1 | For "all X identified" criteria, enumerate ground truth first, then ask critic to *deny* coverage. |

### Keep (kept from v1)

| v1 Component | Why kept |
|---|---|
| `Rubric` / `RubricCriterion` types | Right shape; matches Prometheus-2 / RaR literature. |
| Best-of-iterations selection (with caveats) | Right safety net (AlphaCode pattern); but selection key needs paired re-scoring across iterations to defeat score drift. Academic §4 invariant 2. |
| Producer-cloning per iteration | Correct (Self-Refine, DSPy Refine). Concurrency-safe. |
| Criterion-level feedback | Strictly dominates holistic (Madaan 2023, Prometheus-2). Default `GAP_LIST` was right. |
| REFLECTION memory channel | More faithful to original Reflexion than `langgraph-reflection`'s message-append shortcut. Production §6. |
| Trace-complete iteration | Required by Yu 2025. |
| Concurrent per-criterion judging via `asyncio.gather` | Right shape for the embarrassing-parallelism. |

## Final shape (if Phase 0 says build it)

```
ReflexiveRunner (Runner-level wrapper, ~600 LoC)
    │
    ├── producer: BaseAgent                        ← existing
    │
    ├── rubric: Rubric                             ← REQUIRED at construction
    │   └── (no derivation; if caller doesn't supply, fail closed)
    │
    ├── verifier: VerifierHierarchy                ← NEW
    │   ├── StructuralVerifier per criterion       ← regex/substring/schema/citation
    │   ├── GroundedVerifier                       ← substring vs source spans (kaos-citations)
    │   └── LLMVerifier                            ← rubric_judge fallback, last resort
    │       └── INVARIANT: model_family ≠ producer.family
    │
    ├── kernel: LoopRunner (existing, generic)
    │   ├── make_state                             ← (current_deliverable, history)
    │   ├── step                                   ← verifier + memory.append(REFLECTION) + producer.run
    │   ├── build_result                           ← best-of-iterations by paired-rescore
    │   └── on_step_error                          ← existing route() / circuit-break
    │
    ├── early-stop policy                          ← patience-bounded (NEW)
    │   └── stop when (improvement < ε) and (CI overlaps)
    │
    ├── re-judge policy                            ← incremental (NEW)
    │   └── after iter 1: freeze passed criteria, re-judge only failed
    │
    ├── critic-context isolator                    ← invariant (NEW, tested)
    │   └── critic prompt = (rubric, final_deliverable) ONLY
    │
    └── trace-complete output
        ├── per-criterion verdicts
        ├── per-iteration deliverables
        ├── structural-coverage % metric
        ├── critic test-retest CI
        └── stop reason (PATIENCE / BUDGET / FAILURE / COMPLETED)
```

## Cost model (revised)

For the Harvey CoC benchmark (8 documents, 55 criteria) at full
configuration:

| Item | Estimate (Sonnet producer + Haiku critic) |
|---|---|
| Producer (Sonnet, 30k input + 5k output, **prompt-cached**) | ~$0.18 across 3 iterations |
| Critic, iteration 1 (Haiku, 55 criteria × 1.5k tokens) | ~$0.08 |
| Critic, iter 2 (only failed criteria, ~30 expected) | ~$0.04 |
| Critic, iter 3 (only failed criteria, ~15 expected) | ~$0.02 |
| Structural verifier (zero LLM cost) | $0 |
| **Per-task total** | **$0.32** |
| Without prompt caching | ~$1.05 |
| Without incremental re-judging | ~$0.45 |
| Without both | ~$1.40 |

Per Phase 0's $0.30 budget, the system fits **only** with prompt caching
+ incremental re-judging. v1 had neither.

For real M&A scale (200 criteria, 100 contracts, 200k-token corpus):
~$3.40/task with both optimizations, $15-$30 without. The deal-room
ceiling is $60-120 per matter (per Harvey production pricing); fits.

## Validation plan (revised)

Two tiers. The first must pass before any of the second runs.

### Tier A — Critic reliability gate (mandatory pre-launch)

1. **Test-retest reliability**: 50 frozen `(deliverable, rubric)` pairs,
   each scored by the critic 5×. Krippendorff's α ≥ 0.8 required.
2. **Inter-judge agreement**: same 50 pairs scored by `claude-haiku-4-5`
   AND `gemini-2.5-flash` AND `gpt-5.4-mini`. Spearman ≥ 0.5 required.
   Below this, the LLM-critic stack is too unreliable to use as
   iteration signal.
3. **Calibration vs human gold** (when available): 20 deliverables on a
   Harvey-style frozen set with attorney-graded ground truth. Report
   exact-match agreement; expect 64-68% per Yu 2025 legal-domain
   numbers. Document, don't gate (we don't have attorney labels yet).

### Tier B — Outer-loop runs (only after Tier A passes)

The Phase 0 head-to-head, then re-run with the v2 architecture:

| Run | Configuration | What it tests |
|---|---|---|
| Phase-0-A | Strong-single, sonnet, no loop | Baseline ceiling without architecture |
| Phase-0-B | Same-model loop (haiku × haiku) | Literature prediction: no lift |
| Phase-0-C | Cross-model loop (haiku producer, sonnet critic) | Lift from cross-family critic alone |
| Phase-0-D | BestOfN with sonnet critic | Compute-optimal shape |
| **Phase-2-V2** | Full v2 ReflexiveRunner: cross-family critic + verifier hierarchy + structural pre-checks + incremental re-judging + patience-bounded stop | Does adding verifier hierarchy + structural-first beat plain cross-model loop? |
| Phase-2-V2-broad | Same on 2 additional benchmarks (multiformat, cross-doc) | Does it generalise beyond Harvey CoC, or is it benchmark-specific? |

**Acceptance gate for shipping v2 as default:**
- v2 must beat the best Phase-0 config by ≥ 5 points on Harvey CoC
- AND must beat the best Phase-0 config by ≥ 3 points on at least
  ONE of (multiformat, cross-doc)
- AND total cost-per-task must stay under $0.50 with prompt caching
- AND structural-verifier coverage must be ≥ 30% on the rubric

If any gate fails, ship the best Phase-0 config instead of v2.

## What we're NOT building (and why)

These were in v1 or implied; v2 explicitly rejects them.

| Not building | Reason |
|---|---|
| `RubricDeriver` | Wei 2026 names this. Harvey doesn't do it. Defer to v2.1 with mandatory human review. |
| Same-family critic option | Hard-invariant. Don't expose it in the API. |
| Recursive `ReflexiveRunner(producer=ReflexiveRunner(...))` | Compounds bias multiplicatively (academic §4, invariant 6 in v1 was wrong). Forbidden at construction. |
| Mode-switching strategies based on critic guidance | "Free-form mode switches under same-critic guidance is the failure mode CRITIC names" (academic Q4). Switches must be triggered by *grounded* signals (BM25 score, structural verifier failure), not by LLM critic suggestions. |
| Open-ended drafting / synthesis tasks | All-pass rubric scoring breaks down for tasks without clean criteria. Scope v2 to extraction / due-diligence / compliance — checklist-shaped tasks (skeptic §6). |

## Open questions for v3 (deferred, not punted)

1. **Process Reward Models (PRMs) at inference time.** AgentPRM (Feb
   2025) trains a step-level critic. We don't have the training
   infrastructure. Worth tracking; not v2.
2. **Attorney-graded ground truth.** Tier A.3 needs this. Sourcing it
   is its own work.
3. **Fine-tuned domain critics.** Prometheus-2 (EMNLP 2024) is
   open-source 7B/8x7B; could replace `claude-haiku-4-5` as the LLM
   verifier. Cheaper, externally trained against human judgments,
   defeats preference leakage. v2 leaves a hook for this.
4. **Verifiable reward training.** RLVR is the field's direction
   (DeepSeek R1, Tülu 3, Qwen-Agent). We are an inference-time system;
   this is for whoever ships kaos-agent training.

## Reading order for the reviews

If you're catching up:
- **Skeptic review** is the diagnosis correction. Read first. Sets
  up the Phase 0 experiment.
- **Academic review** is the literature anchor. Cite-rich; the source
  for each invariant.
- **Production review** is the architecture / cost / shipping reality.
  Source for `Runner` vs subclass, prompt caching, incremental
  re-judging.

## Decision

**Phase 0 starts now.** No new architecture lands until Phase 0 results
arrive. The v2 architecture above is the *option* that gets built only
in the C-wins or B-wins branches of Phase 0; A-wins or D-wins triggers
a different (smaller) build.

This is the honest synthesis. v1 underestimated how much of the lift
is upstream of the loop (model choice, retrieval, output budget) and
overestimated how much is downstream (self-critique architecture).

---

## Cross-reference table — where each review's points landed

| Review | Point | v2 disposition |
|---|---|---|
| Skeptic | Output truncation diagnosis | **Phase 0.1** (mandatory pre-work) |
| Skeptic | Stronger producer first | **Phase 0.3 config A** |
| Skeptic | One-shot critique tool, not subclass | Adopted: `ReflexiveRunner` not `ReflexiveAgent` |
| Skeptic | Cost ceiling math wrong | Adopted: revised cost model with caching + incremental |
| Skeptic | Decisive constant-cost experiment | Adopted as Phase 0.3 |
| Skeptic | "If 2 (same-model) wins, build proposal; else don't" | Adopted as Phase 0 branching |
| Academic | 4 papers on same-family critic failure | Hard-invariant: critic.family ≠ producer.family |
| Academic | Verifier hierarchy (structural before LLM) | Adopted as `VerifierHierarchy` core |
| Academic | Test-retest reliability gating | Adopted as Tier-A pre-launch gate |
| Academic | Patience-bounded stop, not fixed max_iter | Adopted |
| Academic | Drop `RubricDeriver` | Adopted |
| Academic | Asymmetric verification for coverage criteria | Adopted in `VerifierHierarchy` |
| Academic | Chain-of-Verification (decouple from draft) | Adopted via critic-context isolation |
| Academic | Best-of-iterations needs paired re-score | Adopted; iteration scores come from same critic call |
| Academic | Stagnation by criterion-set, not text | Adopted |
| Academic | Forbid same-family stacking | Adopted: invariant 6 v1 explicitly forbidden in v2 |
| Production | `BaseAgent` subclass is wrong layer | Adopted: `ReflexiveRunner` |
| Production | Drop `RubricDeriver` | Adopted (also flagged by academic) |
| Production | Critic-context isolation (Devin pattern) | Adopted as tested invariant |
| Production | Prompt caching mandatory | Adopted as cost invariant |
| Production | Incremental re-judging | Adopted (`O(failed_criteria)` per iteration) |
| Production | All-pass `min_pass_rate=1.0` is wrong default | Adopted: continuous (pos+neg)/total_pos |
| Production | Stagnation detector design (~20 LoC) | Adopted: criterion-set equality + diff |
| Production | Rubric-on-first-pass injection | Adopted: producer sees rubric upfront |
| Production | Cost reality: $0.32 with caching, $1.40 without | Adopted in cost model |
