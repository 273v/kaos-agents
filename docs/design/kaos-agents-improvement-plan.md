# kaos-agents — concrete improvement plan

**Status:** committed work order, 2026-05-06.

Synthesises the v2 ReflexiveAgent design with the underlying diagnosis
of the Harvey LAB CoC baseline and the existing kaos-agents technical
debt. This is not a wish list — every item lists what it does, where
it lives, what it deletes, and a rough day estimate.

The ordering principle: **highest leverage first**. Producer fixes
before retrieval fixes before architecture. Bug fixes that lift the
benchmark by 10+ points come before any 600-LoC subsystem.

---

## Tier 1 — Producer + retrieval fixes (1 week)

These have nothing to do with architecture. They are bugs and missed
configuration. They will move the Harvey CoC baseline more than any
new component would.

### 1.1 Fix output truncation [P0, 1 day]

**Problem.** The Harvey CoC deliverable
(`docs/benchmarks/harvey-coc-2026-05-06.json:23`) cuts off
mid-sentence at 15.7 KB. Multiple criteria explicitly cite the
truncation as the failure cause. Validation retry messages in the
benchmark log say: *"the model hit max_tokens before closing the
JSON. Try increasing max_tokens."*

**Fix.**
- Audit every `max_tokens` setting in the producer call path:
  - `kaos_agents/patterns/research.py` — RAG `_RESEARCH_REACT_INSTRUCTION`
    paths
  - `kaos_llm_core/programs/call.py` — `Call.__call__` default
  - `kaos_llm_core/programs/react.py` — ReAct loop max_tokens
  - `kaos_agents/agent.py` — `BaseAgent._simple_respond` and
    streaming handlers
- Add a unit test `tests/integration/test_long_deliverable.py` that
  produces a 50KB deliverable from a 200KB corpus and asserts the
  output is not truncated mid-sentence (last char is `.` or `\n`).
- Default `max_tokens` for the research-pattern's final-respond Call
  must be at least `model_context_window // 4` (Anthropic Sonnet 4.6
  = 200K → 50K output budget).
- Surface as `KAOS_AGENT_MAX_OUTPUT_TOKENS` settings field with
  per-pattern override.

**Acceptance.** Re-run Harvey CoC; deliverable_chars ≥ 30,000;
no criteria fail with reasoning citing truncation.

**Predicted lift:** 5-10 points on the benchmark by fixing this
alone.

### 1.2 Verify retrieval tools were actually invoked [P0, 1 day]

**Problem.** The Section 14.2 / Section 1.01 / Section 8.01(j) / $36.2M
failures are precisely the failure modes that
`kaos-retrieval-synonyms` (vocabulary expansion) and
`kaos-retrieval-hyde` (embedding-based) are designed to catch. They
auto-inject for RESEARCH-pattern agents
(`kaos_agents/retrieval_agent.py:42-46`). But the haiku producer
might never have invoked them.

**Fix.**
- Add a `harvey_coc_benchmark.py` trace assertion: when the corpus
  exceeds 100 chunks, at least one of `kaos-retrieval-synonyms` or
  `kaos-retrieval-hyde` is invoked at least once during the agent
  run.
- Add a `KAOS_AGENT_RETRIEVAL_TOOL_TRACE` setting that records all
  retrieval-tool invocations to the agent's trace tree.
- If the assertion fails on baseline: the bug is in the
  `_RESEARCH_REACT_INSTRUCTION` prompt — it doesn't push the agent
  to consider vocabulary-expansion. Update the instruction to say
  "If your first BM25 retrieval has weak hits (best score < 5.0),
  invoke `kaos-retrieval-synonyms` before answering."

**Acceptance.** On Harvey CoC baseline, ≥ 50% of question paths
invoke a non-BM25 retrieval tool.

### 1.3 Bump `_reflect_on_coverage` rounds [P1, 1 hour]

**Problem.** `kaos_agents/context/retrieval.py:279` does 2 reflection
rounds; the proposal already exists, just under-tuned.

**Fix.**
- Bump constant from 2 to 4 (or wire to a setting:
  `KAOS_AGENT_RETRIEVAL_REFLECTION_ROUNDS`).
- Re-measure on Harvey CoC. Free experiment.

**Acceptance.** No regression on multiformat baseline. Lift on Harvey
CoC.

### 1.4 Plumb `bm25_score_floor` through RAG [P1, half day]

**Problem.** From the prior-session follow-up: setting was added to
`KaosAgentSettings.bm25_score_floor` but never plumbed through to
`kaos-llm-core` RAG corpus.search().

**Fix.**
- `kaos_agents/patterns/research.py:_build_corpus_bm25` — pass
  `score_floor=settings.bm25_score_floor` to `corpus.search()`.
- Add live test in `tests/integration/test_research_live.py` that
  verifies low-score hits are pruned when floor > 0.
- Default 0.0 (no behavior change unless user opts in).

### 1.5 OCR confidence filter through verifier [P1, half day]

**Problem.** Prior-session follow-up. `KAOS_AGENT_OCR_MIN_CONFIDENCE`
plumbed into the data layer but the verifier doesn't filter on it.
Means low-OCR-quality citations can verify and pollute findings.

**Fix.**
- `kaos_llm_core/programs/rag.py` verifier path: read
  `passage.provenance.confidence` (already populated; from prior
  session N6 work) and reject citations below
  `settings.ocr_min_confidence`.
- Live test on a court-PDF fixture with known OCR garbage.

### 1.6 Phase 0 head-to-head experiment [P0, 1 day]

**Problem.** Need to settle the literature's prediction empirically
before any architecture commits. Per `reflexive-agent-v2.md` Phase 0.3.

**Fix.**
- Add `tests/benchmarks/phase_0_constant_cost.py` that runs Harvey CoC
  4 ways at $0.30/task target each, 3 reps each:
  - **A**: Sonnet single-pass, no loop
  - **B**: Haiku × Haiku, 3 iterations (same-model loop)
  - **C**: Haiku producer, Sonnet critic, 2 iterations (cross-model)
  - **D**: Haiku × 6 parallel, Sonnet picks best (BestOfN-with-verifier)
- Reuse existing `kaos_llm_core.programs.best_of_n.BestOfN` for D.
- Output `docs/benchmarks/phase-0-{date}.json` with full per-criterion
  + per-config + per-rep results.

**Acceptance.** Run completes in < 90 minutes total. Reports tell
us which Phase-1 build to do.

**Pre-registered prediction:** A ≥ D > C >> B.

---

## Tier 2 — Architecture aligned with literature (1-2 weeks, conditional on Phase 0)

The shape depends on Phase 0. The committed work is whichever branch
fires; the others are not built.

### 2.A — If "Strong-single" (A) wins

**Build.** ~100 LoC. Per-task-shape model preset routing.

- `kaos_agents/presets.py` (new file): `PRESETS = {"ma-review": Sonnet,
  "qa": Haiku, "drafting": Opus, ...}`.
- CLI: `kaos-agent chat --preset ma-review`.
- The `ProviderConfig.BALANCED` machinery
  (`kaos_agents/providers.py:56`) already supports per-role models;
  this is a routing table on top.

**Don't build.** No verifier hierarchy, no loop runner, no rubric
infrastructure. Phase 0 said the loop doesn't add value — building
it would be ignoring the data.

### 2.B — If "BestOfN-with-verifier" (D) wins

**Build.** ~450 LoC. Compute-optimal shape per Snell 2024 / Brown 2024.

| Component | Path | LoC |
|---|---|---|
| `BestOfNRunner` | `kaos_agents/best_of_n_runner.py` (new) | ~250 |
| `VerifierHierarchy` (see Tier 3.1) | `kaos_agents/verifiers/hierarchy.py` (new) | ~200 |

**Shape.**
```python
class BestOfNRunner:
    """Run agent N times in parallel; pick best by verifier."""
    
    async def run(self, task: str, *, session_id: str) -> AgentResponse:
        # Diverse temps for sample diversity
        samples = await asyncio.gather(*[
            self._producer.turn(task, session_id=f"{session_id}_s{i}", 
                                temperature=0.7 + i * 0.05)
            for i in range(self._n_samples)
        ])
        scores = await asyncio.gather(*[
            self._verifier(task, s.text) for s in samples
        ])
        return samples[max(range(len(scores)), key=lambda i: scores[i])]
```

### 2.C — If "Cross-model loop" (C) wins

**Build.** ~600 LoC. ReflexiveRunner per `reflexive-agent-v2.md`.

| Component | Path | LoC |
|---|---|---|
| `ReflexiveRunner` (Runner-level, not Agent subclass) | `kaos_agents/reflexive_runner.py` (new) | ~250 |
| `VerifierHierarchy` | `kaos_agents/verifiers/hierarchy.py` (new) | ~200 |
| `Rubric` / `RubricCriterion` first-class types | `kaos_agents/rubric.py` (new, see Tier 3.4) | ~80 |
| `EarlyStopPolicy` (patience-bounded) | `kaos_agents/reflexive/early_stop.py` (new) | ~60 |
| `StagnationDetector` (criterion-set equality) | `kaos_agents/reflexive/stagnation.py` (new) | ~40 |

Key invariants — all asserted at construction:
- `critic.model_family != producer.model_family`
- Critic prompt = `(rubric, deliverable)` only — tested in
  `tests/unit/test_critic_isolation.py`
- Prompt caching enabled on producer system prompt + corpus
- After iteration 1, only failed criteria re-judged

### 2.D — If "Same-model loop" (B) wins (literature is wrong on our task)

**Build.** Full v2 ReflexiveRunner per `reflexive-agent-v2.md`. ~800 LoC.

But: don't ship as default until reproduced on 2 additional benchmarks
(multiformat, cross-doc).

---

## Tier 3 — Verifier infrastructure (1-2 weeks, regardless of Phase 0)

These are the load-bearing components per the academic review. They
ship in some form regardless of which Phase-0 branch wins, because
they enable **structural verification before LLM calls**.

### 3.1 VerifierHierarchy [3 days]

**Add.** `kaos_agents/verifiers/hierarchy.py`.

```python
class VerifierResult(NamedTuple):
    passed: bool
    confidence: float
    level: VerifierLevel  # STRUCTURAL | GROUNDED | LLM
    reasoning: str
    cost_usd: float

class StructuralVerifier(Protocol):
    """Sub-millisecond, zero-LLM. Regex / substring / schema match."""
    def verify(self, criterion: RubricCriterion, deliverable: str) -> VerifierResult | None: ...

class GroundedVerifier(Protocol):
    """Microsecond, zero-LLM. Substring vs source spans."""
    # Reuses kaos-citations substring verifier (already in codebase).

class LLMVerifier:
    """Last resort. ~$0.001-$0.005/criterion. Cross-family enforced."""

class VerifierHierarchy:
    """Cascading per-criterion verification.
    
    Each criterion declares its expected verifier class. If a structural
    or grounded verifier resolves the criterion, no LLM call is made.
    
    Reports % structural coverage as a benchmark metric.
    """
```

**Per-criterion-class verifiers** (~3 days):
- `NumericVerifier` — regex extraction + tolerance comparison
  ("$36.2M ± 1%"). Source: Dhuliawala 2024 Chain-of-Verification.
- `CitationVerifier` — substring match against source spans via
  `kaos_citations.verify` (already exists).
- `CoverageVerifier` — enumerate ground-truth set from corpus, ask
  critic to *deny* coverage (asymmetric).
- `SchemaVerifier` — JSON schema match for structured outputs.

**Acceptance.**
- ≥ 30% of Harvey CoC rubric criteria evaluable structurally.
- Trace shows per-criterion verifier-level used.

### 3.2 Critic-context isolation [1 day]

**Add.** `kaos_agents/verifiers/sandbox.py`.

A wrapper that takes a critic call's context and strips:
- Producer's reasoning chain
- Producer's intermediate tool calls
- REFLECTION items written by the producer
- Any state from the producer's session

Critic sees only `(rubric, final_deliverable_text)`. Tested with
fixture deliverables containing producer-trace markers; assert
markers don't appear in critic prompt.

### 3.3 Cross-family critic enforcement [half day]

**Add.** `kaos_agents/verifiers/family_check.py`.

```python
def assert_critic_family_separation(producer_model: str, critic_model: str) -> None:
    """Hard-fail if critic and producer share model family.
    
    Raises ConfigurationError with explanation. Per academic review §2.3:
    same-family critics inflate apparent pass rates via self-preference bias
    (Panickssery 2024, Li 2025 Preference Leakage).
    """
```

Mapping table: `claude-*` family, `gpt-*` family, `gemini-*` family,
`grok-*` family. Fail fast at `__init__`.

### 3.4 First-class `Rubric` types [1 day]

**Promote** the implicit shape from `rubric_judge.py` and the Harvey
LAB benchmark fixture into a real type in `kaos_agents/rubric.py`.

```python
@dataclass(frozen=True, slots=True)
class RubricCriterion:
    id: str
    description: str
    match: str
    weight: float = 1.0
    expected_verifier: VerifierClass | None = None  # NumericVerifier, CitationVerifier, ...
    
@dataclass(frozen=True, slots=True)
class Rubric:
    criteria: tuple[RubricCriterion, ...]
    
    @classmethod
    def from_harvey_task_json(cls, path: Path) -> Rubric: ...
    
    @classmethod
    def from_extraction_recipe(cls, recipe_name: str) -> Rubric: ...
```

Replaces ad-hoc dict shape currently used in
`harvey_coc_benchmark.py:_judge_all`.

### 3.5 Continuous (pos+neg)/total_pos scoring [half day]

**Change.** Replace v1's `min_pass_rate` all-pass default with the
BigLaw Bench formula:

```python
def biglaw_score(verdicts: tuple[CriterionVerdict, ...]) -> float:
    """Harvey BigLaw Bench scoring: (pos + neg) / total_pos.
    
    pos = sum of weights of passed criteria
    neg = -1 × sum of weights of HALLUCINATED criteria  
    total_pos = sum of all positive-weight criteria
    
    Continuous, range [-1.0, 1.0]. Hallucinations get negative scores.
    """
```

Per production review §4. Requires `RubricCriterion.weight` to support
positive (must-have) and negative (penalty for hallucination) weights.

### 3.6 Move `rubric_judge` from `benchmarks/` to `verifiers/` [half day, refactor]

**Change.** Currently at
`kaos_agents/benchmarks/rubric_judge.py`. It's not benchmark code —
it's a verifier. Move to `kaos_agents/verifiers/llm_verifier.py`.
Update imports in `harvey_coc_benchmark.py`.

---

## Tier 4 — Cost optimization (1 week, after architecture)

Per production review §5: prompt caching + incremental re-judging
are non-negotiable for the proposed cost ceiling.

### 4.1 Prompt caching wiring [2 days]

**Add.** Cache markers on:
- Producer system prompt
- Producer's corpus chunks (the dominant token cost)
- Critic system prompt (`_RUBRIC_SYSTEM` in
  `kaos_agents/verifiers/llm_verifier.py`)

`kaos-llm-client` already supports Anthropic prompt caching (added in
`build-with-claude/prompt-caching` GA April 2026). Wire through
`ProviderConfig` so callers can enable per-role.

Test: same task with caching off vs on; assert cached run cost is
≤ 30% of uncached run cost (literature says 5-10× reduction).

### 4.2 Incremental re-judging [1 day]

**Add.** `IncrementalCritic` wrapping `VerifierHierarchy`.

```python
class IncrementalCritic:
    """Cache verdicts across iterations. Only re-judge failed criteria."""
    
    async def __call__(self, deliverable: str, rubric: Rubric, 
                       prior_verdict: RubricVerdict | None = None) -> RubricVerdict:
        if prior_verdict is None:
            return await self._verifier.verify_all(deliverable, rubric)
        # Re-judge only criteria that previously failed
        to_judge = tuple(c for c in rubric.criteria 
                         if prior_verdict.criterion_passed(c.id) is False)
        new_verdicts = await self._verifier.verify_all(deliverable, 
                                                        Rubric(criteria=to_judge))
        return prior_verdict.merge(new_verdicts)
```

Cost: `O(failed_criteria)` per iteration instead of `O(|criteria|)`.

### 4.3 Patience-bounded early stop [half day]

**Add.** `kaos_agents/reflexive/early_stop.py`.

```python
class PatienceEarlyStop:
    """Stop when iteration gain < epsilon AND CI overlaps."""
    epsilon: float = 0.05
    patience: int = 1  # iterations without improvement before stopping
    
    def should_stop(self, history: list[float]) -> bool: ...
```

Replaces fixed `max_iterations=3`.

---

## Tier 5 — Benchmark + validation infrastructure (1 week)

### 5.1 Test-retest reliability gate [2 days]

**Add.** `tests/benchmarks/critic_reliability.py`.

50 frozen `(deliverable, rubric)` pairs. Each scored 5×. Compute
Krippendorff's α. Block CI on α < 0.8 per academic review §3.2.

Pairs sourced from existing `harvey-coc-2026-05-06.json` deliverable
+ Harvey LAB rubric.

### 5.2 Inter-judge agreement test [1 day]

**Add.** Same fixture as 5.1. Score with `claude-haiku-4-5`,
`gemini-2.5-flash`, `gpt-5.4-mini`. Report Spearman; fail if < 0.5.

### 5.3 Phase-0 + phase-2-v2 benchmarks [2 days]

**Add.** `tests/benchmarks/reflexive_validation.py` running the full
A/B/C/D × baseline-vs-v2 grid on Harvey CoC + multiformat + cross-doc.
~6 hours of compute; gated on `--include-live --include-network`.

### 5.4 Trace metrics [1 day]

**Add to existing trace tree:**
- `structural_coverage_pct`: fraction of criteria evaluated
  structurally vs LLM
- `judge_confidence_distribution`: histogram of per-criterion confidence
- `iteration_gain_curve`: weighted_pass_rate per iteration
- `same_family_warning`: flag set if family check disabled (only for
  research, never for production)

---

## What gets REMOVED

| What | Where | Why |
|---|---|---|
| `RubricDeriver` plan | (was in v1, never built) | Wei 2026 names this as anti-pattern. Harvey doesn't do it. |
| Same-family critic option | API surface of `ReflexiveRunner` | Hard-invariant: don't expose what we don't want users to do. |
| Recursive `ReflexiveRunner` stacking | `__init__` validation | Compounds preference bias multiplicatively. |
| `adaptive_retrieve()` deprecated path | `kaos_agents/context/retrieval.py:adaptive_retrieve` | CLAUDE.md says it's deprecated and worse than plain BM25. Verify it's not in any non-test import path; remove if so. |
| `min_pass_rate=1.0` all-pass default | `RubricCritic.__init__` | Wrong on a 50-criterion rubric with score drift. Continuous scoring instead. |
| Fixed `max_iterations=3` default | reflexive runner | Patience-bounded early stop is the right shape. |
| `ProviderConfig` presets that route same-family | `kaos_agents/providers.py` | Audit existing `FAST`/`BALANCED`/`STRONG` — must not default to same family critic + producer. |
| `evaluate_semantic` in `planning/evaluate.py` when called with same-family | (case-by-case) | Same anti-pattern at the planning layer. Audit; flip to cross-family or to structural-only. |

---

## What gets CHANGED (refactors)

| Refactor | Current | Target | Rationale |
|---|---|---|---|
| `rubric_judge.py` location | `benchmarks/` | `verifiers/llm_verifier.py` | It's not a benchmark. |
| Reflection at `BaseAgent` level | mixed in `BaseAgent.run()` paths | Runner-level wrapper (`ReflexiveRunner`, `BestOfNRunner`) | Production review §1: layering. |
| Producer prompt rubric injection | Not done today | Producer sees rubric upfront on first pass | Production review §7: question-specific rubrics in prompt outperform alternatives. |
| `_reflect_on_coverage` | Retrieval-level only | Promote to general-purpose `verify_coverage(corpus, query, hits)` usable at deliverable level | Existing pattern, broader application. |
| `kaos-extract-corpus` recipe schemas | No `verifier` per column | Each `ColumnSpec` declares its expected verifier (numeric/citation/schema) | Lets the extraction recipes plug into VerifierHierarchy directly. |
| MCP `kaos-agent-rubric-critique` tool | Doesn't exist | Add, exposes RubricCritic as a callable tool | Lets agents that aren't wrapped in ReflexiveRunner still invoke verification mid-run via the existing REPLAN path (skeptic build #3). |

---

## Sequencing — what I'd do tomorrow

**Week 1:**
- Mon: 1.1 (output truncation) + start 1.2 (retrieval trace assertion)
- Tue: 1.3 (reflection rounds), 1.4 (bm25 floor), 1.5 (OCR confidence)
- Wed: 1.6 (Phase 0 head-to-head benchmark code)
- Thu-Fri: Run Phase 0; analyze; decide A/B/C/D branch

**Week 2:**
- Whichever Phase-1 branch fired (most likely D — BestOfNRunner)
- Plus 3.1 (VerifierHierarchy core)

**Week 3:**
- 3.2 (critic isolation), 3.3 (family check), 3.4 (Rubric types),
  3.5 (continuous scoring), 3.6 (move rubric_judge)
- 4.1 (prompt caching)

**Week 4:**
- 4.2 (incremental re-judging), 4.3 (early stop)
- 5.1, 5.2 (reliability gates)
- 5.3 (full validation grid)
- 5.4 (trace metrics)

**Week 5:**
- Documentation, migration guides
- Final acceptance gate per `reflexive-agent-v2.md` Tier B

---

## Out of scope (deferred to v3)

- Process Reward Models at inference (AgentPRM)
- Fine-tuned domain critics (Prometheus 2 swap)
- Verifiable reward training (RLVR)
- Attorney-graded ground truth labelling
- `RubricDeriver` with mandatory human review

These are real opportunities but each is its own multi-week
workstream. v2 ships without them.

---

## Risk register

| Risk | Likelihood | Mitigation |
|---|---|---|
| Phase 0 says A wins; team wants to build the loop anyway | Med | Pre-registered prediction in v2 doc; tier 1.6 commits to Phase 0 results in writing. |
| Output truncation fix doesn't lift baseline as much as expected | Low | 1.1 is well-grounded in benchmark logs. If lift < 3 points, look for second truncation site. |
| BestOfNRunner is right shape but Sonnet critic is too expensive at scale | Med | Falls back to Haiku critic at lower-confidence-flag — degraded mode. Document the tradeoff. |
| VerifierHierarchy structural coverage is < 30% on Harvey CoC | Med | Expand `NumericVerifier` and `CitationVerifier` first; fall back to LLM only on truly free-form criteria (which are minority on Harvey-shape rubrics). |
| Test-retest reliability fails at α < 0.8 | Med | Iterate critic prompt; try Prometheus-2 as critic; ultimately accept that LLM-judge is best-effort and downgrade weight on subjective criteria. |
| Cross-family critic adds latency that breaks UX | Low | Critic concurrency via `asyncio.gather` keeps wall-clock bounded by single criterion's judge call (~1s). |

---

## Acceptance — what "shipped" means

**Tier 1 (week 1):**
- Harvey CoC pooled pass rate ≥ 35% on the post-fix run (baseline 18.2%
  → fix-only ≥ 35%, before any Phase 1 architecture).
- All retrieval tools demonstrably invoked when the corpus exceeds 100
  chunks.
- Phase 0 results published.

**Tier 2 (week 2):**
- Whichever branch fired ships with passing live integration tests.
- The unfired branches are not built.

**Tier 3 (week 3):**
- VerifierHierarchy ships with ≥ 30% structural coverage on Harvey CoC.
- All cross-family invariants enforced and tested.
- `Rubric` is a first-class type used by `harvey_coc_benchmark.py` and
  `cross_doc_benchmark.py`.

**Tier 4 (week 4):**
- Cost per task with caching ≤ $0.50 on Harvey CoC.
- Incremental re-judging gives ≥ 2× cost reduction on iterations 2-3.

**Tier 5 (week 5):**
- Test-retest α ≥ 0.8 on the critic.
- Full v2 acceptance gate from `reflexive-agent-v2.md` Tier B passes.
- Documentation updated; migration guide for users on `BaseAgent.run()`
  → `Runner.run_with_*()`.

If any acceptance gate misses, that branch doesn't ship as default —
it ships as opt-in until the gate passes.
