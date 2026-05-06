# Iterative-findings agent pattern — implementation plan

Status: design accepted, implementation pending
Owner: kaos-agents
Date opened: 2026-05-06
Related: `docs/design/kaos-agents-improvement-plan.md`,
`docs/benchmarks/harvey-coc-pipeline-comparison-2026-05-06.md`,
`docs/benchmarks/harvey-mtd-chat-sonnet-2026-05-06.json`

## 0. TL;DR

Port the production-validated kelvin agent loop (3 years, 151 commits) into
the kaos stack as a new `findings` agent pattern, backed by four typed
primitives in kaos-llm-core (`Finding`, `FilterSegmentsSignature`,
`CollectFindings`, `InterpretFindingsSignature`) and one Program
(`IterativeFindings`) built on the existing `LoopRunner` kernel.

The shape:

```
loop:
  segments  = retrieve(query, corpus, top_k=50)
  findings += filter_segments(question, segments) -> List[Finding]
  synthesis = interpret(question, findings, prior_attempts)
  if synthesis.confidence >= 8 or no suggested_next_queries: break
  query = synthesis.suggested_next_queries[0]
```

This sits between `RAG` (one-shot retrieve+answer with citation verification)
and `chat` simple-respond (open-loop, no evidence layer). It is the right
shape for **long-form deliverables grounded in a corpus** — exactly the
work-type Harvey LAB benchmarks evaluate, and the work-type our existing
patterns under-serve.

## 1. Motivation

### 1.1 What kelvin did right

Two parallel code-archaeology passes through `../kelvin-nlp/` and
`../kelvin-agent/` confirmed a clean separation that the kaos stack lost:

* `kelvin/nlp/llm2/utilities/records.py::Finding` is the typed unit of
  "this segment is relevant because X" — `text + reason + start_pos +
  end_pos + id` (BLAKE2b content hash, used in citations).
* `kelvin/nlp/llm2/executors/FilterExecutor` is a batched LLM call that
  takes a list of segments and a question, and returns `(segment, relevant,
  reason)` for each.
* `kelvin/agent/actions/search/findings/collect_findings.py::CollectFindingsAction`
  composes retrieval + filter into a single step that returns
  `List[Finding]`.
* `kelvin/agent/actions/interpretation/interpret_last_action.py::InterpretLastResultAction`
  is the synthesis step. Its output schema is the load-bearing piece:

  ```python
  output_schema = (
      """{"interpretation": str, "score": int, "suggested_prompts": list[str],"""
      """ "title": str, "finding_id": list[str]}"""
  )
  ```

  - `interpretation` — the memo/answer
  - `score` — 1–10 self-rated confidence
  - `suggested_prompts` — what to ask next (drives gap-driven re-query)
  - `finding_id` — which Findings the synthesis used (audit trail)

* `kelvin/agent/actions/planning/input/retry_plan.py` adds the explicit
  guard `"DO NOT REPEAT THE PRIOR PLAN. Identify information needed to
  achieve the goal. Be more comprehensive than the prior plan."` —
  empirically necessary to force divergence on retry.

* Termination is bounded: `score <= 7` triggers one `RetryPlan` per turn,
  not unbounded recursion. (Threshold tuned 9 → 8 → 7 over the project's
  life; commit `b3c2afe`.)

### 1.2 The gap in kaos today

| Step | kelvin (3 years ago) | kaos-agents today |
|---|---|---|
| Segmentation w/ provenance | `kelvin/nlp/segments/*` byte-offset tuples | kaos-content `SectionChunker`, AST refs (better) |
| Multi-strategy retrieval | `KelvinSpan.rank_span_bm25/tfidf/similarity` | kaos-nlp-core BM25 + embeddings (better) |
| `Finding(text, reason, span, source, id)` | First-class dataclass | **Missing** — `Cited[T]`/`Claim` are output types, not intermediate-evidence types |
| Batched segment filter | `FilterExecutor` | **Missing** |
| `retrieve → filter → List[Finding]` | `CollectFindingsAction` | **Missing** — `RAG` retrieves then *answers*; nothing produces a `List[Finding]` |
| Synthesis with confidence + gaps | `InterpretLastResultAction` | **Missing** — `RefineDeliverable` synthesises but has no confidence/gap fields |
| Iterate-until-confident loop | `chat_agent.handle_input` w/ `RetryPlan` | **Missing** — every kaos pattern is 1–2 steps |

Concretely, the litigation-MTD bench at 2026-05-06 showed:

* `research`-pattern (RAG) → 8.8% (RAG produces short cited answers, not memos)
* `chat`-pattern (simple_respond, no evidence layer) → 85.3%
* Harvey CoC `hybrid_v2` (ad-hoc kelvin-shaped pipeline) → 61.8%

Chat-pattern works on litigation-MTD because the synthesis step is unconstrained
prose, but it has no evidence accumulation, no gap detection, and no iteration —
so it caps at "what fits in one good draft" and loses on cross-document checks
(C-042 ERP unreviewed, C-007 dual-definition consistency, etc.).

The iterative-findings pattern is what bridges these: kelvin-shaped evidence
discipline + synthesis-quality prose + gap-driven re-query.

## 2. Design

### 2.1 Layered, additive

Six layers, each independently shippable and testable. Layers 1–4 land in
kaos-content / kaos-llm-core (no agent runtime change). Layer 5 lands in
kaos-llm-core as a pure Program. Layer 6 wires the pattern in kaos-agents.

```
Layer 6: kaos-agents / patterns / findings.py            (pattern wrapper)
Layer 5: kaos-llm-core / programs / iterative_findings.py (LoopRunner-driven)
Layer 4: kaos-llm-core / signatures / interpret_findings.py
Layer 3: kaos-llm-core / programs / collect_findings.py
Layer 2: kaos-llm-core / signatures / filter_segments.py + Filter program
Layer 1: kaos-content / model / evidence.py              (Finding type)
```

### 2.2 Layer 1 — `Finding` type (kaos-content)

File: `kaos-content/kaos_content/model/evidence.py` (new).

```python
@dataclass(frozen=True, slots=True)
class Finding:
    """A relevant segment surfaced by the filter pass.

    A Finding is the output of step 2 (filter) and the input to step 3
    (synthesis). It is NOT a `Claim` — a Claim asserts a proposition;
    a Finding records "this segment is relevant to the question because…".
    """

    text: str
    """The exact source text of the segment."""

    reason: str
    """Filter-LLM's one-sentence justification for relevance."""

    span: Span
    """source_uri + char_span; reuses kaos_llm_core.signatures.grounding.Span."""

    id: str
    """Content-addressed id (BLAKE2b hex of text + span). Used in citations."""

    confidence: float
    """Filter-LLM's relevance confidence in [0.0, 1.0]."""

    block_ref: str | None = None
    """JSON-pointer block_ref into the source ContentDocument when AST-grounded."""

    metadata: Mapping[str, Any] = field(default_factory=dict)
    """Free-form: page, section_label, chunk_index, etc. — non-load-bearing."""

    @classmethod
    def from_segment(
        cls, *, text: str, reason: str, source_uri: str,
        char_start: int, char_end: int, confidence: float,
        block_ref: str | None = None, metadata: Mapping[str, Any] | None = None,
    ) -> Finding: ...

    def citation_marker(self) -> str:
        """Return the inline citation token, e.g. `(F:a3f1b2)`."""
        return f"(F:{self.id[:8]})"
```

Tests (`kaos-content/tests/unit/test_evidence.py`):
- construction with required fields
- content-hash determinism (same text+span → same id)
- `citation_marker` shape
- round-trip through `model_dump`/`model_validate`-style serialization

Live test: none required — pure data class.

QA gate: `ruff format && ruff check --fix && ty check && pytest`.

### 2.3 Layer 2 — segment-filter primitive (kaos-llm-core)

Files:
- `kaos-llm-core/kaos_llm_core/signatures/filter_segments.py` (new)
- `kaos-llm-core/kaos_llm_core/programs/filter.py` (new — thin Program over `Call`)

Signature:

```python
class SegmentInput(BaseModel):
    id: str
    text: str
    source_uri: str
    char_start: int
    char_end: int

class FilterDecision(BaseModel):
    segment_id: str
    relevant: bool
    reason: str = Field(description="One sentence. Empty if not relevant.")
    confidence: float = Field(ge=0.0, le=1.0)

class FilterSegmentsSignature(Signature):
    """For each segment, decide whether it is relevant to the question and explain why.

    Decision rules:
    1. A segment is relevant if it contains facts, definitions, dates,
       parties, financial terms, or operative legal language that bears
       directly on the question.
    2. Background context with no operative content is NOT relevant.
    3. The reason must cite the specific element (a date, a defined term,
       a section number, a quoted phrase) — not generic ("discusses X").
    4. Be precise: relevant=true requires confidence >= 0.6.
    """

    question: str = InputField(description="The question or task driving the filter.")
    segments: list[SegmentInput] = InputField(...)
    decisions: list[FilterDecision] = OutputField(
        description="One decision per segment, in input order."
    )
```

Program:

```python
class Filter(Program):
    """Batched segment filter with bounded concurrency."""

    def __init__(self, *, model: str, batch_size: int = 25, concurrency: int = 5) -> None:
        super().__init__()
        self.call = Call(FilterSegmentsSignature, model=model)
        self.batch_size = batch_size
        self.concurrency = concurrency

    async def forward(*, question: str, segments: list[SegmentInput]) -> FilterResult:
        """Fan out batches under a semaphore; merge decisions in input order."""
        ...
```

`FilterResult` is a frozen dataclass: `(decisions: tuple[FilterDecision, ...], usage: InvocationUsage)`.

Tests:
- Unit (FunctionClient): batched fan-out preserves input order.
- Live: 30 paragraphs from Harvey MTD complaint + question
  *"What allegations support fraudulent inducement?"* → assert at least
  4 decisions are `relevant=True` with non-empty reasons that mention
  Sousa / Jan 10 / payroll integration.

QA gate: same.

### 2.4 Layer 3 — `CollectFindings` program (kaos-llm-core)

File: `kaos-llm-core/kaos_llm_core/programs/collect_findings.py` (new).

```python
@dataclass(frozen=True, slots=True)
class CollectFindingsResult:
    findings: tuple[Finding, ...]
    decisions: tuple[FilterDecision, ...]   # full audit trail incl. rejected
    n_segments_seen: int
    usage: InvocationUsage


class CollectFindings(Program):
    """retrieve(query) -> filter(segments) -> List[Finding]."""

    def __init__(
        self,
        *,
        retriever: Retriever,
        filter_program: Filter,
        top_k: int = 50,
        min_confidence: float = 0.6,
    ) -> None:
        super().__init__()
        self.retriever = retriever
        self.filter = filter_program
        self.top_k = top_k
        self.min_confidence = min_confidence

    async def forward(*, question: str, corpus: Any) -> CollectFindingsResult:
        # 1. Retrieve top_k passages from the corpus (BM25 by default;
        #    callers can pass HybridRetriever for BM25 + embedding rerank).
        passages = await self.retriever.retrieve(question, top_k=self.top_k)

        # 2. Convert to SegmentInput (preserve span provenance).
        segments = [SegmentInput.from_passage(p) for p in passages]

        # 3. Filter — batched LLM call(s) under concurrency cap.
        result = await self.filter(question=question, segments=segments)

        # 4. Build Finding objects from positive decisions.
        findings = tuple(
            Finding.from_segment(
                text=seg.text,
                reason=dec.reason,
                source_uri=seg.source_uri,
                char_start=seg.char_start,
                char_end=seg.char_end,
                confidence=dec.confidence,
                block_ref=getattr(seg, "block_ref", None),
            )
            for seg, dec in zip(segments, result.decisions, strict=True)
            if dec.relevant and dec.confidence >= self.min_confidence
        )
        return CollectFindingsResult(...)
```

Notes:
- `top_k=50` — wider than RAG's 5. The filter pass is cheap (batched, low
  per-segment cost) and compresses recall→precision.
- The retriever is **injected**, not hardcoded. Live tests use BM25;
  production callers can swap to `HybridRetriever` (BM25 + embedding
  rerank) for cross-domain recall.
- Provenance flows end-to-end: `Passage.span → SegmentInput.char_start/end
  → Finding.span`. No flat-text re-search.

Tests:
- Unit (FunctionClient retriever + FunctionClient filter): exact span
  preservation; min_confidence threshold rejects low-confidence positives.
- Live: Harvey MTD complaint + question → 5–15 findings, each with
  non-empty reason + valid Span that round-trips against source text.

### 2.5 Layer 4 — `InterpretFindings` Signature + Call (kaos-llm-core)

File: `kaos-llm-core/kaos_llm_core/signatures/interpret_findings.py` (new).

```python
class PriorAttempt(BaseModel):
    query: str
    findings_added: int
    confidence: int
    open_questions: list[str]


class InterpretFindingsSignature(Signature):
    """Synthesise findings into an answer; report confidence and gaps.

    Mirrors kelvin's InterpretLastResultAction output shape, ported to a
    typed Signature so kaos-llm-core's structured-output codec handles
    validation without ad-hoc JSON parsing.

    Decision rules:
    1. The answer must address the user's question end-to-end. Cite each
       Finding it relies on by `(F:<id-prefix>)`. Do not cite Findings
       you didn't actually use.
    2. Confidence is 1–10:
       - 9–10: every claim in the answer is grounded in a cited Finding;
               no obvious gaps.
       - 7–8:  one or two minor unknowns, mostly answered.
       - 4–6:  major gaps remain; partial answer.
       - 1–3:  insufficient evidence to answer.
    3. `open_questions` is what's missing semantically (e.g. "the
       counterparty's revenue concentration"). `suggested_next_queries`
       is what to TYPE INTO BM25 next (3–10 word retrieval queries:
       "Voltan revenue percentage", "Crestline ERP termination").
       Both are required when confidence < 9.
    4. Never repeat a query that appears in `prior_attempts`. If you
       cannot generate a new query, return an empty
       `suggested_next_queries`. The loop will terminate.
    """

    question: str = InputField(...)
    findings: list[Finding] = InputField(...)
    prior_attempts: list[PriorAttempt] = InputField(default_factory=list)

    answer: str = OutputField(description="The synthesised answer / memo.")
    confidence: int = OutputField(ge=1, le=10)
    used_finding_ids: list[str] = OutputField(...)
    open_questions: list[str] = OutputField(default_factory=list)
    suggested_next_queries: list[str] = OutputField(default_factory=list)
```

Wired as a plain `Call(InterpretFindingsSignature, model=...)`. No new
Program needed — the loop in Layer 5 owns iteration.

Tests:
- Unit: confidence bounds, used_finding_ids ⊆ input findings ids.
- Live: 10 hand-curated findings on the MTD case → confidence 8–10,
  answer >= 1500 chars, suggested_next_queries empty when answer covers
  the question or non-empty otherwise.

### 2.6 Layer 5 — `IterativeFindings` Program (kaos-llm-core)

File: `kaos-llm-core/kaos_llm_core/programs/iterative_findings.py` (new).

Built on `LoopRunner` (the same kernel `RefineDeliverable` and `ReAct` use).

```python
@dataclass(frozen=True, slots=True)
class IterativeFindingsResult:
    answer: str
    confidence: int
    findings: tuple[Finding, ...]            # accumulated across rounds
    used_finding_ids: tuple[str, ...]
    open_questions: tuple[str, ...]
    rounds: int
    stop_reason: StopReason                  # CONFIDENT | EXHAUSTED | MAX_ROUNDS | BUDGET
    usage: InvocationUsage


class IterativeFindings(Program):
    """retrieve → filter → interpret → loop until confident or exhausted."""

    def __init__(
        self,
        *,
        collect: CollectFindings,
        interpret: Call,                     # Call(InterpretFindingsSignature)
        max_rounds: int = 5,
        confidence_threshold: int = 8,
        max_cost_usd: float | None = 5.0,
    ) -> None:
        super().__init__()
        self.collect = collect
        self.interpret = interpret
        self.max_rounds = max_rounds
        self.confidence_threshold = confidence_threshold
        self.max_cost_usd = max_cost_usd

    async def forward(*, question: str, corpus: Any) -> IterativeFindingsResult:
        runner = LoopRunner(max_iterations=self.max_rounds, ...)
        state = _State(question=question, query=question, findings={}, attempts=[])

        async def step(it: int) -> StepOutcome:
            collect_result = await self.collect(
                question=state.query, corpus=corpus,
            )
            # Dedupe by Finding.id — never re-add the same span.
            for f in collect_result.findings:
                state.findings.setdefault(f.id, f)

            interpret_result = await self.interpret(
                question=state.question,
                findings=list(state.findings.values()),
                prior_attempts=state.attempts,
            )

            state.attempts.append(PriorAttempt(
                query=state.query,
                findings_added=len(collect_result.findings),
                confidence=interpret_result.confidence,
                open_questions=interpret_result.open_questions,
            ))

            if interpret_result.confidence >= self.confidence_threshold:
                return StepOutcome.stop(StopReason.CONFIDENT, payload=interpret_result)
            if not interpret_result.suggested_next_queries:
                return StepOutcome.stop(StopReason.EXHAUSTED, payload=interpret_result)
            state.query = interpret_result.suggested_next_queries[0]
            return StepOutcome.continue_(payload=interpret_result)

        return await runner.run(step, build_result=_build_result)
```

Notes:
- **Dedupe by Finding.id** is mandatory — without it, a stuck loop fills
  the synthesis context with the same evidence and the LLM hallucinates
  progress. Kelvin learned this the hard way (commit `965cad3`).
- **Re-query never repeats**. `prior_attempts` is part of the
  `InterpretFindings` input so the LLM sees what's already been tried;
  the Signature's decision rule #4 is the explicit guard.
- **Budget cap**. `max_cost_usd` short-circuits the loop with
  `StopReason.BUDGET`. Cost flows through the standard `Invocation.usage`
  rollup.

Tests:
- Unit: deterministic FunctionClient cycle through 3 rounds with
  confidence 5 → 7 → 9; assert dedupe, assert query divergence, assert
  CONFIDENT stop.
- Live: Harvey MTD task → confidence ≥ 8 within 3 rounds, ≥ 12 findings,
  answer ≥ 4000 chars.

### 2.7 Layer 6 — `findings` agent pattern (kaos-agents)

File: `kaos-agents/kaos_agents/patterns/findings.py` (new).

```python
class FindingsAgent(BaseAgent):
    """kaos-agents wrapper around `IterativeFindings`.

    Loads the agent's session memory (DOCUMENTS section) into a corpus,
    runs `IterativeFindings`, and emits typed events:
    - one `IntentClassified` (always RESEARCH for this pattern)
    - per round: `StepStart(round=N, query=...)`,
                 N x `CitationFound` (one per Finding),
                 `StepComplete(confidence=...)`
    - final: `TextDelta` (the answer), `TurnComplete`.

    Every Finding is also persisted to MemoryType.FINDINGS so subsequent
    turns can reuse evidence (kelvin equivalent: `target_documents` +
    `finding_log`).
    """

    PATTERN_NAME = "findings"
    ...
```

Wiring:
- `Agent.create(pattern="findings", ...)` selects this pattern.
- `Runner` dispatches the same way it does for chat/research/plan-execute.
- The `--pattern findings` flag added to `harvey_lab_benchmark.py` and
  `kaos-agent chat`.

Tests:
- Unit: pattern selection from Agent.create, event emission.
- Live (the acceptance gate): `harvey_lab_benchmark.py
  --pattern findings --task tests/fixtures/harvey-lab/analyze-counterparty-motion-to-dismiss
  --model anthropic:claude-sonnet-4-6` reaches ≥ 85% pass rate (parity
  with chat-pattern) AND emits ≥ 1 round of structured findings.

## 3. Validation strategy

### 3.1 Per-layer

Each layer ships with its own QA gate (ruff format + ruff check + ty
check + pytest unit + at least one live integration test). No layer is
"done" without a green live test.

### 3.2 End-to-end benchmarks

Two existing fixtures, run with the new pattern, judged with the same
rubric judges as before:

| Benchmark | Current best (free-form) | Target (findings) |
|---|---|---|
| Harvey CoC (M&A, 8 contracts, 55 criteria) | hybrid_v2 = 61.8% | ≥ 70% |
| Harvey MTD (litigation, 6 docs, 34 criteria) | chat-pattern = 85.3% | ≥ 88% |

Targets are deliberately modest — the win is qualitative (citations,
audit trail, gap detection) more than headline pass rate. If the new
pattern does NOT match chat-pattern's pass rate on MTD, that's a bug —
file it and fix.

### 3.3 Cross-domain check

Re-run BEIR `nfcorpus`, `scifact`, `fiqa` with the underlying
`CollectFindings` retrieval to confirm the wider top_k + filter
combo doesn't regress NDCG@10 vs plain BM25. (Plain BM25 is the
production default per `kaos-agents/CLAUDE.md` — adaptive retrieval
already lost this gate once.)

## 4. Migration / rollout

* Land Layers 1–5 first. They are pure additions to kaos-content +
  kaos-llm-core; nothing in kaos-agents changes. Existing
  `chat`/`research`/`plan-execute` pattern users see no behavior change.
* Layer 6 adds `findings` as a third pattern alongside the existing
  three. No defaults move yet — opt-in by `pattern="findings"`.
* After two weeks of real-world usage and a passing benchmark gate, open
  a separate PR to consider making `findings` the default for
  long-corpus, long-deliverable tasks (probably keyed off corpus size and
  intent classification, not a flat default change).

## 5. Open questions

1. **Where does `Finding` actually live?** Argued for kaos-content
   (`evidence.py`) above because Finding's natural shape is "a slice of
   a ContentDocument" and kaos-content already owns Span-like
   provenance. Alternative: kaos-llm-core (next to `Cited[T]`/`Claim`).
   Decision criterion: does anything outside an LLM context need to
   construct a Finding? If yes (e.g. rule-based or kaos-graph-driven
   finders) → kaos-content. If no → kaos-llm-core. Initial bet: kaos-content.

2. **Should `CollectFindings` ALWAYS go wider than RAG?** Default
   `top_k=50`. For tiny corpora (< 50 segments) the wider top_k is a
   no-op; for huge ones (> 500) it might miss recall. Plan: ship the
   default at 50, add a `top_k_per_round` override, watch the live
   numbers.

3. **Finding-level reranking?** kelvin had `RankingExecutor` (a
   re-ranking step between retrieve and filter). We can plug
   `kaos-nlp-transformers` embedding rerank in front of filter. Skip in
   v1; revisit if BM25-then-filter recall regresses on the BEIR check.

4. **Confidence threshold = 8 right?** kelvin tuned 9 → 8 → 7 over the
   project. We start at 8 and treat it as a tunable. Add a per-pattern
   setting: `KAOS_AGENT_FINDINGS_CONFIDENCE_THRESHOLD`.

5. **Does this replace `RefineDeliverable`?** No. `RefineDeliverable`
   refines the *output* against a *rubric/critic*. `IterativeFindings`
   accumulates *evidence* against *gaps*. Different signal, different
   pattern. They compose: a deliverable can be the output of an
   `IterativeFindings` run, and then refined by `RefineDeliverable`
   against a rubric.

6. **What about kaos-graph cross-document checks?** Out of scope here.
   Plan record it as a follow-up: a `GraphRetriever` implementation of
   the `Retriever` Protocol that surfaces SPARQL-derived passages
   ("which contracts mention 'reverse triangular merger'?") drops
   straight into Layer 3 with no pattern changes.

## 6. Implementation phases (task-tracker entries)

The plan slices into eight tracked tasks (the `F` series). Each is
independently shippable and live-tested.

* **F1** — `Finding` dataclass + tests (kaos-content)
* **F2** — `FilterSegmentsSignature` + `Filter` program + live test (kaos-llm-core)
* **F3** — `CollectFindings` program + live test (kaos-llm-core)
* **F4** — `InterpretFindingsSignature` + Call + live test (kaos-llm-core)
* **F5** — `IterativeFindings` Program + LoopRunner integration + unit + live test (kaos-llm-core)
* **F6** — `findings` agent pattern (kaos-agents) + `--pattern findings` flag in benchmark + Harvey MTD acceptance run
* **F7** — Harvey CoC live run with `findings` pattern; record vs. hybrid_v2
* **F8** — BEIR cross-domain regression check + final QA + commit batch + design-doc closeout

Each F* task carries its own test gate. A task is not "completed"
without a green `--include-live --include-network` validator run on the
affected packages, in line with kaos-modules' standing testing policy.

## 7. References

* `../kelvin-agent/kelvin/agent/patterns/chat_agent.py` — the loop
* `../kelvin-agent/kelvin/agent/actions/search/findings/collect_findings.py`
* `../kelvin-agent/kelvin/agent/actions/conversation/inputs/filter_results.py`
* `../kelvin-agent/kelvin/agent/actions/interpretation/interpret_last_action.py`
* `../kelvin-agent/kelvin/agent/actions/planning/input/retry_plan.py`
* `../kelvin-nlp/kelvin/nlp/llm2/utilities/records.py` — `Finding` dataclass
* `../kelvin-nlp/kelvin/nlp/llm2/executors/filter_executor.py`
* `../kelvin-nlp/kelvin/nlp/types/kelvin_span.py` — multi-strategy span
  search (architectural inspiration; we keep our scattered equivalents)
* `kaos-llm-core/kaos_llm_core/programs/loop_runner.py` — the iteration kernel we reuse
* `kaos-llm-core/kaos_llm_core/programs/refine.py` — the closest existing analogue
* `docs/benchmarks/harvey-coc-pipeline-comparison-2026-05-06.md` — the
  free-form / structured / hybrid baseline this pattern competes with
