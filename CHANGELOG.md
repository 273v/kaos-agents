# Changelog

All notable changes to `kaos-agents` are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0a14] — 2026-05-17

### Changed

- **kaos-core floor raised to `>=0.1.0a10`** to pick up the URI
  contract redesign (file://, vfs://, `context.default_vfs_namespace`).
  Pass-through for kaos-agents internals — no synthetic bare-name
  resolver calls in the agent runtime. See
  `kaos-modules/docs/plans/uri-contract-redesign.md`.

## [0.1.0a13] — 2026-05-17

### Added — no-evidence refusal gate (P0 hallucination defence)

The chat pattern now refuses to ship an LLM-drafted final answer when
**every** tool call attempted in the turn returned ``is_error=True``
AND the user explicitly referenced files (via attached documents,
filename tokens in the message text, OR filename tokens in the
agent's own tool-call arguments). Instead the agent emits a
structured "I tried to read these files, every tool call failed,
I will NOT fabricate an answer" message and a
``GroundingRefusalTriggered`` event.

Stage 0.2 of ``kaos-modules/docs/plans/vfs-blind-tools-audit-and-fix-plan.md``.
Belt-and-suspenders guarantee against a production incident already
seen in the wild (session ``01KRVYAEA3B1HG95DBAG6H0DJ3``): 5 NDA
.docx uploads, every ``kaos-office-parse-docx`` returned
"File not found" because the tools are VFS-blind, and the agent
fabricated a Delaware-vs-Michigan jurisdiction analysis citing
files it never read. Violates the 273V kaos-* legal-research bar
where "confidently wrong" ranks as the WORST failure class.

New API surface:

- ``kaos_agents.grounding.no_evidence_gate`` module
- ``evaluate_no_evidence_gate(observations, user_message, attached_documents) → NoEvidenceVerdict``
- ``render_refusal_text(verdict)`` — composes the chat-friendly refusal message
- ``ToolObservationSummary`` — pattern-independent projection of a tool call's outcome
- ``NoEvidenceVerdict`` — frozen value type carrying the gate's decision + payload
- ``extract_referenced_files(message)`` — extracts filename-shaped tokens for the gate (or external callers)

Wired into ``kaos_agents.patterns.chat.ChatAgent`` between the
trajectory's tool-call emission and the final ``TextDelta``. When
the gate refuses, the LLM's drafted answer is replaced with the
honest refusal text and the structured event fires for downstream
consumers (the SPA's ``ToolCallBlock``, OTel hooks, plan-execute
replan logic).


## [0.1.0a12] — 2026-05-17

### Changed — promote LLM-context caps to env-overridable settings

Stage B4 (partial) of the cross-package
`no-hardcoded-caps-and-artifact-first-tool-results` plan in the
kaos-modules monorepo. Moves three module-level constants out of
`_constants.py` and `tools/retrieval.py` into typed fields on
`KaosAgentSettings`, per the plan's architectural principle 4
("Typed module settings"). These caps are LLM-context guards and
UI-preview limits, not data-loss caps — the artifact tier system in
kaos-core handles the data-loss case — but they're configuration and
should be env-overridable per deployment.

**New settings fields** (`KaosAgentSettings`, env prefix
`KAOS_AGENT_`):

| Field | Default | Was |
|---|---|---|
| `result_summary_max_chars` | 200 | `RESULT_SUMMARY_TRUNCATE` in `_constants.py` |
| `eval_result_max_chars` | 2000 | `EVAL_RESULT_MAX_CHARS` in `_constants.py` |
| `rerank_passage_max_chars` | 2000 | `_RERANK_PASSAGE_TRUNCATE` in `tools/retrieval.py` |

Env-var overrides:

```bash
export KAOS_AGENT_RESULT_SUMMARY_MAX_CHARS=400
export KAOS_AGENT_EVAL_RESULT_MAX_CHARS=4000
export KAOS_AGENT_RERANK_PASSAGE_MAX_CHARS=4000
```

**Call-site migrations** (no behavior change at default values):

- `runtime/runner.py:641` — `result_text[:RESULT_SUMMARY_TRUNCATE]`
  → `result_text[: self._settings.result_summary_max_chars]`; import
  dropped.
- `patterns/plan_execute.py:341, 352` — same `str(step_result)`
  truncation now uses `self._settings.result_summary_max_chars`;
  import dropped.
- `tools/retrieval.py:974` — `RerankTool.execute` now resolves
  `KaosAgentSettings.from_context(context).rerank_passage_max_chars`
  at call time, so per-request `_meta.kaos_config` overrides take
  effect.

**Deletions** (no behavior change at default values):

- `kaos_agents/_constants.py`: `RESULT_SUMMARY_TRUNCATE` +
  `EVAL_RESULT_MAX_CHARS` deleted (replaced with an explanatory
  comment pointing at `KaosAgentSettings`).
- `kaos_agents/tools/retrieval.py`: `_RERANK_PASSAGE_TRUNCATE` deleted
  (replaced with explanatory comment).

### Deferred to a follow-up release

`ArtifactToCorpusHook` (Stage B4 part 2) — the auto-promote-text-ish-
artifacts-into-SessionMemory.DOCUMENTS hook. The hook is a new
feature that deserves its own design iteration: dedupe semantics,
mime-type policy, and per-session cost implications need shakedown
before flipping `auto_corpus_from_artifacts=True` becomes the
default. Stage C (kaos-ui SPA artifact rendering) does NOT depend on
the hook — Stage C reads `artifact_id` from
`ToolCallSummary.structured_content` which is already produced by
Stage B2 (kaos-source 0.1.0a7) and Stage B3 (kaos-web 0.1.0a5).

### Constants audit

```bash
$ git grep -E 'RESULT_SUMMARY_TRUNCATE|EVAL_RESULT_MAX_CHARS|_RERANK_PASSAGE_TRUNCATE' kaos_agents/
# Only doc-string mentions in settings.py + an explanatory comment in
# _constants.py; no production callsites.
```

### Dependencies

No version pin changes.


## [0.1.0a11] — 2026-05-17

### Fixed — ``PatternMismatch`` event now reaches the yielded stream, not just in-process collectors (#42)

The 0.1.0a10 release shipped the dispatch redirect correctly (PLAN /
RESEARCH intent on ``ChatAgent`` redirected through ``_handle_tool_use``
so real tool calls fired), but the ``PatternMismatch`` typed event was
invisible to anyone outside the agent process. Root cause:
``BaseAgent._detect_pattern_mismatch`` called
``emitter.emit(PatternMismatch, ...)`` and discarded the return value.
``EventEmitter.emit`` instantiates the event and *only* pushes it to an
active ``collect_events()`` collector — it does not yield it from the
``_dispatch_streaming`` async generator. Unit tests opened a collector
so they passed; production SSE / OTel / live-test consumers iterate the
generator output and never saw the event. Net: the dispatcher worked
but its diagnostic was silent.

Fix:

* ``BaseAgent._detect_pattern_mismatch`` now returns
  ``(redirect_handler, mismatch_event)`` instead of just the handler.
* ``BaseAgent._dispatch_streaming`` yields the ``mismatch_event``
  before invoking the redirect handler, so the event flows through
  the same ``async for`` loop as every other ``KaosEvent``.

Live regression coverage (``@pytest.mark.live``,
``ANTHROPIC_API_KEY`` required):

* ``tests/integration/test_plan_intent_dispatches_with_tools.py`` —
  both tests now PASS on Sonnet. Pre-0.1.0a10 ChatAgent + the
  v2-matrix Test 7 prompt: ``intent=plan/0.97``, ``tools=0``,
  fabricated answer. Post-0.1.0a11: ``intent=plan/0.95``,
  ``PatternMismatch=1``, ``tools=2``, real Federal Register
  document number cited (Regulation S-P), per-turn cost $0.11
  vs. pre-fix ~$0.005.
* New ``tests/integration/test_dispatch_extended.py`` — 3 additional
  live tests proving (a) Haiku 4.5 dispatches identically (model-
  agnostic), (b) trivial ``RESPOND``-intent prompts do NOT fire
  ``PatternMismatch`` (gate is selective), and (c) follow-up turns
  in the same session still dispatch correctly (no state leak).

Also relaxed ``test_plan_pattern_plan_intent_runs_plan_execute_with_tools``
which asserted ``Span(JUDGE, COMPLETE) >= 1`` — ``Span(JUDGE)`` is
conditional (fires only when the planner needs semantic re-eval on
a step that returned ``matched=False``); a clean-execution plan
against Sonnet + Federal Register tools never triggers it. The test
now documents the conditional behaviour rather than asserting
something model-specific.

L8 regression net: full
``tests/integration/test_planning_live.py`` +
``tests/integration/test_router_live.py`` (20 live tests covering
Wishes #2 / #4 / #5 / #7 / #8 from 0.1.0a8/a9) re-run and pass — the
dispatcher rewire didn't break the planning loop, judge spans, route
events, conditional steps, or LLM-based synthesis.


## [0.1.0a10] — 2026-05-17

### Fixed — PLAN/RESEARCH intent dispatch no longer silently degrades to ``_handle_respond`` (#40)

``ChatAgent`` (the default agent for sessions opened with
``pattern="chat"`` — the FastAPI ``MessageRequest.pattern`` default)
silently degraded to ``BaseAgent._handle_respond`` when the per-turn
``IntentExtractor`` returned ``IntentType.PLAN`` or
``IntentType.RESEARCH``. The handler was a one-line "override in
PlanExecuteAgent" placeholder that dispatched a plain
``Call(RespondSignature)`` — no tool catalog, no plan graph, no
``compose()``. The agent answered confidently from training data;
``AgenticLoop``'s ``GoalChecker`` correctly diagnosed *"agent
asserted facts without using available web tools"* and then the
turn ended anyway. SPA × kaos-agents R1-REAL v2-matrix Tests 3 + 7
hit this on every Sonnet PLAN-intent run (``intent=plan/0.97``,
``tools=0``, ``judge_spans=0``, 56 ``citation_found`` events parsed
from training-data text).

This release closes the dispatch hole at the source. ``BaseAgent``
now ships:

* A new :meth:`BaseAgent._detect_pattern_mismatch` instance method
  called from :meth:`BaseAgent._dispatch_streaming` before the
  fall-through can fire. When the per-turn intent demands
  ``_handle_plan`` / ``_handle_research`` AND the running agent
  class hasn't overridden the BaseAgent default, the dispatcher
  emits a typed :class:`~kaos_agents.events.lifecycle.PatternMismatch`
  event and redirects to ``_handle_tool_use`` so at least ReAct
  fires.
* A new :meth:`BaseAgent._handler_is_default` helper that walks
  ``type(self).__mro__`` to decide whether a handler is the
  unmodified ``BaseAgent`` implementation or a subclass override.
* A new typed event
  :class:`kaos_agents.events.lifecycle.PatternMismatch` with fields
  ``{classified_intent, agent_pattern, recommended_pattern,
  fallback_handler, rationale}``. Registered in
  :data:`kaos_agents.events.ALL_EVENT_TYPES` + the snake_case
  type-name registry; SSE and OTel consumers can pattern-match on
  it. Recommended use: a future ``pattern="auto"`` wrapper would
  consume this event to switch agents mid-run.
* A new :attr:`KaosAgentSettings.debug_prompts` field (env var
  ``KAOS_AGENT_DEBUG_PROMPTS`` per the Configuration Hierarchy in
  ``kaos-modules/CLAUDE.md`` — *not* an ``os.environ`` read at
  call sites). Diagnostic surface for the next time the dispatch
  layer regresses; full Span-attribute coverage lands in a
  follow-up release.

Tests:

* ``tests/unit/test_pattern_mismatch.py`` — 11 unit tests with
  stub agents that exercise every branch of the new dispatcher
  without a live LLM call.
* ``tests/integration/test_plan_intent_dispatches_with_tools.py``
  — 2 live regression tests (``@pytest.mark.live``). The
  bug-guard test boots a ``ChatAgent`` with real Federal Register
  tools, issues the v2-matrix Test 7 prompt against Sonnet, and
  asserts (a) exactly one ``PatternMismatch`` event fires + (b)
  the redirect actually engages at least one tool. The
  happy-path counter-test instantiates a ``PlanExecuteAgent``
  and asserts ``PlanProposed`` + ``Span(TOOL_CALL)`` +
  ``Span(JUDGE)`` all fire.
* 2531 existing unit tests still pass.

Behaviour unchanged for callers that already opened sessions with
``pattern="plan"`` / ``pattern="research"`` — those agents override
the handler and the new detector returns ``None`` for them.

See ``kaos-modules/docs/plans/kaos-agents-autonomy-improvement-1.md``
for the full diagnosis + fix design, and
``kaos-modules/docs/plans/kaos-agents-autonomy-roadmap.md`` (the
parent roadmap) for how this fits the broader autonomy work.


## [0.1.0a9] — 2026-05-17

### Changed — Drop the ``Decision.DEEPEN`` dead-code branch (#34)

``Decision.DEEPEN`` was indistinguishable from ``Decision.REPLAN`` in
practice. Both branches called ``graph.mark_failed`` +
``_skip_dependents`` + ``_skip_remaining`` and returned
``StopReason.NEEDS_REPLAN`` from ``compose`` (the pre-0.1.0a9
``compose.py`` condition was literally ``if decision.decision in
(Decision.REPLAN, Decision.DEEPEN)``). DEEPEN was a confusing alias
rather than a distinct control-flow path, and the unused
``KAOS_AGENT_DEEPEN_THRESHOLD`` env var contributed to the false
impression that lowering it would change plan-execute behavior in
R1-REAL v2 matrix runs (it didn't — those failures fire via the
``matched=False`` branch instead).

This release removes:

* The ``DEEPEN`` value from ``Decision`` (``kaos_agents/types/plan.py``).
* ``deepen_threshold`` field on ``KaosAgentSettings`` and the
  ``KAOS_AGENT_DEEPEN_THRESHOLD`` env var (``settings.py``).
* ``_DEFAULT_DEEPEN_THRESHOLD`` + the ``deepen_threshold`` argument on
  ``route()`` (``planning/route.py``), plus the low-confidence branch
  that returned ``Decision.DEEPEN``. Low-confidence cases now flow
  through the existing REPLAN branch with no behavior change.
* The ``deepen_threshold`` pass-through on ``compose``,
  ``execute_adaptive``, ``execute_decompose``, and
  ``PlanExecuteAgent._handle_plan_streaming``.

A future ADaPT-style implementation that substep-decomposes the
failed step (via the existing ``PlanGraph.insert_subplan`` helper)
can re-introduce DEEPEN with non-trivial semantics. See the docstring
comments in ``planning/route.py`` for the follow-up path.

### Added — LLM-synthesise plan findings into a narrative answer (#35)

Pre-0.1.0a9, ``_synthesize_results`` in
``patterns/plan_execute.py`` was a raw f-string dump of
``{step_id → str(tool_output)[:300]}``. Users saw literal
``{"results": [{"document_number": ...}]}`` JSON blobs as "the
answer" to long-horizon plans — accurate structurally but unreadable,
and never addressing the user's actual question.

New ``kaos_agents/patterns/synthesize.py``:

* ``SynthesizeFindingsSignature`` with docstring rules: (a) lead
  with the answer, not "I searched..."; (b) cite step ids inline
  as ``[step-1-2fcdae]`` so the SPA run inspector can link them;
  (c) preserve uncertainty when ``stop_reason != "success"`` (state
  the gap explicitly rather than papering over it); (d) refuse JSON
  dumps; (e) acknowledge all-empty / all-error results without
  fabricating.
* ``synthesize_findings(goal, result, *, model, step_descriptions)``
  uses ``Call.invoke`` for cost accounting — returns
  ``(narrative, InvocationUsage)``.
* ``should_attempt_llm_synthesis(result)`` gate so we don't spend
  tokens on plans where every step errored.

``_handle_plan_streaming`` now calls ``synthesize_findings`` when
the gate returns True, emits a
``UsageObserved(source="plan-execute-synthesis")`` so the
``TurnSummary`` aggregate (and the SPA's per-turn cost line)
includes the synthesis call, and falls back to the preserved-as-pure
``_format_plan_response`` formatter on any exception or empty
narrative. Degraded environments without an LLM client keep the
0.1.0a7 partial-findings UX recovery for free.

### Added — ``Span(JUDGE, ...)`` and ``Span(ROUTE, ...)`` observability events (#36)

Pre-0.1.0a9 the Evaluate primitive's LLM judge ran invisibly (only
the final ``Judgment`` appeared in ``ComposeResult.traces``
``PrimitiveTrace``) and Route decisions were likewise locked inside
``PrimitiveTrace``. SSE consumers (SPA run inspector, OTel exporter)
saw no events for either — when ``matched=False`` killed a plan
there was no way to ask "what did the judge see?" or "what triggered
REPLAN here?" without reading VFS-persisted ``SessionMemory``.

Surface changes:

* ``SpanSubject`` gains ``JUDGE = "judge"`` and ``ROUTE = "route"``
  enum values. ``Span`` docstring's attribute-conventions table
  documents the expected keys for both.
* ``evaluate_semantic`` accepts optional ``emitter`` + ``step_id``
  kwargs. When provided, emits a ``Span(JUDGE, START)`` before the
  Call.invoke with ``{step_id, expected, result_preview}`` (200-char
  truncation), then either ``Span(JUDGE, COMPLETE)`` on success
  with ``{matched, confidence, reasoning, mode}`` or
  ``Span(JUDGE, ERROR)`` on failure. ``emitter=None`` is a no-op
  for backwards compat.
* ``compose`` accepts optional ``emitter`` kwarg; emits
  ``Span(ROUTE, START)`` + ``Span(ROUTE, COMPLETE)`` after every
  Route decision with ``{step_id, decision, reason,
  judgment_matched, judgment_confidence, replan_count}``.
* ``execute_decompose`` / ``execute_adaptive`` / ``execute_direct``
  thread ``emitter`` through. ``PlanExecuteAgent._handle_plan_streaming``
  passes its turn emitter into ``execute_adaptive``.

### Changed — Route split: low-confidence judge rejection → CONTINUE (#37)

``route.py``'s ``not judgment.matched`` branch used to fire
``Decision.REPLAN`` unconditionally. The SPA R1-REAL v2 matrix
Tests 3 + 7 hit this on every long-horizon plan: 3-5 successful
FR/EDGAR tool calls → judge says ``matched=False`` on the
synthesizable JSON payload because the step's free-form
``expected`` field talked about "the specific rule" while the tool
returned a list of candidates → plan bails before the synthesiser
(#35) gets a chance.

This release splits the rejection branch by judge confidence:

* ``matched=False`` AND ``confidence >= confidence_threshold`` →
  REPLAN. Judge is confident the tool's output is wrong (real
  failure signal, preserves pre-0.1.0a9 behavior for this case).
* ``matched=False`` AND ``confidence <  confidence_threshold`` →
  CONTINUE with a logged warning. Judge says "doesn't match
  expected but I'm not sure"; let the plan-execute synthesiser
  frame the result as partial findings.

The ``confidence_threshold`` knob (default 0.5, env
``KAOS_AGENT_CONFIDENCE_THRESHOLD``) now does double duty — it
gates both the ``matched=True`` low-confidence REPLAN and the new
``matched=False`` low-confidence CONTINUE.

``compose.py`` companion change: the ``mark_failed`` branch no
longer fires on ``not judgment.matched`` alone. Only hard tool
errors (``act_result.is_error``) mark the step failed. A successful
tool with a fussy judge stays ``COMPLETED`` — the judgment is
preserved on the node but the result lives in ``step_results`` so
the synthesiser (#35) / SPA run inspector can render it.

### Added — Conditional plan steps (``Step.abort_if`` + ``Step.pivot_to``) (#38)

R1-REAL Test 3's prompt shape had no semantic encoding in
pre-0.1.0a9 plan-execute:

    "Build me a 5-step research plan on CFPB's 1033 open-banking
    rule, execute it, but if any step reveals the rule was vacated
    or stayed, abandon the remaining steps and pivot to the
    litigation status."

This release adds:

* ``Step.abort_if: str`` (natural-language predicate over prior
  outputs) and ``Step.pivot_to: str`` (optional follow-up goal).
  Both default to empty strings — backwards compat for every
  existing caller.
* ``PlanGraph.add_step`` persists both fields; ``get_step``
  exposes them.
* New ``kaos_agents/planning/evaluate_condition.py``:
  ``EvaluateConditionSignature`` (strict-literal-match rules,
  default to false, asymmetric cost) +
  ``evaluate_condition(condition, prior_outputs, *, model,
  holds_confidence_threshold=0.6)``. Returns ``(False, "")``
  without an LLM call for empty inputs or on Call exception.
  Returns ``(True, evidence)`` only when the judge says
  ``holds=True`` AND ``confidence >= threshold`` (catches LLM
  hedging).
* ``compose.py`` runs the abort-if check sequentially across ready
  steps BEFORE executing them. Steps with a holding condition are
  marked SKIPPED (evidence becomes the skip reason); dependents
  skip too. When a skipped step also carries ``pivot_to``, the
  whole graph stops with the new ``StopReason.PIVOTED``.
* ``StopReason.PIVOTED = "pivoted"`` new enum value.
* ``PlanExpandSignature`` docstring gains rule 6 telling the
  planner to populate ``abort_if`` / ``pivot_to`` when the goal
  carries stop-or-pivot language. ``GeneratedStep`` pydantic model +
  ``_parse_steps`` plumb both fields onto ``Step`` instances.

What's NOT in this release: end-to-end "PIVOTED → strategy
re-expands on pivot_to" wiring. ``execute_decompose`` currently
propagates ``StopReason.PIVOTED`` to its caller without doing the
re-expand; the synthesiser surfaces the pivot context and the user
issues a follow-up turn. Full strategy re-expand is a follow-up.

### Tests

* ``tests/unit/patterns/test_synthesize.py`` — 11 new tests for
  the LLM synthesiser (formatter rendering, gating logic, end-to-end
  Call.invoke wiring).
* ``tests/unit/events/test_judge_route_spans.py`` — 8 new tests
  for the JUDGE/ROUTE span emission (enum extension, emitter=None
  backwards compat, START/COMPLETE pair, START/ERROR pair, input
  truncation, compose signature).
* ``tests/unit/planning/test_conditional_steps.py`` — 15 new tests
  for ``Step.abort_if`` / ``Step.pivot_to`` (type defaults,
  PlanGraph persistence, StopReason enum, formatter invariants,
  evaluate_condition gating + LLM stub paths).
* ``tests/unit/test_route.py`` — 3 new ``TestRouteSoftMissContinues``
  tests for the low-confidence rejection branch (boundary at
  ``confidence_threshold``, low-confidence → CONTINUE,
  high-confidence → REPLAN). ``TestRouteDeepen`` renamed to
  ``TestRouteLowConfidence`` to reflect the DEEPEN removal.


## [0.1.0a8] — 2026-05-16

### Fixed — disk-backed VFS + parse cache now work on Windows

The 0.1.0a5 disk-VFS-default switch exposed two latent Windows
filesystem bugs that the prior in-memory default had masked. Both
fixes percent-encode (or substitute) the offending separator so the
on-disk path component is valid on every OS we target.

- **`SessionStore` session paths.** Tenant-scoped session ids use
  ``<tenant>:<session>`` (see ``scope_session_id``), and the ``:``
  is a reserved character on the Windows filesystem (NTFS alternate
  data streams). The disk backend would write
  ``./.kaos-vfs/.../sessions/<tenant>:<session>/memory.json``
  happily on POSIX and fail with ``NotADirectoryError [WinError 267]``
  on Windows. New ``_safe_component`` helper in
  ``kaos_agents.memory.store`` percent-encodes the reserved set
  via :func:`urllib.parse.quote` so the directory name is valid on
  every OS; ``_unsafe_component`` is its inverse so
  ``list_sessions`` returns the caller-supplied id verbatim. The
  session id passed to ``save`` / ``load`` / ``exists`` / ``delete``
  / ``list_sessions`` is unchanged.
- **`kaos-agent chat` parse cache.** The cache key was
  ``f"{sha256}:{chunk_size}"`` and was composed directly into a
  filename (``<cache>/blobs/<key>.json``). On Windows the ``:``
  broke the write silently, so the second run of the same corpus
  found zero cached blobs and re-parsed every file. Separator is
  now ``-`` (hex + dash + decimal stays within the safe set on
  every OS). Existing local caches will miss once and rebuild.

8 new regression tests in
``tests/unit/test_store.py::TestSessionPathEncoding`` lock in the
encoding invariants for ``:``, the full Windows reserved set
(``< > : " | ? * \``), the roundtrip, and a tenant-scoped
``save → load → list_sessions`` cycle. The existing 5
``TestCacheRoundTrip`` cases continue to pass.

Closes the kaos-agents CI Windows-x64 / Python 3.13 failures
observed on PRs #19, #25, and #27 (``tests/unit/test_api_auth.py``
``TestTenantScoping::test_same_tenant_round_trip``,
``TestAuthBypassedEndpointsStillWork::test_send_message_with_token_200``,
and ``tests/unit/test_cli_chat.py::TestCacheRoundTrip::test_hit_matches_miss``).


## [0.1.0a7] — 2026-05-17

### Fixed — Surface partial findings when plan-execute stops early (#28)

When a plan-execute run terminated with ``stop_reason != SUCCESS`` but
``step_results`` was non-empty (``NEEDS_REPLAN`` past max replans,
``MAX_COST``, ``MAX_STEPS``, ``MAX_WALL_CLOCK``, ``FAILURE``), the
``_handle_plan_streaming`` branch emitted only the terse line
``"Plan execution stopped: <reason>. Completed N steps."`` and
silently dropped every tool result the agent had already produced.

R1-REAL SPA verification matrix v2, Tests 3 & 7 (2026-05-17) exposed
this: long-horizon Federal Register / EDGAR research runs fired 3–5
real tools across multiple plan steps, hit ``NEEDS_REPLAN`` at the
Route ``matched=False`` branch, and surfaced the terse status text
with zero findings to the user.

Now the same runs emit:

```
_Plan stopped early (reason: needs_replan, 3 step(s) completed).
Partial findings:_

Plan completed with the following results:

**step-1-2fcdae**: Found 9 Federal Register document(s)…
{ "results": [ { "document_number": "2024-30494", "title": "EDGAR
Filer Access and Account Management" … } ] }
```

The response-formatting logic moved from inline branches inside
``_handle_plan_streaming`` to a pure
``_format_plan_response(result: ComposeResult) -> str`` helper so the
four termination branches (success-with-results, success-empty,
stopped-with-partials, stopped-empty) are unit-testable without
spinning up an Agent. The error-only-results fallback (``"Plan
completed but results were empty or all errored."``) is preserved
across the success and stopped-early branches.

Behavior change is purely additive — the existing
"Plan execution stopped: …" wording is still emitted for the
no-progress case so callers can distinguish "stopped with partial
work" from "stopped with nothing".

### Added — Working replan loop in ``execute_decompose`` (#30)

``execute_decompose`` previously executed exactly one Compose call. When
Compose returned ``NEEDS_REPLAN``, the strategy propagated it as-is
even though ``compose.py`` line 188 documented the gap (``"Return
NEEDS_REPLAN so the strategy layer can re-expand"``). No strategy
wired the re-expand up.

This release adds a bounded replan loop with regression guards:

* **Re-expand only when nothing succeeded.** If the current attempt
  produced any successful step results, the loop exits so the caller's
  response formatter surfaces the partials. Re-planning on partial
  success historically biased the LLM toward LLM-only "analysis" steps
  (turned a 1-step / 2-tool-call plan into an 18-step / 0-tool-call
  plan in the prior failed attempt).
* **Cumulative ``step_results``.** The final ``ComposeResult`` carries
  the union of every attempt's successful step results so the
  formatter can synthesise them.
* **Structured ``prior_failures``.** New
  ``PlanGraph.get_failures() -> dict[step_id, {tool_name, step_type,
  result}]`` helper feeds a new
  ``_format_prior_failures`` formatter that emits only FAILED entries
  plus an explicit "prefer direct tool calls over LLM-only analysis"
  hint. Successful tool output is structurally impossible to feed back.
* **``NEEDS_REPLAN`` → ``MAX_REPLANS`` promotion.** When
  ``budget.replans`` exhausts on a run that never converged, the final
  ``stop_reason`` is rewritten so callers can distinguish "ran out of
  retries" from "control-flow signal the strategy never wired up".

Loop bound: one initial attempt + up to ``budget.max_replans`` retries
(default ``KAOS_AGENT_PLAN_MAX_REPLANS=3``). Cumulative cost / token /
wall-clock budgets still apply across attempts via the shared
``PlanBudget`` instance.

The existing ``tests/integration/test_strategies_live.py::
TestDecomposeStrategy::test_multi_step_goal`` accepts ``MAX_REPLANS``
in addition to ``NEEDS_REPLAN`` / ``FAILURE`` / ``SUCCESS``.

### Added — ``corpus_attached`` / ``corpus_size`` signal in ``IntentSignature`` (#29)

When a session has documents attached, indirect references like
``"summarize that"``, ``"the file"``, ``"extract terms from the
PDF"``, or bare verbs like ``"summarize"`` silently routed to ``CHAT``
pre-release — the intent classifier had no signal that a corpus was
attached and treated the message as conversational. The agent then
answered from training data instead of the attached document (SPA
R1-REAL UX-C2).

New surface:

* ``IntentSignature`` gains two new ``InputField``s:
  ``corpus_attached: bool`` (default ``False``) and
  ``corpus_size: int`` (default ``0``, ``ge=0``).
* The Signature docstring gains rule 6: *"When ``corpus_attached=true``
  and the message refers to documents indirectly (pronouns / 'the
  file' / 'extract terms' / bare follow-ups), prefer
  ``pattern=RESEARCH``."*
* ``IntentExtractor.forward`` accepts and coerces the new kwargs and
  threads them into the inner ``Call.invoke``.
* ``AgentLoop.prepare_turn`` computes the count from
  ``SessionMemory.section_item_count(DOCUMENTS)`` via a new
  ``_corpus_size_from_memory`` static helper and passes both fields
  into ``IntentExtractor.invoke`` on every turn.

Backwards compatible: every existing caller continues to pass
``False`` / ``0`` by default; the new docstring rule only fires when
the classifier sees ``corpus_attached=true``. Applications that call
``IntentExtractor.invoke`` directly (bypassing ``AgentLoop``) see no
change unless they opt in.

### Tests

* ``tests/unit/test_plan_response_formatter.py`` — 11 new tests across
  4 classes for the formatter branches.
* ``tests/unit/planning/test_decompose_replan.py`` — 17 new tests across
  4 classes for the replan loop, including explicit regression
  coverage for the "any tool call succeeded → don't retry" guard.
* ``tests/unit/intent/test_extractor.py`` — 7 new tests across two
  classes for the IntentExtractor / IntentSignature corpus inputs.
* ``tests/unit/loop/test_agent_loop.py`` — 7 new tests for
  ``_corpus_size_from_memory`` and the prepare_turn threading path.


## [0.1.0a6] — 2026-05-16

### Changed — Strengthen planner + critic Signature docstrings (M2 of thin-worker-prompt.md)

The kaos-agents Signature decision points are the canonical home
for tool-selection rules and verdict shortcuts; the worker prompt
on the kaos-ui side is supposed to carry only context. Between
2026-05-13 and 2026-05-16 the kaos-ui worker prompt accumulated
~570 tokens of English behavior rules duplicating decisions these
Signature docstrings already encode. The right reaction was a
docstring edit + an eval, not a worker-prompt patch. This release
makes that the path of least resistance.

**`_TurnToolPolicySignature` — lookup-beyond-corpus rule.** Added
to the docstring's corpus-kinds hints:

  *"When `corpus_headlines` is non-empty AND the question asks
  about facts that likely go beyond the attached files — names,
  current roles or titles, public records, recent events, prices,
  'who is X', 'look up Y', 'find the source for Z' — include BOTH
  `documents` AND `web` in `wanted_groups`. The agent searches the
  docs first (cheaper, deterministic); if the answer isn't there,
  it already has web in scope to escalate without a replan."*

Closes the 2026-05-16 "who teaches 800" → "look up who that is"
failure where the planner picked documents-only, the agent
searched the PDF, didn't find the answer, and hallucinated a
faculty record instead of escalating to web search.

**`_GoalCheckerSignature` — confident-hallucination shortcut.**
Added to the docstring's concrete shortcuts:

  *"Agent's response asserts a specific person's identity, current
  role/title, recent date, price, legal status, or any other
  public-record fact, AND `tool_calls_made` is empty (no successful
  tool call produced evidence) → `needs_more_work` with
  `next_action` = 'search the web for [the asserted fact] before
  answering'. Confident hallucination of look-up-able facts is the
  single highest-impact failure this critic catches."*

The shortcut explicitly excludes definition / arithmetic / language
/ summarization tasks that legitimately don't need a tool call.
The verdict feeds into `AgenticLoop`'s replan path: the critic's
`next_action` becomes the next iteration's `thinking_note`, so the
agent gets structured guidance to call web tools without any new
English in the worker prompt.

### Added — Signature-level live eval suite

`tests/integration/test_signature_evals_live.py` (6 cases, ~$0.0012
per full run). Each case is sourced from a 2026-05-16 failing
session — when a future docstring edit regresses a behavior, the
relevant case fails with a pointer at the right Signature file.

| Layer | Case | Asserts |
|---|---|---|
| Planner | `lookup_cfpb_2026_no_corpus` | `wanted_groups ⊇ {"web"}` for "look up the latest CFPB enforcement" |
| Planner | `who_teaches_800_with_pdf` | `wanted_groups ⊇ {"documents"}` for "who teaches 800" with PDF attached |
| Planner | `lookup_continuation_with_pdf` | `wanted_groups ⊇ {"documents", "web"}` (the M2.1 rule) |
| Critic | `hallucinated_person_no_tools` | "J. Bommarito is the lead instructor…" + 0 tool calls → `needs_more_work` with `next_action` naming `search`+`web` |
| Critic | `clarification_when_docs_attached` | "I need more specific information…" + 0 tool calls → `needs_more_work` with `next_action` naming `search`+`document` |
| Critic | `grounded_answer_with_tool_call` | "The most recent FR rule on dairy is 90 FR 12345…" + 1 successful tool call → `satisfied` |

`@pytest.mark.live`-marked so the default unit gate stays free.

Cross-reference: `kaos-modules/docs/plans/thin-worker-prompt.md`
§2.5 (kaos-ui-hack ↔ kaos-agents-Signature mapping), §3 (designed
architecture diagram), §4.2 (M2.1 + M2.2 rule text), §4.3 (M3 eval
case spec).

### Fixed — `create_app()` and `Runner` now default to disk-backed VFS

`_resolve_vfs(runtime=None)` in both `kaos_agents/api/server.py` and
`kaos_agents/runtime/runner.py` was constructing an in-memory
`VirtualFileSystem` when no runtime was provided. This silently lost
every persisted conversation on uvicorn restart, contradicting the
kaos-core platform default (`StorageBackend.DISK` rooted at
`.kaos-vfs/`).

The fix replaces the explicit in-memory `VFSConfig` with
`VirtualFileSystem()`, which picks up the kaos-core disk default.
Matches `KaosRuntime()`, the `kaos-mcp` resource adapter, and the
rest of the kaos-* ecosystem.

**Behavior change:**

| Caller | Before | After |
|---|---|---|
| `create_app(runtime=KaosRuntime(vfs=disk_vfs))` | disk-backed | disk-backed (unchanged) |
| `create_app(runtime=None)` | **in-memory** (data lost on restart) | **disk-backed** (matches platform default) |
| `create_app(runtime=KaosRuntime(vfs=mem_vfs))` | in-memory | in-memory (unchanged) |
| `Runner(agent=..., runtime=None, vfs=None)` | **in-memory** | **disk-backed** |

Callers that explicitly wanted in-memory (most commonly tests) now
must construct it themselves — `KaosRuntime.test_mode()` is the
documented pattern in `CLAUDE.md` for kaos-agents test isolation.

Closes #16. New `TestResolveVFS` regression tests in
`tests/unit/test_api.py` lock in both branches. The `app` fixture
in `tests/unit/test_api.py` and the two `create_app()` sites in
`tests/unit/test_streaming_metrics.py` now construct an explicit
`KaosRuntime.test_mode()` so the test working directory does not
accumulate `.kaos-vfs/` artifacts on every run.


## [0.1.0a5] — 2026-05-16

### Fixed — `[/response]` scratchpad-tag leak in respond handler

`BaseAgent._simple_respond` previously overrode the default
`JSONCodec` with `ChatCodec()` for the single-output
`RespondSignature`. The historic justification — Anthropic Sonnet
4.6 was observed to truncate ~30K-char prompts at ~3K characters
when the output was JSON-wrapped — does not reproduce on current
Claude 4.x / GPT-5.x / Gemini 2.5 models, all of which support
first-class structured output via the provider's JSON-schema /
function-calling path.

The workaround was leaking visibly to downstream UI surfaces.
ChatCodec instructs the model with an opener-only `[response]`
field marker; Haiku 4.5 (and other instruction-tuned models)
generalize that to XML-style and emit a matching `[/response]`
closer that `ChatCodec.decode` does not strip. The closer landed
inside the field value and rendered verbatim in chat UIs.

### Changed

- `kaos_agents/runtime/agent.py` — drop the `ChatCodec()` override
  in `_simple_respond`. The handler now uses the default
  `JSONCodec` (native structured output). The historic justification
  is preserved as a comment with a deprecation note.
- `kaos_agents/runtime/agent.py` — defense-in-depth scratchpad
  closer strip (`_STRIP_SCRATCHPAD_RE`) applied to the response
  text post-decode so a future non-JSON codec regression — or a
  model that hallucinates closers inside a JSON string value —
  cannot reach the response body. Conservative regex: only matches
  whole `[/\w+]` / `</\w+>` lines whose name is a slug.

### Tested

- 2424 unit tests pass (no regressions).
- 34 BaseAgent + AgenticLoop unit tests verified explicitly.
- Manual verification: Haiku 4.5 `compare these` against two PDFs
  no longer emits `[/response]` closer.

## [0.1.0a4] — 2026-05-15

### Added — AgenticLoop pattern: plan → elevate → execute → check → replan

Closes the "agent gives up because web search is disabled" failure
mode. The loop sits one level above the existing per-turn
`TurnToolPolicy` planner, composes a new `GoalChecker` Critic and a
SessionPolicy with three-tier auto-elevation, and orchestrates plan
→ ReAct → check → replan iterations until the user's goal is
satisfied (or a hard guard trips). Working backwards from the
single failure mode the user named, the design is
foundation-first: every primitive composes with existing kaos-agents
machinery (the per-turn planner, TurnToolPolicy, SessionToolSet,
the event taxonomy).

**`kaos_agents.types.session_policy.SessionPolicy`** — two-tier
ceiling + loop config:

- `allowed_groups` (working set) + `soft_ceiling` (auto-elevation
  max). Persona presets — `for_persona("research"|"drafting"|
  "forensics")` — set documented soft ceilings.
- Three-tier elevation taxonomy mirroring Claude Code's permission
  modes: `green-auto` (web, documents, citations, retrieval, vfs,
  forensics — silent elevation), `yellow-confirm` (browser,
  authoring, netinfra — inline approval card), `red-blocked`
  (programs, agents — never auto-elevate).
- Three independent loop limiters: `max_loop_iterations` (3),
  `max_loop_cost_usd` ($0.25), `max_loop_wall_clock_seconds` (60s).
- Immutable updates: `with_added_groups` / `with_removed_groups`;
  `to_session_tool_set` adapter for downstream `filter_tools`.
- 38 truth-table tests pin the taxonomy + tier mapping + persona
  invariants.

**`kaos_agents.planning.goal_check`** — the Critic Signature + a
three-way discriminated-union output:

- `GoalCheckSatisfied` (loop returns) / `GoalCheckNeedsMoreWork`
  (loop replans with `next_action` as agent thinking block, NOT
  fake user message) / `GoalCheckInsufficientEvidence` (corpus
  lacks; refusal-with-explanation, gray badge — not red).
- Modeled on Everlaw Deep Dive's `insufficient_evidence` gold-
  standard refusal UX (competitive doc §18).
- On provider exception / missing `[llm]` extra, defaults to
  `needs_more_work` — NEVER to `satisfied` (false satisfaction
  silently ships a bad answer).
- 13 contract tests including the canonical "provider exception
  must default to needs_more_work" regression.

**`kaos_agents.patterns.agentic_loop.run_agentic_turn`** — pure
async generator that yields the event stream for one user turn.
Worker is injected as a callable (decouples kaos-agents from any
specific ReAct implementation; the single-user-chat backend will
wire its existing `stream_chat` proxy in Stage L).

- 8 contract tests covering: single-iteration happy path, green-
  auto elevation, yellow-confirm capability request,
  needs_more_work replan + max_iterations cap, cost_exceeded
  mid-loop, stuck_no_progress (state-mutation detection), user
  interrupt (asyncio cancel re-raises after emitting
  `LoopTerminated(reason="user_interrupt")`), worker-event
  pass-through.
- Three-tier elevation logic, three independent limiters,
  state-mutation stuck-detection.

**`kaos_agents.events.policy`** — four new SSE-streamable events:

- `ToolPolicyElevated` — auto-elevation just happened silently.
- `CapabilityRequested` — yellow-confirm group needs approval.
- `GoalChecked` — Critic verdict with `kind` + `next_action` /
  `missing` / `confidence`.
- `LoopTerminated` — always the last event, carries
  `reason` ∈ {satisfied, insufficient_evidence, max_iterations,
  cost_exceeded, wall_clock_exceeded, stuck_no_progress,
  user_interrupt} + cumulative cost + wall-clock + elevation count.

Total event taxonomy: **19 types** (was 15).

Design references (competitive landscape research):
- Harvey Deep Research (`kaos-modules/docs/competitive/landscape.md`
  §"Harvey AI") — execute-then-show-plan transparency pattern.
- Everlaw Deep Dive
  (`kaos-modules/docs/competitive/capabilities/18-refuses-when-uncertain.md`)
  — three-way discriminated-union output as the trust differentiator.
- LangGraph cycle optimization — state-mutation stuck-detection
  (rajatpandit.com/optimizing-langgraph-cycles).
- Claude Code auto mode — three-tier permission taxonomy
  (anthropic.com/engineering/claude-code-auto-mode).
- Pydantic AI usage_limits — three-independent-limiter pattern.

Tests:
- 38 new SessionPolicy tests
- 13 new GoalChecker tests
- 8 new AgenticLoop orchestrator tests
- 4 new event fixtures added to test_events.py
- **2424 total unit tests pass** (was 2337); ruff + ty clean.

The loop is NOT yet wired into any consumer — the chat router
swap is Stage L (single-user-chat backend update). This release
ships the primitives so the backend can adopt them without
duplicating the design.

## [0.1.0a3] — 2026-05-15

### Added — derivation-based tool-group taxonomy + SessionToolSet defaults + TurnToolPolicy promotion (PRD PR 2)

The taxonomy and the planner ship together. The taxonomy is the
foundation; the planner is what surfaces it per-turn to the LLM.

**`kaos_agents.registry.tool_group_classifier`** — owns the canonical
11-group catalogue used by ceiling enforcement, the per-turn planner,
and every SPA tool-policy UI surface. Built as a **derivation over
existing `ToolMetadata` fields**, not a parallel name-prefix taxonomy:

  - **`derive_group(meta: ToolMetadata) -> str | None`** — pure
    function reading `category`, `capability`, `annotations.openWorldHint`,
    `annotations.readOnlyHint`, `tags`, and `module_name`. First-match-wins
    on a small truth table (11 rules; tag-based narrowings take
    precedence over category-based defaults). Returns `None` for
    tools that don't fit any group.
  - **`RECOGNIZED_TAGS = {"browser", "netinfra", "forensics", "retrieval"}`** —
    the four tag values the derivation reads as narrowing signals.
    Tools may carry additional free-form tags (`"experimental"`,
    `"deprecated"`, domain labels) without affecting group assignment.
  - **`KAOS_TOOL_GROUP_DESCRIPTIONS`** — one-paragraph description
    per group, used as the SettingsSheet group label.
  - **`register_kaos_tool_groups(runtime, registry=None)`** — walks
    every tool registered on a runtime, calls `derive_group` on each,
    and writes one `ToolGroup` per non-empty bucket into the registry.
    Returns `{group_name: tool_count}`.

Why derivation, not prefix patterns: a new tool added in any kaos-*
repo auto-classifies on the next runtime walk — **zero kaos-agents
release needed**. Third-party tools self-declare via the standard
`category` + `capability` + `tags` fields. The 11 groups are derived
views over existing ground truth, not new ground truth.

**`kaos_agents.planning.policy`** — TurnToolPolicy promoted from the
kaos-ui single-user-chat example into kaos-agents proper:

  - **`TurnToolPolicy`** frozen value type — `kept_groups` (planner's
    intersect with ceiling), `dropped_groups` (planner wanted these
    but the ceiling denied — surfaces in the SPA's "wanted but
    blocked" UX), `rationale`, `confidence`, `fell_back_to_ceiling`,
    `cost_usd`, `latency_ms`. The pre-promotion `turn_groups` field
    survives as a property alias on `kept_groups` for back-compat.
  - **`plan_turn_tool_policy(**inputs)`** — async entrypoint.
    Best-effort with abdicate-to-ceiling semantics: low confidence,
    provider failure, missing `[llm]` extra, or disjoint
    wanted/ceiling sets all fall back to the full ceiling. Never
    raises.
  - **Signature inputs** (PRD round-2 decision #7): `user_message`,
    `recent_turns`, `corpus_headlines`, **`corpus_kinds: list[str]`**
    (Magika-style content classification for uploaded files —
    `["pdf", "spreadsheet", "html"]`), **`session_intent: str | None`**
    (preset chip selection — `"research"` / `"drafting"` / `"forensics"`),
    **`raw_turn_groups: list[str] | None`** (last turn's wanted set
    for cross-turn coherence), `ceiling_groups`, `available_groups`.
  - **Three-way BM25 disambiguation** in the Signature few-shots
    (round-2 decision #6): `kaos-source-bm25-search` searches the
    Free Law Project corpus; `kaos-nlp-core-bm25-search` searches
    session memory; `kaos-retrieval-bm25` delegates to the
    RetrievalAgent for broader recall.
  - The 8 RetrievalAgent tools (`kaos-retrieval-bm25`, `-synonyms`,
    `-hyde`, `-evaluate`, `-rerank`, `-corpus-info`, `-corpus-manifest`,
    `-answer`) now carry `tags=["retrieval"]` so they auto-classify
    into the `retrieval` group.

- **`SessionToolSet` ceiling defaults** in
  `kaos_agents.types.session_tool_set`:
  - **`DEFAULT_ALLOWED_GROUPS`** — the 7-group "research" preset
    every fresh session starts with: `web`, `browser`, `documents`,
    `citations`, `vfs`, `forensics`, `retrieval`. Excludes
    `netinfra` (DNS/WHOIS — opt-in for diligence), `authoring`
    (writers — opt-in for drafting), `programs` (kaos-llm-core
    typed-program + alpha-* — opt-in for power users), and `agents`
    (self-recursive — opt-in *and* always-denied).
  - **`DEFAULT_DENIED_TOOLS`** — the 4 self-recursive kaos-agents
    tools (`kaos-agent-chat`, `kaos-agent-plan`,
    `kaos-agent-findings`, `kaos-agent-corpus-filter`). Registered
    in the runtime so power-user topologies can wire them as
    sub-agents, but denied at the ceiling so accidental opt-in
    can't trigger infinite recursion.
  - **`SessionToolSet.auto_narrow: bool = True`** — per-session
    toggle for the per-turn `TurnToolPolicy` planner. When `True`,
    the chat router narrows the ceiling to just the groups this
    message needs (cost + hallucination reduction). When `False`,
    the full ceiling passes to ReAct.
  - **`SessionToolSet.default()`** — classmethod returning the
    canonical fresh-session config (the 7-group ceiling + the 4
    denied tools + `auto_narrow=True`). Use this instead of
    `SessionToolSet()` (which returns the unrestricted config) when
    creating a new session.

Motivated by `kaos-modules/docs/internal/dynamic-tool-planning-prd.md`
§4 ("PR 2 — kaos-agents default ceiling + ToolGroupRegistry rewrite")
and the live session bug it documents: a session that asked the
agent to search the web was unable to because the default ceiling
omitted `web`. The default ceiling now matches what an 80%-case
legal-research session expects.

Tests:
  - 33 new tests in `tests/unit/test_tool_group_classifier.py`
    pinning the derivation truth table — one parametrized case per
    rule + ordering tests (tag-beats-category, citations-beats-web,
    authoring-beats-documents, etc.) + a partitioning happy-path
    test over a representative 13-tool runtime.
  - 13 new tests in `tests/unit/test_turn_tool_policy.py` pinning
    the planner's contract: confident narrowing, ceiling
    intersection, `dropped_groups` for "wanted but blocked",
    disjoint-set fallback, low-confidence fallback, threshold
    override, empty-ceiling short-circuit, provider-exception
    fallback, `corpus_kinds` / `session_intent` / `raw_turn_groups`
    passthrough, omitted-input defaults, and frozen-dataclass
    immutability.
  - 7 new tests in `tests/unit/test_session_tool_set.py` pinning
    the `DEFAULT_ALLOWED_GROUPS` / `DEFAULT_DENIED_TOOLS` /
    `SessionToolSet.default()` / `auto_narrow` defaults.

Purely additive: existing `SessionToolSet()`-without-args still
returns the unrestricted config (allow-all). Callers that want the
canonical fresh-session ceiling explicitly call `.default()`.
The pre-promotion `app.services.turn_tool_policy` module in
single-user-chat remains importable until the consumer migrates in
Stage D.

## [0.1.0a2] — 2026-05-15

### Fixed

- **`ChatAgent` ReAct now drops one bad tool and retries instead of
  failing the whole turn when a provider rejects a single tool's
  JSON Schema.** Previously, when OpenAI returned HTTP 400
  `invalid_function_parameters` for a specific function in the
  catalog, the broad `except Exception` in
  `_handle_tool_use_streaming` caught the error, lost ALL tools for
  the turn, and fell back to `_simple_respond` with no tools — the
  agent then hallucinated answers or apologized for "the tool layer
  failed". Now the chat pattern parses the offending function name
  or `tools[N].function.parameters` index out of the provider error
  text, drops that single tool from the catalog, and re-instantiates
  ReAct. Up to 5 schema-rejection drops are tolerated per turn
  before falling through to the existing `react-fallback` path;
  loop protection refuses to drop the same tool twice or drain the
  list to empty. Non-schema exceptions (rate limits, network) still
  fall through directly. The shared parser lives at
  `kaos_agents/patterns/_tool_schema.py` and is exhaustively
  tested against a verbatim openai:gpt-5.5 400 payload.

### Added
- **PA5: `AgentChatTool` auto-hydrates VFS artifact references from the user
  message.** Upstream tools (e.g. `kaos-pdf-parse`) return manifest URIs
  like `kaos://artifacts/<id>`; the natural follow-up message "what's in
  that doc?" now triggers a VFS read of the artifact body into
  `SessionMemory.DOCUMENTS` before the chat pattern dispatches. The new
  `kaos_agents.runtime.artifact_hydration` module scans incoming messages
  for `kaos://artifacts/<id>` URIs, `artifact://<id>` shorthand, and
  `ArtifactManifest` JSON blobs; resolves them via `runtime.artifacts`;
  and injects bodies respecting the standard inline-threshold tiers
  (inline < 16 KB, summary < 256 KB, handle-only above). Already-hydrated
  artifacts are detected via `MemoryItem.metadata["hydrated_artifact_id"]`
  and skipped on subsequent turns. Hydration is best-effort: any failure
  is logged at WARNING and the turn proceeds. The chat tool's
  `structuredContent` payload now carries an `hydrated_artifacts: list[…]`
  field when hydration fired, so observers can see which references were
  picked up.

### Changed
- **PA10: pytest marker definitions clarified.** Every custom marker used
  under `tests/` (`unit`, `integration`, `live`, `network`, `slow`,
  `benchmark`) was already registered in
  `[tool.pytest.ini_options].markers` and `--strict-markers` is enabled,
  so collection has been clean of `PytestUnknownMarkWarning` since the
  initial OSS release. This update sharpens the marker descriptions
  (e.g. `live` requires real API keys vs. `network` which only requires
  outbound HTTP) and adds a comment block documenting the single source
  of truth.

## [0.1.0a1] — 2026-05-13

First public alpha release.

### Changed
- Viewer JSON renderer upgraded to an interactive tree: syntax
  highlighting, click-to-expand/collapse at every depth, copy-path
  on hover, long-string folding, redaction-aware badges, theme-
  aware colors. Replaces the previous `<pre>` JSON dump. Renders
  ~100 KB payloads without UI lag via lazy below-default-depth
  nodes (KC18-B).
- Quickstart replaced: README now demonstrates the package's actual
  value prop (FindingsAgent reviewing 5 mutual NDAs with provenance,
  cost cap, refusal contract, and audit trail) instead of a one-shot
  LLM call. Quickstart loads real curated NDAs from
  `kaos_agents.examples.nda_review.ndas` via importlib.resources, so
  it works from any pip-install. Live integration test enforces the
  example can't drift from reality (KC17-P0-6).
- `SECURITY.md` and `CONTRIBUTING.md` rewritten for the actual
  kaos-agents surface — HTTP API auth (P0-3), tool approvals (P0-2),
  memory deletion (P1-1), prompt-injection envelope, recorder
  retention, cost caps. Replaces verbatim copies of kaos-web's
  browser-tooling docs that didn't apply (KC17-P1-2).
- **`SessionMemory.sections` is now a public read-only property (KC17-P2-4).** The HTTP API
  in `kaos_agents/api/server.py` previously read `memory._sections` — a leading-underscore,
  `__slots__`-private attribute — to enumerate configured section types in the wire payload.
  Any future memory-layout change would have silently broken the public surface. The new
  `memory.sections` property returns a defensively-copied `tuple[MemoryType, ...]` keyed in
  configuration order. The HTTP API now uses it; the private `_sections` attribute remains
  for internal mutation only.

### Fixed
- **Atomic `SessionStore.save` survives SIGTERM mid-save (KC17-P1-3).** Pre-KC17 `save()`
  wrote `memory.json` and `graph.ttl` as two non-atomic `vfs.write()` calls. A SIGTERM
  between them left a torn on-disk state that the next `load()` consumed as corrupt JSON.
  Both writes now route through `_atomic_write`: temp+fsync+`os.replace` on disk-backed VFS
  (POSIX-atomic on the same filesystem), with a best-effort directory `fsync` for Linux
  durability. Non-disk backends (memory) fall back to direct write — torn states aren't
  reachable for in-process bytes.
- **DELETE session + memory-clear actually remove persisted memory (KC17-P1-1).** The HTTP
  API's `DELETE /v1/sessions/{id}` and the MCP `kaos-agent-memory-clear` tool previously called
  `vfs.cleanup_context(session_id)` only — leaving
  `kaos-agents/sessions/{id}/memory.json` (and `graph.ttl`) on disk. After a successful
  DELETE, `SessionStore.exists()` stayed True and a follow-up `GET /v1/sessions/{id}` returned
  200. Both paths now call `SessionStore.delete(session_id)` which sweeps `memory.json` AND
  `graph.ttl` (and any future per-session siblings), then call `cleanup_context` for VFS
  scratch (run state, artifacts) — privacy / right-to-delete now matches the contract.

### Security
- **HTTP API auth + tenant scoping + CORS hardening (KC17-P0-3).** The FastAPI surface previously
  shipped with NO auth, NO tenant scoping, and CORS wildcard + credentials. POST
  `/v1/runs/{run_id}/approve` was a human-in-the-loop bypass for anyone who could reach the port.
  - `create_app()` refuses to start unless `KAOS_AGENTS_API_TOKEN` is set OR
    `KAOS_AGENTS_API_ALLOW_UNAUTH_LOCALHOST=1`. Pre-KC17 the API would happily run on `0.0.0.0`
    with no token.
  - Bearer-token auth via `Authorization: Bearer <KAOS_AGENTS_API_TOKEN>` with constant-time
    compare. Wrong token → 401.
  - Tenant scoping: sessions are namespaced by SHA-256(token)[:12]. Token A's session is 404 (not
    403) to token B — explicit "no existence leak across tenants" contract.
  - Localhost-dev mode (`KAOS_AGENTS_API_ALLOW_UNAUTH_LOCALHOST=1`) permits unauthenticated
    requests from 127.0.0.1 / ::1 only; emits a per-request warning log.
  - CORS default is `[]` (no cross-origin). Explicit origin list via
    `KAOS_AGENTS_API_CORS_ALLOW_ORIGINS` (comma-separated). Wildcard `*` with credentials is
    rejected at config time (the W3C CORS spec forbids it; Starlette permits but browsers reject).
- **Default-deny destructive tool approvals (KC17-P0-2).** `Runner` and `tool_bridge` previously
  treated `permission_policy=None` as "skip all checks" — meaning an HTTP API or MCP caller could
  invoke a tool annotated `destructiveHint=True` with no approval gate. Now `None` installs
  `PermissionPolicy.default_safe()` which escalates destructive / `humanConfirmationRequired` tools
  to ASK. Tests, internal benchmarks, and other callers that genuinely need to bypass all checks
  must set `Runner(unsafe_bypass=True)` explicitly. Production deployments MUST NOT use the bypass.
- **XML-escape candidate text in the FindingsAgent injection envelope (KC17-P2-3).** The renderer
  for both filter and synthesis stages now passes `cand.text` through `xml.sax.saxutils.escape`
  before interpolating it into the `<untrusted_document_content>` envelope, so a candidate
  containing a literal `</untrusted_document_content>` tag can no longer close its own envelope
  from inside. Defense-in-depth (heuristic detector + signature directive) was already in place;
  this fix removes a structural-integrity gap.

### Added
- `kaos_agents/examples/viewer/` — single-page HTML viewer for the
  recorder JSONL telemetry. Tailwind + Alpine, no build step. Drag-
  drop a `.jsonl` file to inspect every LLM call with summary stats,
  filterable / sortable table, side-by-side inputs/outputs detail
  panel with markdown render, and group-by-trace_id view. Launch
  via `python -m kaos_agents.examples.viewer` (KC18).
- `kaos_agents/examples/nda_review/hello.py` — Hello-World "easy
  version" of the NDA review: defaults-only, asks for a markdown
  summary table across the 5 NDAs via `ResearchAgent.turn()`. ~$0.10
  on `claude-haiku-4-5`. Best first-impression demo; the README
  quickstart now leads with this. The senior-counsel version
  (recall-first per-doc `FindingsAgent` with provenance + cost cap
  + refusal contract + audit trail) remains at `quickstart.py`
  (KC17-P0-6b). Live regression at
  `tests/integration/test_hello_nda_review_live.py`.
- `research_profile = "strict"` setting (env: `KAOS_AGENT_RESEARCH_PROFILE`)
  for legal / regulated deployments. Raises BM25 score floor, verifier
  confidence threshold, and refuses unverified answers via a typed
  `InsufficientEvidence` collapse instead of warn-and-return. Default profile
  behavior unchanged (KC17-P2-1).
- KaosRuntime VFS isolation: `KaosRuntime(vfs=...)` kwarg + `KaosRuntime.test_mode(in_memory=True)` classmethod
  + `runtime.artifacts` as `cached_property`. Closes the disk-VFS cross-run leakage footgun in live tests.
  Live composition tests are now isolated by default. (Sprint-1 #1, commit d0ba060.)
- Auth/rate-limit/transport failures surface as `isError=True` with credential-named recovery hints
  via `kaos_agents.errors.classify_agent_failure()`. (Sprint-1 #2, commit dee0c9a.) Closes
  skeptic-prod-ops Probe 4b.
- Three-layer OWASP LLM01 defense for `FindingsAgent`: pre-flight heuristic flag, XML isolation
  envelope around all candidate text, defensive signature docstring. Plus live test against Sonnet 4.6
  including a synthesis-targeted payload variant. (Sprint-1 #3, commit fb82f64.) Closes skeptic-prod-ops Probe 1.
- `FindingsRefusal` structured value type with three stable refusal reasons
  (`no_candidates_enumerated`, `no_relevant_candidates`, `budget_exceeded`). Refusal surfaces via
  `FindingsResult.refusal` and `AgentFindingsTool.structuredContent["refusal_reason"]`. (Sprint-1 #4,
  commit 916cb67.) Closes skeptic-trust Probe 2 empty-answer UX bug.
- `FindingsAgent.temperature=0.0` by default; deterministic finding_ids via SHA256(block_ref,
  char_span, normalized_text)[:12]; `runs >= 2` union mode for multi-run consistency. 5-run Jaccard
  rises from 0.84-0.92 (skeptic-trust baseline) to 0.955-1.000 on Anthropic Haiku 4.5. (Sprint-2 #5,
  commit f752ecf.) Closes skeptic-trust Probe 1 (consistency).
- `select_by="semantic"` selector with LLM-driven query rewrite + 8-term sanitized expansion union;
  low-recall warning on token selector when < 5 candidates for >= 6-word question. (Sprint-2 #6, commit
  0ffb020.)
- `FindingsAgent.max_cost_usd` strict wave-level cap (Phase-2 filter + Phase-3 synthesis); honest
  `budget_exceeded` reporting across `AgentChatTool` (soft, 2x overshoot bound), `AgentPlanTool`
  (strict per-step), `AgentFindingsTool` (strict wave-level), `AgentCorpusFilterTool` (post-hoc).
  (Sprint-3 #9, commit 21463ba.) Closes skeptic-prod-ops Probe 2.
- `AgentResponse.cost_usd` + `AgentResponse.total_tokens` as first-class frozen attributes. Same
  numbers ship as `ToolResult.structuredContent["cost_usd"]` and `["total_tokens"]` across all four
  agent tools. (Sprint-3 #10, commit a338d1e.)
- Property-style test asserting the three event→AgentResponse
  drain paths (Runner.turn / agent.py / events_to_response.py)
  produce identical normalized output, closing the unenforced
  "must agree" invariant before consolidation lands in 0.1.0a2
  (KC17-P1-4, PA14).

### Fixed
- Base install now imports cleanly without optional extras. `kaos_agents.Actor` and the rest of the
  `kaos_llm_core`-dependent public surface (`Runner`, `BaseAgent`, `FindingsAgent`, `ReflexionLoop`,
  `RouterAgent`, `SessionMemory`, `ChatAgent`, `PlanExecuteAgent`, `ResearchAgent`, `Perceiver`,
  `IntentExtractor`, `TerminationJudge`, …) plus `kaos_agents.api.create_app` (FastAPI-dependent)
  are now lazily resolved via PEP 562 `__getattr__`. Consumers without `[llm]` / `[api]` extras
  still `import kaos_agents` successfully and can use the always-on surface (`Agent`, `AgentPattern`,
  `KaosAgentSettings`, `KaosEvent`, `PermissionPolicy`, the trigger types, the event serdes, …);
  they only hit a clear install-hint `ImportError` when they actually touch an optional name.
  Closes KC17-P0-1.
- Package root re-exports the three pattern classes the README markets but `__init__.py` previously
  hid behind submodules: `FindingsAgent`, `ReflexionLoop`, `RouterAgent` are now importable from
  `kaos_agents` directly. Closes KC17-P0-5.
- `tests/integration/test_mcp_extract_live.py` now carries `pytestmark = pytest.mark.live` so the
  default `pytest -m "not live and not network and not slow"` run no longer spends real Anthropic
  tokens on every CI invocation. Recipe-name assertion bumped from `court-opinion-v1` to
  `court-opinion-v2` to match the shipping recipe schema id. Closes KC17-P2-2.
- sdist no longer ships unredacted telemetry recordings from
  `tests/integration/runs/` or privileged-marker benchmark JSONs
  from `docs/benchmarks/`. The 9 Harvey-Lab raw JSONs (which
  contain LLM-generated deliverable text with "PRIVILEGED AND
  CONFIDENTIAL — ATTORNEY-CLIENT COMMUNICATION / ATTORNEY WORK
  PRODUCT" boilerplate) moved to a gitignored
  `docs/benchmarks/_private/`; their public pass-rate /
  cost summary remains in
  `harvey-coc-pipeline-comparison-2026-05-06.md`. Multiformat
  benchmark `corpus_dir` paths rewritten to repo-relative. New
  `scripts/check_sdist.py` gate fails any future release that
  regresses; CI release jobs should call it after `uv build`.
  Sdist drops from 17 MB / 752 files to 2.0 MB / 564 files.
  Closes KC17-P0-4.

### Documentation
- Per-file fixture provenance manifests added to every leaf data directory under `tests/fixtures/`:
  `harvey-lab/<task>/`, `harvey-lab/<task>/documents/`, and `images/` now each carry a
  source-URL + license + retrieved + SHA-256 table per file, satisfying
  `docs/oss/50-data-and-fixtures/provenance-policy.md:16`. Closes KC17-P1-5.

### Changed
- Streaming recorder JSONL schema bumped to v3: header line written + fsync'd on `__aenter__`,
  per-invocation lines streamed + fsync'd during run, optional trailer at exit. Audit trail
  now survives SIGTERM and pod eviction. (Sprint-3 #8, commit b8f5998.) Closes skeptic-prod-ops Probe 4c.
- `parse_html` default `pre_content_mode='prose'` (was `'code'`); K3 SentencesWith* tools emit a
  shape-mismatch warning when paragraphs are sparse and `<pre>` blocks dominate. (Sprint-2 #7,
  commit fe73833.) Federal Register / news / Wikipedia / web-search HTML pipelines no longer
  silently produce zero entity hits.
- Bumped `kaos-llm-core>=0.1.0a4` dependency for the `gpt-5.5` pricing entry parity fix (KC16-2).

### Security
- Test capture JSONLs no longer committed to the public repo; `.gitignore` covers
  `tests/integration/runs/*.jsonl`. Production users running the recorder in regulated environments
  MUST point output at encrypted-at-rest storage (KaosVFS with encryption, S3 SSE-KMS, etc.) — see
  README "Known limitations" for the data-plane discussion. (KC16-4.)

### Known Limitations

This is an honest list, not a buried disclaimer. v0.1.0a1 is an alpha;
the items below are tracked work that did not block release but a
regulated-industry adopter must know about.

- **OpenAI reasoning models (gpt-5.5, o3, o4, …) are not supported** for findings-based extraction
  in v0.1.0a1. `FindingsAgent` sends `temperature=0` unconditionally and these models reject it
  with HTTP 400. Cost accounting on gpt-5.5 also reports `$0` despite real billing (pending kaos-llm-core
  0.1.0a4). Use Anthropic Haiku 4.5 / Sonnet 4.6 or OpenAI `gpt-5.4-mini` instead. Tracked as PA16
  for v0.1.0a2. (KC16-2, KC16-3.)
- **Chat-path cost cap is honest but soft.** `AgentChatTool(max_cost_usd=X)` may overshoot the cap
  by up to 2x in a single turn (one classify + one ReAct iteration). `budget_exceeded` flag is
  truthful. For strict per-call caps use `kaos-agent-findings` (wave-level) or `kaos-agent-plan`
  (per-step). Tracked as PA13 for v0.1.0a2. (KC16-6.)
- **`ResearchAgent` / RAG path has no cost cap.** Tracked as PA11 for v0.1.0a2. (KC16-5.)
- **Findings consistency on `openai:gpt-5.4-mini` is ~0.75 Jaccard** (vs the 0.95 Anthropic floor).
  Two associates running the same query may see materially different surviving sets. Use the
  `runs >= 2` union mode on this provider for audit-grade work. (KC16-7.)
- **`anthropic:claude-sonnet-4-6` consistency typically holds at 0.92-0.96** but PA15 observed one
  0.621 outlier across three runs. Anthropic does not advertise `temperature=0` as bit-deterministic.
  For audit-grade extraction prefer Haiku or use `runs >= 2`. (KC16-12.)
- **Cross-provider coverage in v0.1.0a1 is limited to 3 verified rows.** Anthropic Haiku 4.5,
  Anthropic Sonnet 4.6, OpenAI gpt-5.4-mini — all green. OpenAI gpt-5.5 — RED (see above). Google,
  xAI, Groq, Mistral, OpenRouter — UNVERIFIED for v0.1.0a1 against the Sprint 1-3 contracts.
  Tracked as PA15 follow-ups. (KC16-15.)
- **Audit-trail JSONL captures persist full document bodies, conversation context, and
  agent-generated content** to disk. In regulated deployments (SOC2 / HIPAA / FINRA / GLBA) these
  files are subject to the same retention, encryption-at-rest, and access-control requirements as
  the source documents themselves. The recorder writes to a `Path` you supply — do NOT point it at
  unencrypted disk in production. The recorder also only captures calls routed through
  `kaos-llm-core`; direct provider SDK calls in user-supplied tools are invisible to the trail.
  (KC16-4, KC16-13.)
- **Persistence model: disk-first by default.** `KaosRuntime()` uses a disk-backed VFS rooted at
  `.kaos-vfs/`. Session memory persists across container restarts. For stateless / per-request
  deployments use `KaosRuntime.test_mode()` (in-memory + `IsolationMode.GLOBAL`). For multi-tenant
  deployments, scope the VFS root per-tenant. (KC16-21.)
- **`FindingsAgent.max_chunks` / `max_candidates` ceilings** were added in v0.1.0a1 to defend
  against accidental `select_by='every_sentence'` calls on giant corpora (default 200 chunks, 5000
  candidates). The cost cap is the primary defense; these are belt-and-suspenders. Lift them
  explicitly when you have a known-bounded large-corpus job. (KC16-9.)
- **K5 summary-aware triage and raw BM25 are different rankers, not equivalents.** At n>=16 docs
  they share <70% of their top-5 results; at n=64 they share ~10%. Treat K5 as a complementary
  signal, not a drop-in BM25 replacement. (KC16-14.)

### Removed
- `License :: ...` Trove classifier (PEP 639 supersedes; `license = "Apache-2.0"` is now the
  canonical declaration).

[Unreleased]: https://github.com/273v/kaos-agents/compare/v0.1.0a1...HEAD
[0.1.0a1]: https://github.com/273v/kaos-agents/releases/tag/v0.1.0a1
