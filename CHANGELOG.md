# Changelog

All notable changes to `kaos-agents` are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
