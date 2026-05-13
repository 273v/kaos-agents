# Python Design And Architecture Standards

These standards apply to Python code in `kaos-agents`.

`kaos-agents` is a pure-Python package. It publishes the `kaos_agents`
import package and the `kaos-agent`, `kaos-extract`, and
`kaos-agents-serve` CLI entry points.

## Package Shape

- Keep the import package name aligned with the distribution name:
  `kaos-agents` publishes import package `kaos_agents`.
- Declare the public API in `kaos_agents.__all__`. The six agent
  patterns (`ChatAgent`, `PlanExecuteAgent`, `ResearchAgent`,
  `FindingsAgent`, `ReflexionLoop`, `RouterAgent`), `Runner`,
  `SessionMemory` / `SessionStore`, `AgentResponse`, the runtime
  primitives (`KaosAgent`, `KaosPattern`, `KaosHook`, `KaosEvent`),
  and the tool registration entry points
  (`register_agent_tools`, `register_extract_tools`,
  `register_graph_tools`) are part of that surface.
- Keep `kaos_agents/py.typed` in the wheel.
- Keep import-time work minimal: no network calls, filesystem scans,
  LLM provider initialization, runtime construction, logging setup, or
  expensive model loads at import time.
- Use absolute imports for package code.
- Keep base dependencies small: `kaos-core`, `kaos-content`,
  `kaos-graph`, `kaos-nlp-core`, `pydantic`, `pydantic-settings`.
  Capabilities that drag heavier or specialised deps belong behind
  extras (`[llm]`, `[mcp]`, `[api]`, `[otel]`, `[rerank]`, and each
  per-tool-module extra).
- Prefer a small top-level package surface that re-exports stable,
  documented names only.

## Public API

Treat all of these as public API once released:

- Names exported from `kaos_agents.__all__`.
- The three `register_*_tools()` entry points and the 14 MCP tools
  they install (agent / extraction / graph groups).
- `kaos-agent`, `kaos-extract`, and `kaos-agents-serve` CLI commands,
  flags, `--json` output, and exit behavior (including the documented
  exit code `2` for cost-cap refusal).
- `AgentResponse`, `Runner`, `SessionMemory`, `SessionStore`, the six
  pattern classes, and the runtime ABCs (`KaosAgent`, `KaosPattern`,
  `KaosHook`, `KaosEvent`).
- `KaosAgentSettings`, the `KAOS_AGENT_*` environment-variable
  namespace, and the `KAOS_LLM_CORE_RECORDER_DIR` /
  `KAOS_AGENT_RECORDER_REDACT` audit-trail switches.
- The FastAPI HTTP surface (`create_app`, route shapes under
  `POST /v1/sessions/{id}/messages`, session CRUD, memory query /
  search routes) and the SSE / JSONL / WebSocket wire serializers.
- The audit recorder JSONL schema (schema-v4, KC16-4) — header,
  per-call lines, optional trailer, field-level redaction sentinels.
- The 15 `KaosEvent` subclasses and the `serialize_event` /
  `deserialize_event` registry that backs them.
- JSON Schema, MCP-compatible shapes, and `ToolResult` content
  contracts produced by every shipped tool, including the
  `cost_usd` / `total_tokens` top-level convention.

Changing or removing public API requires a changelog entry and a version
bump consistent with the package's pre-1.0 stability policy.

## Dependency Boundaries

- Keep runtime dependencies minimal and justified.
- Do not make `kaos-agents` depend on optional providers or heavy
  capabilities at import time. `kaos-llm-client` / `kaos-llm-core`
  (`[llm]`), `kaos-mcp` (`[mcp]`), `fastapi` (`[api]`),
  `opentelemetry-api` (`[otel]`), `kaos-nlp-transformers` (`[rerank]`),
  and every tool-bearing sibling must be lazy-imported behind their
  extras.
- Agents that never call an LLM must still run without the `[llm]`
  extra — the `InvocationUsage` value type is duplicated in
  `kaos_agents.types.usage` to preserve this invariant.
- Do not make tests pass by relying on undeclared transitive
  dependencies.
- Do not import between sibling extraction packages from inside
  `kaos_agents`. Tool modules surfaced through `kaos-agents-serve
  --with-X` are wired through the registration entry points, not by
  cross-importing.
- Do not use private APIs from dependencies unless the risk is
  recorded and covered by tests.

## Data Modeling

- Use Pydantic for external boundaries: configuration
  (`KaosAgentSettings`), FastAPI request / response models, MCP tool
  payloads, the `KaosEvent` family (frozen pydantic models with the
  serde registry), and CLI `--json` output.
- Use `@dataclass(frozen=True, slots=True)` for value and result
  types: `AgentResponse`, `InvocationUsage`, `FindingCandidate`,
  `FilteredFinding`, `FindingsResult`, `FindingsRefusal`, plan steps,
  permission rules, intent records.
- Keep parsing and validation at boundaries. Internal functions should
  receive typed, normalized values.
- Prefer explicit result types over loosely shaped dictionaries.
- Avoid returning ambiguous tuples from public APIs.

## Functions And Classes

- Prefer functions for stateless transformations (event serde, recipe
  formatting, span-to-OTel mapping, memory token estimation).
- Use classes when there is persistent state, lifecycle management
  (`Runner`, `SessionMemory`, `BaseAgent`), shared configuration,
  registries, hooks, or an explicit protocol (`KaosAgent`,
  `KaosPattern`, `KaosHook`).
- Keep constructors cheap. Agents are frozen config; the `Runner` is
  the execution engine that holds the runtime, context, VFS, and
  hooks.
- Avoid inheritance unless the abstraction is stable and tested through
  multiple implementations. The agent patterns are the canonical
  example — they compose, they don't subclass each other.
- Prefer protocols and small composition points (hook registration,
  pattern registration, recipe loading) over deep class hierarchies.

## Configuration

- Use `KaosAgentSettings` (a `kaos_core.ModuleSettings` subclass) for
  package configuration.
- Read environment variables at the edge (CLI, tool registration,
  agent construction), not deep in algorithmic code.
- Keep the `KAOS_AGENT_*` resolution order documented and covered by
  tests when it changes. The CLI `--max-cost` flag takes precedence
  over `KAOS_AGENT_MAX_COST_USD`; document any new precedence chains
  the same way.
- Represent secrets with `pydantic.SecretStr`.
- Do not print, log, serialize, or include secrets in exception strings.
- Preserve redaction behavior in CLI output, structured logging, and
  the audit recorder.

## Error Handling

- Use the package-specific exception hierarchy in `kaos_agents.errors`
  for user-facing failure modes.
- Tool error messages must include (1) what went wrong, (2) how to fix
  it, (3) an alternative tool when applicable. Agent-facing prompts.
- Surface refusals through the typed `FindingsRefusal` contract (five
  reasons: budget_exceeded, no_candidates, all_filtered_out,
  insufficient_evidence, permission_denied). Refusals are not errors
  and should not be swallowed.
- Do not expose stack traces, credentials, internal paths, or provider
  payloads in user-facing errors.
- Preserve original exceptions with exception chaining when debugging
  context matters.
- Validate untrusted inputs (model identifiers, tool names, recipe
  names, session IDs) early and fail with bounded, predictable errors.

## Async And Concurrency

- The agent runtime, the FastAPI surface, and the LLM transport all
  expose `async` APIs. `Runner.run()` returns
  `AsyncIterator[KaosEvent]`; `Runner.turn()` returns the aggregated
  `AgentResponse`.
- Use synchronous APIs for CPU-bound transformations (token counting,
  BM25 search) — run them under `asyncio.to_thread` when offloaded.
- Bound concurrency with `asyncio.Semaphore` when concurrent execution
  is introduced (the findings filter fan-out is the current example).
- Apply timeouts to every external call. Tool dispatch is bounded by
  `KAOS_AGENT_TOOL_TIMEOUT_SECONDS`; LLM transport timeouts live on
  `kaos-llm-client`.
- Make cancellation safe: persist memory at turn end (or `fsync` per
  recorder line), close HTTP pools, drain event streams. The KC17-P1-3
  atomic-save contract guarantees `SessionStore.save()` survives
  `SIGTERM` mid-write.

## Files, Paths, And Inputs

- Accept `str` and `PathLike` inputs where file paths are part of the
  public API (recorder output directory, CLI `--files` glob, viewer
  JSONL argument).
- Normalize paths at boundaries.
- Do not follow symlinks, traverse directories, or read arbitrary files
  unless the API explicitly permits it.
- Put cost, step-count, depth, recursion, and time limits on untrusted
  inputs. Cost ceilings (`max_cost_usd`), plan-step caps
  (`KAOS_AGENT_PLAN_MAX_STEPS`), and the
  `FindingsAgent.max_chunks` / `max_candidates` belt-and-suspenders
  defaults all live on `KaosAgentSettings` or the pattern signatures.
- Prefer streaming for large event streams and audit-trail captures.

## CLI Design

- Every `kaos-agent`, `kaos-extract`, and `kaos-agents-serve` command
  must support `--help`.
- Commands that produce machine-consumable output must support `--json`.
- JSON output must remain stable once released.
- CLI errors should be concise and actionable. Cost-cap refusal exits
  with code `2` (distinct from code `1` for real errors) so scripts
  can branch on the outcome.
- CLI examples in README and docs must be tested or manually verified
  before release.

## Documentation Expectations

- README quick starts must be runnable from a fresh environment. The
  `nda_review.hello` example loads bundled NDA fixtures via
  `importlib.resources` so `pip install 'kaos-agents[llm,office]'` is
  enough to run it.
- Examples should use public APIs only (`Runner`, the six pattern
  classes, `register_*_tools`, `KaosAgentSettings`).
- Advanced docs belong under `docs/`.
- Any advertised runtime behavior, CLI command, MCP tool, FastAPI
  route, recorder schema convention, security control, or
  permission-policy default must have at least one test at the
  appropriate tier.
