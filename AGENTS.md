# AGENTS.md

Repository-local guidance for coding agents working on `kaos-agents`.
This file is the canonical cross-tool instruction file for this
repository.

## Scope

- Follow this file for all automated coding-agent work in this
  repository.
- Keep changes focused and public-repository appropriate.
- Preserve user changes already present in the worktree.
- For contributor process, use [CONTRIBUTING.md](CONTRIBUTING.md).
- For Claude Code-specific package notes, use [CLAUDE.md](CLAUDE.md).
- For detailed engineering rules, use:
  - [Python design and architecture](docs/standards/python-design-and-architecture.md)
  - [Code quality standards](docs/standards/code-quality-standards.md)
  - [Engineering process](docs/standards/engineering-process.md)
  - [Tests, fixtures, and CI](docs/standards/tests-fixtures-ci.md)

## Project Identity

- Distribution: `kaos-agents`.
- Import package: `kaos_agents`.
- Runtime: Python 3.13+ (3.14 in CI).
- Package type: pure Python, typed, Apache-2.0 licensed.
- CLI entry points: `kaos-agent`, `kaos-extract`, `kaos-agents-serve`.
- Core purpose: agentic runtime for KAOS — six patterns (Chat,
  PlanExecute, Research, Findings, Reflexion, Router), 14 MCP tools,
  `SessionMemory` with 14 sections + per-session RDF graph, a
  streaming audit recorder, a FastAPI HTTP surface, and an interactive
  single-page HTML viewer for replaying captured runs.
- Public contracts include `kaos_agents.__all__`, the three CLI
  surfaces and their `--json` output, the 14 MCP tool names and
  schemas, the FastAPI route shapes (`POST /v1/sessions/{id}/messages`
  and friends), the SSE / JSONL / WebSocket wire serializers, the 15
  `KaosEvent` subclasses, the recorder schema-v4 JSONL contract, and
  the `KAOS_AGENT_*` / `KAOS_LLM_CORE_RECORDER_*` environment
  variables.

For deeper architectural detail (the 8-step turn loop, retrieval as a
delegated sub-agent, K-series surfaces, recipe library), see
[CLAUDE.md](CLAUDE.md) and the README's pattern + MCP tool tables.

## Setup

Use `uv` for local environments, dependency resolution, builds, and
tool execution.

```bash
uv sync --group dev
uvx pre-commit install
```

Public extras are optional and must stay lazy:

- `llm` for LLM transport via `kaos-llm-client` + typed Programs via
  `kaos-llm-core`. Required to make any `.turn()` call against a real
  provider.
- `mcp` for the FastMCP bridge (consumed by `kaos-agents-serve`).
- `api` for the FastAPI HTTP surface.
- `otel` for the OpenTelemetry export hook.
- `rerank` for cross-encoder rerank + dense embeddings.
- Per-tool-module extras (`pdf`, `office`, `source`, `web`,
  `citations`, `tabular`) — each maps 1:1 to a `kaos-agents-serve
  --with-X` flag.

## Local Checks

Run the focused quality gate before handing off code changes:

```bash
uv run ruff format --check kaos_agents tests
uv run ruff check kaos_agents tests
uv run ty check kaos_agents tests
uv run pytest -m "not live and not network and not slow" --no-cov
```

Use `ty`, not mypy. Inline type suppressions use `# ty: ignore[...]`
with the narrowest practical rule.

When packaging, release metadata, README rendering, or build behavior
changes, also run:

```bash
uv build
uvx --from twine twine check --strict dist/*
```

For docs-only changes, run at least `git diff --check` and a practical
Markdown / link sanity check.

## CI Gates

The protected `main` branch enforces six required status checks:

- `Lint` — ruff format + ruff check + ty check.
- `Pre-commit hooks` — the same hook set as local pre-commit.
- `Test (linux-x64 / Python 3.13)`.
- `Test (linux-x64 / Python 3.14)`.
- `Test against minimum dependencies` (`min-deps`).
- `Build distribution + smoke test` (wheel + sdist + twine `--strict`
  + clean-venv import smoke).

A scheduled OpenSSF Scorecard workflow (`scorecard.yml`) and a live
integration job (`Live integration tests`) run on demand or against
keys provided in the protected environment — neither is a required PR
check.

## Live Tests

Live-tier tests (`pytest -m live`) hit real provider APIs and incur
real spend. The full live suite runs against Anthropic Haiku 4.5 /
Sonnet 4.6 and OpenAI `gpt-5.4-mini`; a full sweep costs roughly
$0.50 of provider spend.

- Required env: `ANTHROPIC_API_KEY`. Optional: `OPENAI_API_KEY`.
- Never run the live tier without explicit user approval — it is
  cost-incurring and rate-limited.
- The unit gate is the default; live is opt-in via the `-m live`
  marker.

## Permission Policy

`kaos-agents` ships a default-deny posture on destructive tool calls
(KC17-P0-2). `PermissionPolicy` enforces:

- `readOnlyHint=True` tools auto-approve.
- `destructiveHint=True` tools require explicit approval — the runner
  raises `ToolCallApprovalRequired` and pauses durably via the
  `RunState` machinery until the caller resolves it.
- The default policy denies `kaos-agent-memory-clear` outright.
- Per-session `SessionToolSet` rules can narrow the allowed surface
  further (Track 4 T4-5).

Do not bypass `permission_policy` in tests or examples without an
explicit threat-model justification. If you need a permissive policy
for a fixture, instantiate one inline rather than mutating the
package-level defaults.

## Cost Cap

Cost-cap enforcement is a contract, not a hope (KC17-P0-3, Sprint-3
#9). Every cost-bearing pattern surfaces `cost_usd` and `total_tokens`
on the typed `AgentResponse` and at the top of
`ToolResult.structuredContent`.

Enforcement granularity per tool:

- `kaos-agent-plan` — strict per-step.
- `kaos-agent-findings` — strict per-wave; aborts before next chunk
  dispatch.
- `kaos-agent-corpus-filter` — post-hoc single call.
- `kaos-agent-chat` — soft post-turn (bounded by one classify + one
  ReAct iteration).
- `kaos-agent-research` (RAG) — **not wired yet** (PA11 follow-up).
  Scope deployments to the strict-cap tools if you need a hard
  ceiling.

When you breach a cap, surface `budget_exceeded=True` on the response;
do not silently truncate.

## Refusal Contract

`FindingsAgent` surfaces a typed `FindingsRefusal` with one of five
stable reasons:

- `budget_exceeded` — the wave-level cap fired.
- `no_candidates` — the selector produced nothing.
- `all_filtered_out` — every candidate fell below the relevance
  threshold.
- `insufficient_evidence` — synthesis could not assemble a citable
  answer.
- `permission_denied` — a downstream tool call was refused.

Agents must surface refusals to the caller, not hallucinate an answer
or swallow them. When extending the pattern set, follow the same
typed-refusal convention rather than returning empty strings or `None`.

## Audit Recorder

Every LLM call routed through `kaos-llm-core` is captured by the F2
streaming recorder when `KAOS_LLM_CORE_RECORDER_DIR` is set. Schema-v4
(KC16-4) writes:

- A header line at `__aenter__` (with `fsync()`).
- One JSONL line per LLM invocation, streamed and `fsync()`-flushed
  per line so the trail survives `SIGTERM` / pod eviction / OOM-kill.
- An optional trailer at exit.

Field-level redaction is on by default at 2048 chars. Document bodies,
conversation context, candidates, and instructions are replaced with
`<redacted:N-chars>` sentinels before the line hits disk. Set
`KAOS_AGENT_RECORDER_REDACT=0` only for synthetic / public-domain
fixtures during development.

In production: leave redaction on, or point the recorder at
encrypted-at-rest storage (KaosVFS-with-encryption, S3 SSE-KMS). The
captured JSONLs become a secondary data plane subject to the same
retention / encryption / access-control requirements as the source
documents.

Coverage gap to be aware of: user-supplied tools that call
`anthropic.Anthropic()` or `openai.OpenAI()` directly bypass the
recorder (KC16-13). Route LLM calls through `kaos-llm-core` to keep
the trail complete.

## Tools You Use

- `uv` — environments, dependency resolution, builds, publishing
  commands. Never use `pip` directly against the project venv.
- `ruff format` and `ruff check` — formatting + linting.
- `ty check` — type checking. `ty`, not mypy.
- `pytest` — tests. Use markers (`-m unit`, `-m live`, etc.) to scope
  runs.
- `pre-commit` — local hook runner; pinned to match the dev group.
- `gh` — GitHub API calls, PR creation, branch protection inspection.

## Documentation Pointers

- [README.md](README.md) — install, quickstart, patterns, MCP tools,
  known limitations.
- [CLAUDE.md](CLAUDE.md) — architecture deep dive, 8-step turn loop,
  K-series surfaces, recipe library, isolation patterns for live tests.
- [CONTRIBUTING.md](CONTRIBUTING.md) — contributor process, DCO, PR
  expectations.
- [SECURITY.md](SECURITY.md) — vulnerability disclosure policy.
- [docs/standards/](docs/standards/) — cross-cutting OSS standards.
- [docs/design/](docs/design/) — design notes and audit findings.

## Never Do

- Bypass branch protection on `main` without explicit admin approval —
  the per-module repo runs with `enforce_admins=true` for a reason.
- Commit secrets, tokens, API keys, `.env` files, or unredacted
  audit-trail captures. The KC17-P0-4 sdist exclude list is the
  second wall; do not weaken it.
- Mix style-only changes with behavior changes.
- Weaken the KC17-P0-2 default-deny permission policy.
- Disable cost-cap enforcement to "make a test pass." Fix the test or
  the cap, not the gate.
- Modify the per-module `273v/kaos-agents` repo from a non-feature
  branch — every change lands via PR, even when admin-merge is needed
  for an urgent fix.
- Add new runtime dependencies without checking the
  optional-extras boundary. Anything LLM / MCP / FastAPI / OTel /
  rerank / sibling-tool-bearing belongs behind an extra, not in the
  base install.
- Move public tags or force-push shared branches.

## Commits, PRs, And Releases

- Use conventional commit style and sign commits with `git commit -s`.
- Keep docs-only, code, tests, packaging, and release changes
  separated when possible.
- PRs should state what changed, why, how it was tested, and whether
  public API, CLI, MCP tool surface, FastAPI route shape, recorder
  schema, package metadata, or fixtures changed.
- User-visible behavior changes need a `CHANGELOG.md` entry.
- Releases require green formatting, linting, typing, tests, build,
  strict metadata check, and a fresh install smoke test as described
  in the standards.
