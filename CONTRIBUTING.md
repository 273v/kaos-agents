# Contributing

Thank you for contributing. Keep changes focused, tested, signed off,
and documented. Participation in this project is governed by the
[project conduct expectations](CODE_OF_CONDUCT.md).

## Setup

```bash
uv sync --group dev
uvx pre-commit install
```

The pre-commit hook runs the same ruff / ty checks as CI. Installing
it shortens the local feedback loop; CI remains the final gate.

`kaos-agents` requires Python 3.13 or newer. The import package is
`kaos_agents`; CLI entry points are `kaos-agent` (chat REPL) and
`kaos-agents-serve` (MCP / HTTP server).

The base install pulls in `kaos-core`, `kaos-content[markdown,html]`,
`kaos-graph`, and `kaos-nlp-core`. The optional extras you will most
often want during development:

- `[llm]` — `kaos-llm-client` + `kaos-llm-core`. Required for real
  LLM dispatch (every pattern beyond the no-LLM smoke path).
- `[api]` — `fastapi[standard]`. Required for `create_app` and the
  HTTP API tests.
- `[mcp]` — `kaos-mcp`. Required for the `kaos-agents-serve` stdio
  and streamable-HTTP transports.
- `[otel]` — `opentelemetry-api`. For exercising the OTel hook.
- `[rerank]` — `kaos-nlp-transformers[torch]`. For dense rerank.

Per-tool-module extras (`[pdf]`, `[office]`, `[source]`, `[web]`,
`[citations]`, `[tabular]`) each enable the corresponding
`kaos-agents-serve --with-X` flag. `[full]` covers everything.

The local HTTP API refuses to start without an auth source. For
development, set `KAOS_AGENTS_API_ALLOW_UNAUTH_LOCALHOST=1` before
running `kaos-agents-serve --api` or `uv run uvicorn ...` — see
[SECURITY.md](SECURITY.md) for the contract.

## Before opening a PR

Run the local quality gate:

```bash
uv run ruff format --check kaos_agents tests
uv run ruff check kaos_agents tests
uv run ty check kaos_agents tests
uv run pytest -m "not live and not network and not slow" --no-cov
```

When packaging, metadata, README rendering, or release behavior
changes, also run:

```bash
uv build
uvx --from twine twine check --strict dist/*
python scripts/check_sdist.py    # KC17-P0-4 gate
```

`scripts/check_sdist.py` is the post-`uv build` gate that fails if
the sdist regrows past 2.0 MB or starts shipping
`tests/integration/runs/*.jsonl` or the privileged-marker benchmark
JSONs under `docs/benchmarks/_private/`. CI release jobs run it
after `uv build`.

Type checking uses `ty`, not mypy. Inline ignores use
`# ty: ignore[...]`; `# type: ignore[...]` is mypy syntax and is
not a substitute for a `ty` ignore.

## Standards

Read the standards before making non-trivial changes:

- [Code quality standards](../docs/guides/code-quality.md)
- [CLI standard](../docs/guides/cli-standard.md) — `--json`
  envelope, page numbering, error handling
- [Tool design](../docs/guides/tool-design.md) — MCP tool
  annotations, error messages, pagination, naming
- [MCP data flow](../docs/guides/mcp-data-flow.md) — artifact
  patterns, URI conventions

`CLAUDE.md` (this package) lists the SDKL checklists every change
must satisfy, plus the Rust-adjacent QA checklist for native
boundary work.

## Test categories

Tests are marked so the default unit gate stays cheap and the live
tier still runs end-to-end against real providers. Three tiers:

| Marker | When it runs | What it asserts |
|--------|--------------|-----------------|
| (unit, default) | every PR, fastest | imports, dataclass invariants, pure functions, in-process Runner with `KaosRuntime.test_mode()` |
| `network` | `--include-network` | hits the public internet (no API keys) |
| `live` | `--include-live` | hits a real LLM / search provider with credentials |

The default gate skips both:

```bash
uv run pytest -m "not live and not network and not slow" --no-cov
```

The full validator runs everything:

```bash
./scripts/validate-platform.sh --profile ubuntu-26.04 \
  --include-network --include-live
```

Live tests:

- New public API needs at least one live integration test through
  its real entry point. Mocked-only tests are documentation, not
  evidence (see `feedback_no_fake_tests.md`).
- Live tests routed through `kaos-llm-core` are captured by the
  audit-trail recorder under `tests/integration/runs/<date>/`. See
  [SECURITY.md](SECURITY.md) for the redaction / retention contract
  (KC16-4 / KC17-P0-4).
- Use `KaosRuntime.test_mode()` (in-memory VFS, `IsolationMode.GLOBAL`)
  for any test that asserts a tool call happened or that persists
  anything into `SessionMemory`. The disk-backed default VFS leaks
  session memory across runs and silently false-greens composition
  tests — see `CLAUDE.md` "Isolation patterns for live tests."

## Adding an MCP tool

Follow `docs/guides/tool-design.md`. Non-negotiable for kaos-agents:

- **Annotations are mandatory.** Set `ToolAnnotations` explicitly;
  never leave it as `None`. `readOnlyHint=True` auto-allows under
  `PermissionPolicy.default_safe()`; `destructiveHint=True` and
  `humanConfirmationRequired=True` escalate to ASK. See
  [SECURITY.md](SECURITY.md) for the full evaluation order.
- **Error messages are agent prompts.** Every error must include
  (1) what went wrong, (2) how to fix it, (3) the alternative
  tool / approach where applicable.
- **Name `kaos-agent-<verb>` or `kaos-{module}-<verb>`.** Lowercase
  hyphenated; minimum three segments.
- **Cost transparency.** Tools that drive LLM calls must surface
  `cost_usd` and `total_tokens` at the top level of
  `ToolResult.structuredContent` (Sprint-3 #10). Multi-stage tools
  (`kaos-agent-findings`) also emit per-stage breakdowns.

## Adding an agent pattern

Existing patterns: `ChatAgent`, `PlanExecuteAgent`, `ResearchAgent`,
`FindingsAgent`, `ReflexionLoop`, `RouterAgent`. They live under
`kaos_agents/patterns/`. The contract:

- Subclass `BaseAgent` (`kaos_agents/runtime/agent.py`).
- Drive the LLM through `kaos-llm-core` Programs (`Call`, `ReAct`,
  `RAG`, `Refine`). Do not call providers directly — the recorder,
  cost accounting, and retry policy all hook through
  `kaos-llm-core`.
- Honor `max_cost_usd` truthfully. If the cap fires, return a
  partial result with `budget_exceeded=True` and (for
  `FindingsAgent`-style patterns) a `FindingsRefusal` with reason
  `budget_exceeded`.
- Use `@dataclass(frozen=True, slots=True)` for every value /
  result type. Mutable dataclasses only for builders /
  accumulators.

## Pull requests

Pull requests should explain:

- what changed
- why it changed
- how it was tested (which marker tiers ran)
- whether public API, CLI behavior, MCP tool surface, package
  metadata, fixtures, or release artifacts changed
- whether `CHANGELOG.md` needs an `[Unreleased]` entry

Bug fixes need regression tests. User-visible behavior changes
need docs and a CHANGELOG entry under `[Unreleased]`.

Before requesting review, confirm:

- [ ] One logical change per PR.
- [ ] Branch rebased on `main`.
- [ ] Tests added or updated when behavior changes.
- [ ] Local quality gate run.
- [ ] Public API, CLI, MCP tool surface, package metadata,
      fixtures, and release impact considered.
- [ ] DCO sign-off on every commit (`git commit -s`).

Branch protection on `main` requires the package's 10 CI checks
(per-Python-version `ruff format` / `ruff check` / `ty check` /
unit pytest plus the base-install / build / sdist gates). All
must pass before merge.

## Security-sensitive code review

The following surfaces require two reviewers and CODEOWNERS
sign-off:

- `kaos_agents/api/` (HTTP API auth + tenant scoping)
- `kaos_agents/runtime/permissions.py` (default-safe policy)
- `kaos_agents/runtime/runner.py` (the `unsafe_bypass` escape
  hatch and the permission-policy threading)
- `kaos_agents/memory/store.py` (atomic save + deletion contract)
- `kaos_agents/patterns/findings.py` (prompt-injection envelope +
  refusal contract)
- `tests/integration/_recorder.py` (redaction threshold +
  retention policy)

If a change touches one of these and the matching live test
isn't updated, expect a request to add one before merge.

## Issues

Bug reports should include the `kaos-agents` version, Python
version, operating system, installed extras, a minimal reproducer,
expected behavior, and actual behavior. Do not file public issues
for security reports — follow [SECURITY.md](SECURITY.md) instead.

## Commits

Use conventional commit style and sign commits with `git commit -s`
for the Developer Certificate of Origin:

```text
feat: add new capability
fix: correct broken behavior
docs: update examples
ci: adjust workflow
chore: refresh tooling
```

## Changelog

Update `CHANGELOG.md` for user-visible changes, including public
API, CLI behavior, MCP tool surface, schema output, package
metadata, security behavior, and deprecations.

Entries land under `## [Unreleased]` first. The release commit
promotes the unreleased section to `## [X.Y.Za1]` and adds a fresh
empty `[Unreleased]` skeleton.

## Security

Do not report suspected vulnerabilities in public issues. Follow
[SECURITY.md](SECURITY.md).

## License

By contributing, you agree that your contributions are licensed
under the [Apache License 2.0](LICENSE). The DCO sign-off (`-s`)
on each commit is your attestation that you have the right to
license the work under that license.
