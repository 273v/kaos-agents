# Code Quality Standards

These standards define the minimum quality bar for `kaos-agents` changes.

## Baseline Tools

`kaos-agents` uses:

- `uv` for environments, dependency resolution, builds, and publishing
  commands.
- `ruff format` for formatting.
- `ruff check` for linting.
- `ty check` for type checking.
- `pytest` for tests.
- `twine check --strict` for built distribution metadata checks.

`kaos-agents` is pure Python. Rust, PyO3, `maturin`, and Cargo checks are
not part of this package's active quality gate.

## Formatting

- Formatting is automated and non-negotiable.
- Do not mix style-only rewrites with behavior changes.
- Keep generated files out of hand-edited diffs unless the generation
  step is part of the change.
- Avoid broad reformatting of files unrelated to the PR.

Local format check:

```bash
uv run ruff format --check kaos_agents tests
```

## Linting

- Lint cleanly before review.
- Do not silence a lint rule without a local reason.
- Prefer targeted ignores over file-wide ignores.
- Delete unused code instead of hiding it.
- Keep imports ordered and explicit.

Local lint check:

```bash
uv run ruff check kaos_agents tests
```

## Typing

- Public functions and methods must be typed.
- Complex internal functions should be typed.
- Avoid `Any` unless the boundary is genuinely dynamic (e.g. arbitrary
  tool-result payloads bridged from `kaos-llm-core`).
- Use `typing.Protocol` for structural extension points (`KaosHook`
  callbacks, `KaosAgent` / `KaosPattern` ABCs, the event-emitter
  surface used by the runner).
- Use `Literal`, `TypedDict`, dataclasses, or Pydantic models where they
  make external contracts clearer. `@dataclass(frozen=True, slots=True)`
  is the convention for value and result types (`AgentResponse`,
  `InvocationUsage`, `FindingCandidate`, `FilteredFinding`,
  `FindingsResult`, `FindingsRefusal`).
- Use `# ty: ignore[...]` only with the narrowest possible rule and a
  reason when the reason is not obvious.

Local type check:

```bash
uv run ty check kaos_agents tests
```

## Tests

- Bug fixes require regression tests.
- New public behavior requires tests at the right tier.
- Test names should describe behavior, not implementation.
- Prefer semantic assertions over "not empty" assertions — for
  findings tests, assert specific `block_ref` provenance and
  deterministic `finding_id` shape, not just `len(findings) > 0`.
- Avoid brittle snapshots for large payloads unless they are golden
  fixtures with a review process.
- Do not use network or live credentials in unit tests. Live-tier tests
  (`pytest -m live`) require `ANTHROPIC_API_KEY` and incur real spend;
  they must be opt-in.

Local unit and integration gate:

```bash
uv run pytest -m "not live and not network and not slow"
```

## Public API Discipline

- Public API changes need changelog entries.
- Avoid broad re-exports that make internals public accidentally.
- Deprecate before removal when the stability policy requires it.
- Keep CLI, MCP tool, FastAPI route, SSE / JSONL wire schema, audit
  recorder schema, JSON output, and `KAOS_AGENT_*` env-var contracts
  stable once released.
- Do not rename public objects for aesthetics in patch releases.

## Security Standards

- Never commit secrets, tokens, private keys, credentials, `.env`
  files, or unredacted audit-trail JSONL captures (they can contain
  verbatim LLM inputs / outputs and privileged-marker text).
- Use `pydantic.SecretStr` for API keys.
- Redact secrets in logs, CLI output, JSON output, and errors. The
  recorder's schema-v4 field-level redaction (KC16-4) is the default
  for document bodies, conversation context, candidates, and
  instructions; do not weaken it without a documented threat-model
  justification.
- Add limits for untrusted input: `max_cost_usd` per turn / per plan /
  per findings wave, `FindingsAgent.max_chunks` / `max_candidates`
  ceilings, `KAOS_AGENT_TOOL_TIMEOUT_SECONDS`, ReAct iteration caps.
- Preserve the KC17-P0-2 default-deny posture on destructive tool
  permissions. `PermissionPolicy.deny` rules ship out-of-the-box for
  `kaos-agent-memory-clear` and any tool that carries
  `destructiveHint=True`.
- Preserve cost-cap enforcement (KC17-P0-3, Sprint-3 #9) — `max_cost`
  is a contract, not a hope. Cost-bearing tools must surface
  `cost_usd` and `total_tokens` at the top of
  `ToolResult.structuredContent`.
- Do not add GPL, AGPL, unknown-license, non-commercial, or
  no-derivatives dependencies.
- Run secret scanning before release.

## Dependency Hygiene

- Keep base dependencies minimal (`kaos-core`, `kaos-content`,
  `kaos-graph`, `kaos-nlp-core`, `pydantic`, `pydantic-settings`).
- Do not promote optional integrations into the base install when they
  belong behind an extra. LLM transport (`[llm]`), the MCP bridge
  (`[mcp]`), the FastAPI surface (`[api]`), OTel (`[otel]`), the
  rerank stack (`[rerank]`), and every tool-bearing sibling
  (`[pdf]`/`[office]`/`[source]`/`[web]`/`[citations]`/`[tabular]`)
  must stay opt-in.
- Pin lower bounds intentionally and test them with the `min-deps` CI
  job.
- Do not rely on undeclared transitive dependencies.
- Prefer well-maintained packages with compatible licenses.
- Document risky or unusual dependencies.

## Documentation Quality

- README examples must run. The `kaos_agents.examples.nda_review.hello`
  quickstart is the canonical smoke surface and is covered by a live
  integration test (KC17-P0-6) so it cannot drift.
- Public functions with non-obvious behavior need docstrings.
- CLI flags and JSON output must be documented.
- Error messages should be useful without reading source — the
  three-part what / how / alternative pattern from the KAOS tool
  design guide applies to every user-facing error.
- Keep docs current with code in the same PR.

## Performance Quality

- Do not optimize without a measurement for non-trivial changes.
- Add or update benchmarks for performance-sensitive APIs (the
  `tests/benchmarks/` tier, including the BEIR cross-domain harness
  for retrieval changes).
- Watch memory growth on large corpora and long-running sessions.
- Bound expensive operations (turn count, plan step count, ReAct
  iteration count, findings wave size, recorder line size).
- Preserve streaming behavior where it is part of the design — the
  SSE / JSONL / WebSocket wire and the recorder's per-line `fsync()`
  contract.

## Definition Of Done

A change is done when:

- The implementation is complete and scoped to the stated problem.
- Tests cover the new or changed behavior.
- Formatting, linting, typing, and tests pass.
- Built distributions pass strict metadata checks when packaging is
  affected.
- Security and dependency checks pass when relevant.
- README, docs, and CHANGELOG are updated when public behavior changes.
- The PR explains what changed, why, and how it was verified.
