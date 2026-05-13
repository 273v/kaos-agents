# Tests, Fixtures, And CI Standards

This document defines test tiers, fixture rules, and GitHub Actions
standards for `kaos-agents`.

## Test Tiers

`kaos-agents` ships explicit test tiers. CI's PR gate runs only the
unit tier; the others run on demand or against scheduled live windows.

| Tier | Marker | Network | Credentials | Purpose |
|---|---|---|---|---|
| Unit | `unit` or none | No | No | Fast deterministic behavior. The runner is exercised through an in-memory VFS (`KaosRuntime.test_mode()`) and `SessionMemory` lives entirely in memory. LLM transport is stubbed. |
| Integration | `integration` | Some | No | End-to-end behavior across multiple in-process components (runner + memory + recorder + viewer) without external services. Some integration tests need network for sibling-module fetches; mark those `network` as well. |
| Network | `network` | Yes | No | Outbound HTTP without credentials. |
| Live | `live` | Yes | Yes | Real provider APIs (Anthropic, OpenAI). Requires `ANTHROPIC_API_KEY` and optionally `OPENAI_API_KEY`. Cost-incurring. |
| Slow | `slow` | Maybe | Maybe | Long-running checks (BEIR cross-domain retrieval evals, large-corpus findings sweeps). |
| Benchmark | `benchmark` | No | No | `pytest-benchmark` performance checks under `tests/benchmarks/`. |

Unit-tier CI must not require network, credentials, local services, or
large downloads.

## Test Requirements

- New behavior needs tests at the appropriate tier.
- Bug fixes need regression tests.
- Security fixes need abuse-case tests where safe (cost-cap bypass
  attempts, permission-policy bypass attempts, prompt-injection
  envelopes, recorder leakage).
- README quick starts and CLI examples need smoke coverage or manual
  verification before release. The `nda_review.hello` quickstart is
  protected by a live integration test (KC17-P0-6) so the example
  cannot drift from reality.
- Pattern behavior (Chat / PlanExecute / Research / Findings /
  Reflexion / Router), MCP tool surface, FastAPI routes, recorder
  schema, and viewer rendering need tests at the appropriate tier when
  changed.
- Tests should assert semantics, not just non-empty output. For
  findings tests, assert specific `block_ref` provenance, deterministic
  `finding_id` shape, and `cost_usd` accounting — not just
  `len(findings) > 0`.
- Tests should avoid wall-clock sleeps unless testing timeouts.

## Marker Discipline

- Integration tests must be marked `integration`.
- Live-credential tests must be marked `live`.
- Network-only tests must be marked `network`.
- Benchmark tests must be marked `benchmark`.
- Slow tests must be marked `slow`.
- New marker tiers must be registered in `pyproject.toml`
  (`[tool.pytest.ini_options].markers`).
- CI unit selection must be able to run:

```bash
uv run pytest -m "not live and not network and not slow" --no-cov
```

The command above must not collect tests that need credentials, local
services, or external network.

## Fixtures

`kaos-agents` ships fixtures under `tests/fixtures/` and
`kaos_agents/examples/nda_review/ndas/` (the latter is bundled as
package data and consumed by the README quickstart).

Highlighted fixture surfaces:

- Five real mutual NDAs (DOCX format) under
  `kaos_agents/examples/nda_review/ndas/` — drawn from 273 Ventures'
  own counterparty agreements, redistributable, used by the live
  Hello-World and quickstart.
- In-memory `SessionMemory` builders for unit tests — never persist
  to disk, never collide across tests, and decouple the runner from
  the disk-backed VFS.
- Recorded audit-trail JSONLs under
  `tests/integration/runs/INDEX.jsonl` for replay regression tests.
  Pre-KC16-4 (schema-v3) captures are **excluded from the sdist** —
  see the `[tool.hatch.build.targets.sdist] exclude` block in
  `pyproject.toml` and the KC17-P0-4 release-blocker note.
- `LegalBench` / `Atticus` golden sets referenced by the extraction
  recipes under `kaos_agents/recipes/extraction/`.

Fixtures must be:

- Small enough for normal repository use.
- Redistributable under compatible terms.
- Free of customer data, privileged content, secrets, and PII.
- Documented with source, license, and purpose.
- Stable enough to support deterministic tests.

Do not commit:

- Customer documents.
- Real credentials.
- Unknown-license data.
- Non-commercial or no-derivatives data for redistributed fixtures.
- Large binary corpora that should be downloaded and hash-verified.
- Unredacted audit-trail captures that contain verbatim LLM
  inputs / outputs or attorney-client / privileged-marker text.

## Fixture Provenance

Every fixture directory should include a README or manifest that records:

- File name.
- Source URL or generation method.
- License or public-domain status.
- Retrieval date when relevant.
- SHA256 for externally sourced files.
- Reason the fixture exists.
- Any transformations applied (anonymisation, header redaction,
  whitespace normalisation).

Generated fixtures should include the generator script or enough
description to recreate them.

## Golden Files

Golden files are allowed when output stability matters (MCP tool JSON
output, FastAPI route shapes, recorder JSONL schema, viewer rendering).

Rules:

- Keep golden files small and reviewable.
- Include a command for regenerating them.
- Review diffs semantically.
- Do not bless broad golden changes without explaining the behavior
  change.
- Store comments in a companion README when the file format cannot
  carry comments.

## Property And Consistency Tests

- The Sprint-2 #5 consistency contract (5-run pairwise Jaccard >= 0.95
  on identical query + corpus + model) is enforced by live regression
  tests on Anthropic Haiku 4.5. Drift on other providers is documented
  in the README's "Known limitations" section.
- The recorder serde round-trip property test exercises every
  `KaosEvent` subclass against the `serialize_event` /
  `deserialize_event` registry — extending the event taxonomy
  requires extending the property fixture.
- Cross-domain retrieval changes must run at least 3 BEIR datasets
  (NFCorpus, SciFact, FiQA) before shipping. Cherry-picked
  improvements on 1-2 queries are not evidence of correctness.

Python property testing:

- Prefer Hypothesis for structured inputs.
- Keep failing examples as regression tests.
- Bound generated sizes so local runs stay practical.

## Coverage

- Coverage is a signal, not the goal.
- Unit-only runs (`pytest tests/unit/ -q`) naturally land near 70%
  line coverage because integration paths are gated behind live-API
  markers. The package-wide 80% floor is enforced via
  `--cov-fail-under=80` on the combined invocation in CI, not on the
  unit subset.
- New important branches should be covered.
- Public API, error paths, security limits, refusal paths, and
  recorder serialization deserve explicit tests.
- Do not add trivial tests only to move a percentage.

## CI Workflows

Required PR checks (mirroring the protected `main` branch contexts):

- Lint (ruff format + ruff check + ty check).
- Pre-commit hooks (the same hook set that runs locally).
- Test (Linux / Python 3.13).
- Test (Linux / Python 3.14).
- Test against minimum dependencies (`min-deps`).
- Build distribution + smoke test (wheel + sdist + twine `--strict` +
  clean-venv import smoke).

Recommended scheduled or manual checks:

- Live integration tests against Anthropic Haiku 4.5 / Sonnet 4.6 and
  OpenAI `gpt-5.4-mini` when API keys are available.
- Dependency CVE audit (`pip-audit` against the locked dep set).
- Secret scanning (`gitleaks`).
- OpenSSF Scorecard (`scorecard.yml`).
- Benchmark regression check.

Release workflow checks:

- Clean checkout.
- Build pure-Python wheel and sdist.
- Strict metadata check.
- Fresh install smoke test.
- Publish through OIDC Trusted Publishing.
- Verify published install after release when practical.
- SBOM generation (CycloneDX) attached to the GitHub Release.

## GitHub Actions Standards

- Use least-privilege `permissions`.
- Do not expose secrets to forked PRs.
- Pin third-party actions to trusted versions.
- Prefer OIDC over static credentials.
- Separate build, test, security, and publish jobs.
- Cache dependencies carefully; never cache secrets.
- Keep workflow logs free of credentials and private paths.
- Use environment protection for publishing.

## Local Verification Commands

Base development setup:

```bash
uv sync --group dev
```

Fast local quality gate:

```bash
uv run ruff format --check kaos_agents tests
uv run ruff check kaos_agents tests
uv run ty check kaos_agents tests
uv run pytest -m "not live and not network and not slow" --no-cov
```

Packaging gate when packaging, metadata, README rendering, or release
behavior changes:

```bash
uv build
uvx --from twine twine check --strict dist/*
```

## Release Gate

Before release:

- Unit and integration CI are green.
- Security checks are green.
- Fixtures have provenance if fixtures were added.
- Audit-trail captures used as test fixtures are schema-v4 with
  redaction on (or excluded from the sdist).
- Build artifacts pass metadata checks.
- Fresh install smoke test passes.
- SBOM is attached to the GitHub Release.
