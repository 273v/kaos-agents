# Contributing

Thank you for contributing. Keep changes focused, tested, signed off,
and documented. Participation in this project is governed by the
[project conduct expectations](CODE_OF_CONDUCT.md).

## Setup

```bash
uv sync --group dev
uvx pre-commit install
```

The pre-commit hook runs the same ruff/ty checks as CI. Installing it
shortens the local feedback loop; CI remains the final gate.

`kaos-agents` requires Python 3.13 or newer. It is a pure-Python package
with import package `kaos_web` and CLI entry points `kaos-agents` and
`kaos-agents-serve`.

Optional extras enable additional capabilities:

- `[browser]` — Playwright for JS-rendered pages and the browser-interaction
  MCP tools. After `uv sync --group dev`, install browser binaries with
  `uv run playwright install chromium` (or `firefox` / `webkit`).
- `[dns]` — `dnspython` for the domain-intelligence DNS tools.
- `[mcp]` — `kaos-mcp` for serving tools over the FastMCP bridge.
- `[nlp]` — `kaos-nlp-core` for BM25 search inside extracted documents.

## Before Opening A PR

Run the local quality gate:

```bash
uv run ruff format --check kaos_web tests
uv run ruff check kaos_web tests
uv run ty check kaos_web tests
uv run pytest -m "not live and not network and not slow" --no-cov
```

When packaging, metadata, README rendering, or release behavior changes,
also run:

```bash
uv build
uvx --from twine twine check --strict dist/*
```

Type checking uses `ty`, not mypy. Inline ignores use
`# ty: ignore[...]`; `# type: ignore[...]` is mypy syntax and is not a
substitute for a `ty` ignore.

## Standards

Read the standards before making non-trivial changes:

- [Python design and architecture](docs/standards/python-design-and-architecture.md)
- [Code quality standards](docs/standards/code-quality-standards.md)
- [Engineering process](docs/standards/engineering-process.md)
- [Tests, fixtures, and CI](docs/standards/tests-fixtures-ci.md)

## Pull Requests

Pull requests should explain:

- what changed
- why it changed
- how it was tested
- whether public API, CLI behavior, MCP tool surface, package metadata,
  fixtures, or release artifacts changed
- whether `CHANGELOG.md` needs an `[Unreleased]` entry

Bug fixes need regression tests. User-visible behavior changes need docs
and a CHANGELOG entry under `[Unreleased]`.

Before requesting review, confirm:

- [ ] One logical change per PR.
- [ ] Branch rebased on `main`.
- [ ] Tests added or updated when behavior changes.
- [ ] Local quality gate run.
- [ ] Public API, CLI, MCP tool surface, package metadata, fixtures, and
      release impact considered.
- [ ] DCO sign-off on every commit (`git commit -s`).

## Testing Standards

- New public API needs a test through its real entry point.
- Mocked-only tests are not enough for security-sensitive behavior.
- URL validation, robots.txt handling, retry/rate-limit policy, browser
  context isolation, response size caps, TLS-verification toggles, and
  WHOIS/DNS parsing must test both accepted and rejected cases with
  realistic inputs.
- **Any new outbound fetch site** (HTTP, browser, raw socket, datagram
  endpoint, …) MUST call `kaos_web.security.validate_url(url)` (or
  `validate_host(host)` for host-only inputs) at the boundary BEFORE
  any I/O, and ship a regression test that proves the gate fires for
  a private-IP target. The gate is the single defence against a
  misconfigured caller (typically the HTTP-mode MCP server fronting
  multiple agents) reaching cloud metadata, loopback, or RFC1918
  internal hosts. See `kaos_web/security.py` and the per-site
  `TestUrlPolicyGate` regression classes for the wiring pattern.
- Tests that hit live network must be marked `integration`, `network`,
  or `live` so CI's unit gate (`-m "not live and not network and not
  slow"`) does not collect them.
- Browser tests must not launch a real Chromium process from the unit
  tier — `tests/unit/conftest.py` blocks Playwright launches via an
  autouse fixture; new tests should reuse it rather than add another.

## Issues

Open issues using the templates in
[`.github/ISSUE_TEMPLATE/`](.github/ISSUE_TEMPLATE/). Bug reports should
include the `kaos-agents` version, Python version, operating system,
installed extras, a minimal reproducer, expected behavior, and actual
behavior. Do not file public issues for security reports — follow
[SECURITY.md](SECURITY.md) instead.

## Commits

Use conventional commit style and sign commits with `git commit -s` for
the Developer Certificate of Origin:

```text
feat: add new capability
fix: correct broken behavior
docs: update examples
ci: adjust workflow
chore: refresh tooling
```

## Changelog

Update `CHANGELOG.md` for user-visible changes, including public API,
CLI behavior, MCP tool surface, schema output, package metadata,
security behavior, and deprecations.

## Security

Do not report suspected vulnerabilities in public issues. Follow
[SECURITY.md](SECURITY.md).

## License

By contributing, you agree that your contributions are licensed under
the [Apache License 2.0](LICENSE). The DCO sign-off (`-s`) on each
commit is your attestation that you have the right to license the work
under that license.
