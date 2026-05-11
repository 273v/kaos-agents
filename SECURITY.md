# Security policy

## Reporting a vulnerability

We take security seriously. If you believe you have found a security
vulnerability in `kaos-agents`, please report it privately so we can address it
before public disclosure.

**Please do not file a public GitHub issue for security reports.**

### How to report

Use [GitHub Private Vulnerability Reporting](https://github.com/273v/kaos-agents/security/advisories/new)
to send a report. Alternatively, email **security@273ventures.com**.

Include as much of the following as you can:

- A description of the vulnerability and its impact
- Steps to reproduce, including affected versions
- Any proof-of-concept code, if available
- Suggested mitigations, if you have any

### What to expect

- **Acknowledgement** — within 3 business days of your report.
- **Initial triage** — within 7 business days, including a severity assessment.
- **Fix and disclosure** — coordinated with you. Our target window is 90 days
  from acknowledgement to public disclosure, faster for high-severity issues.
- **Credit** — we credit reporters in the release notes and security advisory
  unless you prefer to remain anonymous.

## Supported versions

`kaos-agents` follows Semantic Versioning. While the project is pre-1.0, only
the latest minor release receives security fixes. After 1.0, the latest two
minor releases will be supported.

| Version | Supported |
|---------|-----------|
| 0.1.x   | Yes       |
| < 0.1   | No        |

## Scope

`kaos-agents` fetches HTTP(S) content (httpx + optional Playwright), extracts
HTML into the `kaos-content` `ContentDocument` AST, runs domain-intelligence
probes (DNS / WHOIS / TLS / TCP banner / UDP / HTTP header analysis), and
exposes 45 MCP tools across four `register_*_tools(runtime)` functions.

In-scope:

- The `kaos-agents` Python package as published on PyPI
- The `273v/kaos-agents` GitHub repository (CI, release, supply chain)
- HTTP / browser-rendered content fetching — malformed responses, oversized
  bodies, redirect loops, malicious HTML / JS, hostile gzip-compressed
  sitemaps, certificate pinning, cookie handling
- Outbound URL / host policy gate (`kaos_web.security.validate_url` /
  `validate_host`) — SSRF protection against loopback, RFC1918 private
  ranges, link-local cloud metadata services (AWS/Azure/GCP IMDS), and
  non-(http|https) schemes; configurable via `KAOS_SECURITY_*` env vars
- Browser-context isolation — `BrowserClient` MUST scope all internal
  state by `(KaosContext.session_id, context_id)`. Any cross-session
  leak of cookies, captured network traffic, page interaction, or
  storage-state extraction is a vulnerability
- Captured-traffic redaction — `Authorization` / `Cookie` / `Set-Cookie` /
  `X-API-Key` / `X-Auth-Token` / `X-CSRF-Token` and pattern-matched
  auth-shaped headers MUST be masked when returned via
  `GetRequestDetailTool` / `ListCapturedResponsesTool`
- Tool boundaries (`register_web_tools`, `register_browser_tools`,
  `register_crawl_tools`, `register_domain_tools`) — input validation,
  response shaping, tool annotation correctness (`destructiveHint` on
  the 6 browser interaction tools, `readOnlyHint` on read tools)
- `SaveAuthStateTool` artifact storage path — must persist via
  `KaosContext.runtime.artifacts.create_from_path` (session-scoped),
  never via a caller-supplied filesystem path
- Cache middleware — must bypass any request bearing auth-shaped headers
  to prevent cross-caller response leak
- URL filter regex engine — caller-supplied include/exclude patterns
  must use the linear-time Rust engine (`kaos-nlp-core`) when available
  to prevent ReDoS
- OIDC trusted-publishing release pipeline (`.github/workflows/release.yml`)

Out of scope:

- Vulnerabilities in third-party dependencies — report upstream
  (`httpx`, `lxml`, `playwright`, `dnspython`, `kaos-core`,
  `kaos-content`, `kaos-mcp`, `kaos-nlp-core`).
- Vulnerabilities in browser engines themselves (Chromium / Firefox /
  WebKit) — report to the respective browser project.
- MCP transport security — that surface lives in `kaos-mcp`; report
  there.
- Issues caused by user-supplied configuration that explicitly disables
  safety features (`KAOS_WEB_DOMAIN_VERIFY_TLS=false` on probes against
  untrusted hosts, `KAOS_SECURITY_BLOCK_PRIVATE_NETWORKS=0`,
  `KAOS_WEB_REDACT_OBSERVED_TRAFFIC=false`, manual rate-limit /
  robots-policy bypass, raising `KAOS_WEB_MAX_BODY_BYTES` past
  reasonable defaults).
- DNS rebinding attacks against the URL/host policy gate — kaos-agents
  does not perform DNS resolution at the policy layer; the gate
  blocks IP-literal inputs and trusts the eventual TCP connect to
  resolve hostname inputs. A connect-time hook in the HTTP/browser
  client would be needed to defend against rebinding; tracked as
  upstream work.
