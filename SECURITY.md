# Security policy

## Reporting a vulnerability

We take security seriously. If you believe you have found a security
vulnerability in `kaos-agents`, please report it privately so we can address
it before public disclosure.

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

## Threat model

`kaos-agents` is an agent runtime: it dispatches LLM calls, runs MCP tools,
persists session memory + a per-session RDF graph to a VFS, and optionally
exposes the runtime over a FastAPI HTTP surface or an MCP server. The
trust boundaries we care about, in order of severity:

1. **LLM-content boundary.** Document text the agent reads MUST be treated
   as untrusted data, never as instructions. Prompt-injection attempts are
   expected.
2. **HTTP API boundary.** The FastAPI surface is the most public surface
   when enabled. A misconfigured caller MUST NOT reach another tenant's
   session, run, or persisted memory.
3. **Tool-execution boundary.** Destructive / human-in-the-loop tools MUST
   NOT execute without an explicit approval decision. The default is
   "ask," not "skip checks."
4. **Persistence boundary.** Session memory + the per-session knowledge
   graph live on disk via the VFS. A SIGTERM mid-save MUST NOT produce
   torn on-disk state. A `DELETE` MUST actually delete.
5. **Audit-trail boundary.** Live-test recorder JSONLs may contain
   document content; they MUST NOT be committed to the public repo and
   MUST be redacted above the configured size threshold.

Out of scope for this package: browser sandboxing, DNS resolution, URL /
host validation, captured-traffic redaction — those surfaces live in
`kaos-web` and are owned by its `SECURITY.md`. `kaos-agents` never opens
a socket of its own.

## HTTP API authentication and tenant scoping (KC17-P0-3)

The FastAPI surface in `kaos_agents/api/server.py` requires an auth
source at startup. There is no insecure default.

`create_app()` calls `KaosAgentsApiSettings.is_auth_configured()` and
raises `InsecureApiConfigurationError` unless one of these is set:

- `KAOS_AGENTS_API_TOKEN=<bearer-token>` — production. Clients send
  `Authorization: Bearer <token>`. Constant-time compared via
  `hmac.compare_digest` in `KaosAgentsApiSettings.check_token`
  (`kaos_agents/api/settings.py:159-166`). Wrong token → 401.
- `KAOS_AGENTS_API_ALLOW_UNAUTH_LOCALHOST=1` — local development only.
  Permits unauthenticated requests from `127.0.0.1` / `::1`; every
  request emits a `WARNING` log line
  (`kaos_agents/api/server.py:294-311`). Has no effect when a token
  is also set.

When a token is configured, every session and run is namespaced by
`tenant_id = SHA-256(token).hexdigest()[:12]`
(`kaos_agents/api/settings.py:168-181`).
`scope_session_id` prefixes every caller-supplied session id with the
tenant id before any VFS read or write
(`kaos_agents/api/settings.py:206-229`). Cross-tenant access returns
**404, not 403** — explicitly to avoid leaking the existence of a
session or run across tenants
(`kaos_agents/api/server.py:476-494`).

CORS defaults to an empty allow-list. Configure with
`KAOS_AGENTS_API_CORS_ALLOW_ORIGINS=https://a.example.com,https://b.example.com`.
Wildcard `*` combined with credentials is rejected at config time
(`kaos_agents/api/settings.py:128-144`) — the W3C CORS spec forbids
it; Starlette will accept it but browsers will reject the response.

The `X-Forwarded-For` header is honored ONLY when the immediate peer
is localhost (`kaos_agents/api/server.py:244-257`). A public-facing
peer cannot spoof its client IP into the localhost-dev allow-list.

## Tool approvals and destructive operations (KC17-P0-2)

`Runner(permission_policy=None)` installs `PermissionPolicy.default_safe()`
(`kaos_agents/runtime/runner.py:131-132`); the same default applies at
the `tool_bridge` boundary (`kaos_agents/actions/tool_bridge.py:100-103`).

The default-safe evaluation order
(`kaos_agents/runtime/permissions.py:88-120`):

1. `humanConfirmationRequired=True` → **ASK** (escalate to human approval)
2. Explicit deny rule → **DENY**
3. Explicit allow rule → **ALLOW**
4. `readOnlyHint=True` → **ALLOW** (no friction on safe reads)
5. `destructiveHint=True` → **ASK**
6. No annotations / no flags → **ALLOW** (permissive)

Pre-KC17-P0-2, `permission_policy=None` meant "skip all checks." A
caller that reached the HTTP API or the MCP server could fire a
`destructiveHint=True` tool with no approval gate. That is closed.

There is one escape hatch: `Runner(unsafe_bypass=True)`
(`kaos_agents/runtime/runner.py:121-132`). It exists for in-process
tests and internal benchmarks where every tool call is known-safe.
The constructor logs `"Runner: unsafe_bypass=True ignores the
provided permission_policy. This MUST NOT be used in production."`
when the flag is set. Production deployments MUST NOT enable it.

The HTTP API also accepts a per-request glob list of tools to require
approval for: `MessageRequest.require_approval_for_tools`
(`kaos_agents/api/server.py:77-87`). Each glob becomes a
`PermissionRule(action=ASK)` for that request only.

## Prompt-injection defense

The `FindingsAgent` (`kaos_agents/patterns/findings.py`) reads
arbitrary document text on behalf of the user. Three layers defend
the LLM-content boundary:

1. **Heuristic detector** (`is_injection_suspected` at
   `kaos_agents/patterns/findings.py:1081-1090`). Pattern-matches
   the OWASP LLM01 top-10 framings before the candidate ever
   reaches an LLM. Flagged candidates pass through with
   `injection_suspected=True` and emit a structured `WARNING` log
   line. Detection does NOT drop the candidate — the LLM filter
   still adjudicates relevance — but the flag is preserved through
   to the audit trail.
2. **XML isolation envelope** (`_wrap_untrusted_text` at
   `kaos_agents/patterns/findings.py:1139-1168`). Every candidate
   body is wrapped in `<untrusted_document_content
   finding_id="..." injection_suspected="...">…</untrusted_document_content>`
   for both the filter and synthesis stages. The signature
   docstring instructs the LLM to treat anything inside the
   envelope strictly as data.
3. **KC17-P2-3 structural integrity.** Candidate text is passed
   through `xml.sax.saxutils.escape` before interpolation. A
   payload containing a literal `</untrusted_document_content>` no
   longer closes its own envelope — `<`, `>`, and `&` become entity
   equivalents, so the LLM still sees the original content but the
   structural metacharacters are neutralized.

The combined defense is live-tested against the OWASP LLM01 top-10
payload set on `anthropic:claude-sonnet-4-6` (KC16-10) plus a
synthesis-stage variant (Sprint-1 #3, commit `fb82f64`).

## Structured refusal contract

`FindingsAgent` does not silently return an empty result. When the
agent declines to answer, it stamps a `FindingsRefusal` value type
on `FindingsResult.refusal`
(`kaos_agents/patterns/findings.py:303-394`). Five stable reasons:

- `no_candidates_enumerated` — Phase 1 selector emitted zero
  candidates. Remediation: broaden the selector / vocabulary.
- `no_relevant_candidates` — Phase 2 filter judged every candidate
  irrelevant. The canonical "the answer is not in this document"
  signal.
- `budget_exceeded` — `max_cost_usd` was hit before synthesis
  could run. Remediation: raise the cap or accept the partial
  surviving findings.
- `too_many_candidates` — Phase 1 enumeration exceeded
  `max_candidates`. Hard ceiling that fires before any LLM spend.
- `too_many_chunks` — Phase 2 chunk plan exceeded `max_chunks`.
  Hard ceiling that fires before any LLM spend.

KC17-P2-1 adds an `InsufficientEvidence` collapse path for the
research-strict legal profile (see "Research-strict profile"
below). The contract is described in `docs/guides/tool-design.md`:
every error includes (1) what went wrong, (2) how to fix it, (3)
the alternative tool / approach where applicable.

## Memory deletion and right-to-delete (KC17-P1-1)

`SessionStore.delete(session_id)`
(`kaos_agents/memory/store.py:259-293`) actually deletes. It sweeps
both:

- `kaos-agents/sessions/{id}/memory.json` (the JSON memory snapshot)
- `kaos-agents/sessions/{id}/graph.ttl` (the RDF knowledge graph
  when present)

Both paths the HTTP API and the MCP server expose for deletion go
through `SessionStore.delete`:

- `DELETE /v1/sessions/{id}` → `SessionStore.delete` then
  `vfs.cleanup_context` (`kaos_agents/api/server.py:585`).
- `kaos-agent-memory-clear` MCP tool → `SessionStore.delete` then
  `vfs.cleanup_context` (`kaos_agents/tools/registry.py:989-992`).

Pre-KC17-P1-1, the deletion paths called `vfs.cleanup_context` only,
which evicted VFS caches but left `memory.json` + `graph.ttl` on
disk. A subsequent `SessionStore.exists()` returned True and a
follow-up `GET /v1/sessions/{id}` returned 200. That is closed:
after a successful DELETE, `SessionStore.exists()` returns False
and a follow-up GET returns 404. The deletion is idempotent.

## Durable session storage (KC17-P1-3)

`_atomic_write` (`kaos_agents/memory/store.py:50-120`) writes both
`memory.json` and `graph.ttl` via temp-file + fsync + `os.replace`
on disk-backed VFS, with a best-effort directory `fsync` for Linux
durability. `os.replace` is POSIX-atomic on the same filesystem,
so a SIGTERM mid-save leaves either both-old or both-new bytes on
disk — never a partial-write JSON the next `load()` reads as
`json.JSONDecodeError`.

Non-disk VFS backends (memory) short-circuit through a direct
`vfs.write` — torn states are unreachable for in-process bytes.

## Audit trail recorder (KC16-4, Sprint-3 #8)

Live integration tests can record every LLM call to a JSONL audit
trail via `tests/integration/_recorder.py`. Files land in
`tests/integration/runs/<date>/<test-nodeid>.jsonl`. Each file
carries `schema_version=4` and advertises `streaming=true` plus
`redaction_enabled` and `redaction_threshold_chars=2048`
(`tests/integration/_recorder.py:117`, `:488`, `:498-499`).

Crash-safe writes: the header is fsynced on `__aenter__`; every
completed `Invocation` is appended + flushed + fsynced before the
call returns to the test (`schema_version=4` contract).

Redaction: string values longer than `redaction_threshold_chars`
(default 2048) serialize as
`{"_redacted": true, "len_chars": N}` instead of raw text. The
header carries the threshold so a downstream consumer can audit
the policy applied to that file.

Retention + privacy:

- The full live-test capture directory is `.gitignore`-d
  (`tests/integration/runs/**/*.jsonl`; the rolling
  `INDEX.jsonl` is the only file kept under version control).
- The same recorder is published under
  `kaos_llm_core.observability.env_recorder` in
  `kaos-llm-core>=0.1.0a7` for production use; the SDK consumer
  controls the output `Path`.
- **Audit-trail JSONLs persist full document bodies + agent
  output below the redaction threshold.** In regulated
  deployments (SOC 2 / HIPAA / FINRA / GLBA) point the recorder
  at encrypted-at-rest storage — see the README "Known
  limitations" entry on the data plane.
- 142 pre-KC16-4 unredacted captures from
  `tests/integration/runs/2026-05-11/` remain in git history;
  they will be scrubbed in the KC17-FU1 history-flatten task.

## Cost caps and budget contract (Sprint-3 #9, #10)

`max_cost_usd` is honored truthfully across all four agent tools
(`kaos_agents/patterns/findings.py:1338-1456`):

- `kaos-agent-chat`: soft cap, may overshoot up to 2× in a single
  turn. `budget_exceeded` is still reported truthfully.
- `kaos-agent-plan`: strict per-step cap.
- `kaos-agent-findings`: strict wave-level cap (Phase 2 filter +
  Phase 3 synthesis). When the cap fires mid-flight the result
  carries `budget_exceeded=True`, the partial surviving findings,
  and `FindingsRefusal(reason="budget_exceeded")`.
- `kaos-agent-corpus-filter`: post-hoc cap.

Every `AgentResponse` carries `cost_usd: float` (USD) and
`total_tokens: int` as first-class frozen attributes (Sprint-3 #10).
The same numbers ship as `ToolResult.structuredContent["cost_usd"]`
/ `["total_tokens"]` across all four agent MCP tools. The
`ResearchAgent` / RAG path has no cost cap in v0.1.0a1 — tracked
as PA11.

## Research-strict legal profile (KC17-P2-1)

`KAOS_AGENT_RESEARCH_PROFILE=strict` enables a regulated-industry
preset. It raises the BM25 score floor, raises the verifier
confidence threshold, and refuses unverified answers via a typed
`InsufficientEvidence` collapse instead of warn-and-return
(`kaos_agents/settings.py:402-440`). The three underlying knobs
(`bm25_score_floor`, `verifier_min_confidence`,
`refuse_unverified_answers`) remain available for fine-grained
override. Recommended for legal / regulated deployments.

## Optional dependencies and base-import discipline (KC17-P0-1)

`pip install kaos-agents` (base, no extras) imports cleanly. The
`__init__` resolves `kaos_llm_core`- and `fastapi`-dependent names
lazily via PEP 562 `__getattr__`
(`kaos_agents/__init__.py`). Optional extras:

| Extra | Pulls in | What it enables |
|-------|----------|-----------------|
| `[llm]` | `kaos-llm-client`, `kaos-llm-core` | Real LLM dispatch (Call, ReAct, RAG) |
| `[mcp]` | `kaos-mcp` | `kaos-agents-serve` (stdio / HTTP MCP) |
| `[api]` | `fastapi` | `create_app` HTTP surface |
| `[otel]` | `opentelemetry-api` | `OTelHook` span export |
| `[rerank]` | `kaos-nlp-transformers[torch]` | Cross-encoder rerank + dense embeddings |

Plus per-tool-module extras: `[pdf]`, `[office]`, `[source]`,
`[web]`, `[citations]`, `[tabular]`. Each maps 1:1 to a `--with-X`
flag on `kaos-agents-serve`. `[full]` is the convenience meta-extra
covering everything.

A consumer that imports an optional name without the matching
extra gets a clear install-hint `ImportError` from
`kaos_agents.action`'s `__getattr__` — never a silent module
error.

## Reporting CVEs in transitive dependencies

CI runs `pip-audit` against the lockfile. For a CVE affecting a
shipped dependency (kaos-core, kaos-llm-core, kaos-graph, pydantic,
fastapi, …), open a regular GitHub issue linking the CVE plus the
affected `pyproject.toml` entry. For fixes that need to land in the
dependency itself, file upstream first; we'll bump the floor here
once a patched release is available.

## What is NOT in this package's scope

Don't file these against `kaos-agents`; route them to the
package that owns the surface:

- **Browser sandboxing / Playwright context isolation /
  captured-traffic redaction.** Owned by `kaos-web`. The agent
  never opens a browser — it dispatches `kaos-web-*` tools when
  the operator opts in via `[web]`.
- **DNS resolution, WHOIS, TLS probes, URL / host validation,
  SSRF defense.** Owned by `kaos-web`. The agent does not perform
  network I/O of its own.
- **MCP transport security** (stdio framing, streamable-HTTP
  hardening). Owned by `kaos-mcp`.
- **PDF parser hardening / OCR.** Owned by `kaos-pdf`.
- **LLM provider transport / API-key handling.** Owned by
  `kaos-llm-client`.

This file describes the kaos-agents-specific surface only.
