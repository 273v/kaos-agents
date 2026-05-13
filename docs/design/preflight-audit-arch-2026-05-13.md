# Pre-publish architecture audit — kaos-agents v0.1.0a1

**Audit date:** 2026-05-13
**Auditor:** opus pre-release sub-agent #2 (post-KC16)
**Verdict:** SHIP-WITH-CAVEAT

The codebase is in genuinely good shape for an `0.1.0a1` alpha. All Sprint 1-3
correctness/transparency work landed honest contracts; gates are green
(ruff/format clean, ty clean, 2138 unit tests pass with 5 graph-rdf skips);
the README is unusually honest about provider compatibility, cost-cap
gaps, recorder coverage gaps, and the persistence-model footgun. The
top findings are mostly papercuts that survive into 0.1.0a1 because no
one has had to maintain a published surface yet — they will hurt more
in 0.1.0b1 when SemVer starts mattering.

## Top 10 findings (rank-ordered by severity)

1. **[FIX-BEFORE-TAG] README promises 6 patterns at root; only 3 are exported** —
   `README.md:99-105` advertises `ChatAgent`, `PlanExecuteAgent`,
   `ResearchAgent`, `FindingsAgent`, `ReflexionLoop`, `RouterAgent` as
   the public pattern surface. Of those six, only the first three are
   exported in `kaos_agents/__init__.py:157` and
   `kaos_agents/patterns/__init__.py:27-31`. `from kaos_agents import
   FindingsAgent` raises `ImportError`. Users must learn the deep path
   `kaos_agents.patterns.findings.FindingsAgent`. This is the single
   biggest README ↔ code honesty gap. **Fix:** export the three missing
   classes (and `FindingsResult`, `FindingsRefusal`, `FindingsWarning`,
   `FilteredFinding`, `FindingCandidate` — the result-type surface
   is the actual contract a downstream consumer branches on) from
   the package root and from `patterns/__init__.py`. 5-minute change,
   removes a 100% guaranteed pip-install confusion.

2. **[FIX-BEFORE-TAG] SessionStore.save is not atomic — two-phase write
   torns under concurrent access** — `kaos_agents/memory/store.py:68-106`.
   The save path writes `session.json` (line 80), then conditionally
   writes `graph.ttl` (line 92). There is no temp-file + rename, no
   write barrier, no lock. A concurrent `load()` of the same `session_id`
   that interleaves between those two writes gets fresh-JSON + stale-TTL.
   Worse: a `SIGTERM` between the two writes leaves persistent torn
   state on disk that subsequent runs will load and trust. Combined with
   the README's "session memory persists across container restarts" pitch
   (line 282) this is a real concern. **Fix:** write to `*.tmp` + rename
   (atomic on POSIX same-fs); document the single-writer assumption
   explicitly in `SessionMemory`'s "not thread-safe" disclaimer
   (currently line 42 only talks about Python threads, not concurrent
   processes / container restarts). Alternatively, fold the graph into
   the JSON payload so it's one write.

3. **[FIX-BEFORE-TAG] Three parallel `events → AgentResponse` drain
   implementations** — PA14 confirmed. `kaos_agents/runtime/runner.py:570-650`
   (`Runner.turn`) and `kaos_agents/runtime/events_to_response.py:34-110`
   (`events_to_response`) do essentially the same accumulation; the
   `runner.py:638` comment even admits "the two implementations must
   agree" — that's an invariant the type system isn't enforcing. The
   third site is `runtime/agent.py:540-553`, which builds the TurnSummary
   the other two then read. When a Sprint-4 contract adds a new field
   to `AgentResponse` someone will update one site and forget the other.
   **Fix:** delete `Runner.turn`'s reimplementation; call
   `events_to_response()` instead. The 80-line block at runner.py:570-650
   would collapse to ~15 lines.

4. **[SHIP-WITH-CAVEAT] Private-attribute abstraction leakage across
   modules** — Six call sites in three different modules reach into
   `SessionMemory`'s `__slots__`-private attrs: `memory/store.py:86-87`
   (`memory._graph`), `memory/search.py:123-124` (`memory._sections`),
   `memory/session.py:420,432,437` (self-reads in `from_dict`, fair),
   and **`api/server.py:362, 385` (`memory._sections` — from the public
   FastAPI server)**. The API-server reach-in is the worst offender:
   it means an external operator who pip-installs `kaos-agents[api]`
   relies on a leading-underscore attribute that the `SessionMemory`
   docstring says is private. **Fix:** add `SessionMemory.sections` and
   `SessionMemory.graph_or_none` public read-only properties; update
   the four cross-module call sites; drop the `_` from intended public
   surface or add the proper accessor.

5. **[SHIP-WITH-CAVEAT] `patterns/findings.py` is 2218 LOC of "wrapper"** —
   The README markets `FindingsAgent` as a wrapper around any inner
   agent. The module contains: 5 dataclass result types, 7 selector
   factory functions, 4 helper functions for usage extraction, 3
   private rendering functions, the 660-line FindingsAgent class, plus
   `_filter_chunk` and `_synthesize` Signature-bearing helpers. It is
   not "a wrapper" — it is the most architecturally important pattern
   in the package and the one with the most surface area. The size
   itself is not the problem; the problem is that this is the only
   pattern with this much density and no test of its internal
   modularity. **Fix:** split into `findings/agent.py` (FindingsAgent
   class), `findings/selectors.py` (the 7 selector factories +
   sanitize/expand helpers), `findings/types.py` (the 5 result
   dataclasses), `findings/synthesis.py` (`_filter_chunk` +
   `_synthesize` + the rendering helpers). Pre-tag is the right time;
   post-tag this gets harder.

6. **[SHIP-WITH-CAVEAT] 1742 LOC `cli/chat.py` god-module + ~100 `print()`
   calls in CLI** — The chat CLI is the largest non-pattern module and
   handles arg parsing, ANSI rendering, event-stream tailing, SSE
   ingestion, session storage, the recipe loader, and the interactive
   REPL. There are 100 `print(` calls across the package, ~70 of them
   in `cli/chat.py`. Some are stderr-correct (`api/serve.py` is fine);
   the chat CLI mixes stdout colorization with hot-path event rendering
   in one function. This isn't a correctness bug today but it makes
   the CLI surface very hard to test (the unit test file is
   `test_cli_chat.py` — review-pending). **Fix (post-tag):** extract
   `_render_*` functions into a `cli/rendering.py` module.

7. **[SHIP-WITH-CAVEAT] 245 `Any` usages across `kaos_agents/`** —
   `kaos_agents/runtime/runner.py:159, 194, 211, 225, 746, 771, 819, 854,
   903` all use `Any` for `trigger`. Some are honest (the
   trigger types form a sum type and ty doesn't yet do exhaustive
   matching well). Many are not — e.g. `_lookup_annotations` returns
   `Any` instead of `ToolAnnotations | None`. Each `Any` is a place
   where the public-typed surface lies to its consumer. **Fix
   (post-tag):** scan for `Any` returns on public methods; tighten where
   the inner type is known.

8. **[SHIP-WITH-CAVEAT] `Runner.turn`'s doc admits a prior bypass shipped
   to production users with safety policy silently disabled** —
   `runner.py:540-551`: "WS-0.2: this path was previously a bypass
   that called `internal.turn(...)` directly, skipping the Runner's
   hook dispatch + permission policy evaluation. The MCP tool surface
   and the JSON API both use `turn()`, so the bypass silently disabled
   safety policy for every non-streaming caller." This is admirable
   transparency in the docstring, **but pre-tag is the right moment
   to ask:** what other silent-bypass regressions are possible? There
   is no invariant test that proves every `turn()` path goes through
   hook dispatch. **Fix:** add a `test_runner_turn_hooks_fire_invariant.py`
   that asserts `dispatch_hook` is called at least once for any
   tool-bearing run on the `turn()` entry — pin the contract.

9. **[SHIP-WITH-CAVEAT] kaos-graph hard dependency for every agent
   user** — `pyproject.toml:42` lists `kaos-graph>=0.1.0a3` as a base
   dependency, not an extra. Sessions that never touch the GRAPH
   section still pay the kaos-graph install cost (PyO3 wheel, ~MB-class
   binary). The lazy import in `memory/session.py:107` proves the
   code knows this is optional at runtime — but the install surface
   doesn't reflect it. **Fix (post-tag, breaks API):** move
   `kaos-graph` to the `[graph]` extra; lazy-import; sections gate
   on availability.

10. **[SHIP-CLEAN nit] `# noqa` count = 9, TODO/FIXME/HACK = 0,
    bare-except = 0** — Code-smell counters are extremely healthy.
    The 9 `# noqa` mostly justify long string literals in error
    messages or test fixtures. No `# TODO`/`# FIXME`/`# HACK` in
    `kaos_agents/` is genuinely rare and good. No bare `except:`.
    The 104 `except Exception` are almost all in tool wrappers
    catching for the agent-friendly error contract, which is
    correct.

## Code-shape stats

- Total `kaos_agents/` LOC: **45,600** (282 .py files)
- Largest 5 modules:
  - `patterns/findings.py` — 2218 LOC (see finding #5)
  - `cli/chat.py` — 1742 LOC (see finding #6)
  - `tools/retrieval.py` — 1542 LOC
  - `tools/registry.py` — 1295 LOC (6 MCP tool definitions)
  - `loop/agent_loop.py` — 1064 LOC
- Test count: **unit 160 files / integration 44 non-live / live 68**
- `--no-cov` unit run: **2138 passed, 5 skipped** (24.3s)
- TODO / FIXME / HACK count: **0 / 0 / 0**
- `# noqa` count: **9** (mostly long error strings)
- `print(` count: **100** (stderr-correct in `api/serve.py`; mixed
  in `cli/chat.py`; correct in `optimization/evaluate.py` and
  `escalation/hitl.py`)
- bare `except:` count: **0**
- `except Exception` count: **104** (almost all in tool wrappers
  for the agent-friendly error contract)
- `Any` usages: **245**
- Public exports from `kaos_agents/__init__.py`: **236 names in `__all__`**
- README promises 6 patterns at root; actually exports **3** — see #1

## Layering verdict

**Leaky in two places, clean elsewhere.**

The two leaks are documented in finding #4 — `SessionStore` and
`api/server.py` reaching into `SessionMemory`'s `__slots__` private
attrs. The pattern layer (`kaos_agents/patterns/*.py`) is clean:
`FindingsAgent` composes with rather than subclasses `KaosAgent`
(findings.py:1264, docstring is explicit), `RouterAgent` and
`ReflexionLoop` are also wrappers. MCP integration in
`kaos_agents/tools/` is a clean adapter: tools take a runtime,
build a `Runner` per call, and consume the typed `AgentResponse`
— no inner-runner internals leak across the boundary.

The 3-way `events → response` duplication (finding #3) is layering
debt: it means the contract is replicated rather than centralized,
and the explicit "the two implementations must agree" comment is
admitting it.

## State + concurrency verdict

**Smells, two of them potentially harmful.**

- `SessionMemory` is honestly documented as not thread-safe (line 42-43).
  Good.
- `SessionStore` two-phase write (finding #2) is the worst issue —
  unprotected against concurrent writers or SIGTERM mid-write. This
  is a real production-ops concern on `KaosRuntime()` default disk-backed
  VFS, which the README explicitly recommends as the resilient default.
- The concurrency primitives that DO exist (`asyncio.Semaphore` /
  `asyncio.gather` in `output/critic.py`, `output/composers/*.py`,
  `optimization/evaluate.py`, `patterns/findings.py`) are correct
  bounded-fan-out patterns. No `asyncio.Lock` anywhere — the design
  is single-writer-per-session, which is fine *as a design* but
  needs the SessionStore atomicity fix to honor that contract under
  pause/resume + container-restart scenarios.

## API surface verdict

**Will break.** 236 names in `__all__` for an `0.1.0a1` is a lot of
public surface; some of it is genuinely intentional (the error-kind
constants, the section types, the result dataclasses) and some is
incidental (`current_delegation_depth`, `IntentResultV2` aliased to
`IntentResult` with a `type: ignore[attr-defined]` comment on import).
The README/code mismatch on patterns (finding #1) is the most painful
near-term gap. Post-0.1.0a1 the maintainer should curate `__all__`
down to the names they're actually willing to keep stable across
the alpha series — everything else should be reachable via the
documented submodule path and not promised at root.

## Test discipline verdict

**Load-bearing.** The 68 `*_live*.py` files are not theater. Spot-read
of `tests/integration/test_findings_consistency_live.py` shows real
5-run Jaccard measurement with documented spend ($0.05/run) and a
real assertion contract (>= 0.95 surviving-finding-id Jaccard). The
test-design is unusually disciplined: the cost cap on each live file
is documented in the docstring, the gates have rationale comments
("if cost spikes much above [$0.30], something regressed (provider
switched away from Haiku, chunk_size blew up, etc.)"). The unit ↔
live tier separation is honest and the unit gate (`pytest tests/unit/
-q` → 2138 passed in 24.3s) is the right pre-merge bar.

The benchmarks subdir gets `--no-cov` skip; that's the right call.

## Final paragraph

I would stake my reputation on shipping this as 0.1.0a1 today **with
the README ↔ exports gap fixed first** (finding #1 is a 5-minute fix
and otherwise the first six pip-installers will file the same issue).
Findings #2 and #3 are the architectural debts the maintainer should
write down in their post-tag backlog with explicit ticket IDs — they
will hurt more in 0.1.0a2 than fixing them now. The Sprint 1-3
correctness/transparency work is genuinely good: structured refusal,
honest cost-cap contract, deterministic finding_ids, fsync'd audit
trail, KC16 redaction-by-default. The single most-important thing
the maintainer should know is that the **API surface is the riskiest
honest gap** — finding #1 is documentation-vs-code mismatch and finding
#7 ('245 `Any` usages') is the typed-promise-vs-reality mismatch.
Both will set expectations for what `0.1.0b1` is allowed to break.

## Worst 3 (also surfaced to the calling agent)

| # | Severity | Finding | Fix |
|---|---|---|---|
| 1 | FIX-BEFORE-TAG | README promises 6 patterns at package root; 3 actually exported | Add `FindingsAgent`/`ReflexionLoop`/`RouterAgent` + result types to `kaos_agents/__init__.py` |
| 2 | FIX-BEFORE-TAG | `SessionStore.save` is two writes with no atomicity, no temp+rename, no lock | Atomic temp-file + rename; document single-writer; or fold graph into one JSON payload |
| 3 | FIX-BEFORE-TAG | `events → AgentResponse` accumulator implemented 3 times (PA14) | Delete `Runner.turn`'s 80-line reimplementation; call `events_to_response()` |
