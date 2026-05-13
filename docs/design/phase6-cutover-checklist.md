# Phase 6 — Cutover Checklist

**Status:** prep doc, 2026-05-10. Companion to `rewrite-plan-ten-questions.md`.

This checklist captures everything that must happen during the Phase 6 cutover from the legacy `BaseAgent` path to the new `AgentLoop`. It is the artifact the next session can execute step-by-step. Each step has explicit gates so the cutover can pause at any point.

---

## 1. Where Phase 5 left us

Already landed at the time of this doc:

- v2 path is fully built. `KAOS_AGENT_LOOP=v2` opts in via `Runner(agent, agent_loop_version="v2")` or env var.
- v2 path tested across 9 live parity scenarios (CHAT, PLAN-pattern, multi-turn, both providers); all pass after DEFECT-5 fix.
- 47 + 6 + 9 = 62 live integration tests passing across Phase 3.E / 4.F / 6.A / 6.B.
- 1740 unit + benchmark tests passing.
- Default of `KAOS_AGENT_LOOP` is **still `v1`**. No legacy code deleted.
- `IntentResultV2` is a temporary alias for the new `kaos_agents.intent.IntentResult` to avoid colliding with the legacy `kaos_agents.types.intents.IntentResult`.

Still legacy (not yet touched):
- `kaos_agents/runtime/agent.py::BaseAgent`
- `kaos_agents/patterns/{chat,plan_execute,research}/*` and the `ChatAgent` / `PlanExecuteAgent` / `ResearchAgent` classes
- `kaos_agents/runtime/runner.py::_build_internal_agent` body
- `kaos_agents/types/intents.py::IntentResult` (legacy intent shape)
- `kaos_agents/types/plan.py::PlanBudget` and `kaos_agents/types/plan.py::StopReason`
- `kaos_agents/base/pattern.py::KaosPattern` (the empty ABC)

---

## 2. Pre-cutover gates

**Do not start step 3 until every gate is satisfied.** Each gate is a hard precondition for the corresponding deletion class.

### Gate A — broad live parity proven

- [ ] Run `tests/integration/test_v1_v2_parity_live.py` — all 9 currently pass; must continue to pass.
- [ ] Add live parity for tool-use: `Runner.run` with a tool registry, both v1 and v2 should produce a tool call and respond with the tool's result. (Phase 6.A only covered no-tool CHAT.)
- [ ] Add live parity for RESEARCH: `Runner.run` with a corpus + a research-shaped prompt; both paths should produce citations.
- [ ] Run the existing `tests/integration/test_agent_live.py` and `test_patterns_live.py` under both `KAOS_AGENT_LOOP=v1` and `KAOS_AGENT_LOOP=v2`. Document any divergences.

### Gate B — benchmark soak

- [ ] Run the Harvey CoC benchmark (`tests/integration/test_harvey_coc.py`) under both flags; require v2 within ±5% of v1's baseline score.
- [ ] Run the BEIR cross-domain benchmark (NFCorpus / SciFact / FiQA) under both flags; require ±2% of v1's NDCG@10.
- [ ] Run the CUAD extraction calibration (`scripts/cuad_extraction_benchmark.py`) under both flags; require ±2% of v1's calibration score.
- [ ] Optional: two-week soak with telemetry comparing v1 and v2 in real usage. The plan calls for this; in practice the calibration runs above are the hard gate.

### Gate C — performance budget

- [ ] Latency of `Runner.run` under v2 is within ±10% of v1 on a representative prompt set (median + p95).
- [ ] Memory footprint of an `AgentLoop` instance is within 2x of a `BaseAgent` instance (the new path has more objects but they're small).

### Gate D — observability symmetry

- [ ] OTelHook wired through Runner (`Runner(agent, hooks=(OTelHook(),))`) in v2 mode produces the same span tree breadth as v1: turn → intent → plan → tool calls → completion. Verify in Jaeger / equivalent tracing UI.
- [ ] CostTrackingHook + TrialRunner integration (Phase 5.A) sees the same cost numbers as v1's TurnSummary stream.

If any gate fails, fix the gap before cutover. The corresponding deletion is blocked until its gate passes.

---

## 3. Cutover steps (in order)

### Step 1 — Flip the default

```python
# kaos_agents/runtime/runner.py
self._agent_loop_version = (
    agent_loop_version
    if agent_loop_version is not None
    else os.environ.get("KAOS_AGENT_LOOP", "v2")  # <-- "v1" → "v2"
)
```

Gate A + B must be green. After this commit:
- Existing tests that didn't set the env var now run under v2.
- Any test that relied on v1-specific behavior must be either updated or pinned with `agent_loop_version="v1"`.

Run the full test suite (`uv run pytest tests/unit/ tests/integration/ -m "not live"` first, then the live tier). Fix any breakages by **updating the test**, not by reverting the flag — if a test breaks under v2, that's the gap, find it and fix it.

### Step 2 — Delete legacy event handler shims

Smaller surface area, lowest risk first:

- [ ] Delete `kaos_agents/base/pattern.py::KaosPattern` (the empty ABC). Update any imports — none should remain since no concrete pattern inherits from it.
- [ ] Delete `kaos_agents/runtime/runner.py::_build_internal_agent` body. The v2 dispatch path now handles all turn construction; the helper is dead code once Step 1 ships.
- [ ] Delete `kaos_agents/runtime/agent.py::BaseAgent`. Update test_agent_live.py to use `Runner(agent, agent_loop_version="v2")` instead of constructing `BaseAgent` directly.

### Step 3 — Delete legacy pattern subclasses

- [ ] Delete `kaos_agents/patterns/chat.py::ChatAgent`. Re-export shim emits `DeprecationWarning` and returns a Phase-3 ReActPlanner (already wired via auto-select).
- [ ] Delete `kaos_agents/patterns/plan_execute.py::PlanExecuteAgent`. Same pattern.
- [ ] Delete `kaos_agents/patterns/research/agent.py::ResearchAgent`. Same pattern.
- [ ] Eventually remove the deprecation shims (one release later — track with a `# TODO(post-cutover):` comment).

### Step 4 — Rename `IntentResultV2` → `IntentResult`

- [ ] Delete `kaos_agents/types/intents.py::IntentResult` (the legacy shape).
- [ ] In `kaos_agents/__init__.py`: change `from kaos_agents.intent import IntentResult as IntentResultV2` to `from kaos_agents.intent import IntentResult`. Remove `IntentResultV2` from `__all__`.
- [ ] Audit all imports of `from kaos_agents.types.intents import IntentResult` — these now break. Replace with `from kaos_agents.intent import IntentResult`.
- [ ] Audit all imports of `from kaos_agents import IntentResultV2` — replace with `IntentResult`.
- [ ] Update CLAUDE.md references: drop the "IntentResultV2 is a temporary alias" caveat.

### Step 5 — Delete duplicate `StopReason` and refactor `PlanBudget`

The two `StopReason` enums and `PlanBudget` have a partial overlap — the audit identified the unification need. Phase 5.E added the trial-publish wiring; Phase 6 finishes the structural unification.

- [ ] Decide enum naming: rename `kaos_agents.types.plan.StopReason` to `PlanStopReason` (since plan-execution semantics differ from optimizer-trial semantics). Re-export `kaos_llm_core.optimization.budget.StopReason` at `kaos_agents` top level.
- [ ] Refactor `PlanBudget` to wrap a `BudgetTracker` field for cost / tokens / wall_clock. Keep agent-specific fields (`max_steps`, `max_replans`, `replans`, `steps_executed`).
- [ ] Update all consumers (`compose.py`, `expand.py`, `route.py`, `evaluate.py`, `act.py`, `strategies/*`).
- [ ] Re-run the full test suite; fix imports.

### Step 6 — Final cleanup

- [ ] Drop the `auto_select_planner` constructor kwarg back-compat path in `Runner` and `AgentLoop` if no consumer uses `auto_select_planner=False` anymore. (Keep if any test still uses it.)
- [ ] Drop `Runner.agent_loop_version` property and the `KAOS_AGENT_LOOP` env-var resolver. v2 is the only path; the flag is no longer meaningful.
- [ ] Drop the `agent_loop_version=` constructor kwarg.
- [ ] Run `uv run ruff format` + `uv run ruff check --fix` + `uv run ty check kaos_agents/` for any cleanup the deletions surfaced.
- [ ] Update `kaos_agents/CLAUDE.md` to reflect the new single-path architecture. Drop legacy-mention paragraphs.
- [ ] Update `docs/design/rewrite-plan-ten-questions.md` Status section: Phase 6 complete.

---

## 4. Rollback contract

If any cutover step blows up, rollback is `git revert <step-commit>`. Each step in section 3 should be its own commit so revert is surgical. The test suite remains the gate: a step is "done" when:

1. Its commit's tests pass.
2. The full unit suite still passes.
3. The relevant live tier (per Gate A) still passes.

Do NOT batch deletions across multiple subsystems in one commit — keeps `git revert` clean.

---

## 5. Post-cutover validation

Once Step 6 lands:

- [ ] Run `./scripts/validate-platform.sh --profile ubuntu-26.04 --include-network --include-live` (the project-wide acceptance gate per `kaos-modules/CLAUDE.md`).
- [ ] Confirm Harvey CoC, BEIR, CUAD baselines still pass within tolerance.
- [ ] Confirm 0 legacy imports remain: `grep -rn "kaos_agents.runtime.agent.BaseAgent\|kaos_agents.patterns.chat.ChatAgent" kaos_agents/ tests/` returns nothing.
- [ ] Tag a release: the cutover is breaking for downstream callers, so this is a new minor version.

---

## 6. Known unknowns

These are gaps the prep work has not addressed, that may surface during cutover:

- **MCP tool-call wire format.** The legacy `BaseAgent` path serializes tool calls slightly differently than the new path's `Span(TOOL_CALL, ...)` events. Some MCP clients may rely on the old shape. Test against `kaos-mcp` integration tests during cutover.
- **Runner.pause / Runner.resume contract.** The v1 path uses `RunState`; v2 has the broader `EscalationContext`. Migration may need a one-shot translator for in-flight pauses persisted to VFS.
- **`Agent.delegated_agents` typed-Any field.** Phase 0.D made `Agent` envelope-able but the existing `delegated_agents: tuple[Any, ...]` is a forward reference. Phase 6 may want to tighten the type to `tuple[AgentEnvelope, ...]` and migrate the legacy `DelegatedAgent` wrappers.
- **Existing recipes** (`recipes/*.json`). Verify `Agent.from_envelope` reconstructs them losslessly. Phase 0.D's tests cover the core round-trip; recipes have richer schemas (golden_sets etc.) that may need a Phase 6 migration script.

---

## 7. Status table (fill in as cutover progresses)

| Step | Gate | Status | Commit |
|------|------|--------|--------|
| Gate A — broad live parity | — | not started | — |
| Gate B — benchmark soak | — | not started | — |
| Gate C — performance budget | — | not started | — |
| Gate D — observability symmetry | — | not started | — |
| Step 1 — flip default | A+B+C | not started | — |
| Step 2 — delete shims | Step 1 | not started | — |
| Step 3 — delete patterns | Step 2 | not started | — |
| Step 4 — rename IntentResult | Step 3 | not started | — |
| Step 5 — refactor PlanBudget / StopReason | Step 4 | not started | — |
| Step 6 — final cleanup | Step 5 | not started | — |
| Post-cutover validation | Step 6 | not started | — |
