# Live test telemetry runs

Every `@pytest.mark.live` test produces one JSONL file under this
directory, captured by the autouse fixture in
`tests/integration/conftest.py` (mechanics in
`tests/integration/_recorder.py`).

## Layout

```
runs/
├── README.md              ← this file
├── INDEX.jsonl            ← append-only index: one line per recorded test run
└── <YYYY-MM-DD>/          ← UTC date the run started
    └── <sanitized_nodeid>.jsonl
```

## File format

Each per-test JSONL has two kinds of lines:

**Line 1 — header.** Test identity + environment + outcome aggregate:

```json
{
  "kind": "header",
  "test_nodeid": "tests/integration/test_router_live.py::TestRouterAgentLive::test_routes_legal_question_to_legal_specialist",
  "start_ts_utc": "2026-05-11T09:15:00.202638+00:00",
  "end_ts_utc": "2026-05-11T09:15:05.040243+00:00",
  "elapsed_s": 4.8376,
  "outcome": "passed",
  "error": null,
  "traceback": null,
  "call_count": 3,
  "total_cost_usd": 0.00241,
  "total_input_tokens": 1573,
  "total_output_tokens": 288,
  "git": {"sha": "...", "short_sha": "...", "branch": "main", "dirty": "no"},
  "python_version": "3.13.5",
  "schema_version": 1,
  "markers": ["asyncio", "skipif", "live"],
  "anthropic_key_present": true,
  "openai_key_present": true
}
```

**Lines 2..N — one per LLM call.** The full `ExecutionTrace` from
kaos-llm-core, untruncated:

```json
{
  "kind": "invocation",
  "call_seq": 1,
  "invocation_id": "...",
  "model": "anthropic:claude-haiku-4-5",
  "output": {"specialist_name": "legal", "confidence": 0.99, "reasoning": "..."},
  "error": null,
  "trace": {
    "trace_id": "...",
    "call_name": "_RoutingSignature",
    "signature": "_RoutingSignature",
    "inputs": {"message": "...", "specialists": "..."},
    "outputs": {"specialist_name": "...", "confidence": ..., "reasoning": "..."},
    "model": "anthropic:claude-haiku-4-5",
    "codec": "JSONCodec",
    "input_tokens": 674,
    "output_tokens": 64,
    "total_tokens": 738,
    "cost_usd": 0.0007952,
    "latency_ms": 1168.79,
    "retries": 0,
    "examples_used": 0,
    "children": [],
    "error": null,
    "timestamp": "2026-05-11T09:15:00.207611Z"
  },
  "usage": {"input_tokens": 674, "output_tokens": 64, "total_tokens": 738, "cost_usd": 0.0007952}
}
```

**`INDEX.jsonl`** is an append-only summary index — one line per test
run. Tail it (`tail -f INDEX.jsonl`) to watch live test costs in
real time, or grep it for behavior comparison across runs.

## Why this exists

The G6 / G7 live tests (and all future LLM-driven integration tests)
exercise real models against real prompts. Without persisted output,
"the tests passed on commit X" is unverifiable history. With it:

- Regressions are diffable. The captured `outputs` on commit X vs
  commit Y for the same `inputs` show exactly what changed in
  classifier behavior, critic scoring, prompt rendering, etc.
- Audit-trail compliant. SOC 2 CC7.2, FINRA 4511, HIPAA §164.312(b)
  all want logged inputs/outputs of automated decisions. The full
  prompt + full response of every model call is preserved here,
  keyed to a git SHA + UTC timestamp.
- Cost tracking. Aggregating `total_cost_usd` across the
  `INDEX.jsonl` lines gives the per-day / per-commit live-test
  spend. Spikes signal a runaway loop.

## Reproducing / inspecting

The companion CLI (`tests/integration/runs_cli.py`) handles the
common queries. Run it directly via `uv run --no-sync python`:

```bash
# Run the full live tier — appends to today's date dir + INDEX.jsonl
uv run --no-sync pytest tests/integration/ -m live --no-cov

# Today's runs, sorted by cost
uv run --no-sync python tests/integration/runs_cli.py list --sort cost

# Filter by substring / outcome / commit / date
uv run --no-sync python tests/integration/runs_cli.py list \
    --grep reflexion --outcome failed
uv run --no-sync python tests/integration/runs_cli.py list --commit 088136c

# Pretty-print one run (header + every call + inputs/outputs)
uv run --no-sync python tests/integration/runs_cli.py show strict_rubric

# Side-by-side diff two runs of the same test (most-recent pair by default)
uv run --no-sync python tests/integration/runs_cli.py diff strict_rubric

# Cost / outcome rollups
uv run --no-sync python tests/integration/runs_cli.py summary --by day
uv run --no-sync python tests/integration/runs_cli.py summary --by commit
uv run --no-sync python tests/integration/runs_cli.py summary --by test
```

Raw `jq` still works for ad-hoc queries that don't fit the CLI:

```bash
# Today's spend
jq -s 'map(.total_cost_usd) | add' tests/integration/runs/INDEX.jsonl

# Full content of one run
cat tests/integration/runs/2026-05-11/<file>.jsonl | jq
```

Opt-out: `KAOS_TESTS_NO_RECORD=1 pytest ...`.

## Retention

This directory is committed to git so behavioral history travels
with the repo. Files are small (~5-20 KB per test, ~200 KB for the
full G6/G7 live tier, ~1.4 MB for the combined G+ladder+parity
corpus). Even at sustained CI velocity this is sub-100 MB per year.

**Local repo retention: 90 days by default.** Run the prune script
from the monorepo root:

```bash
# Dry-run (default) — show what would be removed
uv run --no-sync --project kaos-agents python \
    scripts/prune-test-runs.py --days 90

# Actually delete
uv run --no-sync --project kaos-agents python \
    scripts/prune-test-runs.py --days 90 --apply
```

The script deletes whole date directories older than `--days` days
ago AND rewrites `INDEX.jsonl` in lockstep (atomic via tmpfile +
`os.replace`). Idempotent: re-running after a successful prune is
a no-op.

**Audit-grade archive (regulatory regimes that require longer
retention than the repo's 90-day window):** push pruned runs to S3
with Object Lock in compliance mode before deletion. Recommended
parameters depend on the regime — SOX 7 years, FINRA 4511 6 years,
HIPAA §164.316(b)(2) 6 years from the date of creation or the date
when it was last in effect (whichever is later). Bucket policy
must deny `s3:DeleteObject` / `s3:PutObjectRetention` to all
principals to make the retention non-bypassable. (Out of scope for
the local prune script — wire up via your CI pipeline.)
