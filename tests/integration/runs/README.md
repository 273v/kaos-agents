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

```bash
# Run the full live tier — appends to today's date dir + INDEX.jsonl
uv run --no-sync pytest tests/integration/ -m live --no-cov

# Today's spend
jq -s 'map(.total_cost_usd) | add' \
    tests/integration/runs/INDEX.jsonl

# All tests on a given commit
grep '"git_short_sha":"088136c"' tests/integration/runs/INDEX.jsonl

# Full content of one run
cat tests/integration/runs/2026-05-11/<file>.jsonl | jq

# Compare critic scores across two runs of the same test
diff \
    <(jq '.output.score' < runs/2026-05-11/...reflexion....jsonl) \
    <(jq '.output.score' < runs/2026-05-12/...reflexion....jsonl)
```

Opt-out: `KAOS_TESTS_NO_RECORD=1 pytest ...`.

## Retention

This directory is committed to git so behavioral history travels
with the repo. Files are small (~5-20 KB per test, ~200 KB for the
full G6/G7 live tier). At sustained CI velocity, prune entries
older than 90 days via:

```bash
find tests/integration/runs -type d -mtime +90 -name '20*-*-*' -exec rm -rf {} +
```

Or migrate older runs to an audit-grade cold store (S3 + Object Lock).
