# Harvey LAB — vendored task fixtures

These fixtures are pulled verbatim from
[harveyai/harvey-labs](https://github.com/harveyai/harvey-labs)
(MIT-licensed) and used to drive integration tests + benchmarks
against the same realistic legal-work scenarios that Harvey publishes
in its Legal Agent Benchmark.

The upstream license is preserved in `LICENSE.upstream`. Each task
directory mirrors the upstream layout exactly:

```
extract-change-of-control-provisions/
  task.json                # title, instructions, deliverables, criteria
  documents/*.docx         # source documents the agent must read
```

## Why vendor

* Reproducibility — the benchmark must run from a clean clone with no
  network calls.
* Stability — upstream task files may evolve; the integration test
  asserts behavior on a frozen fixture.
* Attribution — keeping the upstream layout makes provenance obvious.

## How to refresh

```bash
gh api repos/harveyai/harvey-labs/contents/tasks/<area>/<task>/task.json \
  -H "Accept: application/vnd.github.raw" > task.json
# repeat per document under documents/
```

The benchmark
(`tests/benchmarks/harvey_coc_benchmark.py`) and integration test
(`tests/integration/test_harvey_coc.py`) consume these fixtures.

## Per-file provenance

Per `docs/oss/50-data-and-fixtures/provenance-policy.md:16`, each
task directory has its own `README.md` with a per-file manifest
table (source URL + license + retrieved date + SHA-256 per file):

- `extract-change-of-control-provisions/README.md` (8 documents)
- `extract-change-of-control-provisions/documents/README.md` (same
  manifest, scoped to the leaf documents directory)
- `analyze-counterparty-motion-to-dismiss/README.md` (6 documents)
- `analyze-counterparty-motion-to-dismiss/documents/README.md`
  (same manifest, scoped to the leaf documents directory)

Refreshing a fixture means re-running `sha256sum <file>` and updating
the matching row in every README that covers that file.
