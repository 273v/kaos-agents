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
