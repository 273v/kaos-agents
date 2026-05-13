# `docs/benchmarks/` — public benchmark numbers

This directory carries the **publishable** benchmark inputs and
outputs for kaos-agents: BM25 / rerank parameter sweeps on BEIR
datasets, multi-format end-to-end accuracy, and the public
markdown summaries of the Harvey-Lab pipeline comparisons.

Everything here is shipped with the sdist.

## Public files

- `bm25-tune-*.json` / `nfcorpus-*.json` / `scifact-*.json` /
  `fiqa-*.json` — retrieval-evaluation outputs. No customer data
  on disk; corpora are BEIR public datasets.
- `multiformat-e2e-*.json` — RAG accuracy on the local
  `kaos-llm-core/tests/fixtures/multiformat-corpus` fixture
  (paths stored as repo-relative — see KC17-P0-4).
- `harvey-coc-pipeline-comparison-2026-05-06.md` /
  `validation-pass-2026-05-06.md` /
  `multiformat-chunking-tradeoff.md` — human-readable summaries
  of the experiments. Pass-rate numbers + cost numbers only; no
  generated deliverable text.
- `nfcorpus-adaptive.json` / `technique-isolation-2026-04-17.json`
  — small ablation outputs.

## Private files — `_private/`

`docs/benchmarks/_private/` (gitignored + excluded from the
sdist via `[tool.hatch.build.targets.sdist].exclude`) holds the
**full** Harvey-Lab benchmark JSONs that include the
LLM-generated deliverable text. The deliverables carry the
boilerplate that real legal drafts carry — `PRIVILEGED AND
CONFIDENTIAL — ATTORNEY-CLIENT COMMUNICATION / ATTORNEY WORK
PRODUCT`. Even though the source contracts in
`tests/fixtures/harvey-lab/` are synthetic (no real client
data), shipping those markers in a public PyPI sdist trips the
`docs/oss/50-data-and-fixtures/pii-and-customer-scan.md`
privileged-marker block rule.

The summary numbers from those runs ARE shipped in
`harvey-coc-pipeline-comparison-2026-05-06.md` — that is the
public, citable artifact. The raw JSONs stay local and travel
out-of-band with whoever needs to reproduce the deliverable
text (e.g., for internal review or a regulatory subpoena
response).

If you are reproducing the Harvey benchmarks locally, the
agent CLI writes new outputs into this directory by default:

```bash
mkdir -p docs/benchmarks/_private
uv run python -m kaos_agents.benchmarks.harvey \
  --task extract-change-of-control-provisions \
  --out docs/benchmarks/_private/harvey-coc-$(date -I).json
```

See `kaos-agents/docs/audit-02/kaos-agents.md` (P0-4) for the
release-blocker context that motivated this split.
