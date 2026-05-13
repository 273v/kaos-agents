# Fixture provenance — NDA quickstart corpus

Five mutual NDAs that drive the
`kaos_agents.examples.nda_review.quickstart` demo. The same five files
also back the use-case ladder's T11 (tabular extraction) and T12 (LLM-
as-judge memo) tests under
`tests/integration/ladder/fixtures/nda/` — both copies are byte-
identical. Lawyer-authored ground truth for these contracts (10 fields
per NDA + cross-corpus deviation memo) lives in
`tests/integration/ladder/fixtures/nda_ground_truth.py`.

These DOCX files ship as package data so the quickstart works from a
plain `pip install kaos-agents` — no separate fixture download
required.

| File | Source | License | Retrieved | SHA-256 |
|------|--------|---------|-----------|---------|
| EMNA Mutual NDA.docx | Curated for kaos-agents v0.1.0a1 quickstart by 273V. Anonymized template-style NDAs; all parties are fictional. | Apache-2.0 (this repo) | 2026-05-10 | 60d00131fac23befe3203ae3914444b482afee421a2543b15244a35d248e736d |
| MNDA - Acme.docx | Curated for kaos-agents v0.1.0a1 quickstart by 273V. Anonymized template-style NDAs; all parties are fictional. | Apache-2.0 (this repo) | 2026-05-10 | 368ccbaabd5bfae3f0ab66783d91a1dc2547e57031ae9173dd6ba61aad067466 |
| MNDA - BI.docx | Curated for kaos-agents v0.1.0a1 quickstart by 273V. Anonymized template-style NDAs; all parties are fictional. | Apache-2.0 (this repo) | 2026-05-10 | d8bd22bbf9019d8a1f1e384e88e371eee8e01b77c9b169dcc3d1f5d6c2ac5f42 |
| MNDA - CC Final 2.docx | Curated for kaos-agents v0.1.0a1 quickstart by 273V. Anonymized template-style NDAs; all parties are fictional. | Apache-2.0 (this repo) | 2026-05-10 | dc0ba08582c711913b55a84f7c73bdcc40d384770cd9bb8bb7ff7ca06910b839 |
| MNDA - DynaMo.docx | Curated for kaos-agents v0.1.0a1 quickstart by 273V. Anonymized template-style NDAs; all parties are fictional. | Apache-2.0 (this repo) | 2026-05-10 | f5ae68ae7eda5c7839ded804b291f03ebd163602a32cd6069d66e4289eccffa9 |

## Notes

- All five contracts are **mutual** NDAs (every fixture has
  `mutual=True` in
  `tests/integration/ladder/fixtures/nda_ground_truth.py`). The
  quickstart demonstrates an associate-level batch review and
  intentionally surfaces only the deviations, not the fields each
  contract shares.
- Counterparties (ExMachi Bank N.A., Acme Co., Beta Inc., CyberCorp
  Co., DynaMo GmbH) and signatory names (Jeremy Doe, Jane Doe, John
  Doe, Jorge Doe) are **fictional placeholders** introduced during
  anonymization. The EMNA fixture intentionally preserves a real
  template-copy-paste artifact (signature block reads "DynaMo" even
  though the named counterparty is ExMachi Bank N.A.) so the demo can
  surface the kind of anomaly a careful reviewer would flag before
  signing.
- `Retrieved` corresponds to the git commit date when each file was
  vendored into the kaos-agents repository
  (`git log -1 --format=%cI -- tests/integration/ladder/fixtures/nda/<file>`).
- License: same Apache-2.0 terms as the rest of the kaos-agents
  package. See `/LICENSE`.

## Regenerating hashes

If the fixtures are updated, regenerate the SHA-256 column with::

    cd kaos_agents/examples/nda_review/ndas
    sha256sum *.docx

and update the `Retrieved` column to the new vendoring date via
`git log -1 --format=%cI -- <path>`.
