# Fixture provenance — change-of-control documents

Eight `.docx` source documents that drive the change-of-control
extraction task. The task definition + per-criterion expected
behaviour lives in `../task.json`. Provenance covered alongside that
file in `../README.md`; this file repeats the per-file manifest for
the leaf directory so the policy
(`docs/oss/50-data-and-fixtures/provenance-policy.md:16`) is met for
this directory in isolation.

| File | Source URL | License | Retrieved | SHA-256 |
|------|------------|---------|-----------|---------|
| apex-kenji-jv-agreement.docx | https://github.com/harveyai/harvey-labs/blob/main/tasks/contracts/extract-change-of-control-provisions/documents/apex-kenji-jv-agreement.docx | MIT | 2026-05-06 | c5796b20fcb7540db03a07fffa2447c20a962faf8a339af17c6d88df2494ef2f |
| credit-agreement-summit.docx | https://github.com/harveyai/harvey-labs/blob/main/tasks/contracts/extract-change-of-control-provisions/documents/credit-agreement-summit.docx | MIT | 2026-05-06 | ca11cdc417641617fc907525c4703ff6f02172d82b66d4a4ba4dc327493c81a9 |
| hendricks-license-agreement.docx | https://github.com/harveyai/harvey-labs/blob/main/tasks/contracts/extract-change-of-control-provisions/documents/hendricks-license-agreement.docx | MIT | 2026-05-06 | 1e317ede8aab02b59a6dda51608cfbf203eb2b7823b4741188dee245891b7259 |
| hesse-employment-agreement.docx | https://github.com/harveyai/harvey-labs/blob/main/tasks/contracts/extract-change-of-control-provisions/documents/hesse-employment-agreement.docx | MIT | 2026-05-06 | 7a67ef910239f184fa68e9e7d707493b0da5e0713a6f3fd9bcadc572870c5303 |
| hq-lease-crescent-ridge.docx | https://github.com/harveyai/harvey-labs/blob/main/tasks/contracts/extract-change-of-control-provisions/documents/hq-lease-crescent-ridge.docx | MIT | 2026-05-06 | f582c0e10ca2551c6bfe6f1042cf278463add7e927bcf5d00479676823c14316 |
| northland-refining-msa.docx | https://github.com/harveyai/harvey-labs/blob/main/tasks/contracts/extract-change-of-control-provisions/documents/northland-refining-msa.docx | MIT | 2026-05-06 | 029efb46629b3fd5d3931321cf2aa9d0eb05f8e9b5129cf3c11e372bf2d9523d |
| pacwest-supply-agreement.docx | https://github.com/harveyai/harvey-labs/blob/main/tasks/contracts/extract-change-of-control-provisions/documents/pacwest-supply-agreement.docx | MIT | 2026-05-06 | 0563ce3ed028de67bc6ea21b4be46936cef1f6b46b0e10d65a83f6fb5ae7b128 |
| product-liability-policy.docx | https://github.com/harveyai/harvey-labs/blob/main/tasks/contracts/extract-change-of-control-provisions/documents/product-liability-policy.docx | MIT | 2026-05-06 | 7456f987283a72476c2c1c33c4e169ba513a162d781eeb489c919273428a6b03 |

## Notes

- Source URL paths assume the upstream task lives under
  `tasks/contracts/extract-change-of-control-provisions/`. If those
  URLs 404 (Harvey may reorganize the taxonomy), walk
  https://github.com/harveyai/harvey-labs/tree/main/tasks for the
  current path.
- All files are MIT-licensed; the upstream license text is at
  `../../LICENSE.upstream`.
- `Retrieved` corresponds to the git commit date when each file was
  vendored into this repo (`git log -1 --format=%cI -- <path>`).
