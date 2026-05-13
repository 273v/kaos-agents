# Fixture provenance — counterparty-motion-to-dismiss documents

Six source documents (.docx + .eml) that drive the counterparty
motion-to-dismiss analysis task. The task definition + per-criterion
expected behaviour lives in `../task.json`. Provenance covered
alongside that file in `../README.md`; this file repeats the per-file
manifest for the leaf directory so the policy
(`docs/oss/50-data-and-fixtures/provenance-policy.md:16`) is met for
this directory in isolation.

| File | Source URL | License | Retrieved | SHA-256 |
|------|------------|---------|-----------|---------|
| datacore-motion-to-dismiss.docx | https://github.com/harveyai/harvey-labs/blob/main/tasks/litigation/analyze-counterparty-motion-to-dismiss/documents/datacore-motion-to-dismiss.docx | MIT | 2026-05-06 | 7f0cdb4a7b463b8810a112bc18b39f21e532e8c9f3de96ab6ee7054663f3c1a6 |
| msa-pinnacle-datacore.docx | https://github.com/harveyai/harvey-labs/blob/main/tasks/litigation/analyze-counterparty-motion-to-dismiss/documents/msa-pinnacle-datacore.docx | MIT | 2026-05-06 | 13ddee612aa4621fbb61b4f859b39d9a678e2c519ac2fa8eb44464cb3fcf1c6f |
| pinnacle-complaint.docx | https://github.com/harveyai/harvey-labs/blob/main/tasks/litigation/analyze-counterparty-motion-to-dismiss/documents/pinnacle-complaint.docx | MIT | 2026-05-06 | 10c792c7f0c8ae6bc4ea454ab3350fc7a1a85ba8accb10fb05e9c7f887d186dc |
| sousa-nance-email-chain.eml | https://github.com/harveyai/harvey-labs/blob/main/tasks/litigation/analyze-counterparty-motion-to-dismiss/documents/sousa-nance-email-chain.eml | MIT | 2026-05-06 | 2ad3951713b316020e357648ce9c6ccaa796181fb06abe9a36c170d93aaad64b |
| subramanian-email-jan2023.eml | https://github.com/harveyai/harvey-labs/blob/main/tasks/litigation/analyze-counterparty-motion-to-dismiss/documents/subramanian-email-jan2023.eml | MIT | 2026-05-06 | eabd894f4ab3498b602b0886efd31d5b99f1ac2fc41364bcfe529934a89ef0d8 |
| whitford-declaration.docx | https://github.com/harveyai/harvey-labs/blob/main/tasks/litigation/analyze-counterparty-motion-to-dismiss/documents/whitford-declaration.docx | MIT | 2026-05-06 | b6df84068e4e97105e0497d22fa25d1dba7a51eead331da88d2c63d948216546 |

## Notes

- Source URL paths assume the upstream task lives under
  `tasks/litigation/analyze-counterparty-motion-to-dismiss/`. If
  those URLs 404 (Harvey may reorganize the taxonomy), walk
  https://github.com/harveyai/harvey-labs/tree/main/tasks for the
  current path.
- All files are MIT-licensed; the upstream license text is at
  `../../LICENSE.upstream`.
- `Retrieved` corresponds to the git commit date when each file was
  vendored into this repo (`git log -1 --format=%cI -- <path>`).
