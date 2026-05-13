# KR-E pre-publish paranoid scan — kaos-agents v0.1.0a1

**Date:** 2026-05-13
**Surfaces scanned:** working tree (HEAD=08162dbe) / full git history (336 commits, all refs) / built sdist (`kaos_agents-0.1.0a1.tar.gz`, 587 files, 2.98 MB) / built wheel (`kaos_agents-0.1.0a1-py3-none-any.whl`, 244 files, 778 KB)
**Verdict:** GO-WITH-CAVEATS

The artifacts that go to PyPI (wheel + sdist) carry **one true positive**
in the sdist only — an absolute monorepo path in
`docs/design/architecture-audit-structured-output.md:4`. The wheel is
clean. Everything else in Categories A / C / D / E is either clearly
synthetic, vendored from MIT-licensed upstreams with attribution, or
intentional/documented (e.g. maintainer email).

Recommendation: scrub the one path leak before tagging Phase E. The fix
is mechanical (replace `<MAINTAINER_MONOREPO_PATH>/kaos_agents/`
with `kaos_agents/`). All other findings are LOW or below.

---

## Category A — Secrets

| Surface | Tool | Findings |
|---|---|---|
| Working tree | `gitleaks detect` | **0** ("no leaks found", 336 commits scanned) |
| Full history (`--log-opts=--all`) | `gitleaks detect` | **0** ("no leaks found") |
| sdist (no-git mode) | `gitleaks detect` | **0** |
| wheel (no-git mode) | `gitleaks detect` | **0** |
| NDA fixtures (targeted) | `gitleaks detect` | **0** |
| Manual grep for `sk-ant-`, `sk-proj-`, `xai-`, `AIza...`, `gh[psoru]_`, `AKIA`, and PEM/SSH private-key headers | git grep | **1 — false positive** |

**The single grep hit:** `tests/integration/test_auth_failure_live.py:54`
seeds `monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-garbage-INVALID-FOR-TESTING-PROBE-4B")`.
This is a deliberately invalid string used to exercise the
auth-failure-surfacing classifier (KC16-era). Not a real key.

**Verdict:** clean.

## Category B — Absolute paths

### Working tree
**1 finding** (after excluding `docs/audit*`, `docs/design`, `CLAUDE.md`, `*.lock`):
- `kaos_agents/cli/chat.py:1012` — `Path.cwd().glob("/home/user/*.pdf")` in a
  code comment, generic `/home/user/` illustration (not maintainer-specific).
  **False positive.**

### sdist — **1 TRUE POSITIVE (HIGH)**
- `docs/design/architecture-audit-structured-output.md:4`
  ```
  **Scope:** `<MAINTAINER_MONOREPO_PATH>/kaos_agents/`
  ```
  Verbatim maintainer absolute path + monorepo path leaked in a design doc that
  ships in the sdist. Should be replaced with `kaos_agents/` (repo-relative) or
  removed before tag.

All `/tmp/...` references in the sdist (PKG-INFO, tests, README) are generic
`/tmp/kaos-runs`, `/tmp/foo.txt`, `/tmp/x.pdf` — standard tempfile examples, no
user identity. No `/Users/`, `/private/`, `/root/`, or `file:///` occurrences in
either artifact.

### wheel
**0 findings.** Completely clean of `/home/`, `/Users/`, `/root/`, `/private/`,
`file:///`.

## Category C — Internal markers

### Privileged / attorney-client markers
- **Working tree:** 11 hits, **all meta-references** (CHANGELOG / docs/benchmarks/README /
  pyproject.toml exclude-comments / scripts/check_sdist.py / kaos_agents/recipes/extraction/privilege-classification.json).
- **sdist:** 4 hits — same meta-references (`CHANGELOG.md:158`,
  `pyproject.toml:182-183`, `docs/benchmarks/README.md:33`). All are *describing
  how privileged content is excluded*, not privileged content itself. The
  privilege-classification recipe schema (kaos_agents/recipes/extraction/privilege-classification.json)
  is a legitimate user-facing feature.
- **wheel:** 0 hits for the ALL-CAPS boilerplate. The `privilege-classification.json`
  recipe references "attorney-client privilege" as part of its feature schema
  (description fields citing FRCP 26(b)(5)(A), Hickman v. Taylor) — this is
  intentional product content, not leaked privileged material.

### 5 NDA DOCX fixtures (`kaos_agents/examples/nda_review/ndas/`)
Walked each via the zipped `word/document.xml`. Party names and signature
blocks (verbatim quotes from the closing pages):

| File | Disclosing party | Counterparty | Counterparty signer | 273V signer |
|---|---|---|---|---|
| `EMNA Mutual NDA.docx` | 273 Ventures, LLC (MI) | ExMachi Bank N.A. | Jeremy Doe, CEO | Michael Bommarito, CEO |
| `MNDA - Acme.docx` | 273 Ventures, LLC (MI) | Acme Co. (NV corp) | Jane Doe, CEO | Michael Bommarito, CEO |
| `MNDA - BI.docx` | 273 Ventures, LLC (MI) | Beta Inc. (DE corp) | John Doe, CEO | Michael Bommarito, CEO |
| `MNDA - CC Final 2.docx` | 273 Ventures, LLC (MI) | CyberCorp Co. (CA corp) | Jorge Doe, CEO | Michael Bommarito, CEO |
| `MNDA - DynaMo.docx` | 273 Ventures, LLC (MI) | DynaMo GmbH (DE) | Jeremy Doe, CEO | Michael Bommarito, CEO |

All counterparties are placeholder names (Acme / Beta / CyberCorp / DynaMo /
ExMachi Bank). All counterparty signers are `*Doe` (Jane/John/Jeremy/Jorge —
synthetic). 273V signer is the documented maintainer (intentional + matches
the README/SECURITY contact). DOCX `core.xml` metadata: `lastModifiedBy =
Michael Bommarito` on every file, `Company` element empty, no other authors,
revisions 4-9, creation dates 2023-09-07. Clean.

The DOCXs are duplicated to `tests/integration/ladder/fixtures/nda/` in the
sdist (`cmp` confirms byte-identical). No additional risk.

### Benchmark JSONs (sdist, `docs/benchmarks/`)
19 files (15 JSON + 4 MD). Grep for `PRIVILEGED|ATTORNEY-CLIENT|/home/[a-z]+/`
across all of them: **0 hits**. The `_private/` directory is correctly
excluded (gate verified — see verdict section). The public benchmark JSONs
contain BEIR/scifact/fiqa/nfcorpus retrieval scores + harvey-coc / multiformat
pipeline metrics; no LLM-generated deliverable text shipped.

### Test fixtures (`tests/fixtures/`, sdist only)
- `tests/fixtures/harvey-lab/` — vendored verbatim from
  `harveyai/harvey-labs` (MIT). `LICENSE.upstream` ships alongside. The 8
  DOCX contracts and 2 EML files use synthetic names (Apex/Kenji,
  Summit credit, Hendricks, Hesse, Crescent Ridge HQ, Northland Refining,
  PacWest, datacore-systems.com, pinnaclestaffing.com — none of these are
  real companies that the maintainer or 273 Ventures has a relationship
  with; all are upstream Harvey AI synthetic content). DOCX `dc:creator`
  on every file = `python-docx` (programmatically generated).
- `tests/fixtures/images/iss068e027836-full-moon-south-texas.jpg` — NASA
  public-domain photo (per CLAUDE.md fixture policy).

### Ladder fixture (sdist only): `tests/integration/ladder/fixtures/sample.pdf`
**Surprise finding (LOW):** this is a public Imperial, CA City Council
notice dated November 4, 2010, containing names of public officials acting
in public capacity (Mayor Geoff Dale, council members, city clerk, etc.).
This is a publicly distributed government document — public-domain by
operation of law, names appearing in their official capacity. Used as a
PDF-extraction test fixture in the ladder benchmark. Not a leak; flagging
for awareness only.

## Category D — PII

### Emails
- **Working tree:** beyond `michael@bommaritollc.com` / standard fictional
  examples, the additional emails are:
  - `it@273ventures.com`, `mike@273ventures.com`, `security@273ventures.com`
    (pyproject.toml, README, SECURITY.md — **documented maintainer
    contacts**, same person as `michael@bommaritollc.com` via the corporate
    domain).
  - `vsousa@datacoresystems.com`, `cnance@pinnaclestaffing.com`,
    `dbaines@datacoresystems.com`, `a.subramanian@datacoresystems.com`
    (Harvey-lab .eml fixtures — synthetic, vendored from `harveyai/harvey-labs`).
- **wheel:** 0 non-standard emails. Only `mike@273ventures.com` /
  `it@273ventures.com` / `security@273ventures.com` in METADATA.
- **sdist:** 4 synthetic Harvey-lab emails in the two .eml fixtures.

### Phones / SSNs / addresses
- **SSN-shaped (`\d{3}-\d{2}-\d{4}`):** 0 hits in either artifact.
- **Phone numbers:** 6 hits in the Harvey-lab .eml fixtures, all in the
  NANP-reserved `555-01XX` test range — `(404) 555-0192`, `(404) 555-0238`,
  `(404) 555-0347`, `(512) 555-0193`. Explicitly fictional per the North
  American Numbering Plan.
- **Real US addresses:** none found beyond the Imperial CA city hall
  address inside the public-domain `sample.pdf` (covered above).

## Category E — Stray debug / internal refs

### TODO/FIXME/XXX/HACK
- **In `kaos_agents/**/*.py` source:** 1 occurrence —
  `kaos_agents/types/agent_tool_spec.py:73` — a `TODO` note about a planned
  `FieldSet` projection, in a docstring. Benign feature-design note, not a
  release blocker.
- **`REMOVE BEFORE SHIP` / `DO NOT SHIP`:** 0 occurrences anywhere.

### `print()` in source (excluding CLI / `__main__` / scripts / examples / viewer)
17 occurrences across 8 files. Triage:
- `kaos_agents/api/serve.py` — 7× to `sys.stderr` for startup banners.
  CLI surface, deliberate.
- `kaos_agents/api/server.py`, `kaos_agents/api/settings.py` — none.
- `kaos_agents/decorators/hook.py:19`, `:28` — inside a docstring example.
- `kaos_agents/escalation/hitl.py:120` — interactive HITL prompt.
- `kaos_agents/optimization/evaluate.py:45` — progress line.
- `kaos_agents/patterns/findings.py:62`, `:64` — appear inside a `__main__`
  demo block.
- `kaos_agents/registry/tool_group_registry.py:43`, `:45` — `list_groups()`
  CLI helper.
- `kaos_agents/tools/optional_modules.py:97`, `:101` — to `sys.stderr` for
  optional-module load reporting.

All deliberate. No stray debug `print()` in library code paths.

### `kaos-modules` monorepo references
- **Working tree (excluding `CLAUDE.md` / audit docs):** 6 hits — comments
  in `kaos_agents/errors.py:128`, `tests/benchmarks/cross_doc_benchmark.py:52-53`,
  `tests/unit/test_auth_failure_surfacing.py:27`, harvey-lab READMEs.
- **sdist:** 10 hits. Most are comments. The notable ones:
  - `docs/design/architecture-audit-structured-output.md:4` — **the absolute
    path leak (HIGH, the one true positive of this scan)**.
  - `docs/design/iterative-findings-pattern.md:603` — narrative mention.
  - `docs/design/phase6-cutover-checklist.md:140` — points at
    `kaos-modules/CLAUDE.md` for the validator gate.
  - `docs/design/kc16-audit-findings.md:137` — references the monorepo as
    historical context for an audit finding.
  - `kaos_agents/errors.py:128` — *also in wheel*, comment "Mirrors the
    table in kaos-modules CLAUDE.md".
  - `tests/...` — incidental in test docstrings.
- **wheel:** 1 hit — `kaos_agents/errors.py:128`, a code comment. Harmless.

### `localhost` / `127.0.0.1` / `0.0.0.0` in source
26 hits, **all in `kaos_agents/api/`** — these are the deliberate
`api_allow_unauth_localhost` policy surface (settings + server + serve CLI),
documented behavior, configurable. Not stray.

## sdist + wheel inventory

### Wheel — 244 files
| Type | Count |
|---|---|
| `.py` | 217 |
| `.json` (recipes) | 13 (6 top-level + 7 extraction) |
| `.docx` (NDA fixtures) | 5 |
| `.md` (NDAs README) | 1 |
| `.html` (viewer) | 1 |
| `.txt` (entry_points) | 1 |
| `.typed` (py.typed marker) | 1 |
| `.dist-info/*` | 5 (METADATA, WHEEL, RECORD, LICENSE, NOTICE, entry_points.txt) |

Every wheel file is expected per `pyproject.toml` `[tool.hatch.build.targets.wheel]
packages = ["kaos_agents"]`. The 5 NDA DOCX + viewer HTML + 13 recipe JSONs
ride in via the `kaos_agents/` package tree, as intended.

### Sdist — 587 files
| Type | Count |
|---|---|
| `.py` | 482 |
| `.md` | 36 |
| `.json` | 31 (recipes + benchmark results + task fixtures) |
| `.docx` | 22 (5 NDA × 2 copies + 12 Harvey-lab task fixtures) |
| `.eml` | 2 (Harvey-lab) |
| `.png` | 4 (viewer screenshots) |
| `.jpg` | 1 (NASA test image) |
| `.pdf` | 1 (Imperial CA ladder fixture) |
| `.upstream` | 1 (LICENSE.upstream) |
| `.toml` `.typed` `.html` `.gitignore` | 1 each |

All inclusions match `[tool.hatch.build.targets.sdist].include`:
`/kaos_agents`, `/README.md`, `/CHANGELOG.md`, `/LICENSE`, `/NOTICE`,
`/CLAUDE.md`, `/docs`, `/tests`. The exclude block correctly suppresses
`tests/integration/runs/**`, `tests/**/*.jsonl`, `tests/scratch/**`,
`docs/benchmarks/_private/**`, `.benchmarks/`, `.kaos-vfs/`.

### KC17-P0-4 sdist gate: **PASS**
```
$ uv run --no-sync python3 scripts/check_sdist.py
OK: sdist kaos_agents-0.1.0a1.tar.gz contains no forbidden files.

$ tar tzf dist/*.tar.gz | grep -c '\.jsonl$'   # 0
$ tar tzf dist/*.tar.gz | grep -c 'runs/'      # 0
$ tar tzf dist/*.tar.gz | grep -c '_private/'  # 0
```

### Unexpected files
None. Every file in the wheel and sdist is accounted for by the
`pyproject.toml` build config.

---

## Verdict — **GO-WITH-CAVEATS**

The artifacts are **publish-safe modulo one mechanical fix**:

**MUST FIX before Phase E tags:**
1. `docs/design/architecture-audit-structured-output.md:4` — replace
   `<MAINTAINER_MONOREPO_PATH>/kaos_agents/`
   with `kaos_agents/` (repo-relative). One-line edit. Rebuild sdist + wheel.
   Re-run `scripts/check_sdist.py`. Re-verify with
   `grep -rn 'mjbommar' /tmp/kr-e-precheck/sdist/` → 0 hits.

**MAY ADDRESS (not blockers):**
- `kaos_agents/errors.py:128`, `tests/unit/test_auth_failure_surfacing.py:27`,
  `tests/benchmarks/cross_doc_benchmark.py:52-53` — references to "kaos-modules"
  in code comments. Now that this is the per-module published repo, these
  could be rewritten to "the KAOS monorepo" or removed. Cosmetic; doesn't
  leak anything sensitive.
- `docs/design/*.md` files that narrate audit history with "kaos-modules"
  context — narrative documentation, not actionable leakage.

**Everything else passes the bar:**
- 0 secrets (gitleaks × 4 surfaces, plus manual grep).
- 5 NDA DOCX fixtures verified synthetic (Acme / Beta / CyberCorp / DynaMo /
  ExMachi Bank counterparties; `*Doe` placeholder signers; Michael Bommarito
  on the 273V side per intent).
- Harvey-lab fixtures are MIT-licensed upstream vendoring with
  `LICENSE.upstream` shipped alongside.
- All phone numbers use the NANP-reserved 555-01XX fictional range.
- No SSNs, no JWTs, no private keys, no AWS/GitHub tokens, no real client emails.
- KC17-P0-4 gate holds: no `_private/`, no `*.jsonl`, no `runs/` in sdist.

Once the one path leak is scrubbed, this artifact set is cleared for PyPI.
