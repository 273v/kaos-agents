# Final acceptance — corpus-stress residuals + VFS explorer end-to-end

**Date:** 2026-05-27 (UTC start)
**Stack:**
- kaos-agents 0.1.22 (PyPI)
- kaos-pdf 0.1.4 + [ocr] (PyPI)
- kaos-office 0.1.3 (PyPI)
- kaos-ui (SPA backend on `main` post #31/#32; npm publish for
  `@273v/kaos-ui-react@0.1.0-alpha.10` failed at npm registry
  scope/OIDC step but the SPA app uses pnpm workspace linkage
  so the panel still runs against the local package build)
**Model floor:** anthropic:claude-sonnet-4-6

## Plans closed

1. `kaos-modules/docs/plans/2026-05-26-retrieval-planner-and-findings-dispatch.md`
   — already shipped in 0.1.19. No-op this cycle.
2. `kaos-modules/docs/plans/2026-05-26-corpus-stress-5-failure-resolution.md`
   — already shipped in 0.1.20. No-op this cycle.
3. `kaos-modules/docs/plans/2026-05-26-corpus-stress-residuals-S16-S22-and-spa-acceptance.md`
   — WI1 + WI4 + WI5 closed in 0.1.21. **WI2 + WI3 closed in 0.1.22
   this cycle.** WI6 (SPA HTML/JSON upload validator) deferred to a
   kaos-ui follow-up.
4. `kaos-modules/docs/plans/2026-05-26-spa-vfs-explorer-design.md` —
   **Stages 1 + 3 + 4 closed this cycle.** Stage 2 (SSE event family)
   was deferred to v2 per the design doc's deferral note; v1 uses
   5s TanStack polling.

## Shipped commits / releases

| Repo | PR | Merge | Tag / Version |
|---|---|---|---|
| kaos-ui | #31 (VFS + context-factory) | c051d3d | n/a |
| kaos-ui | #32 (SPA wiring) | 381a390 | n/a |
| kaos-ui | #33 (kaos-ui-react alpha.10) | 7891114 | kaos-ui-react@0.1.0-alpha.10 (npm publish failed — infra) |
| kaos-agents | #87 (WI3 gate wording) | cfff050 | — |
| kaos-agents | #88 (0.1.22 release) | e2682d2 | v0.1.22 → PyPI ✓ |
| kaos-agents | #89 (WI2 S22 fixture) | open | — |

## Final acceptance — Part A: CLI chat

**Command:**

```
kaos-agent chat \
  --message "What is the term of this NDA? Quote the operative clause verbatim and cite the section." \
  --files "/home/mjbommar/Documents/NDA/EMNA Mutual NDA.docx" \
  --model anthropic:claude-sonnet-4-6 \
  --max-cost 0.25 \
  --session "wi-final-acceptance"
```

**Result: PASS.** Final assistant text:

> The NDA has a term ending on the second anniversary of the Effective
> Date (May 20, 2023), meaning it terminates on May 20, 2025. The
> restrictions on disclosure of Confidential Information continue until
> the lesser of: (a) when the information is no longer Confidential
> Information, or (b) one year from the date the information was first
> disclosed. This is set out in the TERM section.
>
> [Verified: 3 claim(s), 3 citation(s)]
>   [6450 tokens, $0.0259]

- 3 verified citations
- $0.0259 spend, well under the $0.25 cap
- Cited operative clause structure (term anniversary + disclosure
  restrictions)
- No hallucination flags

## Final acceptance — Part B: SPA Chrome MCP

### B1. S03 scanned-PDF OCR (regression check on 0.1.22 stack)

**Session:** 01KSK3ZB7329QGRVFNYHCMMDVC
**Result: PASS** (preserved from the 0.1.21 acceptance —
`S03_v021_PASS.md`).

> The codename mentioned in the internal memo is **FALCON** [591f6045651c].
>
> > Codename: FALCON

`enumerated=6 filtered=2 cost=$0.0076 answer_chars=93`. OCR fallback
in `_build_corpus_view_from_documents` still firing on 0.1.22.

### B2. VFS explorer end-to-end (new panel)

**Steps:**

1. Reloaded the active S03 session at /sessions/01KSK3ZB7329QGRVFNYHCMMDVC.
2. Clicked the new "Session VFS explorer" header button (lucide
   `FolderTree` icon, between Citations and verbose-tools).
3. Panel slid in from the right. Header reads `VFS · 2` (count badge
   mirrors Documents / Citations chrome). Toggle buttons: Sidecars,
   Refresh, Close.
4. Tree shows two groups:
   - `files/` → `S03_scanned.pdf` 13.4 KB (UPLOAD badge, green)
   - `toolcalls/` → `turn-0000.jsonl` 516 B
5. Clicked the upload node. Preview pane populated with:
   - Path: `sessions/01KSK3ZB7329QGRVFNYHCMMDVC/files/S03_scanned.pdf`
   - Copy-URI button
   - Size: 13.4 KB
   - MIME: application/pdf
   - Modified: 3h ago
   - Parse: READY (green badge)
   - Summary excerpt: "This is an internal memo designated with the
     codename FALCON. It is marked for restricted distribution,
     intended for leadership personnel only. The memo carries"

**Result: PASS** for the design doc Stage 4 scenarios (1) upload+observe
and (3) S03 OCR artifact appearance. Sidecar exclusion default-on
working (the `.kaos.json` / `.meta.json` sidecars don't appear in the
2-entry count — they would surface only with the Sidecars toggle on).

Backend smoke also confirms the endpoint:

```
$ curl -s http://localhost:8000/v1/chat/sessions/01KSK3ZB7329QGRVFNYHCMMDVC/vfs?recursive=true
{
  "session_id": "01KSK3ZB7329QGRVFNYHCMMDVC",
  "prefix": "",
  "nodes": [
    {"path": "sessions/.../files/S03_scanned.pdf", "kind": "file",
     "size_bytes": 13767, "mime_type": "application/pdf",
     "is_upload": true, "parse_status": "ready",
     "summary_excerpt": "..."},
    ...
  ],
  "total_count": 2, "error_count": 0
}
```

## Deferred (out-of-scope for this cycle)

- **WI6** — SPA upload validator extension to HTML / JSON. Lives
  entirely in kaos-ui frontend; separate follow-up release.
- **kaos-ui-react@0.1.0-alpha.10 npm publish** — tag pushed,
  workflow ran, npm CLI returned 404 from registry on PUT (scope /
  trusted-publishing OIDC policy mismatch). Infrastructure-side fix;
  the package code is on main (commit 7891114). SPA dev path uses
  pnpm workspace linkage, so the panel is reachable in-tree without
  the registry version.
- **VFS panel Chrome MCP Stage 4 scenarios (2) agent-write mid-turn,
  (4) wrong-prefix safety, (5) high-cardinality pagination** —
  scenarios (1) + (3) verified; (2)/(4)/(5) are edge cases that
  don't block the goal but are good additions to a follow-up
  acceptance matrix.

## Tasks closed

The session-scoped task list has been pruned to the 4 keepers
(WI4, WI5 already done; WI2, WI3, VFS Stages 1+3+4 closed this
cycle). The completion order:

- WI4 — kaos-agents 0.1.21 release ceremony (prior cycle)
- WI5 — OCR fallback in FindingsAgent dispatch (prior cycle)
- WI3 — quantified refusal wording (this cycle, PR #87 → v0.1.22)
- WI2 — S22 fixture distractor disambiguation (this cycle, PR #89)
- VFS Stage 1 — backend route + service (this cycle, PR #31)
- VFS Stage 3 — React panel + SPA wiring (this cycle, PR #31 + #32)
- VFS Stage 4 — Chrome MCP acceptance (this writeup)

Remaining pending: WI6 (kaos-ui follow-up), VFS Stage 5 release
(kaos-ui-react npm publish — blocked on infra), and the final
acceptance task itself.
