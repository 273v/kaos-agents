# Harvey CoC pipeline comparison — 2026-05-06

Four Sonnet runs on the same 8-contract Harvey LAB CoC corpus, scored
against the same 55-criterion rubric by the same haiku-4-5 judge. The
producer pipeline is the only variable.

| Pipeline | Pass rate | Cost | Latency | Deliverable | Notes |
|---|---|---|---|---|---|
| **Baseline (Haiku)** | 10/55 = 18.2% | $0.27 | 51s | 16K (truncated) | Pre-token-cap-fix baseline. |
| `freeform` (Sonnet) | 32/55 = **58.2%** | $1.13 | 265s | 58K | Sonnet research-pattern agent, single comprehensive memo. |
| `structured` (Sonnet) | 26/55 = **47.3%** | $1.77 | 191s | 96K (table) | `extract_corpus(coc_schema)` over 8 docs, table-only deliverable. |
| `hybrid_v1` (Sonnet) | 31/55 = **56.4%** | $2.08 | 222s | 122K | Structured + per-contract narrative call (cells only). |
| **`hybrid_v2`** (Sonnet) | 34/55 = **61.8%** | $2.52 | 310s | 154K | Structured + per-contract narrative call (cells + source_text). **Best so far.** |
| Theoretical union ceiling | 46/55 = 83.6% | n/a | n/a | n/a | If a hybrid could combine every freeform-pass + structured-pass without losing either's wins. |

## Where each pipeline wins (and the hybrid v2 picture)

The two single-pipeline approaches are **complementary**. Hybrid v2
is the first pipeline that captures most of both pipelines' wins
plus 3 brand-new criteria neither single pipeline got.

### Hybrid v2 captures 31/46 (67.4%) of the union

Wins it inherits from the structured pipeline (forced specificity):
risk-rating completeness (C-048-C-051), section-number identification
(C-016/C-020/C-028/C-030/C-031), cross-cutting analyses (C-044 UCC,
C-045/C-046 reverse-triangular), cross-references (C-043 ERP).

Wins it inherits from free-form (deep prose synthesis): per-section
analysis of multi-clause contracts (C-006-C-011 credit agreement),
nested sub-criteria (C-013 carve-out conditions), single-trigger
conflict (C-024).

NEW wins neither single pipeline got (3): C-001, C-019, C-026 — all
"specificity gap" criteria where the structured cells gave the
narrative call enough context to make the explicit assertion the
rubric demanded, but where the free-form prose alone scattered the
facts and the table alone lacked the prose connection.

### Hybrid v2 still misses 21 criteria (6 hard-misses across all 3)

The 21 misses break into two classes:

* **15 selective misses** — passed in either freeform OR structured
  but not in v2. The narrative call's instruction is to combine
  cells + source, but it doesn't always pull every multi-section
  sub-clause from the source. Examples: C-007 dual-definition
  inconsistency (free-form caught it, v2 didn't), C-021 downstream
  Voltan risk (free-form caught it).
* **6 hard misses** failed in all three pipelines. These are the
  cross-document / cross-reference criteria (C-042 Crestline ERP
  unreviewed) and the deepest legal-knowledge criteria. These need
  the kaos-graph layer (E9-E11) and / or required-analysis recipe
  (E5-E7) layers that haven't shipped yet.

## Cost-efficiency comparison

| Pipeline | Cost | Pass count | Cost per pass |
|---|---|---|---|
| Baseline (Haiku) | $0.27 | 10 | **$0.027** |
| freeform Sonnet | $1.13 | 32 | $0.035 |
| structured Sonnet | $1.77 | 26 | $0.068 |
| hybrid_v1 Sonnet | $2.08 | 31 | $0.067 |
| **hybrid_v2 Sonnet** | $2.52 | 34 | $0.074 |

hybrid_v2 has the lowest per-pass-criterion cost among the
high-quality pipelines, even though its absolute cost is the highest.
At realistic deal-room scale (10 of these tasks per matter), hybrid_v2
is ~$25 per matter — well within Harvey's per-matter implicit ceiling
of $60-120.

## Architecture observations

The hybrid_v1 → hybrid_v2 jump (56.4% → 61.8%) came from one change:
threading the source text into the per-contract narrative call. This
is concrete evidence that the schema captures *aggregates* (one CoC
definition, one risk rating per contract) but the rubric tests
*sub-clause depth* (Section 2.09 prepayment chain, four carve-out
conditions, two-definition consistency check). The source text is
where the depth lives.

The implication: **structured extraction is a backbone, not a
substitute for source-text reasoning.** A schema that exhaustively
captured every rubric sub-clause would be 200+ columns and brittle to
schema changes. A simpler schema (28 columns) + per-contract source-
text-aware synthesis is the right architecture.

## Next steps to push toward 80%+ (the union ceiling)

1. **`change-of-control-review` required-analyses recipe (E5-E7)** —
   forces explicit cross-cutting analyses (UCC § 2-210, transaction
   structure, carve-out conditions) regardless of whether the
   per-contract narrative covers them. Estimated +3-5 criteria.
2. **`kaos-graph` cross-document checks (E9-E11)** — surface
   unreviewed dependent systems (C-042 Crestline ERP), risk-rating
   completeness gaps, conflicting provisions across contracts.
   Estimated +2-3 criteria, including hard-misses.
3. **`Grounded` verifier for citation quality (E13)** — catches when
   the per-contract narrative paraphrases instead of quoting verbatim.
   Estimated +1-2 criteria.
4. **MultiChainComparison for cross-cutting analyses (E14)** —
   sample N reasoning chains for the hardest legal-reasoning
   criteria (C-007 dual-definition consistency, C-013-C-015
   carve-out conditions). Quality lift on synthesis-heavy criteria.

If 1+2+3 land cleanly, target is 40-44/55 (~75-80%). If 4 helps on
the hardest 4 criteria, target is 44-48/55 (~80-87%).
