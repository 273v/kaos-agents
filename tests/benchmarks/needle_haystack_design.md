# Needle-in-Haystack Benchmark: Multi-Phase Agent Planning

## Goal

Test the kaos-agents planning layer on a realistic legal research task
that requires multiple phases to solve. The agent must find 2-3 specific
documents in a 997-document corpus, extract terms from each, cross-reference
them, and synthesize a coherent answer with citations.

## Data Source

`alea-institute/kl3m-data-edgar-agreements-sample` — 997 real SEC-filed
contract exhibits (employment agreements, mergers, leases, credit facilities).
Documents are decoded from kl3m tokenizer back to raw text.

## Benchmark Scenarios

### Scenario 1: "Employment Agreement Cross-Reference"

**Question**: "Which employment agreements in this corpus have non-compete
clauses with durations longer than 2 years? For each, identify the executive
name, company, and the exact non-compete duration and geographic scope."

**Why it's hard**:
- 997 documents, only ~74 contain non-compete language
- Of those, only a subset have explicit durations > 2 years
- Agent must: triage → extract → filter → synthesize

**Expected phases**:
1. **Collection**: BM25 triage ("non-compete" OR "non-competition" OR "restrictive covenant") → ~74 docs
2. **Extraction**: Extract non-compete duration, executive, company from each
3. **Filter**: Keep only those with duration > 2 years
4. **Interpretation**: Synthesize answer with citations to specific agreements

### Scenario 2: "Change of Control Consequence Analysis"

**Question**: "Find all agreements that define 'Change of Control' and describe
what happens to the executive's compensation or vesting upon a Change of Control
event. Compare the acceleration provisions across at least 3 different agreements."

**Why it's hard**:
- 138 documents mention "change of control"
- The consequence (acceleration, termination, etc.) is usually in a different
  section than the definition
- Agent must cross-reference definitions with consequence provisions
- Comparison requires structured extraction from multiple documents

**Expected phases**:
1. **Collection**: BM25 triage ("change of control") → ~138 docs
2. **Analysis**: For each, extract the CoC definition + consequence provisions
3. **Review**: Identify which have acceleration vs. termination vs. other consequences
4. **Interpretation**: Compare at least 3 and synthesize differences

### Scenario 3: "Find the Conflicting Terms"

**Question**: "Are there any agreements in this corpus where the same company
appears as a party in both an employment agreement and a separate purchase/merger
agreement? If so, identify the company and describe whether the employment
agreement's change of control provisions would be triggered by the merger."

**Why it's hard**:
- Requires entity extraction across ALL documents to find company overlaps
- Then cross-referencing employment terms with merger terms
- True needle-in-haystack: the overlap may exist in only 2-3 documents out of 997
- Requires multi-hop reasoning

**Expected phases**:
1. **Collection**: Extract party names from all agreements (entity extraction at scale)
2. **Analysis**: Find companies that appear in both employment AND merger/purchase agreements
3. **Cross-reference**: For matches, check if the merger would trigger the employment CoC clause
4. **Interpretation**: Synthesize the conflict analysis with citations

## Metrics

| Metric | Description |
|--------|-------------|
| **Recall** | Did the agent find all relevant documents? |
| **Precision** | How many irrelevant documents were examined? |
| **Plan depth** | How many plan steps were generated? |
| **Plan phases** | Did the agent naturally decompose into collection → analysis → review → interpretation? |
| **Cost** | Total API cost (tokens × price) |
| **Latency** | Wall-clock time |
| **Citation accuracy** | Are extracted terms grounded in actual document text? |

## Implementation

The benchmark loads the 997 documents into SessionMemory DOCUMENTS section,
runs each scenario through the Runner, and measures the metrics above.
Uses the retrieval-augmented context assembly and corpus triage.
