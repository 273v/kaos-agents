"""Agent-level BEIR evaluation — runs the REAL kaos-agent on each query.

Unlike beir_eval.py which tests retrieval functions in isolation, this
benchmark runs the full agent pipeline:

    Runner.run(query) → IntentClassified → RAG → CitationFound / InsufficientEvidence → Answer

Evaluates:
1. Did the agent find relevant documents? (retrieval recall via cited doc IDs)
2. Did the agent answer or correctly refuse? (answer vs. insufficient)
3. Are citations grounded in actual relevant documents? (citation precision)
4. Standard IR metrics (NDCG@10, Recall@10, Recall@50) from cited documents vs qrels

Usage::

    # Run on NFCorpus (all 323 test queries)
    uv run python tests/benchmarks/agent_eval.py --dataset nfcorpus

    # Run on SciFact (all 300 test queries)
    uv run python tests/benchmarks/agent_eval.py --dataset scifact

    # Run on FiQA (all test queries)
    uv run python tests/benchmarks/agent_eval.py --dataset fiqa

    # Limit queries for quick test
    uv run python tests/benchmarks/agent_eval.py --dataset nfcorpus --max-queries 10

    # Save results (auto-saves to docs/benchmarks/ if --json not specified)
    uv run python tests/benchmarks/agent_eval.py --dataset nfcorpus --json results.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from tests.benchmarks.beir_eval import load_beir_dataset
from tests.benchmarks.metrics import compute_ndcg, compute_recall

_BEIR_DATASETS = ("nfcorpus", "scifact", "fiqa")
_LATENCY_WARN_THRESHOLD_S = 30.0


@dataclass(frozen=True, slots=True)
class AgentQueryResult:
    """Result from running the agent on one query."""

    query_id: str
    query_text: str
    ground_truth_count: int
    agent_answered: bool
    agent_refused: bool
    agent_errored: bool
    num_citations: int
    cited_doc_ids: tuple[str, ...] = ()
    answer_length: int = 0
    intent: str = ""
    intent_confidence: float = 0.0
    latency_s: float = 0.0
    ndcg_at_10: float = 0.0
    recall_at_10: float = 0.0
    recall_at_50: float = 0.0
    error_message: str = ""


@dataclass(slots=True)  # Mutable: accumulated
class AgentEvalResult:
    """Aggregate results from running the agent on a dataset."""

    dataset: str = ""
    corpus_size: int = 0
    num_queries: int = 0
    num_answered: int = 0
    num_refused: int = 0
    num_errored: int = 0
    avg_citations: float = 0.0
    avg_latency_s: float = 0.0
    answer_rate: float = 0.0
    ndcg_at_10: float = 0.0
    recall_at_10: float = 0.0
    recall_at_50: float = 0.0
    queries: list[AgentQueryResult] = field(default_factory=list)


async def run_agent_eval(
    corpus: dict[str, str],
    queries: dict[str, str],
    qrels: dict[str, dict[str, int]],
    test_qids: list[str],
    dataset_name: str,
    *,
    max_queries: int | None = None,
    model: str | None = None,
) -> AgentEvalResult:
    """Run the real agent on each query and collect results."""
    from kaos_core.registry.container import KaosRuntime
    from kaos_core.vfs.core import VirtualFileSystem

    from kaos_agents.config import Agent
    from kaos_agents.events import (
        CitationFound,
        EvidenceInsufficient,
        IntentClassified,
        RunError,
        TextDelta,
    )
    from kaos_agents.memory.store import SessionStore
    from kaos_agents.runtime.runner import Runner
    from kaos_agents.tools import register_agent_tools
    from kaos_agents.types.memory import MemoryType

    # Set up infrastructure
    vfs = VirtualFileSystem()
    store = SessionStore(vfs)
    session_id = f"agent-eval-{dataset_name}"

    # Pre-load corpus into session memory
    memory = await store.load_or_create(session_id)
    for doc_id, text in corpus.items():
        memory.add(MemoryType.DOCUMENTS, text, metadata={"uri": doc_id})
    await store.save(memory)

    n_docs = memory.section_item_count(MemoryType.DOCUMENTS)
    sys.stdout.write(f"  Loaded {n_docs} docs into agent session\n")
    sys.stdout.flush()

    # Build agent
    runtime = KaosRuntime.default()
    register_agent_tools(runtime)

    from kaos_agents.settings import DEFAULT_MODEL

    agent = Agent.create(
        instructions=(
            f"You are a research assistant with access to {n_docs} documents. "
            "Answer questions by searching the corpus and citing specific documents. "
            "If you cannot find sufficient evidence, say so explicitly."
        ),
        model=model or DEFAULT_MODEL,
        pattern="research",
    )
    runner = Runner(agent, runtime=runtime, vfs=vfs)

    # Run queries
    eval_qids = test_qids[:max_queries] if max_queries else test_qids
    result = AgentEvalResult(dataset=dataset_name, corpus_size=n_docs)

    for i, qid in enumerate(eval_qids):
        query_text = queries[qid]
        qrel_scores = qrels.get(qid, {})
        gt_count = len(qrel_scores)
        relevant = {did for did, score in qrel_scores.items() if score > 0}

        t0 = time.perf_counter()
        text_parts: list[str] = []
        cited_doc_ids: list[str] = []
        refused = False
        errored = False
        error_msg = ""
        intent = ""
        intent_conf = 0.0

        try:
            async for event in runner.run(query_text, session_id):
                if isinstance(event, IntentClassified):
                    intent = event.intent
                    intent_conf = event.confidence
                elif isinstance(event, CitationFound):
                    if event.source_uri:
                        cited_doc_ids.append(event.source_uri)
                elif isinstance(event, EvidenceInsufficient):
                    refused = True
                elif isinstance(event, TextDelta):
                    text_parts.append(event.content)
                elif isinstance(event, RunError):
                    errored = True
                    error_msg = event.message[:200]
        except Exception as exc:
            errored = True
            error_msg = f"{type(exc).__name__}: {exc}"[:200]

        t1 = time.perf_counter()
        latency = t1 - t0
        answer_text = "".join(text_parts)
        answered = bool(answer_text) and not refused and not errored

        # Compute IR metrics from cited documents against qrels
        q_ndcg = compute_ndcg(cited_doc_ids, qrel_scores, 10)
        q_recall_10 = compute_recall(cited_doc_ids, relevant, 10)
        q_recall_50 = compute_recall(cited_doc_ids, relevant, 50)

        qr = AgentQueryResult(
            query_id=qid,
            query_text=query_text[:100],
            ground_truth_count=gt_count,
            agent_answered=answered,
            agent_refused=refused,
            agent_errored=errored,
            num_citations=len(cited_doc_ids),
            cited_doc_ids=tuple(cited_doc_ids),
            answer_length=len(answer_text),
            intent=intent,
            intent_confidence=intent_conf,
            latency_s=latency,
            ndcg_at_10=q_ndcg,
            recall_at_10=q_recall_10,
            recall_at_50=q_recall_50,
            error_message=error_msg,
        )
        result.queries.append(qr)

        if latency > _LATENCY_WARN_THRESHOLD_S:
            sys.stderr.write(
                f"WARNING: query {qid} took {latency:.1f}s (>{_LATENCY_WARN_THRESHOLD_S:.0f}s)\n"
            )

        status = "ANSWER" if answered else ("REFUSE" if refused else "ERROR")
        sys.stdout.write(
            f"  [{i + 1}/{len(eval_qids)}] {status} cites={len(cited_doc_ids)} "
            f"ndcg={q_ndcg:.3f} {latency:.1f}s: {query_text[:60]}\n"
        )
        sys.stdout.flush()

    # Compute aggregates
    n = len(result.queries)
    result.num_queries = n
    result.num_answered = sum(1 for q in result.queries if q.agent_answered)
    result.num_refused = sum(1 for q in result.queries if q.agent_refused)
    result.num_errored = sum(1 for q in result.queries if q.agent_errored)
    result.avg_citations = sum(q.num_citations for q in result.queries) / n if n else 0
    result.avg_latency_s = sum(q.latency_s for q in result.queries) / n if n else 0
    result.answer_rate = result.num_answered / result.num_queries if result.num_queries else 0
    result.ndcg_at_10 = sum(q.ndcg_at_10 for q in result.queries) / n if n else 0
    result.recall_at_10 = sum(q.recall_at_10 for q in result.queries) / n if n else 0
    result.recall_at_50 = sum(q.recall_at_50 for q in result.queries) / n if n else 0

    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Agent-level BEIR evaluation")
    parser.add_argument("--dataset", choices=_BEIR_DATASETS, required=True)
    parser.add_argument(
        "--max-queries", type=int, default=None, help="Limit queries (default: all)"
    )
    parser.add_argument("--model", default=None, help="LLM model (default: from KaosAgentSettings)")
    parser.add_argument("--json", type=str, default=None)
    args = parser.parse_args(argv)

    sys.stdout.write(f"Loading {args.dataset}...\n")
    sys.stdout.flush()
    corpus, queries, qrels, test_qids = load_beir_dataset(args.dataset)
    n_queries = args.max_queries or len(test_qids)
    sys.stdout.write(f"  Corpus: {len(corpus)} docs, Queries: {n_queries}/{len(test_qids)} test\n")
    sys.stdout.flush()

    result = asyncio.run(
        run_agent_eval(
            corpus,
            queries,
            qrels,
            test_qids,
            args.dataset,
            max_queries=args.max_queries,
            model=args.model,
        )
    )

    sys.stdout.write(f"\n{'=' * 60}\n")
    sys.stdout.write(f"AGENT EVALUATION: {args.dataset}\n")
    sys.stdout.write(f"{'=' * 60}\n")
    sys.stdout.write(f"Corpus:      {result.corpus_size} docs\n")
    sys.stdout.write(f"Queries:     {result.num_queries}\n")
    sys.stdout.write(f"Answered:    {result.num_answered} ({result.answer_rate:.1%})\n")
    sys.stdout.write(f"Refused:     {result.num_refused}\n")
    sys.stdout.write(f"Errored:     {result.num_errored}\n")
    sys.stdout.write(f"Avg cites:   {result.avg_citations:.1f}\n")
    sys.stdout.write(f"Avg latency: {result.avg_latency_s:.1f}s\n")
    sys.stdout.write(f"NDCG@10:     {result.ndcg_at_10:.3f}\n")
    sys.stdout.write(f"Recall@10:   {result.recall_at_10:.1%}\n")
    sys.stdout.write(f"Recall@50:   {result.recall_at_50:.1%}\n")
    sys.stdout.flush()

    # Determine JSON output path
    json_path: str | None = args.json
    if json_path is None:
        today = time.strftime("%Y-%m-%d")
        benchmarks_dir = Path(__file__).resolve().parent.parent.parent / "docs" / "benchmarks"
        json_path = str(benchmarks_dir / f"agent-{args.dataset}-{today}.json")
        sys.stderr.write(f"WARNING: --json not specified; results will be saved to {json_path}\n")

    out_path = Path(json_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    output = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "dataset": args.dataset,
        "model": args.model,
        "corpus_size": result.corpus_size,
        "summary": {
            "num_queries": result.num_queries,
            "num_answered": result.num_answered,
            "num_refused": result.num_refused,
            "num_errored": result.num_errored,
            "answer_rate": result.answer_rate,
            "avg_citations": result.avg_citations,
            "avg_latency_s": result.avg_latency_s,
            "ndcg_at_10": result.ndcg_at_10,
            "recall_at_10": result.recall_at_10,
            "recall_at_50": result.recall_at_50,
        },
        "queries": [asdict(q) for q in result.queries],
    }
    out_path.write_text(json.dumps(output, indent=2, default=str))
    sys.stdout.write(f"\nSaved to {json_path}\n")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
