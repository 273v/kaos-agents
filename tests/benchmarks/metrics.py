"""Shared IR evaluation metrics for BEIR benchmarks.

Single source of truth for NDCG@k, AP@k, and Recall@k computation.
All benchmark scripts import from here — no copy-pasting metric functions.
"""

from __future__ import annotations

import math


def compute_ndcg(ranked_ids: list[str], qrel_scores: dict[str, int], k: int) -> float:
    """Compute Normalized Discounted Cumulative Gain at k.

    Args:
        ranked_ids: Document IDs in ranked order (best first).
        qrel_scores: Ground truth relevance scores per document ID.
        k: Cutoff rank.

    Returns:
        NDCG@k in [0.0, 1.0].
    """
    dcg = 0.0
    for i, doc_id in enumerate(ranked_ids[:k]):
        rel = qrel_scores.get(doc_id, 0)
        dcg += (2**rel - 1) / math.log2(i + 2)

    ideal_rels = sorted(qrel_scores.values(), reverse=True)[:k]
    idcg = sum((2**r - 1) / math.log2(i + 2) for i, r in enumerate(ideal_rels))

    return dcg / idcg if idcg > 0 else 0.0


def compute_ap(ranked_ids: list[str], relevant: set[str], k: int) -> float:
    """Compute Average Precision at k.

    Args:
        ranked_ids: Document IDs in ranked order (best first).
        relevant: Set of relevant document IDs (binary relevance).
        k: Cutoff rank.

    Returns:
        AP@k in [0.0, 1.0].
    """
    hits = 0
    sum_precision = 0.0
    for i, doc_id in enumerate(ranked_ids[:k]):
        if doc_id in relevant:
            hits += 1
            sum_precision += hits / (i + 1)
    return sum_precision / len(relevant) if relevant else 0.0


def compute_recall(ranked_ids: list[str], relevant: set[str], k: int) -> float:
    """Compute Recall at k.

    Args:
        ranked_ids: Document IDs in ranked order (best first).
        relevant: Set of relevant document IDs (binary relevance).
        k: Cutoff rank.

    Returns:
        Recall@k in [0.0, 1.0].
    """
    found = sum(1 for doc_id in ranked_ids[:k] if doc_id in relevant)
    return found / len(relevant) if relevant else 0.0
