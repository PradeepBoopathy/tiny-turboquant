"""Retrieval metrics for compressed-vector and RAG experiments."""

from __future__ import annotations

import math
from typing import Iterable, Sequence, Set


def _as_set(values: Iterable[object]) -> Set[object]:
    return set(values)


def recall_at_k(retrieved: Sequence[object], relevant: Iterable[object], k: int | None = None) -> float:
    """Recall@k for ID-based retrieval.

    Args:
        retrieved: Ordered retrieved IDs.
        relevant: Relevant IDs.
        k: Cutoff. If omitted, uses the full retrieved list.
    """
    relevant_set = _as_set(relevant)
    if not relevant_set:
        return 0.0
    cutoff = len(retrieved) if k is None else max(0, int(k))
    got = set(retrieved[:cutoff])
    return len(got & relevant_set) / len(relevant_set)


def precision_at_k(retrieved: Sequence[object], relevant: Iterable[object], k: int | None = None) -> float:
    relevant_set = _as_set(relevant)
    cutoff = len(retrieved) if k is None else max(0, int(k))
    if cutoff == 0:
        return 0.0
    got = list(retrieved[:cutoff])
    return sum(1 for item in got if item in relevant_set) / cutoff


def mrr_at_k(retrieved: Sequence[object], relevant: Iterable[object], k: int | None = None) -> float:
    """Mean reciprocal rank for a single query."""
    relevant_set = _as_set(relevant)
    cutoff = len(retrieved) if k is None else max(0, int(k))
    for rank, item in enumerate(retrieved[:cutoff], start=1):
        if item in relevant_set:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved: Sequence[object], relevant: Iterable[object], k: int | None = None) -> float:
    """Binary nDCG@k for a single query."""
    relevant_set = _as_set(relevant)
    cutoff = len(retrieved) if k is None else max(0, int(k))
    if cutoff == 0 or not relevant_set:
        return 0.0

    dcg = 0.0
    for rank, item in enumerate(retrieved[:cutoff], start=1):
        if item in relevant_set:
            dcg += 1.0 / math.log2(rank + 1)

    ideal_hits = min(len(relevant_set), cutoff)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


def exact_overlap(a: Sequence[object], b: Sequence[object], k: int | None = None) -> float:
    """Set overlap ratio between two top-k result lists."""
    cutoff = min(len(a), len(b)) if k is None else max(0, int(k))
    if cutoff == 0:
        return 0.0
    sa = set(a[:cutoff])
    sb = set(b[:cutoff])
    return len(sa & sb) / cutoff


def label_match_ratio(retrieved_labels: Sequence[object], expected_label: object, k: int | None = None) -> float:
    """Fraction of top-k labels equal to the expected label."""
    cutoff = len(retrieved_labels) if k is None else max(0, int(k))
    if cutoff == 0:
        return 0.0
    return sum(1 for label in retrieved_labels[:cutoff] if label == expected_label) / cutoff


__all__ = [
    "recall_at_k",
    "precision_at_k",
    "mrr_at_k",
    "ndcg_at_k",
    "exact_overlap",
    "label_match_ratio",
]
