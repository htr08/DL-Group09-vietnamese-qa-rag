"""Chỉ số đánh giá: EM/F1, Recall@k, MRR@10, bootstrap CI."""
from typing import List, Union, Any


def recall_at_k(retrieved_ids: List[Any], relevant_id: Any, k: int) -> float:
    """
    Tính Recall@k cho 1 truy vấn.
    Trả về 1.0 nếu relevant_id xuất hiện trong top-k retrieved_ids, ngược lại trả về 0.0.
    """
    if not retrieved_ids or relevant_id is None:
        return 0.0
    top_k = retrieved_ids[:k]
    return 1.0 if relevant_id in top_k else 0.0


def mrr_at_k(retrieved_ids: List[Any], relevant_id: Any, k: int = 10) -> float:
    """
    Tính Reciprocal Rank (MRR@k) cho 1 truy vấn.
    Nếu relevant_id nằm ở vị trí rank (1-indexed) trong top-k thì trả về 1 / rank,
    ngược lại trả về 0.0.
    """
    if not retrieved_ids or relevant_id is None:
        return 0.0
    top_k = retrieved_ids[:k]
    for rank, doc_id in enumerate(top_k, start=1):
        if doc_id == relevant_id:
            return 1.0 / rank
    return 0.0


def mean_metric(scores: List[float]) -> float:
    """Tính trung bình của một mảng chỉ số."""
    return sum(scores) / len(scores) if scores else 0.0