"""Shared retrieval types."""

from dataclasses import dataclass
from typing import Any, Optional, Union

try:
    from qdrant_client.models import ScoredPoint
except ImportError:
    ScoredPoint = Any

HitType = Union[ScoredPoint, dict, Any]


@dataclass
class MergedHit:
    """Merged retrieval hit after fusion/reranking."""

    key: str
    hit: HitType
    rrf_score: float = 0.0
    dense_rank: Optional[int] = None
    sparse_rank: Optional[int] = None
    dense_score: Optional[float] = None
    sparse_score: Optional[float] = None
    heading_score: float = 0.0
    final_score: float = 0.0
