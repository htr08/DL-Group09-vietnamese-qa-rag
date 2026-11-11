"""Result fusion logic."""

from typing import Dict, List

from .types import MergedHit, HitType
from .utils import hit_key, hit_score


class RRFFuser:
    """Fuse dense and sparse result lists with Reciprocal Rank Fusion."""

    def fuse(
        self, dense_hits: List[HitType], sparse_hits: List[HitType], rrf_k: int
    ) -> List[MergedHit]:
        merged: Dict[str, MergedHit] = {}

        for rank, hit in enumerate(dense_hits, start=1):
            key = hit_key(hit)
            item = merged.get(key)
            if item is None:
                item = MergedHit(key=key, hit=hit)
                merged[key] = item
            item.dense_rank = rank
            item.dense_score = hit_score(hit)
            item.rrf_score += 1.0 / (rrf_k + rank)

        for rank, hit in enumerate(sparse_hits, start=1):
            key = hit_key(hit)
            item = merged.get(key)
            if item is None:
                item = MergedHit(key=key, hit=hit)
                merged[key] = item
            item.sparse_rank = rank
            item.sparse_score = hit_score(hit)
            item.rrf_score += 1.0 / (rrf_k + rank)

        fused = list(merged.values())
        for item in fused:
            item.final_score = item.rrf_score
        fused.sort(key=lambda x: (x.rrf_score, x.dense_score or 0.0), reverse=True)
        return fused
