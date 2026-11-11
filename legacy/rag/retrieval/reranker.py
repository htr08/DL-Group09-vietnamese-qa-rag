"""Heading-aware reranking."""

from __future__ import annotations

import math
import re
from typing import Dict, List, Optional, Set

from ..embeddings import GeminiEmbeddingService
from .query_encoder import tokenize
from .types import MergedHit
from .utils import extract_heading_text, hit_payload


class HeadingReranker:
    """Rerank fused hits by heading relevance to query."""

    def __init__(
        self,
        embedding_service: GeminiEmbeddingService,
        rerank_top_k: int,
        alpha: float,
        beta: float,
        use_runtime_heading_embedding: bool = True,
    ) -> None:
        self.embedding_service = embedding_service
        self.rerank_top_k = rerank_top_k
        self.alpha = alpha
        self.beta = beta
        self.use_runtime_heading_embedding = use_runtime_heading_embedding

    def rerank(
        self,
        question: str,
        query_dense_vector: List[float],
        merged_hits: List[MergedHit],
    ) -> List[MergedHit]:
        if not merged_hits:
            return []

        rerank_candidates = merged_hits[: self.rerank_top_k]
        remainder = merged_hits[self.rerank_top_k :]

        query_tokens_list = tokenize(question)
        query_tokens = set(query_tokens_list)
        semantic_weight, lexical_weight = self._adaptive_heading_weights(
            question, query_tokens_list
        )
        heading_texts = [extract_heading_text(hit.hit) for hit in rerank_candidates]
        heading_vectors = self._resolve_heading_vectors(
            rerank_candidates, heading_texts
        )

        for idx, item in enumerate(rerank_candidates):
            heading_text = heading_texts[idx]
            lexical_sim = self._heading_token_similarity(query_tokens, heading_text)
            semantic_sim = 0.0
            vector = heading_vectors.get(idx)
            if vector:
                semantic_sim = self._normalized_cosine(query_dense_vector, vector)

            heading_score = (
                semantic_weight * semantic_sim + lexical_weight * lexical_sim
            )
            item.heading_score = heading_score
            # Sử dụng công thức nhân (Boost) để Heading chỉ đóng vai trò buff tỉ lệ nhỏ (ví dụ 25%)
            # chứ không đè bẹp điểm RRF của toàn bộ chunk.
            item.final_score = (
                self.alpha * item.rrf_score * (1.0 + self.beta * heading_score)
            )

        rerank_candidates.sort(
            key=lambda x: (x.final_score, x.rrf_score, x.heading_score),
            reverse=True,
        )

        for item in remainder:
            item.final_score = item.rrf_score
        return rerank_candidates + remainder

    def _resolve_heading_vectors(
        self,
        merged_hits: List[MergedHit],
        heading_texts: List[str],
    ) -> Dict[int, List[float]]:
        vectors: Dict[int, List[float]] = {}
        missing_indices: List[int] = []

        for idx, merged in enumerate(merged_hits):
            vector = self._precomputed_heading_vector(merged)
            if vector is not None:
                vectors[idx] = vector
            else:
                missing_indices.append(idx)

        if not self.use_runtime_heading_embedding or not missing_indices:
            return vectors

        missing_texts = [
            heading_texts[idx]
            for idx in missing_indices
            if isinstance(heading_texts[idx], str) and heading_texts[idx].strip()
        ]
        if not missing_texts:
            return vectors

        try:
            embedded = self.embedding_service.embed_documents(missing_texts)
        except Exception:
            return vectors

        emb_idx = 0
        for idx in missing_indices:
            text = heading_texts[idx]
            if not isinstance(text, str) or not text.strip():
                continue
            if emb_idx >= len(embedded):
                break
            vectors[idx] = embedded[emb_idx]
            emb_idx += 1
        return vectors

    @staticmethod
    def _precomputed_heading_vector(merged_hit: MergedHit) -> Optional[List[float]]:
        payload = hit_payload(merged_hit.hit)
        metadata = payload.get("metadata") or {}
        for key in ("heading_embedding", "heading_vector", "section_embedding"):
            value = metadata.get(key)
            if (
                isinstance(value, list)
                and value
                and all(isinstance(x, (int, float)) for x in value)
            ):
                return [float(x) for x in value]
        return None

    @staticmethod
    def _adaptive_heading_weights(
        question: str, tokens: Optional[List[str]] = None
    ) -> tuple[float, float]:
        tokens = tokens if tokens is not None else tokenize(question)
        if not tokens:
            return 0.7, 0.3

        has_model_like = bool(
            re.search(r"\b([A-Z]{2,}\d+|\d{2,}GB|\d{2,}TB|SKU|SM-\w+)\b", question)
        )
        digit_ratio = sum(any(ch.isdigit() for ch in t) for t in tokens) / max(
            1, len(tokens)
        )
        avg_len = sum(len(t) for t in tokens) / len(tokens)

        # Keyword-heavy query: increase lexical contribution.
        if (
            has_model_like
            or digit_ratio >= 0.35
            or (len(tokens) <= 6 and avg_len >= 4.5)
        ):
            return 0.45, 0.55
        return 0.75, 0.25

    def _embed_headings(self, heading_texts: List[str]) -> Dict[int, List[float]]:
        # Kept for backward compatibility; prefer _resolve_heading_vectors.
        index_map = {i: text for i, text in enumerate(heading_texts) if text}
        if not index_map:
            return {}

        texts = [index_map[i] for i in sorted(index_map.keys())]
        try:
            vectors = self.embedding_service.embed_documents(texts)
        except Exception:
            return {}

        out: Dict[int, List[float]] = {}
        for idx, vector in zip(sorted(index_map.keys()), vectors):
            out[idx] = vector
        return out

    @staticmethod
    def _heading_token_similarity(query_tokens: Set[str], heading_text: str) -> float:
        if not query_tokens or not heading_text:
            return 0.0
        heading_tokens = set(tokenize(heading_text))
        if not heading_tokens:
            return 0.0
        overlap = len(query_tokens & heading_tokens)
        if overlap == 0:
            return 0.0
        return overlap / math.sqrt(len(query_tokens) * len(heading_tokens))

    @staticmethod
    def _normalized_cosine(left: List[float], right: List[float]) -> float:
        if not left or not right or len(left) != len(right):
            return 0.0
        dot = sum(a * b for a, b in zip(left, right))
        left_norm = math.sqrt(sum(a * a for a in left))
        right_norm = math.sqrt(sum(b * b for b in right))
        if left_norm == 0.0 or right_norm == 0.0:
            return 0.0
        cosine = dot / (left_norm * right_norm)
        cosine = max(-1.0, min(1.0, cosine))
        return (cosine + 1.0) / 2.0
