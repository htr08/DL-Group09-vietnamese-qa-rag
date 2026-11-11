"""Hybrid search against Qdrant."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple

from ..vectorstore import QdrantVectorStore
from .types import HitType
from .utils import extract_hits, hit_matches_filters

try:
    from qdrant_client import models as qdrant_models
except Exception:
    qdrant_models = None


class HybridSearcher:
    """Execute dense and sparse searches in parallel."""

    def __init__(
        self,
        vector_store: QdrantVectorStore,
        collection_name: str,
        sparse_vector_names: List[str],
    ) -> None:
        self.vector_store = vector_store
        self.collection_name = collection_name
        self.sparse_vector_names = sparse_vector_names

    def search_parallel(
        self,
        dense_vector: List[float],
        sparse_vector: Dict[str, Any],
        dense_limit: int,
        sparse_limit: int,
        metadata_filters: Optional[Dict[str, Any]] = None,
    ) -> Tuple[List[HitType], List[HitType]]:
        """Search dense and sparse branches concurrently."""

        with ThreadPoolExecutor(max_workers=2) as executor:
            dense_future = executor.submit(
                self.dense_search,
                dense_vector,
                dense_limit,
                metadata_filters,
            )
            sparse_future = executor.submit(
                self.sparse_search,
                sparse_vector,
                sparse_limit,
                metadata_filters,
            )

            dense_hits = dense_future.result()
            try:
                sparse_hits = sparse_future.result()
            except Exception:
                sparse_hits = []
        return dense_hits, sparse_hits

    def dense_search(
        self,
        query_vector: List[float],
        limit: int,
        metadata_filters: Optional[Dict[str, Any]] = None,
    ) -> List[HitType]:
        hits = self.vector_store.search(query_vector=query_vector, limit=limit)
        if not metadata_filters:
            return hits
        return [hit for hit in hits if hit_matches_filters(hit, metadata_filters)]

    def sparse_search(
        self,
        sparse_vector: Dict[str, Any],
        limit: int,
        metadata_filters: Optional[Dict[str, Any]] = None,
    ) -> List[HitType]:
        indices = sparse_vector.get("indices") or []
        values = sparse_vector.get("values") or []
        if not indices or not values:
            return []

        # Neu vector_store ho tro sparse_search (nhu InMemory de test), uu tien no
        if hasattr(self.vector_store, "sparse_search"):
            try:
                hits = self.vector_store.sparse_search(sparse_vector, limit)
                return [
                    hit for hit in hits if hit_matches_filters(hit, metadata_filters)
                ]
            except Exception:
                # Fallback to client-based search
                pass

        client = getattr(self.vector_store, "client", None)
        if client is None:
            return []

        sparse_query: Any
        if qdrant_models is not None and hasattr(qdrant_models, "SparseVector"):
            sparse_query = qdrant_models.SparseVector(indices=indices, values=values)
        else:
            sparse_query = {"indices": indices, "values": values}

        if hasattr(client, "query_points"):
            for using in self.sparse_vector_names:
                try:
                    response = client.query_points(
                        collection_name=self.collection_name,
                        query=sparse_query,
                        using=using,
                        limit=limit,
                        with_payload=True,
                    )
                    hits = extract_hits(response)
                    if hits:
                        if not metadata_filters:
                            return hits
                        filtered = [
                            hit
                            for hit in hits
                            if hit_matches_filters(hit, metadata_filters)
                        ]
                        if filtered:
                            return filtered
                except Exception:
                    continue

        if hasattr(client, "search"):
            for using in self.sparse_vector_names:
                try:
                    if (
                        qdrant_models is not None
                        and hasattr(qdrant_models, "NamedSparseVector")
                        and hasattr(qdrant_models, "SparseVector")
                    ):
                        search_vector = qdrant_models.NamedSparseVector(
                            name=using,
                            vector=qdrant_models.SparseVector(
                                indices=indices, values=values
                            ),
                        )
                    else:
                        search_vector = (using, sparse_query)

                    response = client.search(
                        collection_name=self.collection_name,
                        query_vector=search_vector,
                        limit=limit,
                        with_payload=True,
                    )
                    hits = extract_hits(response)
                    if hits:
                        if not metadata_filters:
                            return hits
                        filtered = [
                            hit
                            for hit in hits
                            if hit_matches_filters(hit, metadata_filters)
                        ]
                        if filtered:
                            return filtered
                except Exception:
                    continue
        return []
