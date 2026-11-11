"""Qdrant vector store adapter."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from langchain_core.documents import Document
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from ..config import RAGSettings, rag_settings

try:
    from qdrant_client import models as qdrant_models
except Exception:
    qdrant_models = None


class QdrantVectorStore:
    """Encapsulate Qdrant operations used by ingestion and retrieval."""

    def __init__(
        self,
        settings: RAGSettings = rag_settings,
        qdrant_client: Optional[QdrantClient] = None,
        collection_name: Optional[str] = None,
    ) -> None:
        self.settings = settings
        self.collection_name = collection_name or settings.QDRANT_COLLECTION_NAME
        self.client = qdrant_client or QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.QDRANT_API_KEY,
            timeout=settings.QDRANT_TIMEOUT_SECONDS,
        )

        self.dense_vector_name = str(
            getattr(settings, "QDRANT_DENSE_VECTOR_NAME", "dense")
        )
        sparse_names_setting = getattr(settings, "QDRANT_SPARSE_VECTOR_NAMES", "")
        if isinstance(sparse_names_setting, str) and sparse_names_setting.strip():
            sparse_names = [
                part.strip() for part in sparse_names_setting.split(",") if part.strip()
            ]
        else:
            sparse_names = ["sparse"]
        self.sparse_vector_names = sparse_names
        self.primary_sparse_vector_name = sparse_names[0]

    def ensure_collection(self, vector_size: int) -> None:
        """Create collection when it does not exist."""

        if self.collection_exists():
            self._best_effort_enable_sparse_vectors()
            return

        dense_config = {
            self.dense_vector_name: VectorParams(size=vector_size, distance=Distance.COSINE)
        }
        sparse_config = self._build_sparse_vectors_config()

        try:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=dense_config,
                sparse_vectors_config=sparse_config,
            )
        except Exception as exc:
            exc_str = str(exc).lower()
            if "already exists" in exc_str or "409" in exc_str:
                self._best_effort_enable_sparse_vectors()
                return
            # Fallback for legacy Qdrant/client setups that don't support sparse vectors config.
            try:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
                )
            except Exception as exc2:
                if "already exists" in str(exc2).lower() or "409" in str(exc2):
                    pass
                else:
                    raise
            self._best_effort_enable_sparse_vectors()

    def collection_exists(self) -> bool:
        """Return whether target collection exists."""

        try:
            self.client.get_collection(self.collection_name)
            return True
        except Exception:
            return False

    def upsert_chunks(
        self,
        chunks: List[Document],
        vectors: List[List[float]],
        sparse_vectors: Optional[List[Dict[str, Any]]] = None,
    ) -> List[str]:
        """Upsert chunks with dense (+optional sparse) vectors into Qdrant."""

        chunk_ids: List[str] = []
        dense_only_points: List[PointStruct] = []
        named_dense_points: List[PointStruct] = []
        named_dense_sparse_points: List[PointStruct] = []

        has_sparse = bool(
            sparse_vectors
            and len(sparse_vectors) == len(chunks)
            and self.primary_sparse_vector_name
        )

        for idx, (chunk, vector) in enumerate(zip(chunks, vectors)):
            chunk_id = str(chunk.metadata["chunk_id"])
            chunk_ids.append(chunk_id)

            payload = {
                "page_content": chunk.page_content,
                "metadata": chunk.metadata,
                "filename": chunk.metadata.get("filename"),
                "page": chunk.metadata.get("page"),
                "chunk_id": chunk_id,
                "chunk_index": chunk.metadata.get("chunk_index"),
                "file_type": chunk.metadata.get("file_type"),
            }

            sparse_item = sparse_vectors[idx] if has_sparse else None
            if sparse_item:
                payload["sparse_tokens"] = sparse_item.get("tokens") or []

            dense_only_points.append(
                PointStruct(
                    id=chunk_id,
                    vector=vector,
                    payload=payload,
                )
            )
            named_dense_points.append(
                PointStruct(
                    id=chunk_id,
                    vector={self.dense_vector_name: vector},
                    payload=payload,
                )
            )

            if sparse_item:
                sparse_vector = self._to_sparse_vector_payload(sparse_item)
                named_dense_sparse_points.append(
                    PointStruct(
                        id=chunk_id,
                        vector={
                            self.dense_vector_name: vector,
                            self.primary_sparse_vector_name: sparse_vector,
                        },
                        payload=payload,
                    )
                )

        if not chunk_ids:
            return chunk_ids

        # Prefer full hybrid layout, then fallback to compatible formats.
        candidates: List[List[PointStruct]] = []
        if has_sparse and named_dense_sparse_points:
            candidates.append(named_dense_sparse_points)
        candidates.append(named_dense_points)
        candidates.append(dense_only_points)

        last_exc: Optional[Exception] = None
        for points in candidates:
            try:
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=points,
                    wait=True,
                )
                return chunk_ids
            except Exception as exc:
                last_exc = exc
                continue

        if last_exc is not None:
            raise last_exc
        return chunk_ids

    def search(self, query_vector: List[float], limit: int) -> List[Any]:
        """Run similarity search with compatibility for qdrant-client versions."""

        if hasattr(self.client, "search"):
            try:
                return self.client.search(
                    collection_name=self.collection_name,
                    query_vector=(self.dense_vector_name, query_vector),
                    limit=limit,
                    with_payload=True,
                )
            except Exception:
                return self.client.search(
                    collection_name=self.collection_name,
                    query_vector=query_vector,
                    limit=limit,
                    with_payload=True,
                )

        try:
            result = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                using=self.dense_vector_name,
                limit=limit,
                with_payload=True,
            )
        except Exception:
            result = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                limit=limit,
                with_payload=True,
            )

        return list(getattr(result, "points", []))

    @staticmethod
    def hits_to_documents(hits: List[Any]) -> List[Document]:
        """Convert raw Qdrant hits into LangChain Documents."""

        documents: List[Document] = []
        for hit in hits:
            payload = getattr(hit, "payload", None) or {}
            metadata = dict(payload.get("metadata") or {})
            score = getattr(hit, "score", None)

            metadata.update(
                {
                    "score": score,
                    "filename": payload.get("filename") or metadata.get("filename"),
                    "page": payload.get("page", metadata.get("page")),
                    "chunk_id": payload.get("chunk_id", metadata.get("chunk_id")),
                    "chunk_index": payload.get("chunk_index", metadata.get("chunk_index")),
                }
            )

            documents.append(
                Document(
                    page_content=payload.get("page_content", ""),
                    metadata=metadata,
                )
            )
        return documents

    def _build_sparse_vectors_config(self) -> Dict[str, Any]:
        if not self.sparse_vector_names:
            return {}

        out: Dict[str, Any] = {}
        for name in self.sparse_vector_names:
            if qdrant_models is not None and hasattr(qdrant_models, "SparseVectorParams"):
                out[name] = qdrant_models.SparseVectorParams()
            else:
                out[name] = {}
        return out

    def _best_effort_enable_sparse_vectors(self) -> None:
        sparse_config = self._build_sparse_vectors_config()
        if not sparse_config:
            return
        if not hasattr(self.client, "update_collection"):
            return

        try:
            self.client.update_collection(
                collection_name=self.collection_name,
                sparse_vectors_config=sparse_config,
            )
        except Exception:
            # Collection may be legacy/locked; keep backward compatibility.
            pass

    @staticmethod
    def _to_sparse_vector_payload(sparse_item: Dict[str, Any]) -> Any:
        indices = list(sparse_item.get("indices") or [])
        values = list(sparse_item.get("values") or [])

        if qdrant_models is not None and hasattr(qdrant_models, "SparseVector"):
            return qdrant_models.SparseVector(indices=indices, values=values)
        return {"indices": indices, "values": values}
