"""Retriever dung embedding Gemini va Qdrant vector search."""

from typing import List, Optional

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document
from qdrant_client import QdrantClient

from ..config import RAGSettings, rag_settings
from ..embeddings import GeminiEmbeddingService

from ..models import RAGSource
from ..vectorstore import QdrantVectorStore

# Import cac module cho luong RAG nang cao
from .query_understanding import QueryUnderstanding
from .query_encoder import QueryEncoder
from .searcher import HybridSearcher
from .fusion import RRFFuser
from .reranker import HeadingReranker
from .diversity import select_diverse_documents
from .cache import RetrievalCache
from .source_mapper import documents_to_sources


class MarketingRetriever:
    """Truy xuat top-k chunks bang Hybrid Search, Fusion va Reranking."""

    def __init__(
        self,
        settings: RAGSettings = rag_settings,
        qdrant_client: Optional[QdrantClient] = None,
        embeddings: Optional[GoogleGenerativeAIEmbeddings] = None,
        embedding_service: Optional[GeminiEmbeddingService] = None,
        vector_store: Optional[QdrantVectorStore] = None,
    ) -> None:
        self.settings = settings
        self.collection_name = settings.QDRANT_COLLECTION_NAME
        self.embedding_service = embedding_service or GeminiEmbeddingService(
            settings=settings,
            embeddings=embeddings,
        )
        self.vector_store = vector_store or QdrantVectorStore(
            settings=settings,
            qdrant_client=qdrant_client,
            collection_name=self.collection_name,
        )

        # Khoi tao cac module nam trong Pipeline
        self.query_understanding = QueryUnderstanding()
        self.query_encoder = QueryEncoder(self.embedding_service)
        self.hybrid_searcher = HybridSearcher(
            vector_store=self.vector_store,
            collection_name=self.collection_name,
            sparse_vector_names=[self.settings.QDRANT_SPARSE_VECTOR_NAME],
        )
        self.fuser = RRFFuser()
        self.reranker = HeadingReranker(
            embedding_service=self.embedding_service,
            rerank_top_k=self.settings.RERANK_TOP_K,
            alpha=self.settings.RERANK_ALPHA,
            beta=self.settings.RERANK_BETA,
        )
        self.cache = RetrievalCache()

    def retrieve(self, question: str, top_k: Optional[int] = None) -> List[Document]:
        """Pipeline: Query Gen -> Hybrid Search -> RRF -> Rerank -> Diversity."""

        clean_question = question.strip()
        if not clean_question:
            return []

        # 0. Kiem tra cache truoc khi chay pipeline ton kem
        cached_docs = self.cache.get(clean_question)
        if cached_docs is not None:
            return cached_docs

        if not self.vector_store.collection_exists():
            return []

        limit = top_k or self.settings.DEFAULT_TOP_K

        # 1. Sinh bien the cau hoi va encode cau hoi chinh
        queries = self.query_understanding.generate_variants(clean_question)
        main_query = queries[0]
        dense_vector, sparse_vector = self.query_encoder.encode_parallel(main_query)

        # 2. Chay song song Dense & Sparse Search
        # Dung cua so tim kiem rong hon (it nhat 30) de khong bo sot chunks
        # khi co nhieu tai lieu canh tranh trong collection.
        search_limit = max(limit * 6, 30)
        dense_hits, sparse_hits = self.hybrid_searcher.search_parallel(
            dense_vector=dense_vector,
            sparse_vector=sparse_vector,
            dense_limit=search_limit,
            sparse_limit=search_limit,
        )

        # 3. Gop ket qua bang Reciprocal Rank Fusion
        fused_hits = self.fuser.fuse(dense_hits, sparse_hits, rrf_k=self.settings.RRF_K)

        # 4. Danh gia lai diem so dua tren Tieu de (Heading)
        reranked_hits = self.reranker.rerank(
            question=clean_question,
            query_dense_vector=dense_vector,
            merged_hits=fused_hits,
        )

        # Unpack MergedHit objects to raw hits and update their scores before creating documents.
        final_hits = []
        for item in reranked_hits:
            if hasattr(item.hit, "score"):
                item.hit.score = item.final_score
            
            if hasattr(item.hit, "payload") and isinstance(item.hit.payload, dict):
                metadata = item.hit.payload.get("metadata")
                if not isinstance(metadata, dict):
                    metadata = {}
                    item.hit.payload["metadata"] = metadata
                
                if hasattr(item, "sparse_score") and item.sparse_score is not None:
                    metadata["sparse_score"] = item.sparse_score
                if hasattr(item, "dense_score") and item.dense_score is not None:
                    metadata["dense_score"] = item.dense_score
                    
            final_hits.append(item.hit)

        documents = self.vector_store.hits_to_documents(final_hits)
        # 5. Loc loai bo cac doan van ban qua giong nhau
        final_documents = select_diverse_documents(documents, limit=limit)

        # 6. Luu ket qua vao cache cho lan goi sau
        self.cache.set(clean_question, final_documents)
        return final_documents

    def to_sources(self, documents: List[Document]) -> List[RAGSource]:
        """Chuyen documents truy xuat thanh sources cho response."""
        return documents_to_sources(documents)
