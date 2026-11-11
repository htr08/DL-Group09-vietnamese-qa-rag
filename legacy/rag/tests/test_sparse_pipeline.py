from __future__ import annotations

from langchain_core.documents import Document

from backend.rag.retrieval.query_encoder import SparseTextEncoder
from backend.rag.retrieval.retriever import MarketingRetriever


class _Hit:
    def __init__(self, hit_id: str, score: float, payload: dict):
        self.id = hit_id
        self.score = score
        self.payload = payload


class _FakeEmbeddingService:
    def embed_query(self, text: str):
        return [0.1, 0.2, 0.3]

    def embed_documents(self, texts):
        return [[0.1, 0.2, 0.3] for _ in texts]


class _FakeVectorStore:
    def collection_exists(self):
        return True

    def hits_to_documents(self, hits):
        docs = []
        for hit in hits:
            payload = hit.payload
            metadata = dict(payload.get("metadata") or {})
            metadata["score"] = hit.score
            docs.append(
                Document(
                    page_content=payload.get("page_content", ""),
                    metadata=metadata,
                )
            )
        return docs


class _FakeQueryEncoder:
    def encode_parallel(self, question: str):
        return [0.1, 0.2, 0.3], {"indices": [1, 2], "values": [1.0, 0.7], "tokens": ["vna", "gold"]}


class _FakeSearcher:
    def __init__(self, dense_hits, sparse_hits):
        self._dense_hits = dense_hits
        self._sparse_hits = sparse_hits

    def search_parallel(self, **kwargs):
        return self._dense_hits, self._sparse_hits


def test_sparse_encoder_bm25_for_keyword_query():
    encoder = SparseTextEncoder(backend="bm25")
    vector = encoder.encode_query("VNA Gold")

    assert vector["indices"]
    assert vector["values"]
    assert "vna" in vector["tokens"]
    assert "gold" in vector["tokens"]


def test_retriever_sets_sparse_score_when_sparse_hits_exist():
    hit_1 = _Hit(
        hit_id="chunk-1",
        score=0.91,
        payload={
            "page_content": "goi cuoc VNA Gold dac biet",
            "metadata": {"filename": "offers.txt", "chunk_id": "chunk-1", "chunk_index": 0},
        },
    )
    hit_2 = _Hit(
        hit_id="chunk-2",
        score=0.75,
        payload={
            "page_content": "goi cuoc pho thong",
            "metadata": {"filename": "offers.txt", "chunk_id": "chunk-2", "chunk_index": 1},
        },
    )

    retriever = MarketingRetriever(
        embedding_service=_FakeEmbeddingService(),
        vector_store=_FakeVectorStore(),
    )
    retriever.query_encoder = _FakeQueryEncoder()
    retriever.hybrid_searcher = _FakeSearcher(dense_hits=[hit_1, hit_2], sparse_hits=[hit_1])

    docs = retriever.retrieve("VNA Gold", top_k=2)
    assert docs
    assert any(doc.metadata.get("sparse_score") is not None for doc in docs)
