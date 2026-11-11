from langchain_core.documents import Document

from backend.rag.retrieval.retriever import MarketingRetriever


class _FakeEmbedding:
    def embed_query(self, text: str):
        return [0.0, 1.0]


class _FakeEmbeddingService:
    def embed_query(self, text: str):
        return [0.0, 1.0]


class _FakeVectorStore:
    def __init__(self, docs):
        self._docs = docs

    def collection_exists(self):
        return True

    def search(self, query_vector, limit):
        return ["hit"]  # opaque

    def hits_to_documents(self, hits):
        return self._docs


def test_retriever_returns_documents_and_sources():
    docs = [
        Document(
            page_content="x",
            metadata={
                "filename": "a.txt",
                "chunk_id": "cid1",
                "score": 0.9,
                "chunk_index": 0,
            },
        ),
        Document(
            page_content="y",
            metadata={
                "filename": "a.txt",
                "chunk_id": "cid2",
                "score": 0.8,
                "chunk_index": 1,
            },
        ),
    ]

    retriever = MarketingRetriever(
        embedding_service=_FakeEmbeddingService(),
        vector_store=_FakeVectorStore(docs),
    )

    out_docs = retriever.retrieve("test", top_k=2)
    assert len(out_docs) == 2

    sources = retriever.to_sources(out_docs)
    assert len(sources) == 2
    assert sources[0].filename == "a.txt"
