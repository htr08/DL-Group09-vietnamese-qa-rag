from langchain_core.documents import Document

from backend.rag.vectorstore.qdrant_store import QdrantVectorStore


class _Hit:
    def __init__(self, payload, score):
        self.payload = payload
        self.score = score


def test_hits_to_documents_maps_metadata_and_score():
    hits = [
        _Hit(
            payload={
                "page_content": "chunk text",
                "metadata": {"filename": "a.pdf", "chunk_id": "cid", "chunk_index": 0},
                "filename": "a.pdf",
                "chunk_id": "cid",
                "chunk_index": 0,
            },
            score=0.77,
        )
    ]

    docs = QdrantVectorStore.hits_to_documents(hits)
    assert len(docs) == 1
    doc: Document = docs[0]
    assert doc.page_content == "chunk text"
    assert doc.metadata["filename"] == "a.pdf"
    assert doc.metadata["chunk_id"] == "cid"
    assert doc.metadata["score"] == 0.77
