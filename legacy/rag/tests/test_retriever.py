from unittest.mock import MagicMock
from backend.rag.retrieval.retriever import MarketingRetriever
from backend.rag.config import RAGSettings


def test_retriever_pipeline_full_flow(mock_embedding_service, mock_qdrant_store):
    """
    Integration Test: Kiểm tra toàn bộ luồng Truy xuất dữ liệu:
    Câu hỏi -> Sinh Vector -> Hybrid Search -> RRF Fusion -> Rerank -> Xóa trùng lặp (Diversity)
    """
    # 1. Setup Mock Data từ DB
    mock_qdrant_store.search.return_value = [
        {
            "id": "hit1",
            "score": 0.9,
            "payload": {"chunk_id": "c1", "metadata": {"headings": ["iPhone"]}},
        },
        {
            "id": "hit2",
            "score": 0.8,
            "payload": {"chunk_id": "c2", "metadata": {"headings": ["Samsung"]}},
        },
    ]

    from langchain_core.documents import Document

    mock_qdrant_store.hits_to_documents.return_value = [
        Document(page_content="iPhone Content", metadata={"chunk_id": "c1"}),
        Document(page_content="Samsung Content", metadata={"chunk_id": "c2"}),
    ]

    settings = RAGSettings(DEFAULT_TOP_K=2)
    retriever = MarketingRetriever(
        settings=settings,
        embedding_service=mock_embedding_service,
        vector_store=mock_qdrant_store,
    )

    # Chặn Cache để bắt pipeline chạy 100% logic
    retriever.cache.get = MagicMock(return_value=None)

    # 2. Thực thi luồng
    docs = retriever.retrieve("Tư vấn iPhone 15", top_k=2)

    # 3. Đánh giá luồng tích hợp
    # Đảm bảo Gemini Embedding đã được gọi để mã hóa câu hỏi khách hàng
    mock_embedding_service.embed_query.assert_called_once()

    # Đảm bảo Qdrant Store đã được trigger để tìm kiếm
    mock_qdrant_store.search.assert_called()
    assert len(docs) == 2
