import pytest
from unittest.mock import MagicMock
from langchain_core.documents import Document


@pytest.fixture
def mock_embedding_service():
    """Giả lập Gemini Embedding Service để trả về vector giả ngay lập tức."""
    service = MagicMock()
    # Giả lập Vector 768 chiều
    service.embed_query.return_value = [0.1] * 768
    service.embed_documents.return_value = [[0.1] * 768]
    return service


@pytest.fixture
def mock_qdrant_store():
    """Giả lập Qdrant Vector DB để không cần khởi chạy Docker khi test."""
    store = MagicMock()
    store.search.return_value = []
    store.sparse_search.return_value = []
    store.collection_exists.return_value = True
    return store


@pytest.fixture
def sample_documents():
    """Fixture cung cấp dữ liệu giả chuẩn hóa để test luồng Ingestion."""
    return [
        Document(
            page_content="# iPhone 15\n\nCamera 48MP sắc nét.",
            metadata={"filename": "iphone.md"},
        ),
        Document(
            page_content="# S24 Ultra\n\nHỗ trợ Galaxy AI.",
            metadata={"filename": "s24.md"},
        ),
    ]
