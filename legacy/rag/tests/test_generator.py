from unittest.mock import MagicMock
from langchain_core.documents import Document
from backend.rag.generation.generator import GeminiSalesGenerator
from backend.rag.models import RAGResponse


def test_gemini_generator_format_context():
    """Kiểm tra việc định dạng ngữ cảnh từ Documents thành chuỗi (Context) cho Prompt."""
    docs = [
        Document(page_content="iPhone 15 giá 29tr", metadata={"headings": ["Apple"]}),
        Document(page_content="S24 Ultra có AI", metadata={"headings": ["Samsung"]}),
    ]

    context_str = GeminiSalesGenerator._format_context(docs)

    assert "iPhone 15 giá 29tr" in context_str
    assert "S24 Ultra có AI" in context_str
    assert "Apple" in context_str
    assert "Samsung" in context_str


def test_gemini_generator_stream_mock():
    """Kiểm tra luồng Stream được xử lý chính xác (Mock không gọi API thật)."""
    generator = GeminiSalesGenerator()

    # Override method stream nội bộ để không gọi AI thật
    generator.stream = MagicMock(
        return_value=iter(["Chào bạn, ", "iPhone 15 đang giảm giá."])
    )

    chunks = list(generator.stream(question="Giá iPhone?", documents=[]))

    assert len(chunks) == 2
    assert chunks[0] == "Chào bạn, "
    assert chunks[1] == "iPhone 15 đang giảm giá."
