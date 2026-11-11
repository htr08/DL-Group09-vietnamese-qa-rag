import os
import pytest
from qdrant_client import QdrantClient

from backend.rag.pipeline import MarketingRAGEngine
from backend.rag.config import rag_settings

# Skip test nếu không có API key thật trong môi trường để chạy E2E
pytestmark = pytest.mark.skipif(
    not os.environ.get("GEMINI_API_KEY") and not getattr(rag_settings, "GEMINI_API_KEY", None),
    reason="Thiếu GEMINI_API_KEY để chạy E2E test",
)

@pytest.fixture
def in_memory_qdrant():
    """Tạo QdrantClient in-memory để test E2E không cần database thực tế."""
    return QdrantClient(location=":memory:")

@pytest.fixture
def temp_marketing_doc(tmp_path):
    """Tạo một file marketing ảo tạm thời."""
    doc_path = tmp_path / "test_marketing.txt"
    content = """
    # iPhone 15 Pro Max
    Chương trình khuyến mãi giảm giá 20% cho iPhone 15 Pro Max.
    Đặc điểm nổi bật:
    - Chip A17 Pro mạnh mẽ.
    - Khung viền Titanium nhẹ và bền.
    - Cổng USB-C tốc độ cao.
    
    # Samsung Galaxy S24 Ultra
    Điện thoại Samsung cao cấp với Galaxy AI.
    - Bút S-Pen tích hợp.
    - Camera 200MP zoom quang 5x.
    - Dịch thuật trực tiếp cuộc gọi.
    """
    doc_path.write_text(content, encoding="utf-8")
    return str(doc_path)

def test_e2e_rag_pipeline(in_memory_qdrant, temp_marketing_doc):
    """
    Test toàn bộ luồng RAG:
    1. Khởi tạo Engine với Qdrant in-memory.
    2. Ingest file tài liệu (Processor -> Splitter -> Embedding -> Qdrant).
    3. Trích xuất thông tin bằng câu hỏi (Retriever -> Generator).
    """
    # Override settings để dùng collection riêng cho test
    rag_settings.QDRANT_COLLECTION_NAME = "test_e2e_collection"
    
    # 1. Khởi tạo Engine
    engine = MarketingRAGEngine(
        settings=rag_settings,
        qdrant_client=in_memory_qdrant
    )
    
    # 2. Ingest Document
    ingest_result = engine.process_document(
        file_path=temp_marketing_doc,
        filename="test_marketing.txt"
    )
    
    assert ingest_result.status == "success", f"Ingest thất bại: {ingest_result.message}"
    assert ingest_result.chunks_count > 0, "Không tạo được chunk nào"
    
    # 3. Test Synchronous QA
    question = "iPhone 15 Pro Max có những điểm nổi bật gì?"
    response = engine.ask_bot_sync(question)
    
    assert response.answer, "Không nhận được câu trả lời từ bot"
    assert len(response.sources) > 0, "Không lấy được nguồn (sources) nào"
    
    # Kiểm tra nội dung câu trả lời có chứa thông tin từ file hay không
    answer_lower = response.answer.lower()
    assert "titanium" in answer_lower or "a17" in answer_lower or "usb-c" in answer_lower, \
        f"Câu trả lời không chứa thông tin chính xác. Answer: {response.answer}"
        
    # 4. Test Stream QA
    stream_question = "S24 Ultra có hỗ trợ AI không?"
    stream_response_chunks = []
    sources = []
    
    for event in engine.ask_bot_stream(stream_question):
        if "chunk" in event:
            stream_response_chunks.append(event["chunk"])
        if "DONE" in event and event["DONE"]:
            sources = event.get("sources", [])
            
    full_stream_answer = "".join(stream_response_chunks).lower()
    assert len(full_stream_answer) > 0, "Luồng stream trả về rỗng"
    assert "galaxy ai" in full_stream_answer or "ai" in full_stream_answer, \
        f"Câu trả lời stream không có thông tin kỳ vọng. Answer: {full_stream_answer}"
    assert len(sources) > 0, "Luồng stream không trả về sources"
