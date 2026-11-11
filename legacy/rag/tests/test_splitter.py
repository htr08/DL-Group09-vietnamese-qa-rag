from langchain_core.documents import Document
from backend.rag.ingestion.splitter import MarketingTextSplitter


def test_markdown_splitter_preserves_headings():
    """Đảm bảo Splitter bắt đúng Header Markdown và lưu vào Metadata để AI hiểu ngữ cảnh."""
    splitter = MarketingTextSplitter(chunk_size=500, chunk_overlap=0)

    doc = Document(
        page_content="# iPhone 15\n\n## Tính năng\n\nCamera 48MP chụp nét.\n\n## Giá bán\n\nTừ 29 triệu đồng.",
        metadata={"filename": "iphone.md"},
    )

    chunks = splitter.split([doc])

    # Sẽ được chia làm 2 chunks dựa trên heading 2
    assert len(chunks) == 2

    # Test Chunk 1:
    assert "Camera 48MP" in chunks[0].page_content
    assert "iPhone 15 > Tính năng" in " > ".join(chunks[0].metadata.get("headings", []))

    # Test Chunk 2:
    assert "29 triệu" in chunks[1].page_content
    assert "Giá bán" in chunks[1].metadata.get("headings", [])
