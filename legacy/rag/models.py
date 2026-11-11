from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class RAGSource(BaseModel):
    """Nguồn tham chiếu trả về cho frontend."""

    filename: str = Field(..., description="Tên file gốc")
    chunk_id: Optional[str] = Field(default=None, description="ID chunk trong Qdrant")
    score: Optional[float] = Field(
        default=None, description="Điểm similarity từ Qdrant"
    )
    preview: Optional[str] = Field(
        default=None, description="Đoạn trích ngắn để hiển thị"
    )
    metadata: Dict[str, Any] = Field(default_factory=dict)


class DocumentIngestionResult(BaseModel):
    """Kết quả ingest một file vào Qdrant."""

    filename: str
    collection_name: str
    chunks_count: int = 0
    status: str = "completed"
    message: str
    chunk_ids: List[str] = Field(default_factory=list)


class RAGResponse(BaseModel):
    """Response chuẩn cho API chat."""

    answer: str
    sources: List[RAGSource] = Field(default_factory=list)
