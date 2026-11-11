from functools import lru_cache
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict

from ..config import settings as backend_settings


class RAGSettings(BaseSettings):
    """Các tham số dành riêng cho RAG pipeline"""

    # Timeout khi goi Qdrant
    QDRANT_TIMEOUT_SECONDS: int = 30

    # Collection mặc định cho tài liệu marketing/sales của CellphoneS.
    QDRANT_COLLECTION_NAME: str = "cellphones_marketing_docs"

    # Default Gemini embedding model and its expected vector dimension.
    GEMINI_EMBEDDING_MODEL: str = "models/gemini-embedding-001"
    EMBEDDING_DIMENSION: int = 3072

    # Model sinh câu trả lời. Có thể đổi trong .env nếu team dùng model khác.
    GEMINI_CHAT_MODEL: str = "gemini-2.5-flash"
    GEMINI_TEMPERATURE: float = 0.2
    GEMINI_MAX_OUTPUT_TOKENS: int = 8192

    CHUNK_SIZE: int = 300
    CHUNK_OVERLAP: int = 80
    DEFAULT_TOP_K: int = 5
    UPSERT_BATCH_SIZE: int = 64

    # Retrieval: Hybrid + Fusion + Rerank
    RRF_K: int = (
        15  # Ép giá trị xuống 15 để phân hóa cực mạnh thứ hạng (Rank) giữa Dense và Sparse
    )
    RERANK_TOP_K: int = 30
    RERANK_ALPHA: float = 1.0
    RERANK_BETA: float = (
        0.1  # Giảm trọng số của Heading do tài liệu đang bị thiếu/phẳng (tránh làm nhiễu điểm)
    )
    QDRANT_DENSE_VECTOR_NAME: str = "dense"
    QDRANT_SPARSE_VECTOR_NAME: str = "sparse"
    QDRANT_SPARSE_VECTOR_NAMES: str = "sparse"

    # Thiết lập mặc định sử dụng thuật toán BM25 cho Sparse Vector (thay vì fastembed)
    SPARSE_BACKEND: str = "bm25"

    # Doc them tu .env neu team muon override rieng cho RAG (vd: CHUNK_SIZE)
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @property
    def qdrant_url(self) -> str:
        """Trả về URL Qdrant từ settings chung."""

        return backend_settings.QDRANT_URL

    @property
    def QDRANT_API_KEY(self) -> Optional[str]:
        """API key Qdrant từ settings chung (normalize rỗng -> None)."""

        value = getattr(backend_settings, "QDRANT_API_KEY", "") or ""
        return value or None

    @property
    def GEMINI_API_KEY(self) -> str:
        """Gemini API key từ settings chung."""

        return backend_settings.GEMINI_API_KEY

    def validate_gemini_api_key(self) -> None:
        """Bao loi ro rang neu chua cau hinh Gemini."""

        if not self.GEMINI_API_KEY:
            raise ValueError("Thieu GEMINI_API_KEY trong .env (backend/config.py).")


@lru_cache
def get_rag_settings() -> RAGSettings:
    """Cache settings để các service dùng chung một cấu hình."""

    return RAGSettings()


rag_settings = get_rag_settings()
