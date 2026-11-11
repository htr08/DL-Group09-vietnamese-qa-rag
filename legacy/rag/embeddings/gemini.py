"""Gemini embedding service wrapper."""

import time
from typing import List, Optional

from langchain_google_genai import GoogleGenerativeAIEmbeddings

from ..config import RAGSettings, rag_settings

_MAX_RETRIES = 3
_RETRY_DELAY = 1.0


class GeminiEmbeddingService:
    """Encapsulate embedding logic to keep orchestration layers simple."""

    def __init__(
        self,
        settings: RAGSettings = rag_settings,
        embeddings: Optional[GoogleGenerativeAIEmbeddings] = None,
    ) -> None:
        self.settings = settings
        self.settings.validate_gemini_api_key()
        self.client = embeddings or GoogleGenerativeAIEmbeddings(
            model=settings.GEMINI_EMBEDDING_MODEL,
            google_api_key=settings.GEMINI_API_KEY,
        )

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of text chunks."""

        if not texts:
            return []
        return self.client.embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        """Embed a query text with retry on transient failure."""

        last_exc: Optional[Exception] = None
        for attempt in range(_MAX_RETRIES):
            try:
                return self.client.embed_query(text)
            except Exception as exc:
                last_exc = exc
                if attempt < _MAX_RETRIES - 1:
                    time.sleep(_RETRY_DELAY * (attempt + 1))
        raise last_exc  # type: ignore[misc]
