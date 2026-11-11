import logging
from typing import Any, Callable, Dict, Iterator, List, Optional, Union
from qdrant_client import QdrantClient
from langchain_core.messages import BaseMessage

from .config import RAGSettings, rag_settings
from .generation import GeminiSalesGenerator
from .ingestion import MarketingDocumentProcessor
from .models import DocumentIngestionResult, RAGResponse
from .retrieval import MarketingRetriever

logger = logging.getLogger(__name__)

_OVERLOAD_KEYWORDS = (
    "quota",
    "429",
    "resourceexhausted",
    "resource_exhausted",
    "ratelimit",
    "rate_limit",
    "overloaded",
    "ratequota",
)
_TIMEOUT_TYPE_NAMES = ("timeouterror", "deadlineexceeded", "timeout")


def _is_timeout(exc: Exception) -> bool:
    return any(kw in type(exc).__name__.lower() for kw in _TIMEOUT_TYPE_NAMES)


def _is_overload(exc: Exception) -> bool:
    name = type(exc).__name__.lower()
    msg = str(exc).lower()
    return any(kw in name or kw in msg for kw in _OVERLOAD_KEYWORDS)


class MarketingRAGEngine:
    """Class chính gom toàn bộ RAG flow cho CellphoneS AI Sales Assistant."""

    def __init__(
        self,
        settings: RAGSettings = rag_settings,
        qdrant_client: Optional[QdrantClient] = None,
        processor: Optional[MarketingDocumentProcessor] = None,
        retriever: Optional[MarketingRetriever] = None,
        generator: Optional[GeminiSalesGenerator] = None,
    ) -> None:
        self.settings = settings
        self.qdrant_client = qdrant_client

        self._processor = processor
        self._retriever = retriever
        self._generator = generator

    def process_document(
        self,
        file_path: Union[str, List[str]],
        filename: Optional[str] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> DocumentIngestionResult:
        """Xử lý ingestion: load file, split, embed và lưu vào Qdrant."""
        try:
            return self.processor.process_document(
                file_path=file_path,
                filename=filename,
                progress_callback=progress_callback,
            )
        except Exception as exc:
            display_name = filename or str(file_path)
            logger.exception("Document ingestion failed for %s", display_name)
            return DocumentIngestionResult(
                filename=display_name,
                collection_name=self.settings.QDRANT_COLLECTION_NAME,
                chunks_count=0,
                status="failed",
                message=f"Ingestion thất bại: {exc}",
            )

    def ask_bot_sync(self, question: str) -> RAGResponse:
        """Chạy full RAG pipeline không stream (trả về 1 lần)."""
        clean_question = question.strip()
        if not clean_question:
            return RAGResponse(
                answer="Bạn vui lòng nhập câu hỏi để mình có thể tư vấn nhé.",
                sources=[],
            )

        try:
            documents = self.retriever.retrieve(
                question=clean_question,
                top_k=self.settings.DEFAULT_TOP_K,
            )
            sources = self.retriever.to_sources(documents)
            return self.generator.generate(
                question=clean_question,
                documents=documents,
                sources=sources,
            )
        except Exception as exc:
            logger.exception("RAG pipeline error for question: %s", clean_question[:80])
            return RAGResponse(
                answer="Xin lỗi, hệ thống tư vấn đang gặp lỗi khi xử lý câu hỏi.",
                sources=[],
            )

    def ask_bot_stream(
        self,
        question: str,
        chat_history: Optional[List[BaseMessage]] = None,
        system_prompt: Optional[str] = None,
    ) -> Iterator[Dict[str, Any]]:
        """Generator: yield {"chunk": str} liên tục, chunk cuối {"DONE": True, "sources": [...]}."""
        clean_question = question.strip()
        if not clean_question:
            yield {"chunk": "Bạn vui lòng nhập câu hỏi để mình có thể tư vấn nhé."}
            yield {"DONE": True, "sources": []}
            return

        try:
            documents = self.retriever.retrieve(
                question=clean_question,
                top_k=self.settings.DEFAULT_TOP_K,
            )
            sources = self.retriever.to_sources(documents)
            formatted_sources = [
                {
                    "filename": s.filename,
                    "snippet": s.preview,
                    "chunk_id": s.chunk_id,
                    "score": s.score,
                    "metadata": s.metadata,
                }
                for s in sources
            ]

            for text in self.generator.stream(
                question=clean_question,
                documents=documents,
                history=chat_history or [],
                system_prompt=system_prompt,
            ):
                if hasattr(text, "content"):
                    chunk_str = text.content
                elif isinstance(text, dict):
                    chunk_str = text.get("content", "")
                else:
                    chunk_str = str(text)

                if isinstance(chunk_str, list):
                    chunk_str = "".join(
                        part.get("text", "")
                        for part in chunk_str
                        if isinstance(part, dict)
                    )
                elif not isinstance(chunk_str, str):
                    chunk_str = str(chunk_str)

                if (
                    chunk_str.startswith("content=")
                    and "additional_kwargs=" in chunk_str
                ):
                    continue

                if chunk_str:
                    yield {"chunk": chunk_str}

            yield {"DONE": True, "sources": formatted_sources}

        except TimeoutError as exc:
            logger.error("Gemini timeout: %s", exc)
            yield {"chunk": "AI đang quá tải, vui lòng thử lại sau."}
            yield {"DONE": True, "sources": []}
        except Exception as exc:
            if _is_timeout(exc):
                logger.error("Gemini timeout (%s): %s", type(exc).__name__, exc)
                yield {"chunk": "AI đang quá tải, vui lòng thử lại sau."}
            elif _is_overload(exc):
                logger.error("Gemini quota/overload (%s): %s", type(exc).__name__, exc)
                yield {"chunk": "AI đang quá tải, vui lòng thử lại sau."}
            else:
                logger.exception("Unexpected RAG stream error")
                yield {
                    "chunk": "Xin lỗi, hệ thống tư vấn đang gặp lỗi khi xử lý câu hỏi."
                }
            yield {"DONE": True, "sources": []}

    @property
    def processor(self) -> MarketingDocumentProcessor:
        if self._processor is None:
            self._processor = MarketingDocumentProcessor(
                settings=self.settings,
                qdrant_client=self.qdrant_client,
            )
        return self._processor

    @property
    def retriever(self) -> MarketingRetriever:
        if self._retriever is None:
            self._retriever = MarketingRetriever(
                settings=self.settings,
                qdrant_client=self.qdrant_client,
            )
        return self._retriever

    @property
    def generator(self) -> GeminiSalesGenerator:
        if self._generator is None:
            self._generator = GeminiSalesGenerator(settings=self.settings)
        return self._generator
