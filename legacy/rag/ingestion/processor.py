from typing import Callable, Iterable, List, Optional, Union

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document
from qdrant_client import QdrantClient

from ..config import RAGSettings, rag_settings
from ..embeddings import GeminiEmbeddingService
from ..models import DocumentIngestionResult
from ..vectorstore import QdrantVectorStore
from .loader import MarketingDocumentLoader
from .splitter import MarketingTextSplitter
from ..retrieval.query_encoder import SparseTextEncoder


class MarketingDocumentProcessor:
    def __init__(
        self,
        settings: RAGSettings = rag_settings,
        qdrant_client: Optional[QdrantClient] = None,
        embeddings: Optional[GoogleGenerativeAIEmbeddings] = None,
        embedding_service: Optional[GeminiEmbeddingService] = None,
        vector_store: Optional[QdrantVectorStore] = None,
    ) -> None:
        self.settings = settings
        self.collection_name = settings.QDRANT_COLLECTION_NAME
        self.embedding_service = embedding_service or GeminiEmbeddingService(
            settings=settings,
            embeddings=embeddings,
        )
        self.vector_store = vector_store or QdrantVectorStore(
            settings=settings,
            qdrant_client=qdrant_client,
            collection_name=self.collection_name,
        )
        self.loader = MarketingDocumentLoader()
        self.splitter = MarketingTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
        )
        self.sparse_encoder = SparseTextEncoder()
        self.vector_store.ensure_collection(vector_size=settings.EMBEDDING_DIMENSION)

    def process_document(
        self,
        file_path: Union[str, List[str]],
        filename: Optional[str] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> DocumentIngestionResult:
        """Ingest mot hoac nhieu file local vao Qdrant collection."""

        display_name = filename
        if not display_name:
            display_name = (
                file_path
                if isinstance(file_path, str)
                else f"batch_{len(file_path)}_files"
            )

        documents = self.loader.load(
            file_path=file_path,
            filename=filename if isinstance(file_path, str) else None,
        )
        chunks = self.splitter.split(documents)

        if not chunks:
            return DocumentIngestionResult(
                filename=display_name,
                collection_name=self.collection_name,
                chunks_count=0,
                status="skipped",
                message="File khong co noi dung de ingest.",
            )

        chunk_ids = self._upsert_chunks(chunks, progress_callback=progress_callback)

        return DocumentIngestionResult(
            filename=display_name,
            collection_name=self.collection_name,
            chunks_count=len(chunks),
            status="completed",
            message="Da ingest tai lieu vao Qdrant thanh cong.",
            chunk_ids=chunk_ids,
        )

    def _upsert_chunks(
        self,
        chunks: List[Document],
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> List[str]:
        """Embed va upsert chunks theo batch de tranh request qua lon."""

        chunk_ids: List[str] = []
        batch_size = max(1, self.settings.UPSERT_BATCH_SIZE)
        total = len(chunks)
        processed = 0

        for batch in self._batched(chunks, batch_size):
            texts_for_embedding = []
            for doc in batch:
                headings = doc.metadata.get("headings", [])
                heading_str = " > ".join(headings) if headings else ""
                if heading_str and heading_str != "General":
                    texts_for_embedding.append(f"{heading_str}\n{doc.page_content}")
                else:
                    texts_for_embedding.append(doc.page_content)

            dense_vectors = self.embedding_service.embed_documents(texts_for_embedding)
            sparse_vectors = self.sparse_encoder.encode_documents(texts_for_embedding)

            batch_chunk_ids = self.vector_store.upsert_chunks(
                chunks=batch, vectors=dense_vectors, sparse_vectors=sparse_vectors
            )
            chunk_ids.extend(batch_chunk_ids)
            processed += len(batch)
            if progress_callback:
                progress_callback(processed, total)

        return chunk_ids

    @staticmethod
    def _batched(items: List[Document], batch_size: int) -> Iterable[List[Document]]:

        for start in range(0, len(items), batch_size):
            yield items[start : start + batch_size]
