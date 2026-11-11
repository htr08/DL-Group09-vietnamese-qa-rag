"""Map retrieved documents to RAG sources."""

from typing import List

from langchain_core.documents import Document

from ..models import RAGSource


def documents_to_sources(documents: List[Document]) -> List[RAGSource]:
    """Convert retrieved documents to response sources."""

    sources: List[RAGSource] = []
    seen = set()

    for doc in documents:
        metadata = doc.metadata or {}
        chunk_id = metadata.get("chunk_id")
        dedupe_key = chunk_id or (
            metadata.get("filename"),
            metadata.get("page"),
            metadata.get("chunk_index"),
        )
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)

        preview = " ".join(doc.page_content.split())[:240]
        sources.append(
            RAGSource(
                filename=str(metadata.get("filename") or "unknown"),
                page=metadata.get("page"),
                chunk_id=chunk_id,
                score=metadata.get("score"),
                preview=preview,
                metadata={
                    key: value
                    for key, value in metadata.items()
                    if key not in {"filename", "page", "chunk_id", "score"}
                },
            )
        )
    return sources

