"""Package RAG cho CellphoneS AI Sales Assistant.

Ghi chu:
- Khong import cac module nang (qdrant/langchain/...) o import-time.
- Can dung engine thi import truc tiep: `from backend.rag.pipeline import MarketingRAGEngine`.
"""

__all__ = ["MarketingRAGEngine"]


def __getattr__(name: str):
    if name == "MarketingRAGEngine":
        from .pipeline import MarketingRAGEngine

        return MarketingRAGEngine
    raise AttributeError(name)
