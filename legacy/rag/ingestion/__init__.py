"""Ingestion package (lazy imports).

Muc tieu: cho phep import `backend.rag.ingestion.loader` de test nhanh,
khong bat buoc cai day du Qdrant/Gemini deps o import-time.
"""

__all__ = [
    "MarketingDocumentLoader",
    "MarketingDocumentProcessor",
    "MarketingTextSplitter",
    "MinioTempDownloader",
]


def __getattr__(name: str):
    if name == "MarketingDocumentLoader":
        from .loader import MarketingDocumentLoader

        return MarketingDocumentLoader
    if name == "MarketingTextSplitter":
        from .splitter import MarketingTextSplitter

        return MarketingTextSplitter
    if name == "MarketingDocumentProcessor":
        from .processor import MarketingDocumentProcessor

        return MarketingDocumentProcessor
    if name == "MinioTempDownloader":
        from .minio_downloader import MinioTempDownloader

        return MinioTempDownloader
    raise AttributeError(name)
