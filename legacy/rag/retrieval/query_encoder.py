"""Query encoding: dense + sparse."""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple

from ..embeddings import GeminiEmbeddingService

try:
    from underthesea import word_tokenize as _vn_word_tokenize
except Exception:
    _vn_word_tokenize = None

try:
    from pyvi import ViTokenizer as _vi_tokenizer
except Exception:
    _vi_tokenizer = None

try:
    from fastembed import SparseTextEmbedding as _fastembed_sparse_text_embedding
except Exception:
    _fastembed_sparse_text_embedding = None


TOKEN_PATTERN = re.compile(
    r"[0-9A-Za-z\u00C0-\u1EF9]+(?:[-_./][0-9A-Za-z\u00C0-\u1EF9]+)*",
    flags=re.IGNORECASE,
)


def tokenize(text: str) -> List[str]:
    """Tokenize query text for sparse encoding and lexical matching."""

    raw = (text or "").strip()
    if not raw:
        return []

    if _vn_word_tokenize is not None:
        try:
            tokens = _vn_word_tokenize(raw, format="text").split()
            return [token.lower() for token in tokens if token.strip()]
        except Exception:
            pass

    if _vi_tokenizer is not None:
        try:
            segmented = _vi_tokenizer.tokenize(raw).replace("_", " ")
            return [token.lower() for token in segmented.split() if token.strip()]
        except Exception:
            pass

    return [token.lower() for token in TOKEN_PATTERN.findall(raw)]


class SparseTextEncoder:
    """Encode sparse vectors with FastEmbed (if available) or BM25 fallback."""

    def __init__(
        self,
        backend: str = "auto",
        fastembed_model_name: str = "Qdrant/bm25",
    ) -> None:
        requested = (backend or "auto").strip().lower()
        if requested not in {"auto", "fastembed", "bm25"}:
            requested = "auto"

        self.backend = requested
        self._fastembed_model = None

        if requested in {"auto", "fastembed"} and _fastembed_sparse_text_embedding is not None:
            try:
                self._fastembed_model = _fastembed_sparse_text_embedding(
                    model_name=fastembed_model_name
                )
                self.backend = "fastembed"
            except Exception:
                if requested == "fastembed":
                    self.backend = "bm25"

        if self._fastembed_model is None and self.backend != "fastembed":
            self.backend = "bm25"

    def encode_query(self, question: str) -> Dict[str, Any]:
        if self.backend == "fastembed":
            vector = self._fastembed_encode_one(question, is_query=True)
            if vector:
                return vector
        return self._encode_bm25(question)

    def encode_documents(self, texts: List[str]) -> List[Dict[str, Any]]:
        if not texts:
            return []
        if self.backend == "fastembed":
            vectors = self._fastembed_encode_many(texts)
            if vectors and len(vectors) == len(texts):
                return vectors
        return [self._encode_bm25(text) for text in texts]

    def _encode_bm25(self, text: str) -> Dict[str, Any]:
        tokens = tokenize(text)
        if not tokens:
            return {"indices": [], "values": [], "tokens": []}

        counter = Counter(tokens)
        doc_len = max(1, sum(counter.values()))
        avg_doc_len = 8.0
        k1, b = 1.2, 0.75

        accumulator: Dict[int, float] = {}
        for token, tf in counter.items():
            norm = k1 * (1.0 - b + b * (doc_len / avg_doc_len))
            tf_weight = ((k1 + 1.0) * tf) / (norm + tf)
            token_id = self._stable_token_id(token)
            accumulator[token_id] = accumulator.get(token_id, 0.0) + float(tf_weight)

        pairs = sorted(accumulator.items(), key=lambda item: item[0])
        indices = [idx for idx, _ in pairs]
        values = [val for _, val in pairs]
        return {"indices": indices, "values": values, "tokens": tokens}

    def _fastembed_encode_one(self, text: str, is_query: bool) -> Dict[str, Any]:
        model = self._fastembed_model
        if model is None:
            return {}
        try:
            if is_query and hasattr(model, "query_embed"):
                raw = list(model.query_embed([text]))
            elif hasattr(model, "passage_embed"):
                raw = list(model.passage_embed([text]))
            elif hasattr(model, "embed"):
                raw = list(model.embed([text]))
            else:
                return {}
            if not raw:
                return {}
            return self._normalize_sparse_vector(raw[0])
        except Exception:
            return {}

    def _fastembed_encode_many(self, texts: List[str]) -> List[Dict[str, Any]]:
        model = self._fastembed_model
        if model is None:
            return []
        try:
            if hasattr(model, "passage_embed"):
                raw_list = list(model.passage_embed(texts))
            elif hasattr(model, "embed"):
                raw_list = list(model.embed(texts))
            else:
                return []
            return [self._normalize_sparse_vector(item) for item in raw_list]
        except Exception:
            return []

    @staticmethod
    def _normalize_sparse_vector(raw: Any) -> Dict[str, Any]:
        indices: List[int] = []
        values: List[float] = []

        if isinstance(raw, dict):
            indices = list(raw.get("indices") or [])
            values = list(raw.get("values") or [])
        elif hasattr(raw, "indices") and hasattr(raw, "values"):
            indices = list(getattr(raw, "indices") or [])
            values = list(getattr(raw, "values") or [])
        elif isinstance(raw, (tuple, list)) and len(raw) == 2:
            indices = list(raw[0] or [])
            values = list(raw[1] or [])

        if len(indices) != len(values):
            return {"indices": [], "values": [], "tokens": []}

        clean_indices: List[int] = []
        clean_values: List[float] = []
        for idx, val in zip(indices, values):
            try:
                clean_indices.append(int(idx))
                clean_values.append(float(val))
            except Exception:
                continue

        return {"indices": clean_indices, "values": clean_values, "tokens": []}

    @staticmethod
    def _stable_token_id(token: str) -> int:
        digest = hashlib.md5(token.encode("utf-8")).digest()
        return int.from_bytes(digest[:8], byteorder="big", signed=False) % 2147483647


class QueryEncoder:
    """Encode user query into dense and sparse representations."""

    def __init__(
        self,
        embedding_service: GeminiEmbeddingService,
        sparse_encoder: Optional[SparseTextEncoder] = None,
    ) -> None:
        self.embedding_service = embedding_service
        self.sparse_encoder = sparse_encoder or SparseTextEncoder()

    def encode_parallel(self, question: str) -> Tuple[List[float], Dict[str, Any]]:
        """Run dense and sparse encoders concurrently."""

        with ThreadPoolExecutor(max_workers=2) as executor:
            dense_future = executor.submit(self.embedding_service.embed_query, question)
            sparse_future = executor.submit(self.sparse_encoder.encode_query, question)

            dense_vector = dense_future.result()
            try:
                sparse_vector = sparse_future.result()
            except Exception:
                sparse_vector = {"indices": [], "values": [], "tokens": []}
        return dense_vector, sparse_vector

    def encode_sparse(self, question: str) -> Dict[str, Any]:
        """Backward-compatible alias for sparse query encoding."""

        return self.sparse_encoder.encode_query(question)

    def encode_sparse_documents(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Encode sparse vectors for documents/chunks."""

        return self.sparse_encoder.encode_documents(texts)
