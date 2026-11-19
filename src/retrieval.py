"""Truy hồi: BM25, dense (BGE-M3), RRF, xếp hạng lại theo tiêu đề."""
import math
import re
from typing import List, Dict, Tuple, Optional, Any, Set
from rank_bm25 import BM25Okapi

# Stopwords tiếng Việt phổ biến để tránh boost nhầm cho tiêu đề
VIETNAMESE_STOPWORDS: Set[str] = {
    "và", "của", "các", "những", "cho", "về", "trong", "có", "được", "là",
    "khi", "do", "tại", "theo", "với", "từ", "này", "đó", "ra", "đến",
    "đã", "sẽ", "đang", "bị", "một", "nhiều", "như", "thì", "để", "ở"
}

# Hệ số beta của reranker hệ thống nền (RERANK_BETA = 0.1 trong cấu hình gốc); chưa tune trên dev.
DEFAULT_TITLE_BETA = 0.1


def simple_tokenize(text: str) -> List[str]:
    """Tách từ cơ bản dựa trên regex \\w+ và chuyển thành chữ thường (dùng cho BM25)."""
    return re.findall(r"\w+", text.lower())


def reciprocal_rank_fusion(
    ranked_lists: List[List[str]],
    rrf_k: int = 60
) -> List[Tuple[str, float]]:
    """
    Hợp nhất các danh sách ID tài liệu bằng thuật toán Reciprocal Rank Fusion (RRF).
    Công thức: score(d) = sum(1 / (rrf_k + rank(d)))
    
    Args:
        ranked_lists: Danh sách các list doc_id đã được xếp hạng từ các retriever khác nhau.
        rrf_k: Hằng số làm mượt (mặc định 60 theo Cormack et al., hệ thống nền từng dùng 15).
        
    Returns:
        Danh sách (doc_id, rrf_score) được sắp xếp giảm dần theo điểm RRF.
    """
    scores: Dict[str, float] = {}
    for r_list in ranked_lists:
        for rank, doc_id in enumerate(r_list, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (rrf_k + rank)
            
    sorted_docs = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    return sorted_docs


class BM25Retriever:
    """Truy hồi thưa (Sparse) bằng BM25 trên văn bản được tách token theo regex \\w+."""

    def __init__(self, doc_ids: List[str], corpus_texts: List[str]):
        self.doc_ids = doc_ids
        self.tokenized_corpus = [simple_tokenize(text) for text in corpus_texts]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def retrieve(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        tokenized_query = simple_tokenize(query)
        if not tokenized_query:
            return []
        scores = self.bm25.get_scores(tokenized_query)
        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [(self.doc_ids[i], float(scores[i])) for i in ranked_indices]


class DenseRetriever:
    """Truy hồi dày (Dense) bằng embedding BGE-M3 (hỗ trợ cosine similarity)."""

    def __init__(self, model_name: str = "BAAI/bge-m3", device: Optional[str] = None):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name, device=device)
        self.doc_ids: List[str] = []
        self.embeddings = None

    def index(self, doc_ids: List[str], corpus_texts: List[str], batch_size: int = 16) -> None:
        self.doc_ids = doc_ids
        self.embeddings = self.model.encode(
            corpus_texts,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True
        )

    def retrieve(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        import numpy as np
        if self.embeddings is None or len(self.doc_ids) == 0:
            return []
        q_emb = self.model.encode([query], normalize_embeddings=True)[0]
        scores = np.dot(self.embeddings, q_emb)
        ranked_indices = np.argsort(scores)[::-1][:top_k]
        return [(self.doc_ids[i], float(scores[i])) for i in ranked_indices]

    def save_index(self, path_prefix) -> None:
        import numpy as np
        import json as _json
        from pathlib import Path as _Path

        prefix = _Path(path_prefix)
        prefix.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(prefix) + ".npy", self.embeddings)
        with open(str(prefix) + "_ids.json", "w", encoding="utf-8") as f:
            _json.dump(self.doc_ids, f, ensure_ascii=False)

    def load_index(self, path_prefix) -> None:
        import numpy as np
        import json as _json
        from pathlib import Path as _Path

        prefix = _Path(path_prefix)
        emb_path = str(prefix) + ".npy"
        ids_path = str(prefix) + "_ids.json"
        self.embeddings = np.load(emb_path)
        with open(ids_path, encoding="utf-8") as f:
            self.doc_ids = _json.load(f)


def _content_tokens(text: str) -> Set[str]:
    return {w for w in simple_tokenize(text) if w not in VIETNAMESE_STOPWORDS and len(w) > 1}


def rerank_by_title(
    ranked_results: List[Tuple[str, float]],
    doc_titles: Dict[str, str],
    query: str,
    beta: float = DEFAULT_TITLE_BETA
) -> List[Tuple[str, float]]:
    """
    Xếp hạng lại theo độ tương đồng tiêu đề (Ablation component).
    Điểm mới = điểm cũ * (1 + beta * sim), theo dạng nhân của reranker hệ thống nền nên không phụ
    thuộc thang điểm RRF và mức tăng tối đa là beta. sim = |Q ∩ T| / sqrt(|Q| * |T|) trên tập từ
    của câu hỏi Q và tiêu đề T (đã loại stopwords tiếng Việt), như `_heading_token_similarity` của
    hệ thống nền.

    Khác bản gốc: chỉ có thành phần từ vựng (bản gốc cộng thêm độ tương đồng embedding Gemini
    của tiêu đề), tách từ bằng regex \\w+ và xếp hạng lại toàn bộ danh sách thay vì top 30.
    """
    if not ranked_results:
        return []

    query_tokens = _content_tokens(query)
    if not query_tokens:
        return list(ranked_results)

    reranked = []
    for doc_id, score in ranked_results:
        title_tokens = _content_tokens(doc_titles.get(doc_id, ""))
        overlap = len(query_tokens & title_tokens)
        sim = overlap / math.sqrt(len(query_tokens) * len(title_tokens)) if overlap else 0.0
        reranked.append((doc_id, score * (1.0 + beta * sim)))

    return sorted(reranked, key=lambda item: item[1], reverse=True)
