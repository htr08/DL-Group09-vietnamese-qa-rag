"""Truy hồi: BM25, dense (BGE-M3), RRF, xếp hạng lại theo tiêu đề."""
import re
from typing import List, Dict, Tuple, Optional, Any
from rank_bm25 import BM25Okapi


def simple_tokenize(text: str) -> List[str]:
    """Tách từ cơ bản dựa trên regex và chữ thường (dùng cho BM25)."""
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
    """Truy hồi thưa (Sparse) bằng BM25 trên văn bản đã tách từ."""

    def __init__(self, doc_ids: List[str], corpus_texts: List[str]):
        self.doc_ids = doc_ids
        self.tokenized_corpus = [simple_tokenize(text) for text in corpus_texts]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def retrieve(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        tokenized_query = simple_tokenize(query)
        if not tokenized_query:
            return []
        scores = self.bm25.get_scores(tokenized_query)
        # Sắp xếp top_k theo điểm BM25 giảm dần
        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [(self.doc_ids[i], float(scores[i])) for i in ranked_indices]


class DenseRetriever:
    """Truy hồi dense bằng mô hình BGE-M3 (BAAI/bge-m3)."""

    def __init__(self, model_name: str = "BAAI/bge-m3", device: Optional[str] = None):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name, device=device)
        self.doc_ids: List[str] = []
        self.embeddings = None

    def index(self, doc_ids: List[str], corpus_texts: List[str], batch_size: int = 16):
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
        # Cosine similarity khi embeddings đã được normalize là tích vô hướng (dot product)
        scores = np.dot(self.embeddings, q_emb)
        ranked_indices = np.argsort(scores)[::-1][:top_k]
        return [(self.doc_ids[i], float(scores[i])) for i in ranked_indices]

    def save_index(self, path_prefix) -> None:
        """
        Lưu embedding và danh sách doc_id ra đĩa.
        Tạo hai file: <path_prefix>.npy (embeddings) và <path_prefix>_ids.json (doc_ids).
        Gọi sau khi đã index() để không cần encode lại mỗi lần chạy.
        """
        import numpy as np
        import json as _json
        from pathlib import Path as _Path

        prefix = _Path(path_prefix)
        prefix.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(prefix) + ".npy", self.embeddings)
        with open(str(prefix) + "_ids.json", "w", encoding="utf-8") as f:
            _json.dump(self.doc_ids, f, ensure_ascii=False)

    def load_index(self, path_prefix) -> None:
        """
        Nạp embedding và doc_id đã lưu từ đĩa vào retriever.
        Không cần gọi index() nếu index đã được build_index.py tạo sẵn.
        """
        import numpy as np
        import json as _json
        from pathlib import Path as _Path

        prefix = _Path(path_prefix)
        emb_path = str(prefix) + ".npy"
        ids_path = str(prefix) + "_ids.json"
        self.embeddings = np.load(emb_path)
        with open(ids_path, encoding="utf-8") as f:
            self.doc_ids = _json.load(f)


def rerank_by_title(
    ranked_results: List[Tuple[str, float]],
    doc_titles: Dict[str, str],
    query: str,
    boost: float = 0.1
) -> List[Tuple[str, float]]:
    """
    Xếp hạng lại theo độ trùng khớp tiêu đề (Ablation component từ hệ thống nền).
    Nếu tiêu đề chứa các từ trong query, cộng thêm điểm boost.
    """
    query_tokens = set(simple_tokenize(query))
    reranked = []
    for doc_id, score in ranked_results:
        title = doc_titles.get(doc_id, "")
        title_tokens = set(simple_tokenize(title))
        overlap = len(query_tokens.intersection(title_tokens))
        new_score = score + (boost * overlap)
        reranked.append((doc_id, new_score))
    return sorted(reranked, key=lambda item: item[1], reverse=True)
