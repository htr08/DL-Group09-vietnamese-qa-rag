"""Pipeline RAG với hai chế độ: sinh bằng LLM và trích xuất.

Hiện chỉ có chế độ sinh bằng LLM (RAG v0: truy hồi hybrid rồi Qwen sinh câu trả lời).
Chế độ trích xuất (PhoBERT đọc các đoạn truy hồi) được bổ sung ở Tuần 3 T4.
"""
from __future__ import annotations

import time
from typing import Callable, Dict, List, Mapping, Optional

from src.generate import generate_answer
from src.retrieval import hybrid_retrieve


def retrieve_contexts(
    bm25_retriever,
    dense_retriever,
    chunk_texts: Mapping[str, str],
    question: str,
    top_k: int = 5,
    rrf_k: int = 60,
    doc_titles: Optional[Dict[str, str]] = None,
) -> List[dict]:
    """Truy hồi hybrid và trả về `top_k` đoạn dạng {id, score, text}, giảm dần theo điểm."""
    hits = hybrid_retrieve(
        bm25_retriever, dense_retriever, question, top_k=top_k, rrf_k=rrf_k, doc_titles=doc_titles
    )
    return [{"id": doc_id, "score": score, "text": chunk_texts[doc_id]} for doc_id, score in hits]


def answer_question(
    generate_fn: Callable[[list, dict], str],
    bm25_retriever,
    dense_retriever,
    chunk_texts: Mapping[str, str],
    question: str,
    top_k: int = 5,
    rrf_k: int = 60,
    doc_titles: Optional[Dict[str, str]] = None,
    max_retries: int = 2,
) -> dict:
    """RAG v0: truy hồi rồi sinh câu trả lời.

    Kết quả gồm các khóa của `generate_answer` cùng `sources` (mã đoạn; phần tử thứ i ứng với
    nguồn [i+1] trong câu trả lời) và `retrieval_seconds`.
    """
    start = time.perf_counter()
    contexts = retrieve_contexts(
        bm25_retriever, dense_retriever, chunk_texts, question,
        top_k=top_k, rrf_k=rrf_k, doc_titles=doc_titles,
    )
    retrieval_seconds = round(time.perf_counter() - start, 3)
    result = generate_answer(generate_fn, question, [c["text"] for c in contexts], max_retries=max_retries)
    result["sources"] = [c["id"] for c in contexts]
    result["retrieval_seconds"] = retrieval_seconds
    return result
