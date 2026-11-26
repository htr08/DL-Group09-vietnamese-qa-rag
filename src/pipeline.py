"""Pipeline RAG với hai chế độ: sinh bằng LLM và trích xuất.

- Chế độ sinh (`answer_question`): truy hồi hybrid rồi Qwen sinh câu trả lời.
- Chế độ trích xuất (`answer_question_extractive`): truy hồi hybrid rồi PhoBERT chọn một khoảng văn bản
  trong các đoạn truy hồi (xem `src/extractive.py`).
"""
from __future__ import annotations

import time
from typing import Callable, Dict, List, Mapping, Optional

from src.generate import find_invalid_citations, generate_answer
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
    nguồn [i+1] trong câu trả lời), `invalid_citations` (số nguồn [n] không có trong danh sách
    đoạn truy hồi) và `retrieval_seconds`.
    """
    start = time.perf_counter()
    contexts = retrieve_contexts(
        bm25_retriever, dense_retriever, chunk_texts, question,
        top_k=top_k, rrf_k=rrf_k, doc_titles=doc_titles,
    )
    retrieval_seconds = round(time.perf_counter() - start, 3)
    result = generate_answer(generate_fn, question, [c["text"] for c in contexts], max_retries=max_retries)
    result["sources"] = [c["id"] for c in contexts]
    result["invalid_citations"] = find_invalid_citations(result["answer"], len(contexts))
    result["retrieval_seconds"] = retrieval_seconds
    return result


def answer_question_extractive(
    extract_fn: Callable[[str, list], dict],
    bm25_retriever,
    dense_retriever,
    chunk_texts: Mapping[str, str],
    question: str,
    top_k: int = 5,
    rrf_k: int = 60,
    doc_titles: Optional[Dict[str, str]] = None,
) -> dict:
    """Chế độ trích xuất: truy hồi rồi để `extract_fn(question, contexts)` chọn khoảng văn bản.

    `extract_fn` nhận danh sách đoạn {id, score, text} và trả kết quả như `extract_answer`
    (xem `make_extract_fn`). Kết quả thêm `sources` (mã đoạn theo thứ hạng truy hồi) và
    `retrieval_seconds`; `source_rank` của `extract_answer` là thứ hạng 1-based trong `sources`.
    """
    start = time.perf_counter()
    contexts = retrieve_contexts(
        bm25_retriever, dense_retriever, chunk_texts, question,
        top_k=top_k, rrf_k=rrf_k, doc_titles=doc_titles,
    )
    retrieval_seconds = round(time.perf_counter() - start, 3)
    result = extract_fn(question, contexts)
    result["sources"] = [c["id"] for c in contexts]
    result["retrieval_seconds"] = retrieval_seconds
    return result
