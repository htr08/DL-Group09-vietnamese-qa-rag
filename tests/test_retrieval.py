from src.retrieval import (
    simple_tokenize,
    reciprocal_rank_fusion,
    BM25Retriever,
    DenseRetriever,
    rerank_by_title,
)


def test_reciprocal_rank_fusion():
    bm25_list = ["doc_A", "doc_B", "doc_C"]
    dense_list = ["doc_B", "doc_D", "doc_E"]
    
    # doc_B xuất hiện ở cả 2 nguồn (rank 2 và rank 1) -> điểm cao hơn doc_A chỉ có ở BM25
    fused = reciprocal_rank_fusion([bm25_list, dense_list], rrf_k=60)
    top_doc, top_score = fused[0]
    assert top_doc == "doc_B"
    assert len(fused) == 5


def test_bm25_retriever():
    doc_ids = ["d1", "d2", "d3"]
    corpus = [
        "Quy chế đào tạo đại học năm 2014",
        "Sinh viên cần tích lũy đủ tín chỉ",
        "Thông số kỹ thuật điện thoại Samsung"
    ]
    retriever = BM25Retriever(doc_ids=doc_ids, corpus_texts=corpus)
    results = retriever.retrieve("tín chỉ sinh viên", top_k=2)
    assert len(results) > 0
    assert results[0][0] == "d2"


def test_rerank_by_title():
    ranked = [("d1", 1.0), ("d2", 0.95)]
    doc_titles = {
        "d1": "Giới thiệu chung",
        "d2": "Quy định tín chỉ học tập"
    }
    # Query có chữ 'tín chỉ' nên d2 được cộng boost và vượt lên d1
    reranked = rerank_by_title(ranked, doc_titles, query="tín chỉ", boost=0.2)
    assert reranked[0][0] == "d2"


def test_dense_retriever_save_load_index(tmp_path):
    """Kiểm tra save_index / load_index của DenseRetriever không cần model thật."""
    import numpy as np

    # Tạo DenseRetriever với dữ liệu giả, bỏ qua model thật
    retriever = DenseRetriever.__new__(DenseRetriever)
    retriever.doc_ids = ["d1", "d2", "d3"]
    retriever.embeddings = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]])
    retriever.model = None  # không dùng model khi test persistence

    # Lưu và nạp lại
    prefix = tmp_path / "dense"
    retriever.save_index(prefix)

    retriever2 = DenseRetriever.__new__(DenseRetriever)
    retriever2.model = None
    retriever2.embeddings = None
    retriever2.doc_ids = []
    retriever2.load_index(prefix)

    assert retriever2.doc_ids == ["d1", "d2", "d3"]
    assert retriever2.embeddings.shape == (3, 2)
    np.testing.assert_array_almost_equal(retriever2.embeddings, retriever.embeddings)