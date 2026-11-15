import pytest
from src.retrieval import (
    simple_tokenize,
    reciprocal_rank_fusion,
    BM25Retriever,
    DenseRetriever,
    rerank_by_title,
)
from scripts.build_index import chunk_quy_che


def test_reciprocal_rank_fusion():
    bm25_list = ["doc_A", "doc_B", "doc_C"]
    dense_list = ["doc_B", "doc_D", "doc_E"]
    
    # doc_B xuất hiện ở cả 2 nguồn (rank 2 và rank 1) -> điểm cao hơn doc_A chỉ có ở BM25
    fused = reciprocal_rank_fusion([bm25_list, dense_list], rrf_k=60)
    top_doc, top_score = fused[0]
    assert top_doc == "doc_B"
    assert len(fused) == 5


def test_rrf_exact_math():
    """Kiểm tra công thức toán chính xác của RRF với k=60 và k=15 (mục 8.2)."""
    list1 = ["d1", "d2"]
    list2 = ["d2", "d1"]
    
    # Với k=60:
    # d1: 1/(60+1) + 1/(60+2) = 1/61 + 1/62 = 0.0163934 + 0.0161290 = 0.0325224
    # d2: 1/(60+2) + 1/(60+1) = bằng d1
    fused_60 = reciprocal_rank_fusion([list1, list2], rrf_k=60)
    expected_score_60 = (1.0 / 61) + (1.0 / 62)
    assert pytest.approx(fused_60[0][1], rel=1e-5) == expected_score_60

    # Với k=15:
    # d1: 1/(15+1) + 1/(15+2) = 1/16 + 1/17 = 0.0625 + 0.0588235 = 0.1213235
    fused_15 = reciprocal_rank_fusion([list1, list2], rrf_k=15)
    expected_score_15 = (1.0 / 16) + (1.0 / 17)
    assert pytest.approx(fused_15[0][1], rel=1e-5) == expected_score_15


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


def test_rerank_by_title_with_rrf_scale_and_stopwords():
    """Kiểm tra rerank trên thang điểm RRF (0.01-0.03) và lọc stopwords tiếng Việt."""
    ranked = [("d1", 0.032), ("d2", 0.030)]
    doc_titles = {
        "d1": "Giới thiệu chung và các quy định",  # chứa stopwords 'và', 'các'
        "d2": "Quy chế tín chỉ đào tạo"           # chứa từ có nghĩa 'tín chỉ'
    }
    # Query có từ dừng 'và', 'các' kèm từ khóa 'tín chỉ'
    # d2 phải được boost vì trùng từ khóa thực 'tín chỉ', không bị 'và/các' đẩy d1
    reranked = rerank_by_title(ranked, doc_titles, query="và các tín chỉ", boost=0.003)
    assert reranked[0][0] == "d2"


def test_chunk_quy_che_intro_and_clause_ids():
    """Kiểm tra xử lý đoạn dẫn _intro và lấy đúng số Khoản cho Điều 41 và Điều 10."""
    sample_text = """
Điều 10. Cấu trúc chương trình đào tạo
Chương trình đào tạo được sắp xếp theo các khối kiến thức:
1. Khối kiến thức chung được tổ chức giảng dạy thống nhất.
2. Khối kiến thức theo lĩnh vực.

Điều 41. Xử lý học vụ
Sau mỗi học kỳ chính, đơn vị đào tạo thực hiện xử lý học vụ. Kết quả học kỳ phụ tính vào học kỳ trước.
1. Cảnh báo học vụ
Đơn vị đào tạo thực hiện cảnh báo học vụ theo từng học kỳ.
2. Buộc thôi học
Sinh viên bị buộc thôi học nếu thuộc các trường hợp quy định.
"""
    chunks = chunk_quy_che(sample_text)
    chunk_ids = [c[0] for c in chunks]

    # Kiểm tra Điều 10
    assert "dieu10_intro" in chunk_ids
    assert "dieu10_khoan1" in chunk_ids
    assert "dieu10_khoan2" in chunk_ids

    # Kiểm tra Điều 41 (Lỗi 1 nghiêm trọng được reviewer chỉ ra)
    assert "dieu41_intro" in chunk_ids
    assert "dieu41_khoan1" in chunk_ids
    assert "dieu41_khoan2" in chunk_ids

    # Nội dung của dieu41_khoan1 phải là '1. Cảnh báo học vụ' chứ không phải đoạn dẫn
    d41_k1 = next(c[2] for c in chunks if c[0] == "dieu41_khoan1")
    assert "1. Cảnh báo học vụ" in d41_k1


def test_dense_retriever_save_load_index(tmp_path):
    """Kiểm tra save_index / load_index của DenseRetriever không cần model thật."""
    import numpy as np

    retriever = DenseRetriever.__new__(DenseRetriever)
    retriever.doc_ids = ["d1", "d2", "d3"]
    retriever.embeddings = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]])
    retriever.model = None

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
