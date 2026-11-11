from backend.rag.retrieval.fusion import RRFFuser


def test_rrf_fuser_correct_ranking():
    """Kiểm tra thuật toán Reciprocal Rank Fusion (RRF) gộp và xếp hạng chuẩn xác."""
    fuser = RRFFuser()

    # Giả lập kết quả từ Semantic (Vector AI)
    dense_hits = [
        {"id": "doc_A", "score": 0.9, "payload": {"chunk_id": "chunk_A"}},
        {"id": "doc_B", "score": 0.8, "payload": {"chunk_id": "chunk_B"}},
    ]

    # Giả lập kết quả từ BM25 (Từ khóa)
    sparse_hits = [
        {"id": "doc_B", "score": 0.85, "payload": {"chunk_id": "chunk_B"}},
        {"id": "doc_C", "score": 0.7, "payload": {"chunk_id": "chunk_C"}},
    ]

    # Thực thi Fuser
    fused_results = fuser.fuse(dense_hits, sparse_hits, rrf_k=60)

    # Đánh giá: doc_B phải lên TOP 1 vì xuất hiện ở cả 2 nhánh (cross-validation)
    assert len(fused_results) == 3
    assert fused_results[0].key == "chunk_B", "Chunk B phải TOP 1"
    assert fused_results[1].key == "chunk_A", "Chunk A TOP 2"

    # Tính toán chính xác kỳ vọng: doc_B có rank 2 (dense) và rank 1 (sparse)
    expected_score_b = (1.0 / (60 + 2)) + (1.0 / (60 + 1))
    # Kiểm tra sai số float
    assert abs(fused_results[0].rrf_score - expected_score_b) < 1e-6
