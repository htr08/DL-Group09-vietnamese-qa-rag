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
    """Kiểm tra rerank dùng boost mặc định (0.0003) trên điểm thật từ RRF và lọc stopwords tiếng Việt."""
    # Điểm thật sinh ra từ RRF: d1 hạng 1 ở cả 2 nguồn, d2 hạng 2 ở cả 2 nguồn
    bm25_list = ["d1", "d2"]
    dense_list = ["d1", "d2"]
    fused = reciprocal_rank_fusion([bm25_list, dense_list], rrf_k=60)
    # fused = [('d1', 2/61 ≈ 0.032787), ('d2', 2/62 ≈ 0.032258)] -> chênh lệch 0.000529

    doc_titles = {
        "d1": "Giới thiệu chung và các quy định",        # chứa stopwords 'và', 'các'
        "d2": "Quy chế đào tạo tín chỉ đại học"         # chứa các từ khóa thực 'đào', 'tạo', 'tín', 'chỉ'
    }
    # Query có từ dừng 'và', 'các' kèm từ khóa 'đào tạo tín chỉ'
    # d1 không được boost vì chỉ trùng stopwords
    # d2 trùng 4 từ khóa thực nên được boost 4 * 0.0003 = 0.0012 > 0.000529 -> vượt lên hạng 1
    # Gọi hàm KHÔNG truyền tham số boost để kiểm tra giá trị mặc định (boost=0.0003)
    reranked = rerank_by_title(fused, doc_titles, query="và các đào tạo tín chỉ")
    assert reranked[0][0] == "d2"


def test_sliding_window_direct():
    """Kiểm tra gọi trực tiếp sliding_window: đúng kích thước từ, giữ nguyên dấu xuống dòng."""
    from scripts.build_index import sliding_window

    # Văn bản ngắn <= max_words: giữ nguyên 1 chunk, không thêm suffix _w0
    short_text = "Dòng 1\nDòng 2\nDòng 3 có vài từ ngắn."
    res_short = sliding_window(short_text, "dieu1_khoan1", max_words=200)
    assert len(res_short) == 1
    assert res_short[0][0] == "dieu1_khoan1"
    assert "\nDòng 2\n" in res_short[0][1]

    # Văn bản dài > 200 từ: phải tách thành ít nhất 2 cửa sổ _w0, _w1 và giữ \n
    long_lines = [f"Dòng {i}: " + ("từ " * 25) for i in range(10)]
    long_text = "\n".join(long_lines)
    res_long = sliding_window(long_text, "dieu2_khoan1", max_words=200, stride=180)
    assert len(res_long) >= 2
    assert res_long[0][0] == "dieu2_khoan1_w0"
    assert res_long[1][0] == "dieu2_khoan1_w1"
    assert "\n" in res_long[0][1]
    assert "\n" in res_long[1][1]


def test_chunk_quy_che_points_and_window_headers():
    """Kiểm tra tách Điểm (_diem{a-z}) và mọi cửa sổ trượt (_w1) đều giữ tiêu đề Điều."""
    long_diem_b = " ".join([f"noidung{i}" for i in range(250)])
    sample_text = f"""
Điều 8. Kinh phí đào tạo
1. Nguồn kinh phí đào tạo
a) Kinh phí từ ngân sách nhà nước cấp hàng năm.
b) Kinh phí từ học phí và các nguồn thu hợp pháp khác:
{long_diem_b}
"""
    chunks = chunk_quy_che(sample_text)
    chunk_ids = [c[0] for c in chunks]

    # Kiểm tra tách Điểm: có _diema và các cửa sổ của _diemb
    assert "dieu8_khoan1_diema" in chunk_ids
    assert "dieu8_khoan1_diemb_w0" in chunk_ids
    assert "dieu8_khoan1_diemb_w1" in chunk_ids

    # Kiểm tra ngữ cảnh: CẢ HAI cửa sổ _w0 và _w1 đều phải có dòng tiêu đề Điều ở đầu
    chunk_map = {c[0]: c[2] for c in chunks}
    for sub_id in ["dieu8_khoan1_diemb_w0", "dieu8_khoan1_diemb_w1"]:
        lines = [ln.strip() for ln in chunk_map[sub_id].splitlines() if ln.strip()]
        assert lines[0] == "Điều 8. Kinh phí đào tạo", f"{sub_id} bị mất tiêu đề Điều ở dòng đầu"


def test_chunk_quy_che_on_real_pdf():
    """Kiểm tra chunk_quy_che trên file PDF thật: đủ 8 Điều _intro, không trùng ID, cửa sổ _w1 có tiêu đề."""
    from pathlib import Path
    import pytest

    pdf_path = Path(__file__).parents[1] / "data" / "domain" / "dhqghn_quy_che_dao_tao_dai_hoc_2014.pdf"
    if not pdf_path.exists():
        pytest.skip(f"Không tìm thấy file PDF tại {pdf_path}")

    import pdfplumber
    from scripts.build_index import normalize_text

    with pdfplumber.open(pdf_path) as pdf:
        full_text = "\n".join((p.extract_text() or "") for p in pdf.pages[:-3])
    chunks = chunk_quy_che(normalize_text(full_text))
    chunk_ids = [c[0] for c in chunks]

    # 1. Không trùng ID
    assert len(chunk_ids) == len(set(chunk_ids)), "Có chunk_id bị trùng lặp"

    # 2. Đủ 8 Điều có _intro
    for d in [10, 11, 22, 27, 28, 37, 39, 41]:
        assert f"dieu{d}_intro" in chunk_ids, f"Thiếu dieu{d}_intro trong PDF thật"

    # 3. Mọi cửa sổ trượt _w đều có dòng đầu tiên là tiêu đề Điều
    for cid, title, txt in chunks:
        if "_w" in cid:
            lines = [ln.strip() for ln in txt.splitlines() if ln.strip()]
            assert lines[0].startswith("Điều "), f"{cid} trong PDF thật không có tiêu đề Điều: {lines[0]!r}"


def test_chunk_quy_che_intro_and_clause_ids():
    """Kiểm tra xử lý đoạn dẫn _intro và lấy đúng số Khoản cho Điều 41 và Điều 10."""
    sample_text = """
Điều 10. Cấu trúc chương trình đào tạo
Chương trình đào tạo được sắp xếp theo các khối kiến thức:
1. Khối kiến thức chung được tổ chức giảng dạy thống nhất.
2. Khối kiến thức theo lĩnh vực.

Điều 11. Nguyên tắc xây dựng ngành học mới
Đơn vị đào tạo xây dựng đề án mở ngành học mới theo nguyên tắc:
1. Có nhu cầu xã hội cao.
2. Phù hợp với sứ mệnh đào tạo.

Điều 22. Đối tượng được miễn, tạm hoãn học các học phần giáo dục
quốc phòng - an ninh, giáo dục thể chất
1. Đối tượng được miễn.
2. Đối tượng được tạm hoãn.

Điều 27. Hoạt động nghiên cứu khoa học của sinh viên
Hoạt động nghiên cứu khoa học được tổ chức dưới các hình thức:
1. Tham gia câu lạc bộ khoa học.
2. Báo cáo khoa học tại hội nghị.

Điều 28. Tổ chức và quản lý hoạt động nghiên cứu khoa học của
sinh viên
Công tác tổ chức được phân cấp như sau:
1. Đơn vị đào tạo xây dựng kế hoạch.
2. Phòng công tác sinh viên triển khai.

Điều 37. Kỷ luật đối với cán bộ coi thi, tổ chức thi, chấm thi
Người vi phạm sẽ bị xử lý theo các hình thức sau:
1. Khiển trách.
2. Cảnh cáo.
3. Buộc thôi việc.

Điều 39. Đánh giá kết quả học tập
Kết quả học tập được đánh giá theo các tiêu chí sau:
1. Khối lượng kiến thức.
2. Điểm trung bình chung tích lũy.
3. Hạnh kiểm và ý thức.

Điều 41. Xử lý học vụ
Sau mỗi học kỳ chính, đơn vị đào tạo thực hiện xử lý học vụ. Kết quả học kỳ phụ tính vào học kỳ trước.
1. Cảnh báo học vụ
Đơn vị đào tạo thực hiện cảnh báo học vụ theo từng học kỳ.
2. Buộc thôi học
Sinh viên bị buộc thôi học nếu thuộc các trường hợp quy định.
"""
    chunks = chunk_quy_che(sample_text)
    chunk_ids = [c[0] for c in chunks]

    # Kiểm tra TẤT CẢ 8 Điều có đoạn dẫn _intro theo đúng danh sách reviewer chỉ ra
    for dieu_num in [10, 11, 22, 27, 28, 37, 39, 41]:
        intro_id = f"dieu{dieu_num}_intro"
        assert intro_id in chunk_ids, f"Thiếu {intro_id} trong danh sách chunk"

    # Điều 41: Khoản phải lấy đúng số từ regex, không bị lệch do đếm tuần tự
    assert "dieu41_khoan1" in chunk_ids
    assert "dieu41_khoan2" in chunk_ids
    d41_k1 = next(c[2] for c in chunks if c[0] == "dieu41_khoan1")
    assert "1. Cảnh báo học vụ" in d41_k1, "dieu41_khoan1 phải chứa nội dung Cảnh báo học vụ"

    # Điều 10: Khoan 1 và 2 đúng
    assert "dieu10_khoan1" in chunk_ids
    assert "dieu10_khoan2" in chunk_ids

    # Không được có ID trùng nhau
    assert len(chunk_ids) == len(set(chunk_ids)), f"Có ID chunk bị trùng: {[x for x in chunk_ids if chunk_ids.count(x) > 1]}"


def test_chunk_quy_che_sliding_window_for_long_clause():
    """Kiểm tra sliding_window được áp dụng cho Khoản dài không có Điểm."""
    # Tạo Khoản dài hơn 200 từ, không có a), b), c)
    long_content = " ".join([f"từ{i}" for i in range(250)])
    sample_text = f"""
Điều 13. Thiết kế chương trình đào tạo
1. Thiết kế chương trình
{long_content}
"""
    chunks = chunk_quy_che(sample_text)
    chunk_ids = [c[0] for c in chunks]
    # Khoản dài phải bị chia thành các cửa sổ _w0, _w1, ...
    sliding_ids = [cid for cid in chunk_ids if cid.startswith("dieu13_khoan1_w")]
    assert len(sliding_ids) >= 2, f"Khoản dài không có Điểm phải được tách sliding, có: {chunk_ids}"
    # Không có ID trùng nhau
    assert len(chunk_ids) == len(set(chunk_ids))


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
