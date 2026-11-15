"""Chỉ số đánh giá: Recall@k, MRR@10, bootstrap CI."""
from typing import List, Union, Any, Iterable, Set


def _normalize_gold_set(relevant_ids: Union[str, Iterable[str], Any]) -> Set[str]:
    """Chuyển đổi relevant_id đơn lẻ hoặc danh sách thành tập hợp chuỗi đã chuẩn hóa."""
    if relevant_ids is None:
        return set()
    if isinstance(relevant_ids, str):
        return {relevant_ids.strip()}
    if isinstance(relevant_ids, (list, tuple, set)):
        return {str(x).strip() for x in relevant_ids if x is not None}
    return {str(relevant_ids).strip()}


def is_match(retrieved_id: str, gold_id: str, match_level: str = "clause") -> bool:
    """
    Kiểm tra xem một chunk ID được truy hồi có khớp với một gold ID hay không.
    
    Args:
        retrieved_id: ID chunk được hệ thống trả về (ví dụ 'dieu5_khoan2_w0', 'dieu5_khoan2').
        gold_id: ID nhãn chuẩn (ví dụ 'dieu5_khoan2' hoặc 'dieu5').
        match_level: 'clause' (mặc định - so khớp mức Khoản/tiền tố) hoặc 'article'/'dieu' (mức Điều).
    """
    if not retrieved_id or not gold_id:
        return False
    retrieved_id = str(retrieved_id).strip()
    gold_id = str(gold_id).strip()

    # Mức Điều: chỉ cần cùng số Điều (ví dụ dieu5 khớp với dieu5_khoan2)
    if match_level in ("article", "dieu"):
        r_dieu = retrieved_id.split("_")[0]
        g_dieu = gold_id.split("_")[0]
        return r_dieu == g_dieu

    # Mức Khoản (mặc định):
    # 1. Bằng nhau chính xác
    if retrieved_id == gold_id:
        return True
    # 2. Khớp theo tiền tố: retrieved là chunk con của gold (vd gold='dieu5_khoan2', retrieved='dieu5_khoan2_w0' hoặc '_diema')
    if retrieved_id.startswith(gold_id + "_"):
        return True
    # 3. Gold chi tiết hơn chunk (vd gold='dieu5_khoan2_diema', retrieved='dieu5_khoan2')
    if gold_id.startswith(retrieved_id + "_"):
        return True

    return False


def recall_at_k(
    retrieved_ids: List[Any],
    relevant_id: Union[str, Iterable[str], Any],
    k: int,
    match_level: str = "clause"
) -> float:
    """
    Tính Recall@k cho 1 truy vấn.
    Hỗ trợ:
      - relevant_id là 1 chuỗi hoặc tập hợp nhiều gold IDs.
      - Khớp tiền tố (chunk con _w*, _diem* khớp với Khoản cha).
      - Đánh giá mức Điều khi match_level='article'.
    """
    gold_set = _normalize_gold_set(relevant_id)
    if not retrieved_ids or not gold_set:
        return 0.0

    top_k = retrieved_ids[:k]
    for r_id in top_k:
        for g_id in gold_set:
            if is_match(str(r_id), g_id, match_level=match_level):
                return 1.0
    return 0.0


def mrr_at_k(
    retrieved_ids: List[Any],
    relevant_id: Union[str, Iterable[str], Any],
    k: int = 10,
    match_level: str = "clause"
) -> float:
    """
    Tính Reciprocal Rank (MRR@k) cho 1 truy vấn.
    Trả về 1 / rank của vị trí xuất hiện đầu tiên của bất kỳ gold ID nào trong top-k.
    """
    gold_set = _normalize_gold_set(relevant_id)
    if not retrieved_ids or not gold_set:
        return 0.0

    top_k = retrieved_ids[:k]
    for rank, r_id in enumerate(top_k, start=1):
        for g_id in gold_set:
            if is_match(str(r_id), g_id, match_level=match_level):
                return 1.0 / rank
    return 0.0


def mean_metric(scores: List[float]) -> float:
    """Tính trung bình của một mảng chỉ số."""
    return sum(scores) / len(scores) if scores else 0.0
