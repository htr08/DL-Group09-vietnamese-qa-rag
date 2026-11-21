import unicodedata
import pytest

from src.metrics import (
    normalize_answer,
    exact_match_score,
    token_f1_score,
    f1_score,
)


def test_normalize_answer():
    assert normalize_answer("  Hà Nội, Việt Nam! ") == "hà nội việt nam"


def test_normalize_answer_unicode():
    text = unicodedata.normalize("NFD", "Việt Nam")
    assert normalize_answer(text) == "việt nam"


def test_exact_match():
    assert exact_match_score("Hà Nội", ["hà nội"]) is True


def test_exact_match_with_punctuation():
    assert exact_match_score("Hà Nội!", ["Hà Nội"]) is True


def test_exact_match_multiple_answers():
    assert exact_match_score("Hà Nội", ["Hồ Chí Minh", "Hà Nội"]) is True


def test_exact_match_incorrect():
    assert exact_match_score("Hồ Chí Minh", ["Hà Nội"]) is False


def test_token_f1_partial_match():
    assert token_f1_score("Hà Nội", "Thủ đô Hà Nội") == pytest.approx(2 / 3)


def test_token_f1_no_overlap():
    assert token_f1_score("Hà Nội", "Đà Nẵng") == 0.0


def test_token_f1_empty_answers():
    assert token_f1_score("", "") == 1.0


def test_f1_multiple_answers():
    assert f1_score("Hà Nội", ["Đà Nẵng", "Hà Nội"]) == 1.0


def test_f1_no_gold_answers():
    assert f1_score("Hà Nội", []) == 0.0


# Retrieval metric regression tests from main.
from src.metrics import recall_at_k, mrr_at_k, mean_metric, is_match

def test_recall_at_k():
    retrieved = ["doc_1", "doc_2", "doc_3", "doc_4", "doc_5"]
    assert recall_at_k(retrieved, "doc_1", k=1) == 1.0
    assert recall_at_k(retrieved, "doc_3", k=3) == 1.0
    assert recall_at_k(retrieved, "doc_4", k=3) == 0.0
    assert recall_at_k(retrieved, "doc_non_exist", k=5) == 0.0


def test_recall_at_k_prefix_matching():
    # Trường hợp hệ thống trả về sub-chunk _w0 hoặc _diema nhưng gold là Khoản cha
    retrieved = ["dieu5_khoan2_w0", "dieu10_khoan1", "dieu20_khoan2"]
    assert recall_at_k(retrieved, "dieu5_khoan2", k=1) == 1.0
    assert recall_at_k(["dieu20_khoan2_diema"], "dieu20_khoan2", k=1) == 1.0
    # Gold chi tiết hơn
    assert recall_at_k(["dieu5_khoan2"], "dieu5_khoan2_w0", k=1) == 1.0


def test_recall_at_k_multi_gold():
    # Một câu hỏi có thể có nhiều gold chunks
    retrieved = ["dieu8_khoan1", "dieu10_khoan2", "dieu15_khoan1"]
    multi_gold = ["dieu10_khoan2", "dieu20_khoan1"]
    assert recall_at_k(retrieved, multi_gold, k=2) == 1.0
    assert recall_at_k(retrieved, ["dieu99", "dieu100"], k=3) == 0.0


def test_recall_at_k_article_level():
    # Đánh giá ở mức Điều
    retrieved = ["dieu41_khoan3_w0"]
    assert recall_at_k(retrieved, "dieu41", k=1, match_level="article") == 1.0
    assert recall_at_k(retrieved, "dieu41_khoan1", k=1, match_level="article") == 1.0
    assert recall_at_k(retrieved, "dieu42", k=1, match_level="article") == 0.0


def test_mrr_at_k():
    retrieved = ["doc_1", "doc_2", "doc_3"]
    assert mrr_at_k(retrieved, "doc_1", k=10) == 1.0
    assert mrr_at_k(retrieved, "doc_2", k=10) == 0.5
    assert mrr_at_k(retrieved, "doc_3", k=10) == 1.0 / 3.0
    assert mrr_at_k(retrieved, "doc_4", k=10) == 0.0


def test_mrr_at_k_prefix_and_multi_gold():
    retrieved = ["other_doc", "dieu41_khoan2_w0", "dieu5_khoan1"]
    # dieu41_khoan2 ở vị trí số 2 -> MRR = 0.5
    assert mrr_at_k(retrieved, "dieu41_khoan2", k=10) == 0.5
    # Nhiều gold, một cái ở vị trí 2, một cái ở vị trí 3 -> lấy vị trí đầu tiên
    assert mrr_at_k(retrieved, ["dieu5_khoan1", "dieu41_khoan2"], k=10) == 0.5


def test_mean_metric():
    assert mean_metric([1.0, 0.5, 0.0]) == 0.5
    assert mean_metric([]) == 0.0
