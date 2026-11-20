import unicodedata
import pytest

from src.qa_metrics import (
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
