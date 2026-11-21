"""Regression tests không cần GPU, model hay VnCoreNLP."""
import pytest
from src.qa_nbest import decode_window, best_dev_threshold, scores_at_threshold


class FakeTokenizer:
    def decode(self, ids, **kwargs):
        return "legacy"


def test_offset_decode_picks_second_occurrence_not_first():
    # <s> Q </s></s> [word1] [word2] [word3] </s>
    text = "Hà Nội và Hà Nội"
    ids = [0, 100, 2, 2, 101, 102, 103, 2]
    # Các word lần lượt "Hà Nội", "và", "Hà Nội".
    words = [[0, 6], [7, 9], [10, 16]]
    subs = [[4, 5], [5, 6], [6, 7]]
    st = [0.0] * len(ids)
    en = [0.0] * len(ids)
    st[6], en[6] = 8.0, 7.0
    items, null = decode_window(ids, st, en, FakeTokenizer(),
                                word_spans=words, subword_spans=subs,
                                context=text)
    assert items[0]["text"] == "Hà Nội"
    assert items[0]["char_start"] == 10
    assert items[0]["char_end"] == 16
    assert null == 0


def test_rejects_internal_subword_start_despite_high_logit():
    ids = [0, 100, 2, 2, 11, 12, 13, 2]
    st = [0.0] * len(ids)
    en = [0.0] * len(ids)
    st[5] = 99.0  # subword nằm giữa từ đầu
    st[4] = 1.0
    en[5] = 2.0
    en[6] = 3.0
    items, _ = decode_window(ids, st, en, FakeTokenizer(),
                             word_spans=[[0, 4], [5, 8]],
                             subword_spans=[[4, 6], [6, 7]],
                             context="ABCD EFG")
    assert all(c["start_token"] != 5 for c in items)
    assert items[0]["start_token"] == 4


def test_partial_offset_information_rejected():
    ids = [0, 100, 2, 2, 11, 2]
    with pytest.raises(ValueError, match="must be supplied together"):
        decode_window(ids, [0] * 6, [0] * 6, FakeTokenizer(),
                      word_spans=[[0, 3]], context="abc")


def test_window_can_have_no_complete_words():
    ids = [0, 100, 2, 2, 11, 2]
    items, null = decode_window(ids, [0] * 6, [0] * 6, FakeTokenizer(),
                                word_spans=[], subword_spans=[], context="abc")
    assert items == []
    assert null == 0


def test_threshold_math_unchanged():
    rows = [
        {"score_diff": -2.0, "answer_em": 1, "answer_f1": 1,
         "null_em": 0, "null_f1": 0},
        {"score_diff": 3.0, "answer_em": 0, "answer_f1": 0,
         "null_em": 1, "null_f1": 1},
    ]
    t, em, f1 = best_dev_threshold(rows)
    assert -2.0 <= t < 3.0
    assert (em, f1) == (1.0, 1.0)
    assert scores_at_threshold(rows, t) == (1.0, 1.0)
