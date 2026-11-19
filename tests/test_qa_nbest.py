import math
import pytest
from src.qa_nbest import decode_window, scores_at_threshold, best_dev_threshold


class DummyTokenizer:
    def decode(self, ids, **kwargs):
        return " ".join({4:"trong", 5:"quy_chế", 6:"này"}.get(x, "X") for x in ids)


def test_context_only_and_nbest_pairs():
    ids = [0, 9, 2, 2, 4, 5, 6, 2]
    start = [100.0, 99.0, 99.0, 99.0, 5.0, 1.0, 0.0, 100.0]
    end = [100.0, 99.0, 99.0, 99.0, 0.0, 7.0, 0.0, 100.0]
    candidates, null_score = decode_window(ids, start, end, DummyTokenizer(), 10, 2)
    assert null_score == 200
    assert candidates[0]["text"] == "trong quy chế"
    assert candidates[0]["start_token"] == 4
    assert candidates[0]["end_token"] == 5
    assert all(4 <= c["start_token"] <= c["end_token"] <= 6 for c in candidates)
    assert all(c["end_token"] - c["start_token"] + 1 <= 2 for c in candidates)


def test_missing_separator_rejected():
    with pytest.raises(ValueError, match="separator"):
        decode_window([0,4,5,2], [0]*4, [0]*4, DummyTokenizer())


def test_threshold_sweep_selects_best_dev_f1():
    rows = [
        {"score_diff": -1.0, "answer_em":1, "answer_f1":1, "null_em":0, "null_f1":0},
        {"score_diff": +2.0, "answer_em":0, "answer_f1":0, "null_em":1, "null_f1":1},
    ]
    t, em, f1 = best_dev_threshold(rows)
    assert -1.0 <= t < 2.0
    assert em == f1 == 1.0
    assert scores_at_threshold(rows, t) == (1.0, 1.0)


def test_threshold_semantics_equality_means_answer():
    row = [{"score_diff": 0.0, "answer_em":1, "answer_f1":1, "null_em":0, "null_f1":0}]
    assert scores_at_threshold(row, 0.0) == (1,1)
