"""Các test phải chạy không cần GPU/transformers/VnCoreNLP."""
from copy import deepcopy

import pytest

from src.preprocessing_qa import (
    IGNORE_INDEX,
    create_sliding_windows,
    preprocess_example,
)


class FakeTokenizer:
    """Trả về một subword ID cho mỗi token, đủ để test logic QA."""
    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return [100 + (sum(ord(c) for c in text) % 100)]


TOKENIZER = FakeTokenizer()
CONTEXT = "Hà Nội có nhiều trường đại học."
CONTEXT_TOKENS = ["Hà_Nội", "có", "nhiều", "trường", "đại_học", "."]
QUESTION_TOKENS = ["Thành_phố", "nào", "?"]


def make_row(text="Hà Nội", impossible=False):
    start = CONTEXT.index(text)
    return {
        "id": "q1",
        "question": "Thành phố nào?",
        "context": CONTEXT,
        "is_impossible": impossible,
        "answers": {"text": [] if impossible else [text],
                    "answer_start": [] if impossible else [start]},
    }


def inputs(features):
    return [
        (f["input_ids"], f["context_start"], f["context_end"],
         f["word_spans"], f["subword_spans"])
        for f in features
    ]


def test_dev_input_ids_and_window_offsets_never_depend_on_gold():
    rows = [make_row("Hà Nội"), make_row("Nội"), make_row("nhiều"),
            make_row("Hà Nội", impossible=True)]
    results = [
        preprocess_example(r, QUESTION_TOKENS, CONTEXT_TOKENS, TOKENIZER,
                           mode="eval")
        for r in rows
    ]
    assert all(inputs(results[0]) == inputs(r) for r in results[1:])


def test_dev_midword_gold_is_not_silently_labeled_impossible():
    fs = preprocess_example(make_row("Nội"), QUESTION_TOKENS,
                            CONTEXT_TOKENS, TOKENIZER, mode="eval")
    assert fs[0]["label_status"] == "unalignable"
    assert all(f["start_positions"] == IGNORE_INDEX for f in fs)
    assert all(f["is_impossible"] is False for f in fs)


def test_eval_aligned_gold_still_receives_positive_label():
    fs = preprocess_example(make_row("nhiều"), QUESTION_TOKENS,
                            CONTEXT_TOKENS, TOKENIZER, mode="eval")
    assert fs[0]["label_status"] == "aligned"
    assert fs[0]["start_positions"] > 0


def test_train_legacy_refines_midword_gold():
    fs = preprocess_example(make_row("Nội"), QUESTION_TOKENS,
                            CONTEXT_TOKENS, TOKENIZER, mode="train")
    assert fs[0]["label_status"] == "aligned"
    assert fs[0]["start_positions"] > 0
    assert fs[0]["word_spans"][0] == [0, 2]  # Hà, không còn Hà_Nội


def test_predict_requires_no_gold_and_no_is_impossible_field():
    row = {"id": "demo", "context": CONTEXT, "question": "Trường nào?"}
    fs = preprocess_example(row, QUESTION_TOKENS, CONTEXT_TOKENS,
                            TOKENIZER, mode="predict")
    assert fs[0]["label_status"] == "unlabeled"
    assert fs[0]["start_positions"] == IGNORE_INDEX


def test_offset_map_points_to_original_context():
    fs = preprocess_example(make_row("nhiều"), QUESTION_TOKENS,
                            CONTEXT_TOKENS, TOKENIZER, mode="eval")
    f = fs[0]
    assert len(f["word_spans"]) == len(f["subword_spans"])
    assert CONTEXT[slice(*f["word_spans"][0])] == "Hà Nội"
    assert CONTEXT[slice(*f["word_spans"][2])] == "nhiều"
    for lo, hi in f["subword_spans"]:
        assert 0 <= lo < hi <= len(f["input_ids"])


def test_sliding_window_in_eval_does_not_need_answer_positions():
    context = list(range(1000, 1600))
    a = create_sliding_windows([101], context, max_len=64, doc_stride=16)
    b = create_sliding_windows([101], context, max_len=64, doc_stride=16)
    assert a == b and len(a) > 1
    assert all(w["context_end"] <= len(context) for w in a)


def test_bad_mode_is_rejected():
    with pytest.raises(ValueError, match="Invalid mode"):
        preprocess_example(make_row(), QUESTION_TOKENS, CONTEXT_TOKENS,
                           TOKENIZER, mode="foo")
