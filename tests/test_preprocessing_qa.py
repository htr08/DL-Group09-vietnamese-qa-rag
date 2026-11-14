import pytest

from src.preprocessing_qa import (
    create_sliding_windows,
    label_sliding_windows,
    preprocess_example,
)


class DummyTokenizer:
    def encode(self, text, add_special_tokens=False):
        return [len(text)]


def test_sliding_windows_max_length():
    windows = create_sliding_windows(
        list(range(20)),
        list(range(600)),
        max_len=256,
        max_question_len=64,
        doc_stride=128,
    )

    assert len(windows) == 5
    assert all(len(window["input_ids"]) <= 256 for window in windows)


def test_sliding_windows_overlap():
    windows = create_sliding_windows(
        list(range(20)),
        list(range(600)),
        max_len=256,
        max_question_len=64,
        doc_stride=128,
    )

    assert windows[0]["context_end"] - windows[1]["context_start"] == 128


def test_answer_labels_in_overlapping_windows():
    windows = create_sliding_windows(
        list(range(20)),
        list(range(600)),
        max_len=256,
        max_question_len=64,
        doc_stride=128,
    )

    features = label_sliding_windows(
        windows,
        answer_start=230,
        answer_end=234,
        question_length=20,
    )

    positive = [
        feature for feature in features
        if feature["start_positions"] != 0
    ]

    assert len(positive) == 2


def test_impossible_question_labels():
    windows = create_sliding_windows(
        list(range(20)),
        list(range(600)),
    )

    features = label_sliding_windows(
        windows,
        0,
        0,
        20,
        is_impossible=True,
    )

    assert all(feature["start_positions"] == 0 for feature in features)
    assert all(feature["end_positions"] == 0 for feature in features)


def test_preprocess_rejects_empty_context():
    row = {
        "id": "test-1",
        "context": "",
        "is_impossible": True,
        "answers": {"text": [], "answer_start": []},
    }

    with pytest.raises(ValueError, match="Context is empty"):
        preprocess_example(row, [], [], DummyTokenizer())


def test_preprocess_rejects_missing_answer():
    row = {
        "id": "test-2",
        "context": "Ha Noi la thu do Viet Nam.",
        "is_impossible": False,
        "answers": {"text": [], "answer_start": []},
    }

    with pytest.raises(ValueError, match="no answer"):
        preprocess_example(
            row,
            ["Ha"],
            ["Ha", "Noi", "la", "thu", "do", "Viet", "Nam", "."],
            DummyTokenizer(),
        )


def test_preprocess_rejects_mismatched_answer_start():
    row = {
        "id": "test-3",
        "context": "Ha Noi la thu do Viet Nam.",
        "is_impossible": False,
        "answers": {"text": ["Paris"], "answer_start": [0]},
    }

    with pytest.raises(ValueError, match="does not match context"):
        preprocess_example(
            row,
            ["Ha"],
            ["Ha", "Noi", "la", "thu", "do", "Viet", "Nam", "."],
            DummyTokenizer(),
        )


def test_preprocess_rejects_mismatched_answer_arrays():
    row = {
        "id": "test-4",
        "context": "Ha Noi la thu do Viet Nam.",
        "is_impossible": False,
        "answers": {"text": ["Ha Noi"], "answer_start": []},
    }

    with pytest.raises(ValueError, match="different lengths"):
        preprocess_example(
            row,
            ["Ha"],
            ["Ha", "Noi", "la", "thu", "do", "Viet", "Nam", "."],
            DummyTokenizer(),
        )