import pytest
from src.preprocessing_qa import refine_tokens_for_answer
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

from src.preprocessing_qa import align_tokens_to_text, map_answer_to_subwords


def test_align_tokens_to_text():
    text = "Sinh viên ngành Trí tuệ nhân tạo."
    tokens = ["Sinh_viên", "ngành", "Trí_tuệ_nhân_tạo", "."]

    spans = align_tokens_to_text(text, tokens)

    assert [text[start:end] for start, end in spans] == [
        "Sinh viên",
        "ngành",
        "Trí tuệ nhân tạo",
        ".",
    ]


def test_map_answer_to_subwords():
    word_spans = [(0, 2), (3, 6), (7, 10)]
    subword_spans = [(0, 1), (1, 3), (3, 4)]

    start, end = map_answer_to_subwords(
        answer_start=0,
        answer_text="Ha Noi",
        word_spans=word_spans,
        subword_spans=subword_spans,
    )

    assert (start, end) == (0, 2)


def test_refine_tokens_for_answer_splits_boundary_token():
    text = "Giải Nobel Vật lý năm 1938."
    tokens = ["Giải", "Nobel_Vật", "lý", "năm", "1938", "."]

    refined = refine_tokens_for_answer(
        text,
        tokens,
        answer_start=text.index("Vật lý"),
        answer_text="Vật lý",
    )

    assert refined == [
        "Giải",
        "Nobel",
        "Vật",
        "lý",
        "năm",
        "1938",
        ".",
    ]

    spans = align_tokens_to_text(text, refined)

    answer_start = text.index("Vật lý")
    answer_end = answer_start + len("Vật lý")

    assert spans[2] == (answer_start, answer_start + len("Vật"))
    assert spans[3] == (answer_start + len("Vật "), answer_end)


def test_align_tokens_after_vietnamese_accents():
    text = "Công nghệ thông tin và truyền thông"
    tokens = ["Công_nghệ", "thông_tin", "và", "truyền_thông"]

    spans = align_tokens_to_text(text, tokens)

    assert [
        text[start:end]
        for start, end in spans
    ] == [
        "Công nghệ",
        "thông tin",
        "và",
        "truyền thông",
    ]


def test_align_token_with_attached_quote():
    text = 'Tổng thống Mỹ",'
    tokens = ["Tổng_thống", 'Mỹ_"', ","]

    spans = align_tokens_to_text(text, tokens)

    assert [
        text[start:end]
        for start, end in spans
    ] == [
        "Tổng thống",
        'Mỹ"',
        ",",
    ]


def test_align_token_with_closing_parenthesis():
    text = "Tân)"
    tokens = ["Tân_)"]

    spans = align_tokens_to_text(text, tokens)

    assert text[spans[0][0]:spans[0][1]] == "Tân)"


def test_align_token_with_percent():
    text = "57%"
    tokens = ["57%"]

    spans = align_tokens_to_text(text, tokens)

    assert text[spans[0][0]:spans[0][1]] == "57%"


def test_refine_tokens_splits_percent_token():
    text = "Tỷ lệ là 57%."
    tokens = ["Tỷ_lệ", "là", "57%."]
    answer_start = text.index("57")
    answer_text = "57"

    refined = refine_tokens_for_answer(
        text, tokens, answer_start, answer_text
    )

    assert refined == ["Tỷ_lệ", "là", "57", "%."]


def test_refine_tokens_splits_answer_inside_word():
    text = "Tân)"
    tokens = ["Tân_)"]
    answer_start = 0
    answer_text = "Tân"

    refined = refine_tokens_for_answer(
        text, tokens, answer_start, answer_text
    )

    assert refined == ["Tân", ")"]

def test_sliding_windows_cover_long_answer():
    question_ids = list(range(11))
    context_ids = list(range(265))

    windows = create_sliding_windows(
        question_ids,
        context_ids,
        max_len=256,
        max_question_len=64,
        doc_stride=128,
        answer_start=72,
        answer_end=263
    )

    features = label_sliding_windows(
        windows,
        answer_start=72,
        answer_end=263,
        question_length=len(question_ids)
    )

    positive = [
        feature for feature in features
        if feature["start_positions"] != 0
    ]

    assert len(positive) >= 1
    assert all(len(window["input_ids"]) <= 256 for window in windows)


def test_sliding_windows_reject_answer_longer_than_capacity():
    with pytest.raises(ValueError, match="answer_too_long"):
        create_sliding_windows(
            list(range(11)),
            list(range(300)),
            max_len=256,
            doc_stride=128,
            answer_start=0,
            answer_end=241
        )