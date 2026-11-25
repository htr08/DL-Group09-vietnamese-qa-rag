"""Unit test cho chế độ trích xuất (src/extractive.py, pipeline): không cần GPU, Java hay checkpoint."""
import numpy as np
import pytest

from src.extractive import extract_answer, load_extractive_reader, load_qa_model, load_segmenter, segment_text
from src.generate import REFUSAL_TEXT, is_refusal
from src.pipeline import answer_question_extractive


class FakeTokenizer:
    """Mỗi từ ứng với một subword; id đặc biệt theo PhoBERT (pad=1, </s>=2)."""
    pad_token_id = 1
    sep_token_id = 2

    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return [100 + (sum(ord(c) for c in text) % 100)]


TOKENIZER = FakeTokenizer()
CONTEXT_A = "Hà Nội có nhiều trường đại học."
CONTEXT_B = "Huế là cố đô của Việt Nam."


def segment(text):
    return [w for w in text.replace(".", " .").split()]


def make_logits_fn(target_word, strength):
    """Logit start/end cao nhất ở `target_word`; điểm null (vị trí CLS) bằng 0, các vị trí khác -3."""
    calls = []

    def logits_fn(features):
        calls.append(len(features))
        out = np.full((len(features), max(len(f["input_ids"]) for f in features), 2), -3.0)
        out[:, 0, :] = 0.0
        for i, f in enumerate(features):
            words = [f["input_ids"][s] for s, _ in f["subword_spans"]]
            target = TOKENIZER.encode(target_word)[0]
            for (s, t), word_id in zip(f["subword_spans"], words):
                if word_id == target:
                    out[i, s, 0] = strength
                    out[i, t - 1, 1] = strength
        return out

    logits_fn.calls = calls
    return logits_fn


def run(contexts, logits_fn, question="Câu hỏi nào?", **kwargs):
    return extract_answer(segment, TOKENIZER, logits_fn, question, contexts, **kwargs)


def test_returns_span_text_with_source_and_rank():
    contexts = [{"id": "a", "text": CONTEXT_A}, {"id": "b", "text": CONTEXT_B}]
    result = run(contexts, make_logits_fn("Huế", 5.0))
    assert result["answer"] == "Huế"
    assert not result["refusal"]
    assert (result["source_id"], result["source_rank"]) == ("b", 2)
    start, end = result["span"]
    assert CONTEXT_B[start:end] == "Huế"
    assert result["score_diff"] == pytest.approx(-10.0)
    assert result["nbest"][0]["chunk_id"] == "b"


def test_refuses_when_best_span_does_not_beat_null_score():
    # span có điểm -1 + -1 = -2 < điểm null (0): score_diff = 2 > ngưỡng 0.
    result = run([{"id": "a", "text": CONTEXT_A}], make_logits_fn("Nội", -1.0))
    assert result["refusal"] and result["answer"] == REFUSAL_TEXT and is_refusal(result["answer"])
    assert result["source_id"] == "a"  # ứng viên tốt nhất vẫn được giữ để tham khảo
    assert result["score_diff"] > 0


def test_threshold_moves_the_decision():
    contexts = [{"id": "a", "text": CONTEXT_A}]
    logits_fn = make_logits_fn("Nội", -1.0)
    assert run(contexts, logits_fn, null_threshold=0.0)["refusal"]
    assert not run(contexts, logits_fn, null_threshold=5.0)["refusal"]


def test_empty_contexts_or_blank_question_refuse_without_calling_model():
    logits_fn = make_logits_fn("Nội", 5.0)
    for contexts, question in (([], "câu hỏi"), ([{"id": "a", "text": "  \n"}], "câu hỏi"),
                               ([{"id": "a", "text": CONTEXT_A}], "   ")):
        result = run(contexts, logits_fn, question=question)
        assert result["refusal"] and is_refusal(result["answer"])
        assert result["windows"] == 0
    assert logits_fn.calls == []


def test_blank_context_is_skipped_but_rank_keeps_position():
    contexts = [{"id": "x", "text": " "}, {"id": "b", "text": CONTEXT_B}]
    result = run(contexts, make_logits_fn("Huế", 5.0))
    assert (result["source_id"], result["source_rank"]) == ("b", 2)


def test_long_context_uses_several_windows_and_batches():
    long_context = " ".join(f"từ{i}" for i in range(500)) + " Huế " + " ".join(f"chữ{i}" for i in range(20))
    logits_fn = make_logits_fn("Huế", 5.0)
    result = run([{"id": "long", "text": long_context}], logits_fn, batch_size=2)
    assert result["windows"] > 2 and max(logits_fn.calls) <= 2
    assert result["answer"] == "Huế"
    start, end = result["span"]
    assert long_context[start:end] == "Huế"


def test_nan_logits_raise_instead_of_returning_a_silent_answer():
    def nan_logits(features):
        return np.full((len(features), max(len(f["input_ids"]) for f in features), 2), np.nan)

    with pytest.raises(ValueError, match="hữu hạn"):
        run([{"id": "a", "text": CONTEXT_A}], nan_logits)


def test_question_is_normalized_to_nfc_before_segmentation():
    import unicodedata

    seen = []

    def spy_segment(text):
        seen.append(text)
        return segment(text)

    nfd_question = unicodedata.normalize("NFD", "Huế ở đâu?")
    assert nfd_question != unicodedata.normalize("NFC", nfd_question)
    extract_answer(spy_segment, TOKENIZER, make_logits_fn("Huế", 5.0), nfd_question, [{"id": "b", "text": CONTEXT_B}])
    assert seen[0] == unicodedata.normalize("NFC", nfd_question)


def test_invalid_arguments_are_rejected():
    with pytest.raises(ValueError):
        run([{"id": "a", "text": CONTEXT_A}], make_logits_fn("Nội", 1.0), batch_size=0)


def test_segment_text_keeps_rare_character_and_rejects_empty_output():
    class FakeVn:
        def __init__(self, sentences):
            self.sentences = sentences

        def word_segment(self, text):
            return self.sentences

    assert segment_text(FakeVn(["Hà_Nội có", "đại_học"]), "x") == ["Hà_Nội", "có", "đại_học"]
    assert segment_text(FakeVn(["讁 x"]), "𨮁 x") == ["𨮁", "x"]
    with pytest.raises(ValueError):
        segment_text(FakeVn([]), "x")


def test_loaders_report_missing_files_before_importing_heavy_libs(tmp_path):
    with pytest.raises(FileNotFoundError, match="config.json"):
        load_qa_model(tmp_path)
    with pytest.raises(FileNotFoundError, match="VnCoreNLP"):
        load_segmenter(tmp_path)
    with pytest.raises(FileNotFoundError, match="config.json"):
        load_extractive_reader(tmp_path, tmp_path)


class FakeRetriever:
    def __init__(self, ranking):
        self.ranking = ranking

    def retrieve(self, query, top_k=10):
        return self.ranking[:top_k]


def test_answer_question_extractive_passes_ranked_contexts_and_returns_sources():
    chunks = {"a": CONTEXT_A, "b": CONTEXT_B, "c": "đoạn c"}
    # Hai bộ truy hồi cùng thứ hạng nên điểm RRF không hòa; thứ tự không phụ thuộc cách phá hòa của retrieval.
    bm25 = FakeRetriever([("a", 3.0), ("b", 2.0), ("c", 1.0)])
    dense = FakeRetriever([("a", 0.9), ("b", 0.8), ("c", 0.1)])
    seen = {}

    def fake_extract(question, contexts):
        seen["question"], seen["contexts"] = question, contexts
        return {"answer": "Huế", "refusal": False, "source_id": "b", "source_rank": 2}

    result = answer_question_extractive(fake_extract, bm25, dense, chunks, "câu hỏi", top_k=2)
    assert seen["question"] == "câu hỏi"
    assert [c["id"] for c in seen["contexts"]] == ["a", "b"]
    assert [c["text"] for c in seen["contexts"]] == [CONTEXT_A, CONTEXT_B]
    assert result["sources"] == ["a", "b"] and result["sources"][result["source_rank"] - 1] == "b"
    assert result["retrieval_seconds"] >= 0
