"""Unit test cho src/pipeline.py (chế độ sinh): nối truy hồi với sinh, không cần GPU."""
from src.pipeline import answer_question, retrieve_contexts


class FakeRetriever:
    """Retriever giả trả về danh sách (doc_id, điểm) cố định."""

    def __init__(self, ranking):
        self.ranking = ranking

    def retrieve(self, query, top_k=10):
        return self.ranking[:top_k]


CHUNKS = {"a": "đoạn a", "b": "đoạn b", "c": "đoạn c"}
BM25 = FakeRetriever([("a", 3.0), ("b", 2.0), ("c", 1.0)])
DENSE = FakeRetriever([("b", 0.9), ("a", 0.8), ("c", 0.1)])


def test_retrieve_contexts_returns_top_k_with_text():
    contexts = retrieve_contexts(BM25, DENSE, CHUNKS, "câu hỏi", top_k=2)
    assert len(contexts) == 2
    # a và b đều đứng đầu một danh sách; RRF đồng điểm giữ thứ tự xuất hiện đầu tiên (BM25 trước).
    assert [c["id"] for c in contexts] == ["a", "b"]
    assert [c["text"] for c in contexts] == ["đoạn a", "đoạn b"]
    assert contexts[0]["score"] >= contexts[1]["score"]


def test_answer_question_passes_numbered_contexts_and_returns_sources():
    seen = {}

    def fake_generate(messages, sampling):
        seen["user"] = messages[1]["content"]
        return "Trả lời [1]."

    result = answer_question(fake_generate, BM25, DENSE, CHUNKS, "câu hỏi", top_k=2)
    assert "[1] đoạn a" in seen["user"] and "[2] đoạn b" in seen["user"]
    assert result["answer"] == "Trả lời [1]."
    assert result["sources"] == ["a", "b"]  # nguồn [i+1] ứng với sources[i]
    assert result["retrieval_seconds"] >= 0 and not result["cjk"]


def test_answer_question_reports_invalid_citations():
    def fake_generate(messages, sampling):
        return "Theo [1] và [4]."

    result = answer_question(fake_generate, BM25, DENSE, CHUNKS, "câu hỏi", top_k=2)
    assert result["invalid_citations"] == [4]  # chỉ có hai đoạn nên [4] không hợp lệ
