import json
import sys

import pytest

import scripts.run_eval as run_eval


class FakeRetriever:
    def __init__(self, ids):
        self.ids = ids

    def retrieve(self, query, top_k=10):
        return [(doc_id, 1.0 / (i + 1)) for i, doc_id in enumerate(self.ids[:top_k])]


QUERIES = [{"question": "q", "relevant_id": "dieu5_khoan2"}]


def test_article_level_metrics_only_when_requested():
    retriever = FakeRetriever(["dieu5_khoan1", "dieu5_khoan2_w0"])

    default, _ = run_eval.evaluate_retriever(retriever, QUERIES, [1, 3], "fake")
    assert "R@1_dieu" not in default and "MRR@10_dieu" not in default

    metrics, _ = run_eval.evaluate_retriever(retriever, QUERIES, [1, 3], "fake", article_level=True)
    # Mức Khoản: chunk đúng ở hạng 2; mức Điều: chunk cùng Điều ở hạng 1.
    assert (metrics["R@1"], metrics["R@3"], metrics["MRR@10"]) == (0.0, 1.0, 0.5)
    assert (metrics["R@1_dieu"], metrics["R@3_dieu"], metrics["MRR@10_dieu"]) == (1.0, 1.0, 1.0)


def test_hybrid_article_level_metrics():
    retriever = FakeRetriever(["dieu5_khoan1", "dieu5_khoan2"])
    metrics, _ = run_eval.evaluate_hybrid(
        retriever, retriever, QUERIES, [1, 3], rrf_k=60, article_level=True
    )
    assert (metrics["R@1"], metrics["R@3"]) == (0.0, 1.0)
    assert (metrics["R@1_dieu"], metrics["MRR@10_dieu"]) == (1.0, 1.0)


def _write_gold(tmp_path, rows, encoding="utf-8"):
    text = "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n"
    (tmp_path / "gold_dev.jsonl").write_text(text, encoding=encoding)


def test_load_gold_queries_follows_annotation_guide_schema(tmp_path, monkeypatch):
    monkeypatch.setattr(run_eval, "GOLD_DIR", tmp_path)
    _write_gold(tmp_path, [
        {"id": "B01", "question": "Q1", "in_scope": True, "article": 21, "clause": 2},
        {"id": "B02", "question": "Q2", "in_scope": True, "article": 15, "clause": None},
        {"id": "B03", "question": "Q3", "in_scope": False, "article": None, "clause": None},
    ], encoding="utf-8-sig")  # có BOM

    queries = run_eval.load_gold_queries("dev")

    # Điều 15 không chia Khoản nên nhãn ở mức Điều; câu ngoài phạm vi bị tách ra.
    assert [q["relevant_id"] for q in queries] == ["dieu21_khoan2", "dieu15"]


def test_load_gold_queries_raises_when_in_scope_question_has_no_source(tmp_path, monkeypatch):
    monkeypatch.setattr(run_eval, "GOLD_DIR", tmp_path)
    _write_gold(tmp_path, [{"id": "B01", "question": "Q1", "in_scope": True, "article": None, "clause": None}])
    with pytest.raises(ValueError):
        run_eval.load_gold_queries("dev")


def test_load_gold_queries_raises_when_file_is_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(run_eval, "GOLD_DIR", tmp_path)
    (tmp_path / "gold_dev.jsonl").write_text("", encoding="utf-8")
    with pytest.raises(FileNotFoundError):
        run_eval.load_gold_queries("dev")


def test_test_split_requires_explicit_rrf_k(monkeypatch):
    """Trên tập test, k của RRF phải truyền từ dev qua --rrf_k, không được tự chọn lại (§8.2)."""
    monkeypatch.setattr(sys, "argv", ["run_eval.py", "--split", "domain_test", "--method", "hybrid"])
    monkeypatch.setattr(run_eval, "load_gold_queries", lambda split: [])
    monkeypatch.setattr(run_eval, "load_doc_titles", lambda corpus: {})
    monkeypatch.setattr(run_eval, "load_bm25", lambda corpus: object())
    monkeypatch.setattr(run_eval, "load_dense", lambda corpus: object())
    with pytest.raises(ValueError, match="--rrf_k"):
        run_eval.main()
