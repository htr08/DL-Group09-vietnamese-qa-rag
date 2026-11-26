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


def test_evaluate_hybrid_applies_title_rerank_only_when_requested():
    retriever = FakeRetriever(["d1", "d2"])
    titles = {"d1": "Giới thiệu chung", "d2": "Quy chế đào tạo tín chỉ đại học"}
    queries = [{"question": "đào tạo tín chỉ", "relevant_id": "d2"}]

    plain, _ = run_eval.evaluate_hybrid(retriever, retriever, queries, [1, 3], doc_titles=titles)
    reranked, _ = run_eval.evaluate_hybrid(retriever, retriever, queries, [1, 3], doc_titles=titles, do_rerank=True)

    assert plain["R@1"] == 0.0  # có titles nhưng không yêu cầu rerank
    assert reranked["R@1"] == 1.0


def test_k_larger_than_default_depth_is_not_truncated():
    ids = [f"dieu{i}_khoan1" for i in range(30)]
    queries = [{"question": "q", "relevant_id": "dieu24_khoan1"}]  # nằm ở hạng 25

    single, _ = run_eval.evaluate_retriever(FakeRetriever(ids), queries, [25], "fake")
    hybrid, _ = run_eval.evaluate_hybrid(FakeRetriever(ids), FakeRetriever(ids), queries, [25])

    assert single["R@25"] == 1.0
    assert hybrid["R@25"] == 1.0


def _run_main(monkeypatch, tmp_path, argv):
    """Chạy main() với retriever giả; kết quả JSON ghi vào tmp_path."""
    ids = ["a_1", "b_1", "c_1"]
    queries = [{"question": f"q{i}", "relevant_id": rid, "answerable": True} for i, rid in enumerate(ids)]
    monkeypatch.setenv("WANDB_MODE", "disabled")
    monkeypatch.setattr(sys, "argv", ["run_eval.py", *argv])
    (tmp_path / "docs").mkdir()
    monkeypatch.setattr(run_eval, "ROOT", tmp_path)
    monkeypatch.setattr(run_eval, "load_viquad_dev_queries", lambda: queries)
    monkeypatch.setattr(run_eval, "load_doc_titles", lambda corpus: {"a_1": "tiêu đề"})
    monkeypatch.setattr(run_eval, "load_bm25", lambda corpus: FakeRetriever(ids))
    monkeypatch.setattr(run_eval, "load_dense", lambda corpus: FakeRetriever(ids[::-1]))
    run_eval.main()
    return json.loads((tmp_path / "docs" / "results" / "retrieval_viquad_dev.json").read_text(encoding="utf-8"))


def test_main_records_ci_and_rerank_parameters_in_json(monkeypatch, tmp_path):
    result = _run_main(monkeypatch, tmp_path, ["--split", "viquad_dev", "--method", "all"])

    assert result["bootstrap_ci"]["n_resamples"] == run_eval.CI_N_RESAMPLES
    assert result["bootstrap_ci"]["seed"] == run_eval.CI_SEED
    assert result["rerank_beta"] == run_eval.DEFAULT_TITLE_BETA
    assert result["rrf_k"] in (15, 60)
    assert result["commit_hash"]
    rerank_ci = result["bootstrap_ci_rerank"]
    assert rerank_ci["comparison"].startswith("Hybrid + Rerank Title")
    assert rerank_ci["n_resamples"] == run_eval.CI_N_RESAMPLES and rerank_ci["seed"] == run_eval.CI_SEED


def test_paired_ci_info_rejects_different_number_of_questions():
    with pytest.raises(ValueError):
        run_eval.paired_ci_info("a vs b", [1.0, 0.0], [1.0])


def test_main_reports_when_ci_is_skipped(monkeypatch, tmp_path, capsys):
    result = _run_main(monkeypatch, tmp_path, ["--split", "viquad_dev", "--method", "hybrid"])

    assert result["bootstrap_ci"] is None
    assert result["bootstrap_ci_rerank"] is not None  # so với hybrid, không cần retriever đơn
    assert "[CI] Bỏ qua" in capsys.readouterr().out


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
    """Trên tập test, k của RRF phải truyền từ dev qua --rrf_k, không được tự chọn lại (§3.5c)."""
    monkeypatch.setattr(sys, "argv", ["run_eval.py", "--split", "domain_test", "--method", "hybrid"])
    monkeypatch.setattr(run_eval, "load_gold_queries", lambda split: [])
    monkeypatch.setattr(run_eval, "load_doc_titles", lambda corpus: {})
    monkeypatch.setattr(run_eval, "load_bm25", lambda corpus: object())
    monkeypatch.setattr(run_eval, "load_dense", lambda corpus: object())
    with pytest.raises(ValueError, match="--rrf_k"):
        run_eval.main()
