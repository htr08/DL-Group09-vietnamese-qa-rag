"""Unit test cho app/app.py (demo hai chế độ): không cần GPU; phần Gradio bỏ qua nếu thiếu gradio."""
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("demo_app", Path(__file__).resolve().parents[1] / "app" / "app.py")
demo_app = importlib.util.module_from_spec(spec)
spec.loader.exec_module(demo_app)

CHUNKS = {"a": "Điều 1.\nNội dung đoạn a", "b": "Nội dung đoạn b"}


def test_format_generative_result_lists_numbered_sources():
    answer, sources = demo_app.format_result({"answer": "Trả lời [1].", "sources": ["a", "b"]}, CHUNKS)
    assert answer == "Trả lời [1]."
    assert "[1] `a`: Điều 1. Nội dung đoạn a" in sources and "[2] `b`" in sources


def test_format_extractive_result_shows_answer_chunk_and_refusal_state():
    result = {"answer": "Huế", "refusal": False, "source_id": "b", "score_diff": -3.0, "sources": ["a", "b"]}
    answer, sources = demo_app.format_result(result, CHUNKS)
    assert answer == "Huế" and "**b** (đoạn chứa câu trả lời" in sources and "-3.00" in sources
    refused = demo_app.format_result({**result, "refusal": True, "score_diff": 2.0}, CHUNKS)[1]
    assert "ứng viên tốt nhất bị từ chối" in refused


def test_format_handles_refusal_without_candidate_and_unknown_source():
    result = {"answer": "Không tìm thấy thông tin trong tài liệu.", "refusal": True, "source_id": None,
              "score_diff": None, "sources": ["zzz"]}
    answer, sources = demo_app.format_result(result, CHUNKS)
    assert answer.startswith("Không tìm thấy") and "[1] `zzz`" in sources
    assert demo_app.format_result({"answer": "", "sources": []}, CHUNKS) == ("(rỗng)", "(không có nguồn)")


def test_build_demo_constructs_with_both_modes():
    pytest.importorskip("gradio")
    fns = {demo_app.MODE_GENERATE: lambda q: {"answer": "g", "sources": []},
           demo_app.MODE_EXTRACT: lambda q: {"answer": "e", "sources": []}}
    assert demo_app.build_demo(fns, CHUNKS) is not None
