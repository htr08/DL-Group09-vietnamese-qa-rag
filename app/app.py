"""Demo chatbot Gradio với hai chế độ: sinh và trích xuất (bản khung, Tuần 3 T4).

Chạy trên máy có GPU (Kaggle T4), từ thư mục gốc của repo:
    python app/app.py --checkpoint <thư_mục best_model> --vncorenlp data/raw/vncorenlp [--share]
Cần trước đó: `python scripts/build_index.py --corpus domain` (index BM25 và dense) và VnCoreNLP
(`py_vncorenlp.download_model`, cần Java). `--no-llm` bỏ Qwen, chỉ giữ chế độ trích xuất.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Callable, Dict, Mapping

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

MODE_GENERATE = "Sinh bằng LLM (Qwen2.5-7B)"
MODE_EXTRACT = "Trích xuất (PhoBERT)"
SNIPPET_CHARS = 200


def format_result(result: dict, chunk_texts: Mapping[str, str]) -> tuple[str, str]:
    """Đổi kết quả của `answer_question` hoặc `answer_question_extractive` thành (câu trả lời, nguồn)."""
    answer = result["answer"] or "(rỗng)"
    lines = []
    if result.get("source_id"):  # chế độ trích xuất: đoạn chứa câu trả lời
        verdict = "ứng viên tốt nhất bị từ chối" if result["refusal"] else "đoạn chứa câu trả lời"
        lines.append(f"**{result['source_id']}** ({verdict}, điểm chênh `{result['score_diff']:.2f}`)")
    for rank, source_id in enumerate(result.get("sources", []), start=1):
        snippet = " ".join(chunk_texts.get(source_id, "").split())[:SNIPPET_CHARS]
        lines.append(f"[{rank}] `{source_id}`: {snippet}")
    return answer, "\n\n".join(lines) or "(không có nguồn)"


def build_demo(answer_fns: Dict[str, Callable[[str], dict]], chunk_texts: Mapping[str, str]):
    """Giao diện Gradio; `answer_fns` ánh xạ tên chế độ sang hàm (câu hỏi) -> kết quả."""
    import gradio as gr

    def respond(question: str, mode: str):
        if not question.strip():
            return "Hãy nhập câu hỏi.", ""
        if mode not in answer_fns:
            return f"Chế độ không khả dụng: {mode}", ""
        return format_result(answer_fns[mode](question), chunk_texts)

    with gr.Blocks(title="Hỏi đáp Quy chế đào tạo ĐHQGHN") as demo:
        gr.Markdown("# Hỏi đáp Quy chế đào tạo đại học ĐHQGHN\nBản khung: hai chế độ trả lời trên cùng kết quả truy hồi.")
        mode = gr.Radio(list(answer_fns), value=next(iter(answer_fns)), label="Chế độ")
        question = gr.Textbox(label="Câu hỏi", lines=2)
        button = gr.Button("Trả lời")
        answer = gr.Textbox(label="Câu trả lời", lines=4)
        sources = gr.Markdown()
        button.click(respond, [question, mode], [answer, sources])
        question.submit(respond, [question, mode], [answer, sources])
    return demo


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True, help="thư mục best_model của PhoBERT")
    parser.add_argument("--vncorenlp", type=Path, default=ROOT / "data/raw/vncorenlp")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--null-threshold", type=float, default=0.0, help="ngưỡng từ chối của chế độ trích xuất")
    parser.add_argument("--no-llm", action="store_true", help="không nạp Qwen")
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    from src.extractive import CHECKPOINT_FILES, load_extractive_reader, make_extract_fn
    from src.pipeline import answer_question, answer_question_extractive

    for name in CHECKPOINT_FILES:  # báo lỗi trước khi tốn thời gian nạp BGE-M3
        if not (args.checkpoint / name).is_file():
            parser.error(f"checkpoint thiếu {name}: {args.checkpoint}")
    from scripts.run_eval import load_bm25, load_dense

    bm25, dense = load_bm25("domain"), load_dense("domain")
    meta_path = ROOT / "data" / "index" / "domain" / "chunks_meta.json"
    chunk_texts = {item["id"]: item["text"] for item in json.loads(meta_path.read_text(encoding="utf-8"))}
    retrieval = dict(top_k=args.top_k, rrf_k=args.rrf_k)

    reader = load_extractive_reader(args.checkpoint, args.vncorenlp)
    extract_fn = make_extract_fn(reader, null_threshold=args.null_threshold)
    answer_fns = {
        MODE_EXTRACT: lambda q: answer_question_extractive(extract_fn, bm25, dense, chunk_texts, q, **retrieval)
    }
    if not args.no_llm:
        from src.generate import load_model, make_generate_fn

        model, tokenizer = load_model()
        generate_fn = make_generate_fn(model, tokenizer)
        answer_fns = {
            MODE_GENERATE: lambda q: answer_question(generate_fn, bm25, dense, chunk_texts, q, **retrieval),
            **answer_fns,
        }
    build_demo(answer_fns, chunk_texts).launch(share=args.share)


if __name__ == "__main__":
    main()
