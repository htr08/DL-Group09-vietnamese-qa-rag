"""Chế độ trích xuất: PhoBERT đọc các đoạn truy hồi và chọn một khoảng văn bản làm câu trả lời.

Cách chọn và ngưỡng từ chối theo cách của `scripts/run_b1_domain_dev.py` (nhánh
feat/extractive-qa-pipeline), để kết quả của hai đường chạy so sánh được:
- tiền xử lý `mode="predict"` (không dùng gold), mỗi đoạn truy hồi cho một hoặc nhiều cửa sổ;
- điểm "không có đáp án" của câu hỏi là giá trị nhỏ nhất trên mọi cửa sổ của mọi đoạn;
- khoảng văn bản tốt nhất là ứng viên có điểm start + end cao nhất trên mọi đoạn;
- trả lời nếu `score_diff = null_score - best_span_score <= null_threshold`, ngược lại từ chối.

Điểm logit giữa các đoạn khác nhau chưa được hiệu chỉnh; ngưỡng phải chọn trên dev của miền
ứng dụng, không dùng ngưỡng của ViQuAD cho miền Quy chế mà không kiểm tra.
"""
from __future__ import annotations

import json
import math
import time
import unicodedata
from pathlib import Path
from typing import Callable, Dict, List, Sequence

from src.generate import REFUSAL_TEXT
from src.preprocessing_qa import preprocess_example
from src.qa_nbest import decode_window, require_gold_independent_features

DEFAULT_NULL_THRESHOLD = 0.0
CHECKPOINT_FILES = ("config.json", "qa_head.pt", "qa_config.json")


def _refusal(seconds: float, windows: int = 0, threshold: float = DEFAULT_NULL_THRESHOLD) -> dict:
    return {
        "answer": REFUSAL_TEXT, "refusal": True, "source_id": None, "source_rank": None,
        "span": None, "score_diff": None, "null_threshold": threshold, "nbest": [],
        "windows": windows, "seconds": round(seconds, 3),
    }


def extract_answer(
    segment_fn: Callable[[str], List[str]],
    tokenizer,
    logits_fn: Callable[[List[dict]], Sequence],
    question: str,
    contexts: Sequence[dict],
    *,
    null_threshold: float = DEFAULT_NULL_THRESHOLD,
    n_best: int = 20,
    max_answer_length: int = 30,
    batch_size: int = 8,
    keep_nbest: int = 5,
) -> dict:
    """Chọn khoảng văn bản trả lời trong các đoạn `contexts` (mỗi đoạn dạng {id, text, ...}).

    `segment_fn(text)` trả danh sách từ đã tách (VnCoreNLP); `logits_fn(features)` trả mảng
    [len(features), L, 2] gồm logit start và end của từng cửa sổ. Hai hàm được truyền vào để
    kiểm thử không cần Java, PyTorch hay checkpoint.

    Kết quả: `answer` (khoảng văn bản đã gộp khoảng trắng và xuống dòng, hoặc câu từ chối cố định nếu
    không có ứng viên hay `score_diff` vượt ngưỡng; văn bản gốc của khoảng nằm ở `nbest[0]["text"]`), `refusal`, `source_id` và `source_rank` (đoạn chứa câu trả lời,
    rank tính từ 1), `span` (vị trí ký tự trong đoạn đó), `score_diff`, `nbest`, `windows`, `seconds`.
    Khi từ chối vì vượt ngưỡng, `source_id` và `span` vẫn là của ứng viên tốt nhất để người dùng
    tham khảo; `refusal=True` mới là cờ quyết định.
    """
    if n_best < 1 or max_answer_length < 1 or batch_size < 1:
        raise ValueError("n_best, max_answer_length và batch_size phải dương")
    start = time.perf_counter()
    question = unicodedata.normalize("NFC", question)  # như run_b1_domain_dev.py; câu NFD bị từ chối nhầm
    usable = [(rank, c) for rank, c in enumerate(contexts, start=1) if c["text"].strip()]
    if not usable or not question.strip():
        return _refusal(time.perf_counter() - start, threshold=null_threshold)

    question_tokens = segment_fn(question)
    candidates: List[dict] = []
    best_null = float("inf")
    windows = 0
    for rank, context in usable:
        text = context["text"]
        features = preprocess_example(
            {"id": "extractive", "question": question, "context": text},
            question_tokens, segment_fn(text), tokenizer, mode="predict",
        )
        require_gold_independent_features(features, f"{context['id']}")
        windows += len(features)
        for offset in range(0, len(features), batch_size):
            batch = features[offset:offset + batch_size]
            logits = logits_fn(batch)
            for i, feature in enumerate(batch):
                length = len(feature["input_ids"])
                found, null_score = decode_window(
                    feature["input_ids"],
                    [float(x) for x in logits[i][:length, 0]],
                    [float(x) for x in logits[i][:length, 1]],
                    tokenizer, n_best_size=n_best, max_answer_length=max_answer_length,
                    sep_token_id=tokenizer.sep_token_id,
                    word_spans=feature["word_spans"], subword_spans=feature["subword_spans"],
                    context=text,
                )
                best_null = min(best_null, float(null_score))
                for item in found:
                    candidates.append({
                        "text": item["text"], "score": float(item["score"]), "chunk_id": context["id"],
                        "rank": rank, "char_start": item["char_start"], "char_end": item["char_end"],
                    })

    candidates.sort(key=lambda c: c["score"], reverse=True)
    unique, seen = [], set()
    for c in candidates:
        key = (c["chunk_id"], c["char_start"], c["char_end"])
        if key not in seen:
            seen.add(key)
            unique.append(c)
    if not unique:
        return _refusal(time.perf_counter() - start, windows, null_threshold)

    best = unique[0]
    score_diff = best_null - best["score"]
    if not math.isfinite(score_diff):
        raise ValueError(f"score_diff không hữu hạn ({score_diff}); logits có thể chứa NaN hoặc inf")
    refusal = score_diff > null_threshold
    return {
        "answer": REFUSAL_TEXT if refusal else " ".join(best["text"].split()),
        "refusal": refusal,
        "source_id": best["chunk_id"], "source_rank": best["rank"],
        "span": [best["char_start"], best["char_end"]],
        "score_diff": score_diff, "null_threshold": null_threshold,
        "nbest": [{k: c[k] for k in ("text", "score", "chunk_id")} for c in unique[:keep_nbest]],
        "windows": windows, "seconds": round(time.perf_counter() - start, 3),
    }


def segment_text(vn, text: str) -> List[str]:
    """Tách từ bằng VnCoreNLP; giữ cách xử lý ký tự 𨮁 của `scripts/run_b1_domain_dev.py`."""
    tokens = " ".join(vn.word_segment(text)).split()
    if "𨮁" in text:
        tokens = ["𨮁" if t == "讁" else t for t in tokens]
    if not tokens:
        raise ValueError("VnCoreNLP không trả về từ nào")
    return tokens


def load_segmenter(vncorenlp_dir) -> Callable[[str], List[str]]:
    """Nạp VnCoreNLP (cần Java; JVM chỉ khởi động được một lần trong mỗi tiến trình)."""
    vncorenlp_dir = Path(vncorenlp_dir).resolve()
    if not (vncorenlp_dir / "VnCoreNLP-1.2.jar").is_file():
        raise FileNotFoundError(f"Không thấy VnCoreNLP-1.2.jar trong {vncorenlp_dir}")
    import py_vncorenlp

    vn = py_vncorenlp.VnCoreNLP(annotators=["wseg"], save_dir=str(vncorenlp_dir))
    return lambda text: segment_text(vn, text)


def load_qa_model(checkpoint, device: str = "auto") -> Dict:
    """Nạp checkpoint PhoBERT (thư mục `best_model` của `train_qa.py`).

    Checkpoint gồm config.json, qa_head.pt, qa_config.json cùng trọng số và tokenizer của backbone.
    Trả về dict {tokenizer, logits_fn, max_answer_length, device}.
    """
    checkpoint = Path(checkpoint)
    for name in CHECKPOINT_FILES:
        if not (checkpoint / name).is_file():
            raise FileNotFoundError(f"Checkpoint thiếu {name}: {checkpoint}")
    import torch
    from transformers import AutoModel, AutoTokenizer

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    config = json.loads((checkpoint / "qa_config.json").read_text(encoding="utf-8"))
    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint), use_fast=False, local_files_only=True)
    backbone = AutoModel.from_pretrained(str(checkpoint), local_files_only=True).to(device).eval()
    head = torch.nn.Linear(backbone.config.hidden_size, 2).to(device)
    head.load_state_dict(torch.load(checkpoint / "qa_head.pt", map_location="cpu", weights_only=True), strict=True)
    head.eval()

    def logits_fn(features: List[dict]):
        max_len = max(len(f["input_ids"]) for f in features)
        input_ids = torch.full((len(features), max_len), tokenizer.pad_token_id, dtype=torch.long)
        attention = torch.zeros((len(features), max_len), dtype=torch.long)
        for i, f in enumerate(features):
            ids = f["input_ids"]
            input_ids[i, :len(ids)] = torch.tensor(ids, dtype=torch.long)
            attention[i, :len(ids)] = 1
        with torch.inference_mode():
            hidden = backbone(input_ids=input_ids.to(device), attention_mask=attention.to(device)).last_hidden_state
            return head(hidden).float().cpu()

    return {
        "tokenizer": tokenizer, "logits_fn": logits_fn, "device": device,
        "max_answer_length": int(config.get("max_answer_length", 30)),
    }


def load_extractive_reader(checkpoint, vncorenlp_dir, device: str = "auto") -> Dict:
    """Nạp mô hình và bộ tách từ; trả về dict {segment_fn, tokenizer, logits_fn, max_answer_length, device}."""
    model = load_qa_model(checkpoint, device)  # kiểm tra checkpoint trước khi khởi động JVM
    return {"segment_fn": load_segmenter(vncorenlp_dir), **model}


def make_extract_fn(reader: Dict, null_threshold: float = DEFAULT_NULL_THRESHOLD, n_best: int = 20,
                    batch_size: int = 8) -> Callable[[str, Sequence[dict]], dict]:
    """Tạo `extract_fn(question, contexts)` cho `src.pipeline.answer_question_extractive`."""
    def extract_fn(question: str, contexts: Sequence[dict]) -> dict:
        return extract_answer(
            reader["segment_fn"], reader["tokenizer"], reader["logits_fn"], question, contexts,
            null_threshold=null_threshold, n_best=n_best, batch_size=batch_size,
            max_answer_length=reader["max_answer_length"],
        )
    return extract_fn
