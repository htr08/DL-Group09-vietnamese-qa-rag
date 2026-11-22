"""Đánh giá QA checkpoint bằng offset gốc, không train và không gọi VnCoreNLP.

Chạy Dev và tuning threshold: --split dev, --features <clean_dev.jsonl>.
Đường chạy public_test chỉ CHUẨN BỊ cho Tuần 5, tuyệt đối không chạy sớm.
Khi chạy public_test bắt buộc --threshold-file từ Dev; không tuning trên Test.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.qa_nbest import (
    decode_window, scores_at_threshold, best_dev_threshold,
    require_gold_independent_features,
)
from src.qa_metrics import exact_match_score, f1_score


def load_jsonl(path):
    rows = []
    with path.open(encoding="utf-8-sig") as f:
        for lineno, line in enumerate(f, 1):
            if not line.strip():
                continue
            obj = json.loads(line)
            if "id" not in obj:
                raise ValueError(f"Missing id: {path}:{lineno}")
            rows.append(obj)
    if not rows:
        raise ValueError(f"Empty JSONL: {path}")
    return rows


def model_batch(model, head, feats, pad_id, device):
    import torch
    max_len = max(len(r["input_ids"]) for r in feats)
    inp = torch.full((len(feats), max_len), pad_id, dtype=torch.long)
    attn = torch.zeros((len(feats), max_len), dtype=torch.long)
    for i, feat in enumerate(feats):
        ids = feat["input_ids"]
        inp[i, :len(ids)] = torch.tensor(ids, dtype=torch.long)
        attn[i, :len(ids)] = 1
    inp, attn = inp.to(device), attn.to(device)
    with torch.inference_mode():
        hidden = model(input_ids=inp, attention_mask=attn).last_hidden_state
        logits = head(hidden).float().cpu()
    return logits


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split", choices=["dev", "public_test"], default="dev")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--features", type=Path, required=True)
    p.add_argument("--gold", type=Path, required=True)
    p.add_argument("--dev-titles", type=Path, default=ROOT / "data/splits/viquad_dev_titles.txt")
    p.add_argument("--threshold-file", type=Path, default=None,
                   help="Bắt buộc với public_test: summary.json đã tune trên Dev.")
    p.add_argument("--output-dir", type=Path, default=ROOT / "outputs/qa_nbest_clean_dev")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--n-best", type=int, default=20)
    p.add_argument("--max-answer-length", type=int, default=None)
    p.add_argument("--limit-features", type=int, default=None)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    a = p.parse_args()
    if a.batch_size < 1 or a.n_best < 1:
        p.error("batch-size / n-best must be positive")
    if a.limit_features is not None and a.limit_features < 1:
        p.error("limit-features must be positive")
    if a.split == "public_test" and a.threshold_file is None:
        p.error("public_test bắt buộc --threshold-file từ Dev, không tune trên Test")
    if a.split == "public_test" and a.limit_features is not None:
        p.error("Không chạy partial public_test")

    a.checkpoint = a.checkpoint.resolve()
    a.features = a.features.resolve()
    a.gold = a.gold.resolve()
    a.output_dir = a.output_dir.resolve()
    required = [a.features, a.gold, a.checkpoint / "config.json",
                a.checkpoint / "qa_head.pt", a.checkpoint / "qa_config.json"]
    if a.split == "dev":
        required.append(a.dev_titles)
    if a.threshold_file is not None:
        required.append(a.threshold_file)
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    a.output_dir.mkdir(parents=True, exist_ok=True)
    import torch
    from transformers import AutoModel, AutoTokenizer
    device = torch.device("cuda" if a.device == "auto" and torch.cuda.is_available()
                          else "cpu" if a.device == "auto" else a.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    cfg = json.loads((a.checkpoint / "qa_config.json").read_text(encoding="utf-8"))
    max_answer_length = a.max_answer_length or int(cfg.get("max_answer_length", 30))
    features = load_jsonl(a.features)
    gold = load_jsonl(a.gold)
    gold_by_id = {str(row["id"]): row for row in gold}
    if len(gold_by_id) != len(gold):
        raise ValueError("Duplicate gold IDs")
    if a.split == "dev":
        expected_titles = {x.strip() for x in a.dev_titles.read_text(encoding="utf-8-sig").splitlines() if x.strip()}
        actual_titles = {str(r.get("title", "")).strip() for r in gold}
        if len(expected_titles) != 13 or actual_titles != expected_titles:
            raise ValueError("Dev title split does not match saved 13-title split")

    # Feature dùng để chấm phải tạo bằng mode=eval/predict, tuyệt đối không dùng train.
    require_gold_independent_features(features, str(a.features))
    for feat in features:
        if "word_spans" not in feat or "subword_spans" not in feat:
            raise ValueError("Feature thiếu offset. Hãy preprocess lại Dev sạch; không dùng dữ liệu cũ.")
        if feat.get("label_status") is None or feat["label_status"] not in {
            "aligned", "unalignable", "not_covered", "impossible", "unlabeled"
        }:
            raise ValueError("Feature thiếu label_status từ preprocessing mới")
    if a.limit_features is not None:
        features = features[:a.limit_features]
    elif {str(r["id"]) for r in features} != set(gold_by_id):
        raise ValueError("Feature IDs != gold IDs")

    tokenizer = AutoTokenizer.from_pretrained(str(a.checkpoint), use_fast=False)
    backbone = AutoModel.from_pretrained(str(a.checkpoint)).to(device).eval()
    head = torch.nn.Linear(backbone.config.hidden_size, 2).to(device)
    try:
        state = torch.load(a.checkpoint / "qa_head.pt", map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(a.checkpoint / "qa_head.pt", map_location="cpu")
    head.load_state_dict(state, strict=True)
    head.eval()
    predictions = defaultdict(lambda: {"null_score": float("inf"), "candidates": []})
    for offset in range(0, len(features), a.batch_size):
        batch = features[offset:offset + a.batch_size]
        logits = model_batch(backbone, head, batch, tokenizer.pad_token_id, device)
        for i, feat in enumerate(batch):
            question_id = str(feat["id"])
            row = gold_by_id.get(question_id)
            if row is None:
                raise ValueError(f"Unknown feature ID: {question_id}")
            length = len(feat["input_ids"])
            candidates, null = decode_window(
                feat["input_ids"], logits[i, :length, 0].tolist(),
                logits[i, :length, 1].tolist(), tokenizer,
                n_best_size=a.n_best, max_answer_length=max_answer_length,
                sep_token_id=tokenizer.sep_token_id,
                word_spans=feat["word_spans"], subword_spans=feat["subword_spans"],
                context=row["context"],
            )
            rec = predictions[question_id]
            rec["null_score"] = min(rec["null_score"], null)
            rec["candidates"].extend(candidates)
        if offset + a.batch_size >= len(features) or (offset // a.batch_size + 1) % 50 == 0:
            print(f"Features: {min(offset + a.batch_size, len(features))}/{len(features)}", flush=True)

    records = []
    scored = []
    for question_id, rec in predictions.items():
        sorted_candidates = sorted(rec["candidates"], key=lambda c: c["score"], reverse=True)
        unique, seen = [], set()
        for c in sorted_candidates:
            key = (c.get("char_start"), c.get("char_end"))
            if key in seen:
                continue
            seen.add(key)
            unique.append(c)
            if len(unique) >= a.n_best:
                break
        best = unique[0] if unique else None
        score_diff = rec["null_score"] - best["score"] if best else 1e12
        text = best["text"] if best else ""
        out = {
            "id": question_id, "best_answer": text,
            "null_score": rec["null_score"],
            "best_span_score": best["score"] if best else None,
            "score_diff": score_diff,
            "nbest": [{"text": c["text"], "score": c["score"],
                       "char_start": c.get("char_start"), "char_end": c.get("char_end"),
                       "start_token_in_window": c["start_token"],
                       "end_token_in_window": c["end_token"]} for c in unique],
        }
        records.append(out)
        if a.limit_features is None:
            row = gold_by_id[question_id]
            if row["is_impossible"]:
                refs, category = [""], "impossible"
            else:
                refs, category = row["answers"]["text"], "answerable"
            scored.append({
                "id": question_id, "category": category, "score_diff": score_diff,
                "answer_em": float(exact_match_score(text, refs)),
                "answer_f1": float(f1_score(text, refs)),
                "null_em": float(exact_match_score("", refs)),
                "null_f1": float(f1_score("", refs)),
            })
    pred_path = a.output_dir / f"nbest_{a.split}.jsonl"
    with pred_path.open("w", encoding="utf-8") as f:
        for item in records:
            f.write(json.dumps(item, ensure_ascii=False, allow_nan=False) + "\n")
    print("Saved:", pred_path)
    if a.limit_features is not None:
        print("SMOKE ONLY: không tính metric, không tuning threshold")
        return
    if a.threshold_file is not None:
        source = json.loads(a.threshold_file.read_text(encoding="utf-8"))
        if source.get("eval_split", "").lower().find("dev") < 0:
            raise ValueError("Threshold phải lấy từ summary Dev")
        threshold = float(source["tuned_on_dev"]["null_threshold"])
        tuned_em, tuned_f1 = scores_at_threshold(scored, threshold)
    elif a.split == "dev":
        threshold, tuned_em, tuned_f1 = best_dev_threshold(scored)
    else:
        raise RuntimeError("Never tune on Test")
    base_em, base_f1 = scores_at_threshold(scored, 0.0)
    summary = {
        "eval_split": "ViQuAD title-held-out Dev" if a.split == "dev" else "ViQuAD public Test",
        "dev_questions" if a.split == "dev" else "test_questions": len(gold),
        "features": len(features), "checkpoint": str(a.checkpoint),
        "n_best_size": a.n_best, "max_answer_length": max_answer_length,
        "baseline_threshold_0": {"em": base_em, "f1": base_f1},
        "tuned_on_dev": {"null_threshold": threshold, "em": tuned_em, "f1": tuned_f1},
        "per_category": {
            cat: {
                "count": sum(r["category"] == cat for r in scored),
                "em": scores_at_threshold([r for r in scored if r["category"] == cat], threshold)[0],
                "f1": scores_at_threshold([r for r in scored if r["category"] == cat], threshold)[1],
            } for cat in ("answerable", "impossible") if any(r["category"] == cat for r in scored)
        },
        "notes": ["Features tạo bằng mode=eval (không gold-driven token/window)",
                  "Decoded by exact char offsets and word boundaries",
                  "Không tune threshold trên public Test"],
    }
    (a.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    with (a.output_dir / f"{a.split}_predictions.jsonl").open("w", encoding="utf-8") as f:
        for rec in records:
            ans = "" if rec["score_diff"] > threshold else rec["best_answer"]
            f.write(json.dumps({"id": rec["id"], "prediction": ans}, ensure_ascii=False) + "\n")
    print(f"{a.split}: EM={100*tuned_em:.2f}% F1={100*tuned_f1:.2f}% threshold={threshold:.4f}")
    print("Summary:", a.output_dir / "summary.json")


if __name__ == "__main__":
    main()
