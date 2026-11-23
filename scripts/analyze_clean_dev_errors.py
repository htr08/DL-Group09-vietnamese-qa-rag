#!/usr/bin/env python3
"""Xuất lỗi EM thật từ full gold/predictions ViQuAD Clean Dev.

Chạy SAU KHI đã chạy scripts/eval_qa_nbest.py trên checkpoint cần phân tích.
Không chạy model, không tự thay đổi threshold, không dùng dữ liệu public_test.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path


def read_jsonl(path):
    rows = {}
    with path.open(encoding="utf-8-sig") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            obj = json.loads(line)
            key = str(obj["id"])
            if key in rows:
                raise ValueError(f"ID lặp trong {path}:{line_no}: {key}")
            rows[key] = obj
    if not rows:
        raise ValueError(f"File không có dữ liệu: {path}")
    return rows


def file_sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gold", type=Path, required=True, help="Full ViQuAD Clean Dev Gold JSONL")
    p.add_argument("--predictions", type=Path, required=True, help="dev_predictions.jsonl xuất trực tiếp từ evaluator")
    p.add_argument("--summary", type=Path, required=True, help="summary.json của CÙNG lượt eval")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--expected-count", type=int, default=2768)
    p.add_argument("--sample-size", type=int, default=66)
    p.add_argument("--seed", type=int, default=12)
    a = p.parse_args()

    for path in (a.gold, a.predictions):
        if "test" in path.name.lower():
            p.error(f"Chỉ sử dụng Dev, không được dùng Test: {path}")
        if not path.is_file():
            p.error(f"Không tìm thấy file: {path}")
    if not a.summary.is_file():
        p.error(f"Không tìm thấy file summary: {a.summary}")
    if a.sample_size < 1 or a.expected_count < 1:
        p.error("sample-size và expected-count phải > 0")

    # Sử dụng CHÍNH XÁC metric trong code dự án, tránh khác định nghĩa EM/F1.
    root = Path(__file__).resolve().parents[1]
    if (root / "src" / "qa_metrics.py").is_file():
        sys.path.insert(0, str(root))
    try:
        from src.qa_metrics import exact_match_score, f1_score
    except ImportError as e:
        p.error("Đặt file này tại scripts/analyze_clean_dev_errors.py trong repository để import src.qa_metrics: " + str(e))

    gold = read_jsonl(a.gold)
    preds = read_jsonl(a.predictions)
    if len(gold) != a.expected_count:
        raise ValueError(f"Dev Gold phải có {a.expected_count} ID, nhận {len(gold)}")
    if gold.keys() != preds.keys() and set(gold) != set(preds):
        missing = sorted(set(gold) - set(preds))[:5]
        extra = sorted(set(preds) - set(gold))[:5]
        raise ValueError(f"Gold/prediction ID không trùng: missing={missing}, extra={extra}")
    summary = json.loads(a.summary.read_text(encoding="utf-8-sig"))
    if "dev" not in str(summary.get("eval_split", "")).lower():
        raise ValueError("summary không xác nhận đây là Dev; dừng để tránh dùng Test")
    if int(summary.get("dev_questions", -1)) != len(gold):
        raise ValueError("summary và Gold khác số câu hỏi")

    groups = {"wrong_span": [], "false_negative": [], "false_positive": []}
    em_sum = f1_sum = 0.0
    answerable = impossible = 0
    for key, row in gold.items():
        record = preds[key]
        prediction = record["prediction"]
        if not isinstance(prediction, str):
            raise TypeError(f"Prediction không phải string: {key}")
        is_impossible = bool(row["is_impossible"])
        answers = row.get("answers", {})
        if is_impossible:
            refs = [""]
            impossible += 1
        else:
            refs = answers.get("text", [])
            answerable += 1
            if not refs or not all(isinstance(v, str) for v in refs):
                raise ValueError(f"Gold answerable thiếu refs: {key}")
        em = float(exact_match_score(prediction, refs))
        f1 = float(f1_score(prediction, refs))
        em_sum += em
        f1_sum += f1
        if em:
            continue
        category = ("false_positive" if is_impossible else
                    "false_negative" if not prediction.strip() else "wrong_span")
        entry = {
            "id": key,
            "error_type": category,
            "title": row.get("title", ""),
            "question": row.get("question", ""),
            "gold_answers": json.dumps(refs, ensure_ascii=False),
            "prediction": prediction,
            "f1": round(f1, 6),
            "context": row.get("context", ""),
            "manual_label": "",
            "manual_notes": "",
            "review_status": "PENDING",
        }
        groups[category].append(entry)

    em_pct = 100 * em_sum / len(gold)
    f1_pct = 100 * f1_sum / len(gold)
    reported = summary.get("tuned_on_dev", {})
    if "em" in reported and "f1" in reported:
        if abs(em_sum / len(gold) - float(reported["em"])) > 0.0001 or abs(f1_sum / len(gold) - float(reported["f1"])) > 0.0001:
            raise ValueError(f"EM/F1 tính lại ({em_pct:.2f}, {f1_pct:.2f}) khác summary evaluator: "
                             f"({100*float(reported['em']):.2f}, {100*float(reported['f1']):.2f}); kiểm tra Gold/Pred có đúng cùng run không")

    if a.expected_count == 2768 and (answerable, impossible) != (1886, 882):
        raise ValueError(f"Clean Dev sai phân bố answerable/impossible: {answerable}/{impossible}, expected 1886/882")

    total_errors = sum(map(len, groups.values()))
    if total_errors < a.sample_size:
        raise ValueError(f"Chỉ có {total_errors} lỗi, không đủ {a.sample_size} mẫu")
    rng = random.Random(a.seed)
    for rows in groups.values():
        rng.shuffle(rows)
    names = list(groups)
    quota, remainder = divmod(a.sample_size, len(names))
    chosen = []
    leftovers = []
    for i, name in enumerate(names):
        size = min(len(groups[name]), quota + (1 if i < remainder else 0))
        chosen.extend(groups[name][:size])
        leftovers.extend(groups[name][size:])
    rng.shuffle(leftovers)
    chosen.extend(leftovers[:a.sample_size - len(chosen)])
    assert len(chosen) == a.sample_size
    assert len({r["id"] for r in chosen}) == a.sample_size

    a.out_dir.mkdir(parents=True, exist_ok=True)
    columns = ["id", "error_type", "title", "question", "gold_answers", "prediction", "f1", "context", "manual_label", "manual_notes", "review_status"]
    def write_csv(name, rows):
        path = a.out_dir / name
        with path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
        return str(path)

    all_errors = [r for name in names for r in groups[name]]
    write_csv("all_dev_errors.csv", all_errors)
    write_csv(f"{a.sample_size}_dev_errors_to_review.csv", chosen)
    checkpoint = Path(str(summary.get("checkpoint", "")))
    checkpoint_head = checkpoint / "qa_head.pt"
    backbone_weights = next((f for f in (checkpoint / "model.safetensors", checkpoint / "pytorch_model.bin") if f.is_file()), None)
    report = {
        "sha256_qa_head": file_sha(checkpoint_head) if checkpoint_head.is_file() else None,
        "sha256_backbone": file_sha(backbone_weights) if backbone_weights else None,
        "source": "Full Gold + full dev_predictions from one evaluator run; not archival error CSV",
        "checkpoint_reported_by_evaluator": summary.get("checkpoint"),
        "threshold_reported_by_evaluator": reported.get("null_threshold"),
        "num_dev_questions": len(gold),
        "num_answerable": answerable,
        "num_impossible": impossible,
        "em_percent": round(em_pct, 5),
        "f1_percent": round(f1_pct, 5),
        "errors_by_em": {name: len(groups[name]) for name in names},
        "total_em_errors": total_errors,
        "selected_count": len(chosen),
        "selected_types": dict(Counter(r["error_type"] for r in chosen)),
        "seed": a.seed,
        "sha256_gold": file_sha(a.gold),
        "sha256_predictions": file_sha(a.predictions),
        "sha256_summary": file_sha(a.summary),
    }
    (a.out_dir / "audit_summary.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"SAVED: {a.out_dir / 'all_dev_errors.csv'}")
    print(f"SAVED: {a.out_dir / f'{a.sample_size}_dev_errors_to_review.csv'}")


if __name__ == "__main__":
    main()
