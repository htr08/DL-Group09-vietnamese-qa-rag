"""Sinh QA feature theo train/dev; ghi rõ trạng thái gold alignment.

Ví dụ: python scripts/preprocess_qa.py --splits train dev
Không chạy public_test trước lịch đánh giá chính thức.
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import py_vncorenlp
from transformers import AutoTokenizer

from src.preprocessing_qa import IGNORE_INDEX, preprocess_example


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path,
                        default=ROOT / "data" / "processed" / "viquad")
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "data" / "processed" / "qa_features")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--splits", nargs="+", default=["train", "dev"],
                        choices=["train", "dev", "public_test"])
    return parser.parse_args()


def segment_text(model, text):
    tokens = " ".join(model.word_segment(text)).split()
    if "𨮁" in text:
        tokens = ["𨮁" if token == "讁" else token for token in tokens]
    return tokens


def process_split(split, input_path, output_dir, model, tokenizer, limit=None):
    stats = Counter()
    skip_reasons = Counter()
    mode = "train" if split == "train" else "eval"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{split}.jsonl"

    with input_path.open("r", encoding="utf-8") as source, \
         output_path.open("w", encoding="utf-8") as output:
        for line in source:
            if limit is not None and stats["total_examples"] >= limit:
                break
            stats["total_examples"] += 1
            try:
                row = json.loads(line)
                question_tokens = segment_text(model, row["question"])
                context_tokens = segment_text(model, row["context"])
                if not question_tokens:
                    raise ValueError("Question tokens are empty")

                features = preprocess_example(
                    row, question_tokens, context_tokens, tokenizer, mode=mode
                )
                statuses = {f["label_status"] for f in features}
                if len(statuses) != 1:
                    raise AssertionError("Inconsistent label_status across windows")
                status = statuses.pop()
                positive_windows = sum(f["start_positions"] > 0 for f in features)
                if mode == "train" and not row["is_impossible"] and positive_windows == 0:
                    raise ValueError("Answerable example has no positive window")

                for feature in features:
                    if len(feature["input_ids"]) > 256:
                        raise ValueError("Feature exceeds MAX_LEN")
                    for key in ("start_positions", "end_positions"):
                        value = feature[key]
                        if value != IGNORE_INDEX and not 0 <= value < len(feature["input_ids"]):
                            raise ValueError(f"Invalid {key}: {value}")
                    output.write(json.dumps(feature, ensure_ascii=False) + "\n")

                stats["processed_examples"] += 1
                stats["features"] += len(features)
                stats["positive_windows"] += positive_windows
                stats["cls_windows"] += sum(f["start_positions"] == 0 for f in features)
                stats["ignored_windows"] += sum(
                    f["start_positions"] == IGNORE_INDEX for f in features
                )
                stats[f"label_status_{status}"] += 1
                if row["is_impossible"]:
                    stats["impossible_examples"] += 1
                else:
                    stats["answerable_examples"] += 1
            except Exception as error:
                stats["skipped_examples"] += 1
                skip_reasons[f"{type(error).__name__}: {error}"] += 1

    for key in (
        "processed_examples", "skipped_examples", "features", "positive_windows",
        "cls_windows", "ignored_windows", "answerable_examples", "impossible_examples",
        "label_status_aligned", "label_status_impossible", "label_status_unalignable",
        "label_status_not_covered",
    ):
        stats[key] += 0
    # Cấu hình tạo feature: lưu vào report của CÁC LẦN CHẠY MỚI.
    from src.preprocessing_qa import MAX_LEN, DOC_STRIDE
    report = {
        "split": split,
        "mode": mode,
        "input_file": str(input_path),
        "output_file": str(output_path),
        "preprocessing_config": {
            "max_length": MAX_LEN,
            "doc_stride": DOC_STRIDE,
        },
        "statistics": dict(stats),
        "skip_reasons": dict(skip_reasons),
    }
    report_path = output_dir / f"{split}_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n===== {split.upper()} (mode={mode}) =====")
    for key, value in stats.items():
        print(f"{key}: {value}")
    print("Skip reasons:", dict(skip_reasons))
    print("Features:", output_path)
    print("Report:", report_path)
    return stats, dict(skip_reasons)


def main():
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    model_dir = ROOT / "data" / "raw" / "vncorenlp"
    if not input_dir.is_dir():
        raise FileNotFoundError(input_dir)
    if not model_dir.is_dir():
        raise FileNotFoundError(model_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model = py_vncorenlp.VnCoreNLP(annotators=["wseg"], save_dir=str(model_dir))
    tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base", use_fast=False)
    all_stats = {}
    all_errors = {}
    for split in args.splits:
        input_path = input_dir / f"{split}.jsonl"
        if not input_path.is_file():
            raise FileNotFoundError(input_path)
        stats, errors = process_split(
            split, input_path, output_dir, model, tokenizer, args.limit
        )
        all_stats[split] = dict(stats)
        all_errors[split] = errors
    total_examples = sum(s["total_examples"] for s in all_stats.values())
    total_processed = sum(s["processed_examples"] for s in all_stats.values())
    total_skipped = sum(s["skipped_examples"] for s in all_stats.values())
    summary = {
        "splits": all_stats,
        "total_examples": total_examples,
        "total_processed": total_processed,
        "total_skipped": total_skipped,
        "all_examples_accounted_for": total_processed + total_skipped == total_examples,
        "errors_by_split": all_errors,
    }
    summary_path = output_dir / "pipeline_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Pipeline report:", summary_path)
    if not summary["all_examples_accounted_for"]:
        raise RuntimeError("Pipeline example accounting failed")


if __name__ == "__main__":
    main()
