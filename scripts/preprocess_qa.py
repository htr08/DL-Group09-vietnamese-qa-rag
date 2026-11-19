import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import py_vncorenlp
from transformers import AutoTokenizer

from src.preprocessing_qa import preprocess_example


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input-dir",
        type=Path,
        default=ROOT / "data" / "processed" / "viquad"
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "data" / "processed" / "qa_features"
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None
    )

    return parser.parse_args()


def segment_text(model, text):
    tokens = " ".join(model.word_segment(text)).split()

    if "𨮁" in text:
        tokens = ["𨮁" if token == "讁" else token for token in tokens]

    return tokens



def process_split(
    split,
    input_path,
    output_dir,
    model,
    tokenizer,
    limit=None
):
    stats = Counter()
    skip_reasons = Counter()

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{split}.jsonl"

    with input_path.open("r", encoding="utf-8") as source, \
         output_path.open("w", encoding="utf-8") as output:

        for line_number, line in enumerate(source, start=1):
            if limit is not None and stats["total_examples"] >= limit:
                break

            stats["total_examples"] += 1

            try:
                row = json.loads(line)

                question_tokens = segment_text(
                    model,
                    row["question"]
                )

                context_tokens = segment_text(
                    model,
                    row["context"]
                )

                if not question_tokens:
                    raise ValueError("Question tokens are empty")

                features = preprocess_example(
                    row,
                    question_tokens,
                    context_tokens,
                    tokenizer
                )

                fallback_used = any(
                    feature.pop("_fallback_used", False)
                    for feature in features
                )

                positive_windows = sum(
                    feature["start_positions"] != 0
                    for feature in features
                )

                if not row["is_impossible"] and positive_windows == 0:
                    stats["answerable_without_positive_window"] += 1
                    raise ValueError(
                        "Answerable example has no positive window after fallback"
                    )

                for feature in features:
                    if len(feature["input_ids"]) > 256:
                        raise ValueError("Feature exceeds MAX_LEN")

                    if not (
                        0 <= feature["start_positions"] < len(feature["input_ids"])
                    ):
                        raise ValueError("Invalid start position")

                    if not (
                        0 <= feature["end_positions"] < len(feature["input_ids"])
                    ):
                        raise ValueError("Invalid end position")

                for feature in features:
                    output.write(
                        json.dumps(feature, ensure_ascii=False) + "\n"
                    )

                stats["processed_examples"] += 1
                stats["features"] += len(features)
                stats["positive_windows"] += positive_windows
                stats["cls_windows"] += len(features) - positive_windows

                if fallback_used:
                    stats["fallback_examples"] += 1

                if row["is_impossible"]:
                    stats["impossible_examples"] += 1
                else:
                    stats["answerable_examples"] += 1

            except Exception as error:
                stats["skipped_examples"] += 1
                reason = f"{type(error).__name__}: {error}"
                skip_reasons[reason] += 1

    for key in [
        "processed_examples",
        "skipped_examples",
        "features",
        "positive_windows",
        "cls_windows",
        "answerable_examples",
        "impossible_examples",
        "fallback_examples",
        "answerable_without_positive_window"
    ]:
        stats[key] += 0

    report = {
        "split": split,
        "input_file": str(input_path),
        "output_file": str(output_path),
        "statistics": dict(stats),
        "skip_reasons": dict(skip_reasons)
    }

    report_path = output_dir / f"{split}_report.json"

    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    print(f"\n===== {split.upper()} =====")

    for key, value in stats.items():
        print(f"{key}: {value}")

    print("\nSkip reasons:")

    if skip_reasons:
        for reason, count in skip_reasons.most_common():
            print(f"{count}: {reason}")
    else:
        print("None")

    print(f"\nFeatures: {output_path}")
    print(f"Report: {report_path}")

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

    model = py_vncorenlp.VnCoreNLP(
        annotators=["wseg"],
        save_dir=str(model_dir)
    )

    tokenizer = AutoTokenizer.from_pretrained(
        "vinai/phobert-base",
        use_fast=False
    )

    all_stats = {}
    all_errors = {}

    for split in ["train", "dev"]:
        input_path = input_dir / f"{split}.jsonl"

        if not input_path.is_file():
            raise FileNotFoundError(input_path)

        stats, errors = process_split(
            split,
            input_path,
            output_dir,
            model,
            tokenizer,
            args.limit
        )

        all_stats[split] = dict(stats)
        all_errors[split] = errors

    print("\n===== PIPELINE SUMMARY =====")

    for split, stats in all_stats.items():
        print(
            f"{split}: "
            f"{stats['processed_examples']}/"
            f"{stats['total_examples']} processed, "
            f"{stats['skipped_examples']} skipped, "
            f"{stats['fallback_examples']} fallback, "
            f"{stats['features']} features"
        )

        if all_errors[split]:
            print(f"{split} error details:")

            for error, count in all_errors[split].items():
                print(f"  {count}: {error}")

    total_examples = sum(
        stats["total_examples"]
        for stats in all_stats.values()
    )

    total_processed = sum(
        stats["processed_examples"]
        for stats in all_stats.values()
    )

    total_skipped = sum(
        stats["skipped_examples"]
        for stats in all_stats.values()
    )

    total_fallback = sum(
        stats["fallback_examples"]
        for stats in all_stats.values()
    )

    print(f"\nTotal examples: {total_examples}")
    print(f"Total processed: {total_processed}")
    print(f"Total skipped: {total_skipped}")
    print(f"Total fallback: {total_fallback}")

    summary = {
        "splits": all_stats,
        "total_examples": total_examples,
        "total_processed": total_processed,
        "total_skipped": total_skipped,
        "total_fallback": total_fallback,
        "all_examples_accounted_for": (
            total_processed + total_skipped == total_examples
        ),
        "errors_by_split": all_errors
    }

    summary_path = output_dir / "pipeline_summary.json"

    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    print(f"Pipeline report: {summary_path}")

    if not summary["all_examples_accounted_for"]:
        raise RuntimeError("Pipeline example accounting failed")


if __name__ == "__main__":
    main()
