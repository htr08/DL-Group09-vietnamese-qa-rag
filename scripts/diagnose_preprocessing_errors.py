
import json
import sys
from pathlib import Path
from collections import Counter, defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import py_vncorenlp
from transformers import AutoTokenizer

from src.preprocessing_qa import preprocess_example

root = Path(__file__).resolve().parents[1]

model = py_vncorenlp.VnCoreNLP(
    annotators=["wseg"],
    save_dir=str(root / "data/raw/vncorenlp")
)

tokenizer = AutoTokenizer.from_pretrained(
    "vinai/phobert-base",
    use_fast=False
)


def segment(text):
    tokens = " ".join(model.word_segment(text)).split()

    if "𨮁" in text:
        tokens = [
            "𨮁" if token == "讁" else token
            for token in tokens
        ]

    return tokens

for split in ["train", "dev"]:
    input_path = root / f"data/processed/viquad/{split}.jsonl"
    output_path = (
        root / f"data/processed/qa_features_candidate_v2/{split}_errors.jsonl"
    )

    errors = []
    counts = Counter()

    with open(input_path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            context_tokens = None

            try:
                question_tokens = segment(row["question"])
                context_tokens = segment(row["context"])

                features = preprocess_example(
                    row,
                    question_tokens,
                    context_tokens,
                    tokenizer
                )

                positive_windows = sum(
                    feature["start_positions"] != 0
                    for feature in features
                )

                if not row["is_impossible"] and positive_windows == 0:
                    raise ValueError(
                        "Answerable example has no positive window"
                    )

            except Exception as exc:
                reason = f"{type(exc).__name__}: {exc}"
                counts[reason] += 1

                answer_text = row.get("answers", {}).get("text", [])
                answer_start = row.get("answers", {}).get("answer_start", [])

                errors.append({
                    "id": row.get("id"),
                    "reason": reason,
                    "question": row.get("question"),
                    "answer": answer_text[0] if answer_text else None,
                    "answer_start": answer_start[0] if answer_start else None,
                    "context": row.get("context"),
                    "context_tokens": context_tokens
                })

    with open(output_path, "w", encoding="utf-8") as f:
        for item in errors:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"\n===== {split.upper()} =====")
    print("Total errors:", len(errors))

    for reason, count in counts.most_common():
        print(f"{count}: {reason}")

    groups = defaultdict(list)

    for item in errors:
        groups[item["reason"]].append(item)

    print("\n===== ERROR EXAMPLES =====")

    for reason, items in groups.items():
        print(f"\n--- {reason} ({len(items)} samples) ---")

        for item in items[:3]:
            print("ID:", item["id"])
            print("Question:", item["question"])
            print("Answer:", repr(item["answer"]))
            print("Answer start:", item["answer_start"])

            if item["answer"] is not None and item["answer_start"] is not None:
                start = item["answer_start"]
                end = start + len(item["answer"])

                print(
                    "Context around answer:",
                    repr(item["context"][max(0, start - 50):end + 50])
                )

            print("Tokens:", item["context_tokens"])
            print()

    print("Saved:", output_path)
