import json
from pathlib import Path

import py_vncorenlp
from transformers import AutoTokenizer

from src.preprocessing_qa import (
    MAX_LEN,
    MAX_QUESTION_LEN,
    align_tokens_to_text,
    tokenize_words,
    create_sliding_windows,
    map_answer_to_subwords,
    label_sliding_windows,
)

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed" / "viquad"
OUTPUT_DIR = ROOT / "data" / "processed" / "qa_features"
VNCORENLP_DIR = ROOT / "data" / "raw" / "vncorenlp"

MODEL_NAME = "vinai/phobert-base"


def read_jsonl(path):
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def write_jsonl(path, rows):
    temp_path = path.with_suffix(".tmp")
    with temp_path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
    temp_path.replace(path)


def segment_text(segmenter, text):
    tokens = [
        token
        for sentence in segmenter.word_segment(text)
        for token in sentence.split()
    ]

    replacements = {}

    for char in text:
        codepoint = ord(char)

        if codepoint > 0xFFFF:
            corrupted_char = chr(codepoint & 0xFFFF)
            replacements[corrupted_char] = char

    return [
        replacements.get(token, token)
        for token in tokens
    ]


def build_context_cache(contexts, segmenter, tokenizer):
    cache = {}

    for index, context in enumerate(contexts, start=1):
        tokens = segment_text(segmenter, context)
        word_spans = align_tokens_to_text(context, tokens)
        context_ids, subword_spans = tokenize_words(tokenizer, tokens)

        if not context_ids:
            raise ValueError(f"Empty context encoding at context {index}")

        cache[context] = {
            "word_spans": word_spans,
            "context_ids": context_ids,
            "subword_spans": subword_spans,
        }

        if index % 500 == 0:
            print(f"Cached contexts: {index}/{len(contexts)}")

    return cache


def preprocess_row(row, segmenter, tokenizer, context_cache):
    question_tokens = segment_text(segmenter, row["question"])
    question_ids, _ = tokenize_words(tokenizer, question_tokens)
    question_ids = question_ids[:MAX_QUESTION_LEN]

    context_data = context_cache[row["context"]]
    context_ids = context_data["context_ids"]

    windows = create_sliding_windows(question_ids, context_ids)
    is_impossible = row["is_impossible"]

    if is_impossible:
        features = label_sliding_windows(
            windows, 0, 0, len(question_ids), is_impossible=True
        )
    else:
        answers = row["answers"]
        answer_texts = answers["text"]
        answer_starts = answers["answer_start"]

        if not answer_texts or len(answer_texts) != len(answer_starts):
            raise ValueError(f"Invalid answers: {row['id']}")

        answer_text = answer_texts[0]
        answer_start = answer_starts[0]

        if not answer_text or answer_start < 0:
            raise ValueError(f"Invalid answer span: {row['id']}")

        if row["context"][answer_start:answer_start + len(answer_text)] != answer_text:
            raise ValueError(f"Answer text mismatch: {row['id']}")

        start_subword, end_subword = map_answer_to_subwords(
            answer_start,
            answer_text,
            context_data["word_spans"],
            context_data["subword_spans"],
        )

        context_capacity = MAX_LEN - len(question_ids) - 4

        if end_subword - start_subword + 1 > context_capacity:
            raise ValueError(
                f"Answer exceeds window capacity: {row['id']}"
            )

        answer_in_window = any(
            start_subword >= window["context_start"]
            and end_subword < window["context_end"]
            for window in windows
        )

        if not answer_in_window:
            target_start = min(
                start_subword,
                max(0, len(context_ids) - context_capacity)
            )
            target_end = min(
                target_start + context_capacity,
                len(context_ids)
            )

            windows.append({
                "input_ids": (
                    [0] + question_ids + [2, 2]
                    + context_ids[target_start:target_end] + [2]
                ),
                "context_start": target_start,
                "context_end": target_end
            })

        features = label_sliding_windows(
            windows,
            start_subword,
            end_subword,
            len(question_ids),
        )

        if not any(
            feature["start_positions"] > 0
            and feature["end_positions"] > 0
            for feature in features
        ):
            raise ValueError(f"Answer not found in any window: {row['id']}")

    for feature in features:
        feature["id"] = row["id"]
        feature["is_impossible"] = is_impossible

    return features


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    splits = {
        "train": read_jsonl(DATA_DIR / "train.jsonl"),
        "dev": read_jsonl(DATA_DIR / "dev.jsonl"),
    }

    all_contexts = list(dict.fromkeys(
        row["context"]
        for rows in splits.values()
        for row in rows
    ))

    print(f"Train rows: {len(splits['train'])}")
    print(f"Dev rows: {len(splits['dev'])}")
    print(f"Unique contexts: {len(all_contexts)}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    segmenter = py_vncorenlp.VnCoreNLP(
        save_dir=str(VNCORENLP_DIR),
        annotators=["wseg"],
    )

    context_cache = build_context_cache(
        all_contexts,
        segmenter,
        tokenizer,
    )

    for split, rows in splits.items():
        all_features = []

        for index, row in enumerate(rows, start=1):
            features = preprocess_row(
                row,
                segmenter,
                tokenizer,
                context_cache,
            )
            all_features.extend(features)

            if index % 1000 == 0:
                print(f"{split}: processed {index}/{len(rows)} rows")

        output_path = OUTPUT_DIR / f"{split}.jsonl"
        write_jsonl(output_path, all_features)

        positive = sum(
            feature["start_positions"] > 0
            for feature in all_features
        )

        print(f"{split} rows: {len(rows)}")
        print(f"{split} features: {len(all_features)}")
        print(f"{split} positive features: {positive}")
        print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
