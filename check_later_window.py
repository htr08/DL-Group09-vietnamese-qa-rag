import json
from pathlib import Path

import py_vncorenlp
from transformers import AutoTokenizer

from src.preprocessing_qa import preprocess_example

data_path = Path("data/processed/viquad/train.jsonl").resolve()
rows = [
    json.loads(line)
    for line in data_path.read_text(encoding="utf-8").splitlines()
]

tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
vncorenlp_path = str(Path("data/raw/vncorenlp").resolve())
segmenter = py_vncorenlp.VnCoreNLP(
    save_dir=vncorenlp_path,
    annotators=["wseg"],
)

for row in rows:
    if row["is_impossible"] or len(row["context"]) < 3000:
        continue

    question_tokens = [
        token
        for sentence in segmenter.word_segment(row["question"])
        for token in sentence.split()
    ]
    context_tokens = [
        token
        for sentence in segmenter.word_segment(row["context"])
        for token in sentence.split()
    ]

    features = preprocess_example(
        row, question_tokens, context_tokens, tokenizer
    )

    for index, feature in enumerate(features):
        start = feature["start_positions"]
        end = feature["end_positions"]

        if index > 0 and start > 0:
            print("Window:", index)
            print("Predicted tokens:", tokenizer.convert_ids_to_tokens(
                feature["input_ids"][start:end + 1]
            ))
            print("Decoded:", repr(tokenizer.decode(
                feature["input_ids"][start:end + 1]
            )))
            print("Ground truth:", repr(row["answers"]["text"][0]))
            raise SystemExit

print("No later-window example found.")