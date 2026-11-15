import json
import unicodedata
from pathlib import Path
import py_vncorenlp

root = Path.cwd()
data_path = root / "data" / "processed" / "viquad" / "train.jsonl"
vncorenlp_dir = root / "data" / "raw" / "vncorenlp"

rows = [json.loads(line) for line in data_path.open(encoding="utf-8") if line.strip()]
contexts = list(dict.fromkeys(row["context"] for row in rows))

segmenter = py_vncorenlp.VnCoreNLP(
    save_dir=str(vncorenlp_dir),
    annotators=["wseg"]
)

def normalize(text):
    return "".join(
        c for c in unicodedata.normalize("NFD", text.lower())
        if unicodedata.category(c) != "Mn"
    )

for context_index, context in enumerate(contexts, start=1):
    tokens = [
        token
        for sentence in segmenter.word_segment(context)
        for token in sentence.split()
    ]

    normalized = normalize(context)
    positions = [i for i, char in enumerate(normalized) if not char.isspace()]
    searchable = "".join(char for char in normalized if not char.isspace())
    cursor = 0

    for token_index, token in enumerate(tokens):
        target = "".join(normalize(token.replace("_", " ")).split())
        search_cursor = sum(not char.isspace() for char in normalized[:cursor])
        position = searchable.find(target, search_cursor)

        if position < 0:
            print("Context index:", context_index)
            print("Token index:", token_index)
            print("Token:", repr(token))
            print("Normalized token:", repr(target))
            print("Cursor:", cursor)
            print("Search cursor:", search_cursor)
            print("Text around cursor:", repr(normalized[max(0, cursor - 100):cursor + 200]))
            print("Tokens before:", tokens[max(0, token_index - 8):token_index])
            print("Tokens after:", tokens[token_index + 1:token_index + 9])
            print("Token exists in raw context:", token.replace("_", " ") in context)
            raise SystemExit

        cursor = positions[position + len(target) - 1] + 1

    if context_index % 500 == 0:
        print(f"Checked: {context_index}/{len(contexts)}")

print("No alignment error found.")
