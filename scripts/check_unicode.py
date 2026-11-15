import json
from pathlib import Path

path = Path("data/processed/viquad/train.jsonl")
rows = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
contexts = list(dict.fromkeys(row["context"] for row in rows))

found = {}
for index, context in enumerate(contexts, start=1):
    chars = [(i, c, f"U+{ord(c):X}") for i, c in enumerate(context) if ord(c) > 0xFFFF]
    if chars:
        found[index] = chars

print("Contexts with supplementary characters:", len(found))
for index, chars in list(found.items())[:20]:
    print("Context:", index, "Characters:", chars[:10])
