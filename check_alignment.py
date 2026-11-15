import json
from pathlib import Path

path = Path("data/processed/viquad/train.jsonl")

seen = set()

with path.open(encoding="utf-8") as f:
    for line in f:
        row = json.loads(line)
        context = row["context"]

        if context in seen:
            continue

        seen.add(context)

        if "]" in context:
            print("ID:", row["id"])
            print("CONTEXT:", repr(context))
            print("BRACKET POSITIONS:", [
                i for i, char in enumerate(context) if char == "]"
            ])
            print()