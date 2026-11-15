from pathlib import Path
import re

root = Path(__file__).resolve().parents[1]
folder = root / "data" / "processed" / "domain" / "selected_articles"

files = sorted(
    folder.glob("dieu_*.txt"),
    key=lambda path: int(re.search(r"\d+", path.stem).group())
)

for file in files:
    text = file.read_text(encoding="utf-8")

    print(f"\n===== {file.stem.upper()} =====")

    for line in text.splitlines():
        line = line.strip()

        if re.match(r"^\d+\.", line) or re.match(r"^[a-zđ]\)", line, re.IGNORECASE):
            print(line)