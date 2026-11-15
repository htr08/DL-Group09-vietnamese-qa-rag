from pathlib import Path
import re
import sys
import unicodedata

sys.stdout.reconfigure(encoding="utf-8")

root = Path(__file__).resolve().parents[1]
source = root / "data" / "processed" / "domain" / "dhqghn_quy_che_dao_tao_dai_hoc_2014.txt"
output_dir = root / "data" / "processed" / "domain" / "selected_articles"
output_dir.mkdir(parents=True, exist_ok=True)

text = unicodedata.normalize("NFC", source.read_text(encoding="utf-8"))
matches = list(re.finditer(r"(?m)^Điều\s+(\d+)\.", text))

selected = {4, 5, 15, 20, 21, 24, 32, 33, 34, 38, 40, 43}
seen = set()

for i, match in enumerate(matches):
    article_number = int(match.group(1))

    if article_number not in selected or article_number in seen:
        continue

    seen.add(article_number)

    start = match.start()
    end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
    article_text = text[start:end].strip()

    output_path = output_dir / f"dieu_{article_number}.txt"
    output_path.write_text(article_text, encoding="utf-8")

    print(f"Điều {article_number}: {len(article_text)} ký tự")

print(f"\nĐã trích xuất {len(seen)}/{len(selected)} điều.")