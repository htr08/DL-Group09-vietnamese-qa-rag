
from pathlib import Path
import re
import sys

sys.stdout.reconfigure(encoding="utf-8")

folder = Path(__file__).resolve().parents[1] / "data" / "processed" / "domain"

regulation = folder / "dhqghn_quy_che_dao_tao_dai_hoc_2014.txt"
text = regulation.read_text(encoding="utf-8")

print("===== CẤU TRÚC QUY CHẾ =====")

for line in text.splitlines():
    line = line.strip()
    if line.startswith("Chương ") or re.match(r"^Điều\s+\d+", line):
        print(line)

for name in [
    "samsung_galaxy_s24_128gb_device_specs.txt",
    "samsung_galaxy_s24_specifications.txt"
]:
    path = folder / name
    print(f"\n===== {name} =====")
    print(path.read_text(encoding="utf-8"))
