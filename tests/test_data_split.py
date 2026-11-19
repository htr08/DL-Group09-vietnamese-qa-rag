import json
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data/raw/viquad"
SPLIT_FILE = ROOT / "data/splits/viquad_dev_titles.txt"


def load_titles(path):
    with open(path, "r", encoding="utf-8") as f:
        return {
            json.loads(line)["title"]
            for line in f
            if line.strip()
        }


def test_train_dev_no_title_leakage():
    train_path = DATA_DIR / "train.jsonl"
    dev_path = DATA_DIR / "dev.jsonl"

    if not train_path.exists() or not dev_path.exists():
        pytest.skip("Train/Dev dataset chưa tồn tại")

    train_titles = load_titles(train_path)
    dev_titles = load_titles(dev_path)

    overlap = train_titles & dev_titles

    assert not overlap, f"Trùng title: {overlap}"
    assert len(dev_titles) == 13


def test_dev_titles_match_split_file():
    dev_path = DATA_DIR / "dev.jsonl"

    if not dev_path.exists() or not SPLIT_FILE.exists():
        pytest.skip("Thiếu dữ liệu Dev hoặc split file")

    actual_titles = load_titles(dev_path)

    expected_titles = {
        line.strip()
        for line in SPLIT_FILE.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    }

    assert actual_titles == expected_titles