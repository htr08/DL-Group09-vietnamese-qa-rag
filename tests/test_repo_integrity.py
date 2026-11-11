"""Structural checks on repository assets (no heavy dependencies, runs in CI)."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SMOKE_DIR = ROOT / "kaggle" / "smoke_test_dataset"
REQUIRED_KEYS = {"id", "lang_context", "question", "context", "answerable", "gold_contains"}


def _load_smoke_items():
    with open(SMOKE_DIR / "smoke_test.jsonl", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def test_expected_directories_exist():
    for name in ("data", "notebooks", "scripts", "src", "tests", "app", "docs", "report"):
        assert (ROOT / name).is_dir(), f"missing directory: {name}"


def test_smoke_dataset_schema():
    items = _load_smoke_items()
    assert len(items) == 17
    ids = [item["id"] for item in items]
    assert len(ids) == len(set(ids)), "duplicate ids"
    for item in items:
        assert REQUIRED_KEYS <= item.keys(), f"missing keys in {item['id']}"
        assert item["question"].strip() and item["context"].strip()
        assert item["lang_context"] in {"en", "vi"}


def test_smoke_dataset_answerable_flag_matches_gold():
    for item in _load_smoke_items():
        if item["answerable"]:
            assert item["gold_contains"], f"{item['id']}: answerable item needs gold_contains"
        else:
            assert item["gold_contains"] == [], f"{item['id']}: unanswerable item must not have gold_contains"


def test_smoke_dataset_gold_strings_come_from_context():
    # Answers must be recoverable from the context, except for yes/no items
    # whose gold strings are negation phrases.
    negation_items = {"en07"}
    for item in _load_smoke_items():
        if item["answerable"] and item["id"] not in negation_items:
            assert any(g.lower() in item["context"].lower() for g in item["gold_contains"]), (
                f"{item['id']}: no gold string found in the context"
            )


def test_kaggle_notebook_is_valid_nbformat4():
    nb = json.loads((ROOT / "notebooks" / "00_kaggle_qwen_smoke_test.ipynb").read_text(encoding="utf-8"))
    assert nb["nbformat"] == 4
    code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
    assert code_cells
    assert all(c["outputs"] == [] for c in code_cells), "clear notebook outputs before committing"
