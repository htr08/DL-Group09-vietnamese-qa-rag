"""Tải UIT-ViQuAD 2.0 vào data/raw/ và tạo dev split theo title vào data/splits/."""
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
SPLITS_DIR = ROOT / "data" / "splits"

VIQUAD_HF_NAME = "taidng/UIT-ViQuAD2.0"
DEV_RATIO = 0.10
SEED = 42


def download_viquad():
    """Tải ViQuAD 2.0 từ Hugging Face datasets và lưu vào data/raw/."""
    try:
        from datasets import load_dataset
    except ImportError:
        raise SystemExit("Cài datasets trước: pip install datasets")

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Tải {VIQUAD_HF_NAME} từ Hugging Face...")
    ds = load_dataset(VIQUAD_HF_NAME)

    # Lưu train và validation (= test công khai, có nhãn)
    for split_name in ("train", "validation"):
        out_path = RAW_DIR / f"viquad_{split_name}.jsonl"
        rows = ds[split_name]
        with open(out_path, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"  → {out_path} ({len(rows)} dòng)")

    # Không lưu split 'test': không nhãn và có đoạn trùng với train
    print("  (Bỏ qua split 'test': không nhãn, có đoạn trùng với train)")
    return ds


def make_dev_split(ds):
    """
    Tách 10% số bài viết (theo title) của train làm dev nội bộ.
    Seed cố định; lưu danh sách title và metadata vào data/splits/.
    Kiểm tra không có context nào nằm ở cả train lẫn dev.
    """
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)

    train_rows = list(ds["train"])

    # Lấy danh sách title không trùng, sắp xếp để reproducible
    all_titles = sorted(set(row["title"] for row in train_rows))
    n_dev = max(1, int(len(all_titles) * DEV_RATIO))

    rng = random.Random(SEED)
    dev_titles = set(rng.sample(all_titles, n_dev))
    train_titles = set(all_titles) - dev_titles

    # Kiểm tra không có đoạn context nào nằm ở cả train lẫn dev
    dev_contexts = set(
        row["context"] for row in train_rows if row["title"] in dev_titles
    )
    train_contexts = set(
        row["context"] for row in train_rows if row["title"] in train_titles
    )
    overlap = dev_contexts & train_contexts
    if overlap:
        raise RuntimeError(
            f"Có {len(overlap)} đoạn văn nằm ở cả train lẫn dev — kiểm tra lại cách tách!"
        )

    # Lưu danh sách title dev (đã có file này từ trước, ghi đè cho đồng bộ)
    dev_titles_path = SPLITS_DIR / "viquad_dev_titles.txt"
    with open(dev_titles_path, "w", encoding="utf-8") as f:
        for title in sorted(dev_titles):
            f.write(title + "\n")

    # Lưu metadata split để kiểm tra lại sau
    meta = {
        "seed": SEED,
        "dev_ratio": DEV_RATIO,
        "n_total_titles": len(all_titles),
        "n_dev_titles": len(dev_titles),
        "n_train_titles": len(train_titles),
        "n_dev_rows": sum(1 for r in train_rows if r["title"] in dev_titles),
        "n_train_rows": sum(1 for r in train_rows if r["title"] in train_titles),
        "context_overlap_count": len(overlap),
        "note": (
            "Dev tách theo title (bài viết), không theo câu hỏi, "
            "để không có đoạn context nào nằm ở cả train lẫn dev."
        ),
    }
    with open(SPLITS_DIR / "viquad_split_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"\nDev split : {meta['n_dev_titles']} titles / {meta['n_dev_rows']} dòng")
    print(f"Train split: {meta['n_train_titles']} titles / {meta['n_train_rows']} dòng")
    print(f"→ {dev_titles_path}")
    print(f"→ {SPLITS_DIR / 'viquad_split_meta.json'}")
    return dev_titles


if __name__ == "__main__":
    ds = download_viquad()
    make_dev_split(ds)
    print("\nHoàn tất. Chạy pytest để kiểm tra không có đoạn trùng train/dev.")
