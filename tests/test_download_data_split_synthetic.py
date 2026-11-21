"""Kiểm tra chia Train/Dev bằng dữ liệu giả, không cần tải ViQuAD."""

import json

import pytest

from scripts import download_data


@pytest.fixture
def isolated_split(tmp_path, monkeypatch):
    """Cô lập thư mục output, tránh sửa split thật."""
    split_dir = tmp_path / "splits"

    monkeypatch.setattr(download_data, "SPLITS_DIR", split_dir)
    monkeypatch.setattr(download_data, "DEV_RATIO", 0.5)
    monkeypatch.setattr(download_data, "SEED", 42)

    return split_dir


def test_make_dev_split_no_leakage(isolated_split):
    """Hai title khác nhau, hai context khác nhau: chia hợp lệ."""
    rows = [
        {"title": "Bài A", "context": "Nội dung riêng A"},
        {"title": "Bài B", "context": "Nội dung riêng B"},
    ]

    dev_titles = download_data.make_dev_split({"train": rows})

    train_rows = [r for r in rows if r["title"] not in dev_titles]
    dev_rows = [r for r in rows if r["title"] in dev_titles]

    assert len(train_rows) == 1
    assert len(dev_rows) == 1

    assert not (
        {r["title"] for r in train_rows}
        & {r["title"] for r in dev_rows}
    )

    assert not (
        {r["context"] for r in train_rows}
        & {r["context"] for r in dev_rows}
    )

    saved_titles = set(
        (isolated_split / "viquad_dev_titles.txt")
        .read_text(encoding="utf-8")
        .splitlines()
    )
    assert saved_titles == dev_titles

    metadata = json.loads(
        (isolated_split / "viquad_split_meta.json")
        .read_text(encoding="utf-8")
    )
    assert metadata["context_overlap_count"] == 0


def test_make_dev_split_rejects_duplicate_context(isolated_split):
    """Hai title khác nhau nhưng chung context: phải báo leakage."""
    rows = [
        {"title": "Bài A", "context": "Nội dung bị trùng"},
        {"title": "Bài B", "context": "Nội dung bị trùng"},
    ]

    with pytest.raises(RuntimeError, match="cả train lẫn dev"):
        download_data.make_dev_split({"train": rows})