import pytest

import scripts.download_data as download_data


def _fake_dataset():
    rows = [
        {"title": f"T{i:02d}", "context": f"context {i}-{j}"}
        for i in range(20)
        for j in range(2)
    ]
    return {"train": rows}


def test_make_dev_split_writes_titles_when_file_is_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(download_data, "SPLITS_DIR", tmp_path)
    dev_titles = download_data.make_dev_split(_fake_dataset())

    saved = (tmp_path / "viquad_dev_titles.txt").read_text(encoding="utf-8").splitlines()
    assert len(dev_titles) == 2
    assert set(saved) == dev_titles


def test_make_dev_split_keeps_matching_titles_file(tmp_path, monkeypatch):
    monkeypatch.setattr(download_data, "SPLITS_DIR", tmp_path)
    download_data.make_dev_split(_fake_dataset())
    path = tmp_path / "viquad_dev_titles.txt"
    before = path.read_text(encoding="utf-8")

    download_data.make_dev_split(_fake_dataset())

    assert path.read_text(encoding="utf-8") == before


def test_make_dev_split_does_not_overwrite_mismatching_titles_file(tmp_path, monkeypatch):
    monkeypatch.setattr(download_data, "SPLITS_DIR", tmp_path)
    path = tmp_path / "viquad_dev_titles.txt"
    path.write_text("Tiêu đề khác\n", encoding="utf-8")

    with pytest.raises(RuntimeError):
        download_data.make_dev_split(_fake_dataset())

    assert path.read_text(encoding="utf-8") == "Tiêu đề khác\n"
