import pytest

from scripts.download_data import validate_no_title_leakage


def test_train_dev_with_different_titles_passes():
    train = [
        {"title": "Bài viết A"},
        {"title": "Bài viết B"},
    ]
    dev = [
        {"title": "Bài viết C"},
    ]

    validate_no_title_leakage(train, dev)


def test_train_dev_with_overlapping_titles_fails():
    train = [
        {"title": "Bài viết A"},
        {"title": "Bài viết B"},
    ]
    dev = [
        {"title": "Bài viết B"},
    ]

    with pytest.raises(ValueError, match="overlapping titles"):
        validate_no_title_leakage(train, dev)