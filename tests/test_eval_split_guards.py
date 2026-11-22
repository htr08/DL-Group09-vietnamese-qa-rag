"""Kiểm tra CLI đánh giá Public Test mà không chạy model."""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "eval_qa_nbest.py"


def run_eval(*args):
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )


def test_public_test_requires_dev_threshold():
    result = run_eval(
        "--split", "public_test",
        "--checkpoint", "dummy",
        "--features", "dummy",
        "--gold", "dummy",
    )
    assert result.returncode == 2
    assert "--threshold-file" in result.stderr


def test_public_test_rejects_partial_evaluation():
    result = run_eval(
        "--split", "public_test",
        "--checkpoint", "dummy",
        "--features", "dummy",
        "--gold", "dummy",
        "--threshold-file", "dummy.json",
        "--limit-features", "10",
    )
    assert result.returncode == 2
    assert "partial public_test" in result.stderr