"""Chỉ số đánh giá: EM/F1, Recall@k, MRR@10, bootstrap CI."""
from __future__ import annotations

from typing import Callable, NamedTuple, Sequence

import numpy as np

# ---------------------------------------------------------------------------
# Bootstrap CI dùng chung (C)
#
# Mọi chỉ số của dự án (EM, F1, Recall@k, MRR@10, tỷ lệ ảo giác) đều được tính
# trước thành một vector điểm theo từng câu hỏi; các hàm dưới đây nhận vector
# đó và lấy mẫu lại theo câu hỏi (không theo token hay theo đoạn), đúng với quy
# ước của kế hoạch dự án (mục 3.4).
# ---------------------------------------------------------------------------

# Số lần lấy mẫu lại và độ tin cậy mặc định theo kế hoạch (CI 95%).
DEFAULT_N_RESAMPLES = 10_000
DEFAULT_CONFIDENCE = 0.95
DEFAULT_SEED = 0

# Số lần lấy mẫu lại xử lý cùng lúc, để giới hạn bộ nhớ khi tập đánh giá lớn
# (ví dụ 3.814 câu của test ViQuAD).
_BATCH_SIZE = 500


class BootstrapResult(NamedTuple):
    """Ước lượng điểm và khoảng tin cậy bootstrap (percentile)."""

    estimate: float
    low: float
    high: float
    confidence: float
    n_resamples: int


def _as_scores(values: Sequence[float] | np.ndarray, name: str) -> np.ndarray:
    scores = np.asarray(values, dtype=float)
    if scores.ndim != 1:
        raise ValueError(f"{name} phải là vector một chiều, nhận được shape {scores.shape}")
    if scores.size == 0:
        raise ValueError(f"{name} rỗng")
    if not np.all(np.isfinite(scores)):
        raise ValueError(f"{name} chứa giá trị NaN hoặc vô hạn")
    return scores


def _check_params(n_resamples: int, confidence: float) -> None:
    if n_resamples < 1:
        raise ValueError("n_resamples phải ≥ 1")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence phải nằm trong khoảng (0, 1)")


def _resampled_statistics(
    arrays: Sequence[np.ndarray],
    statistic: Callable[[np.ndarray], float] | None,
    n_resamples: int,
    seed: int,
) -> np.ndarray:
    """Giá trị thống kê trên từng mẫu lấy lại.

    Với nhiều vector (so sánh cặp), mọi vector dùng chung một bộ chỉ số lấy mẫu
    nên cặp (a_i, b_i) của cùng một câu hỏi luôn đi cùng nhau. Kết quả có
    shape (len(arrays), n_resamples).
    """
    n = arrays[0].shape[0]
    rng = np.random.default_rng(seed)
    out = np.empty((len(arrays), n_resamples), dtype=float)
    for start in range(0, n_resamples, _BATCH_SIZE):
        stop = min(start + _BATCH_SIZE, n_resamples)
        idx = rng.integers(0, n, size=(stop - start, n))
        for k, arr in enumerate(arrays):
            sample = arr[idx]  # (batch, n)
            if statistic is None:
                out[k, start:stop] = sample.mean(axis=1)
            else:
                out[k, start:stop] = [statistic(row) for row in sample]
    return out


def _percentile_interval(values: np.ndarray, confidence: float) -> tuple[float, float]:
    alpha = (1.0 - confidence) / 2.0
    low, high = np.quantile(values, [alpha, 1.0 - alpha])
    return float(low), float(high)


def bootstrap_ci(
    scores: Sequence[float] | np.ndarray,
    statistic: Callable[[np.ndarray], float] | None = None,
    n_resamples: int = DEFAULT_N_RESAMPLES,
    confidence: float = DEFAULT_CONFIDENCE,
    seed: int = DEFAULT_SEED,
) -> BootstrapResult:
    """Khoảng tin cậy bootstrap (percentile) cho một chỉ số tính theo từng câu hỏi.

    Args:
        scores: điểm của từng câu hỏi (ví dụ F1 của mỗi câu, hoặc 0/1 cho
            Recall@k và ảo giác).
        statistic: hàm nhận vector điểm của một mẫu và trả về một số. Mặc
            định là trung bình (EM, F1, Recall@k, tỷ lệ ảo giác).
        n_resamples: số lần lấy mẫu lại có hoàn lại.
        confidence: mức tin cậy của khoảng, mặc định 0,95.
        seed: hạt giống của bộ sinh số ngẫu nhiên; cùng seed cho cùng kết quả.

    Returns:
        BootstrapResult với ước lượng trên dữ liệu gốc và hai đầu mút của khoảng.
    """
    arr = _as_scores(scores, "scores")
    _check_params(n_resamples, confidence)
    estimate = float(arr.mean() if statistic is None else statistic(arr))
    stats = _resampled_statistics([arr], statistic, n_resamples, seed)[0]
    low, high = _percentile_interval(stats, confidence)
    return BootstrapResult(estimate, low, high, confidence, n_resamples)


def paired_bootstrap_diff_ci(
    scores_a: Sequence[float] | np.ndarray,
    scores_b: Sequence[float] | np.ndarray,
    statistic: Callable[[np.ndarray], float] | None = None,
    n_resamples: int = DEFAULT_N_RESAMPLES,
    confidence: float = DEFAULT_CONFIDENCE,
    seed: int = DEFAULT_SEED,
) -> BootstrapResult:
    """Khoảng tin cậy bootstrap cho chênh lệch ``statistic(a) - statistic(b)``.

    Hai hệ thống phải được đánh giá trên cùng một danh sách câu hỏi theo cùng
    thứ tự. Mỗi lần lấy mẫu lại chọn một tập câu hỏi chung cho cả hai hệ
    thống, nên khoảng tin cậy phản ánh đúng tính ghép cặp. Nếu khoảng không
    chứa 0 thì chênh lệch được coi là có ý nghĩa ở mức ``confidence``.

    Args:
        scores_a, scores_b: điểm từng câu hỏi của hai hệ thống, cùng độ dài.
        statistic, n_resamples, confidence, seed: như ở ``bootstrap_ci``.

    Returns:
        BootstrapResult; ``estimate`` là chênh lệch trên dữ liệu gốc (a trừ b).
    """
    a = _as_scores(scores_a, "scores_a")
    b = _as_scores(scores_b, "scores_b")
    if a.shape != b.shape:
        raise ValueError(
            f"scores_a và scores_b phải cùng độ dài, nhận được {a.shape[0]} và {b.shape[0]}"
        )
    _check_params(n_resamples, confidence)
    if statistic is None:
        estimate = float(a.mean() - b.mean())
    else:
        estimate = float(statistic(a) - statistic(b))
    stats = _resampled_statistics([a, b], statistic, n_resamples, seed)
    low, high = _percentile_interval(stats[0] - stats[1], confidence)
    return BootstrapResult(estimate, low, high, confidence, n_resamples)
