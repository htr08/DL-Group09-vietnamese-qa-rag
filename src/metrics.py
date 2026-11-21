"""Shared retrieval, extractive-QA, and bootstrap evaluation metrics."""
from __future__ import annotations

import unicodedata
from collections import Counter
from typing import Any, Callable, Iterable, List, NamedTuple, Sequence, Set, Union

import numpy as np


# Retrieval metrics (from main)

def _normalize_gold_set(relevant_ids: Union[str, Iterable[str], Any]) -> Set[str]:
    """Chuyển đổi relevant_id đơn lẻ hoặc danh sách thành tập hợp chuỗi đã chuẩn hóa."""
    if relevant_ids is None:
        return set()
    if isinstance(relevant_ids, str):
        return {relevant_ids.strip()}
    if isinstance(relevant_ids, (list, tuple, set)):
        return {str(x).strip() for x in relevant_ids if x is not None}
    return {str(relevant_ids).strip()}


def is_match(retrieved_id: str, gold_id: str, match_level: str = "clause") -> bool:
    """
    Kiểm tra xem một chunk ID được truy hồi có khớp với một gold ID hay không.
    
    Args:
        retrieved_id: ID chunk được hệ thống trả về (ví dụ 'dieu5_khoan2_w0', 'dieu5_khoan2').
        gold_id: ID nhãn chuẩn (ví dụ 'dieu5_khoan2' hoặc 'dieu5').
        match_level: 'clause' (mặc định - so khớp mức Khoản/tiền tố) hoặc 'article'/'dieu' (mức Điều).
    """
    if not retrieved_id or not gold_id:
        return False
    retrieved_id = str(retrieved_id).strip()
    gold_id = str(gold_id).strip()

    # Mức Điều: chỉ cần cùng số Điều (ví dụ dieu5 khớp với dieu5_khoan2)
    if match_level in ("article", "dieu"):
        r_dieu = retrieved_id.split("_")[0]
        g_dieu = gold_id.split("_")[0]
        return r_dieu == g_dieu

    # Mức Khoản (mặc định):
    # 1. Bằng nhau chính xác
    if retrieved_id == gold_id:
        return True
    # 2. Khớp theo tiền tố: retrieved là chunk con của gold (vd gold='dieu5_khoan2', retrieved='dieu5_khoan2_w0' hoặc '_diema')
    if retrieved_id.startswith(gold_id + "_"):
        return True
    # 3. Gold chi tiết hơn chunk (vd gold='dieu5_khoan2_diema', retrieved='dieu5_khoan2')
    if gold_id.startswith(retrieved_id + "_"):
        return True

    return False


def recall_at_k(
    retrieved_ids: List[Any],
    relevant_id: Union[str, Iterable[str], Any],
    k: int,
    match_level: str = "clause"
) -> float:
    """
    Tính Recall@k cho 1 truy vấn.
    Hỗ trợ:
      - relevant_id là 1 chuỗi hoặc tập hợp nhiều gold IDs.
      - Khớp tiền tố (chunk con _w*, _diem* khớp với Khoản cha).
      - Đánh giá mức Điều khi match_level='article'.
    """
    gold_set = _normalize_gold_set(relevant_id)
    if not retrieved_ids or not gold_set:
        return 0.0

    top_k = retrieved_ids[:k]
    for r_id in top_k:
        for g_id in gold_set:
            if is_match(str(r_id), g_id, match_level=match_level):
                return 1.0
    return 0.0


def mrr_at_k(
    retrieved_ids: List[Any],
    relevant_id: Union[str, Iterable[str], Any],
    k: int = 10,
    match_level: str = "clause"
) -> float:
    """
    Tính Reciprocal Rank (MRR@k) cho 1 truy vấn.
    Trả về 1 / rank của vị trí xuất hiện đầu tiên của bất kỳ gold ID nào trong top-k.
    """
    gold_set = _normalize_gold_set(relevant_id)
    if not retrieved_ids or not gold_set:
        return 0.0

    top_k = retrieved_ids[:k]
    for rank, r_id in enumerate(top_k, start=1):
        for g_id in gold_set:
            if is_match(str(r_id), g_id, match_level=match_level):
                return 1.0 / rank
    return 0.0


def mean_metric(scores: List[float]) -> float:
    """Tính trung bình của một mảng chỉ số."""
    return sum(scores) / len(scores) if scores else 0.0


# Extractive QA metrics (from A)
def normalize_answer(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = text.lower()
    text = "".join(
        " " if unicodedata.category(char).startswith("P") else char
        for char in text
    )
    return " ".join(text.split())


def exact_match_score(prediction: str, ground_truths: str | Sequence[str]) -> bool:
    if isinstance(ground_truths, str):
        ground_truths = [ground_truths]

    prediction = normalize_answer(prediction)

    return any(
        prediction == normalize_answer(answer)
        for answer in ground_truths
    )


def token_f1_score(prediction: str, ground_truth: str) -> float:
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()

    if not prediction_tokens or not ground_truth_tokens:
        return float(prediction_tokens == ground_truth_tokens)

    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_common = sum(common.values())

    if num_common == 0:
        return 0.0

    precision = num_common / len(prediction_tokens)
    recall = num_common / len(ground_truth_tokens)

    return 2 * precision * recall / (precision + recall)


def f1_score(prediction: str, ground_truths: str | Sequence[str]) -> float:
    if isinstance(ground_truths, str):
        ground_truths = [ground_truths]

    return max(
        (token_f1_score(prediction, answer) for answer in ground_truths),
        default=0.0,
    )


# Shared bootstrap confidence intervals
DEFAULT_N_RESAMPLES = 10_000
DEFAULT_CONFIDENCE = 0.95
DEFAULT_SEED = 0
_BATCH_SIZE = 500


class BootstrapResult(NamedTuple):
    estimate: float
    low: float
    high: float
    confidence: float
    n_resamples: int


def _as_scores(values: Sequence[float] | np.ndarray, name: str) -> np.ndarray:
    scores = np.asarray(values, dtype=float)

    if scores.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got {scores.shape}")

    if scores.size == 0:
        raise ValueError(f"{name} is empty")

    if not np.all(np.isfinite(scores)):
        raise ValueError(f"{name} contains NaN or infinite values")

    return scores


def _check_params(n_resamples: int, confidence: float) -> None:
    if n_resamples < 1:
        raise ValueError("n_resamples must be >= 1")

    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between 0 and 1")


def _resampled_statistics(
    arrays: Sequence[np.ndarray],
    statistic: Callable[[np.ndarray], float] | None,
    n_resamples: int,
    seed: int,
) -> np.ndarray:
    n = arrays[0].shape[0]
    rng = np.random.default_rng(seed)
    out = np.empty((len(arrays), n_resamples), dtype=float)

    for start in range(0, n_resamples, _BATCH_SIZE):
        stop = min(start + _BATCH_SIZE, n_resamples)
        idx = rng.integers(0, n, size=(stop - start, n))

        for k, arr in enumerate(arrays):
            sample = arr[idx]

            if statistic is None:
                out[k, start:stop] = sample.mean(axis=1)
            else:
                out[k, start:stop] = [
                    statistic(row) for row in sample
                ]

    return out


def _percentile_interval(
    values: np.ndarray,
    confidence: float,
) -> tuple[float, float]:
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
    arr = _as_scores(scores, "scores")
    _check_params(n_resamples, confidence)

    estimate = float(
        arr.mean() if statistic is None else statistic(arr)
    )

    stats = _resampled_statistics(
        [arr], statistic, n_resamples, seed
    )[0]

    low, high = _percentile_interval(stats, confidence)

    return BootstrapResult(
        estimate, low, high, confidence, n_resamples
    )


def paired_bootstrap_diff_ci(
    scores_a: Sequence[float] | np.ndarray,
    scores_b: Sequence[float] | np.ndarray,
    statistic: Callable[[np.ndarray], float] | None = None,
    n_resamples: int = DEFAULT_N_RESAMPLES,
    confidence: float = DEFAULT_CONFIDENCE,
    seed: int = DEFAULT_SEED,
) -> BootstrapResult:
    a = _as_scores(scores_a, "scores_a")
    b = _as_scores(scores_b, "scores_b")

    if a.shape != b.shape:
        raise ValueError(
            "scores_a và scores_b phải có cùng độ dài"
        )

    _check_params(n_resamples, confidence)

    if statistic is None:
        estimate = float(a.mean() - b.mean())
    else:
        estimate = float(statistic(a) - statistic(b))

    stats = _resampled_statistics(
        [a, b], statistic, n_resamples, seed
    )

    low, high = _percentile_interval(
        stats[0] - stats[1], confidence
    )

    return BootstrapResult(
        estimate, low, high, confidence, n_resamples
    )
