
from __future__ import annotations

import unicodedata
from collections import Counter
from typing import Callable, NamedTuple, Sequence

import numpy as np


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
