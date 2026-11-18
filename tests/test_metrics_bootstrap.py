"""Unit test cho bootstrap CI dùng chung (src/metrics.py)."""
import math

import numpy as np
import pytest

from src.metrics import BootstrapResult, bootstrap_ci, paired_bootstrap_diff_ci

# Số lần lấy mẫu nhỏ hơn mặc định để test chạy nhanh trên CI.
N_RES = 2000


def _binary(n, p, seed):
    return (np.random.default_rng(seed).random(n) < p).astype(float)


# --- bootstrap_ci ---------------------------------------------------------


def test_estimate_is_mean_of_original_scores():
    scores = [1, 0, 1, 1, 0]
    result = bootstrap_ci(scores, n_resamples=200)
    assert isinstance(result, BootstrapResult)
    assert result.estimate == pytest.approx(0.6)
    assert result.confidence == 0.95
    assert result.n_resamples == 200


def test_constant_scores_give_degenerate_interval():
    result = bootstrap_ci([0.7] * 40, n_resamples=300)
    assert result.low == pytest.approx(0.7)
    assert result.high == pytest.approx(0.7)


def test_interval_contains_estimate_and_is_ordered():
    result = bootstrap_ci(_binary(100, 0.4, seed=1), n_resamples=N_RES)
    assert result.low <= result.estimate <= result.high
    assert 0.0 <= result.low and result.high <= 1.0


def test_interval_matches_normal_approximation_for_binary_scores():
    # Với điểm 0/1, độ lệch chuẩn của trung bình là sqrt(p(1-p)/n); khoảng
    # percentile phải xấp xỉ p ± 1,96 * SE khi n đủ lớn.
    n = 400
    scores = _binary(n, 0.5, seed=2)
    p = scores.mean()
    se = math.sqrt(p * (1 - p) / n)
    result = bootstrap_ci(scores, n_resamples=5000)
    assert result.low == pytest.approx(p - 1.96 * se, abs=0.01)
    assert result.high == pytest.approx(p + 1.96 * se, abs=0.01)


def test_same_seed_is_reproducible_and_different_seed_differs():
    scores = _binary(60, 0.3, seed=3)
    first = bootstrap_ci(scores, n_resamples=500, seed=7)
    again = bootstrap_ci(scores, n_resamples=500, seed=7)
    other = bootstrap_ci(scores, n_resamples=500, seed=8)
    assert first == again
    assert (first.low, first.high) != (other.low, other.high)


def test_interval_narrows_as_sample_grows():
    small = bootstrap_ci(_binary(50, 0.5, seed=4), n_resamples=N_RES)
    large = bootstrap_ci(_binary(2000, 0.5, seed=4), n_resamples=N_RES)
    assert (large.high - large.low) < (small.high - small.low)


def test_higher_confidence_gives_wider_interval():
    scores = _binary(80, 0.5, seed=5)
    narrow = bootstrap_ci(scores, n_resamples=N_RES, confidence=0.80)
    wide = bootstrap_ci(scores, n_resamples=N_RES, confidence=0.99)
    assert (wide.high - wide.low) > (narrow.high - narrow.low)


def test_empirical_coverage_close_to_nominal():
    # Lặp lại trên nhiều tập mô phỏng có tỷ lệ thật p = 0,6: khoảng 95% phải
    # chứa p ở đa số lần. Ngưỡng 0,88 đủ lỏng để không flaky với 150 lần thử.
    p_true, hits, trials = 0.6, 0, 150
    for t in range(trials):
        r = bootstrap_ci(_binary(100, p_true, seed=1000 + t), n_resamples=400, seed=t)
        hits += r.low <= p_true <= r.high
    assert hits / trials >= 0.88


def test_custom_statistic_median():
    scores = [1.0, 2.0, 3.0, 4.0, 100.0]
    result = bootstrap_ci(scores, statistic=np.median, n_resamples=300)
    assert result.estimate == pytest.approx(3.0)
    assert result.low <= 3.0 <= result.high


def test_batching_covers_all_resamples_when_not_a_multiple_of_batch_size():
    # 1234 không chia hết cho cỡ lô (500): chạy không lỗi và khoảng hợp lệ.
    result = bootstrap_ci(_binary(30, 0.5, seed=6), n_resamples=1234)
    assert result.n_resamples == 1234
    assert result.low <= result.high


def test_accepts_numpy_array_and_list_equally():
    scores = [0.0, 1.0, 1.0, 0.0, 1.0, 1.0]
    assert bootstrap_ci(scores, n_resamples=200) == bootstrap_ci(np.array(scores), n_resamples=200)


@pytest.mark.parametrize(
    "bad",
    [[], [[1.0, 0.0], [1.0, 1.0]], [1.0, float("nan")], [1.0, float("inf")]],
)
def test_invalid_scores_raise(bad):
    with pytest.raises(ValueError):
        bootstrap_ci(bad, n_resamples=10)


@pytest.mark.parametrize("confidence", [0.0, 1.0, -0.1, 1.5])
def test_invalid_confidence_raises(confidence):
    with pytest.raises(ValueError):
        bootstrap_ci([0.0, 1.0], confidence=confidence, n_resamples=10)


def test_invalid_n_resamples_raises():
    with pytest.raises(ValueError):
        bootstrap_ci([0.0, 1.0], n_resamples=0)


# --- paired_bootstrap_diff_ci ----------------------------------------------


def test_paired_estimate_is_difference_of_means():
    a = [1, 1, 0, 1]
    b = [0, 1, 0, 0]
    result = paired_bootstrap_diff_ci(a, b, n_resamples=300)
    assert result.estimate == pytest.approx(0.5)


def test_identical_systems_give_zero_difference_with_degenerate_interval():
    scores = _binary(50, 0.5, seed=10)
    result = paired_bootstrap_diff_ci(scores, scores, n_resamples=300)
    assert result.estimate == 0.0
    assert result.low == 0.0 and result.high == 0.0


def test_clearly_better_system_has_interval_above_zero():
    rng = np.random.default_rng(11)
    b = (rng.random(300) < 0.4).astype(float)
    a = np.where(rng.random(300) < 0.9, 1.0, b)  # a thắng b ở đa số câu
    result = paired_bootstrap_diff_ci(a, b, n_resamples=N_RES)
    assert result.estimate > 0.3
    assert result.low > 0.0


def test_equal_systems_with_independent_noise_interval_contains_zero():
    a = _binary(200, 0.5, seed=12)
    b = _binary(200, 0.5, seed=13)
    result = paired_bootstrap_diff_ci(a, b, n_resamples=N_RES)
    assert result.low < 0.0 < result.high


def test_pairing_gives_tighter_interval_than_independent_resampling():
    # b chỉ khác a ở vài câu: ghép cặp phải cho khoảng hẹp hơn nhiều so với
    # trừ hai khoảng bootstrap độc lập. Đây là bằng chứng hai vector dùng
    # chung một bộ chỉ số lấy mẫu.
    a = _binary(300, 0.5, seed=14)
    b = a.copy()
    b[:10] = 1.0 - b[:10]
    paired = paired_bootstrap_diff_ci(a, b, n_resamples=N_RES)
    ci_a = bootstrap_ci(a, n_resamples=N_RES)
    ci_b = bootstrap_ci(b, n_resamples=N_RES, seed=1)
    independent_width = math.hypot(ci_a.high - ci_a.low, ci_b.high - ci_b.low)
    assert (paired.high - paired.low) < 0.5 * independent_width


def test_paired_is_antisymmetric_in_estimate_and_interval():
    a = _binary(80, 0.6, seed=15)
    b = _binary(80, 0.4, seed=16)
    ab = paired_bootstrap_diff_ci(a, b, n_resamples=500, seed=3)
    ba = paired_bootstrap_diff_ci(b, a, n_resamples=500, seed=3)
    assert ab.estimate == pytest.approx(-ba.estimate)
    assert ab.low == pytest.approx(-ba.high)
    assert ab.high == pytest.approx(-ba.low)


def test_paired_custom_statistic():
    a = [1.0, 2.0, 3.0, 10.0]
    b = [1.0, 1.0, 1.0, 1.0]
    result = paired_bootstrap_diff_ci(a, b, statistic=np.median, n_resamples=200)
    assert result.estimate == pytest.approx(np.median(a) - np.median(b))


def test_paired_length_mismatch_raises():
    with pytest.raises(ValueError, match="cùng độ dài"):
        paired_bootstrap_diff_ci([1.0, 0.0], [1.0], n_resamples=10)


def test_paired_rejects_nan():
    with pytest.raises(ValueError):
        paired_bootstrap_diff_ci([1.0, float("nan")], [1.0, 0.0], n_resamples=10)
