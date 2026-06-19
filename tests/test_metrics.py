import numpy as np
import pytest

from napari_colocalization._metrics import (
    costes_regression,
    costes_threshold,
    li_icq,
    manders,
    overlap,
    pearson,
    spearman,
)


@pytest.fixture
def rng():
    return np.random.default_rng(seed=42)


def test_pearson_identical_is_one(rng):
    a = rng.random((64, 64))
    pcc, _ = pearson(a, a.copy())
    assert pcc == pytest.approx(1.0)


def test_pearson_inverted_is_minus_one(rng):
    a = rng.random((64, 64))
    pcc, _ = pearson(a, -a)
    assert pcc == pytest.approx(-1.0)


def test_pearson_independent_near_zero(rng):
    a = rng.random((128, 128))
    b = rng.random((128, 128))
    pcc, _ = pearson(a, b)
    assert abs(pcc) < 0.05


def test_pearson_3d_works(rng):
    a = rng.random((16, 32, 32))
    pcc, _ = pearson(a, a)
    assert pcc == pytest.approx(1.0)


def test_pearson_mask_excludes_pixels(rng):
    a = rng.random((32, 32))
    b = a.copy()
    b[:16] = rng.random((16, 32))
    mask = np.zeros_like(a, dtype=bool)
    mask[16:] = True
    pcc, _ = pearson(a, b, mask=mask)
    assert pcc == pytest.approx(1.0)


def test_pearson_constant_input_is_nan():
    a = np.zeros((10, 10))
    b = np.ones((10, 10))
    pcc, pval = pearson(a, b)
    assert np.isnan(pcc) and np.isnan(pval)


def test_spearman_one_for_monotonic_nonlinear(rng):
    a = np.linspace(0.1, 10.0, 1000)
    b = a**3
    pcc, _ = pearson(a, b)
    rho, _ = spearman(a, b)
    assert pcc < 0.95
    assert rho == pytest.approx(1.0)


def test_spearman_3d_works(rng):
    a = rng.random((8, 16, 16))
    rho, _ = spearman(a, a)
    assert rho == pytest.approx(1.0)


def test_spearman_constant_input_is_nan():
    a = np.zeros((10, 10))
    b = np.ones((10, 10))
    rho, pval = spearman(a, b)
    assert np.isnan(rho) and np.isnan(pval)


def test_manders_full_overlap_is_one():
    a = np.zeros((10, 10))
    b = np.zeros((10, 10))
    a[2:8, 2:8] = 1.0
    b[2:8, 2:8] = 1.0
    m1, m2 = manders(a, b, threshold_a=0.5, threshold_b=0.5)
    assert m1 == pytest.approx(1.0)
    assert m2 == pytest.approx(1.0)


def test_manders_disjoint_is_zero():
    a = np.zeros((10, 10))
    b = np.zeros((10, 10))
    a[:5] = 1.0
    b[5:] = 1.0
    m1, m2 = manders(a, b, threshold_a=0.5, threshold_b=0.5)
    assert m1 == pytest.approx(0.0)
    assert m2 == pytest.approx(0.0)


def test_manders_partial_overlap():
    a = np.zeros((10, 10))
    b = np.zeros((10, 10))
    a[:, :] = 1.0  # all of a
    b[:5, :] = 1.0  # half of b
    m1, m2 = manders(a, b, threshold_a=0.5, threshold_b=0.5)
    # half of a's intensity colocalises with b
    assert m1 == pytest.approx(0.5)
    # all of b's intensity colocalises with a
    assert m2 == pytest.approx(1.0)


def test_manders_3d_works():
    a = np.zeros((8, 8, 8))
    b = np.zeros((8, 8, 8))
    a[2:6, 2:6, 2:6] = 1.0
    b[2:6, 2:6, 2:6] = 1.0
    m1, m2 = manders(a, b, threshold_a=0.5, threshold_b=0.5)
    assert m1 == pytest.approx(1.0)
    assert m2 == pytest.approx(1.0)


def test_manders_mask_restricts_region():
    a = np.zeros((10, 10))
    b = np.zeros((10, 10))
    a[:, :] = 1.0
    b[:5, :] = 1.0
    mask = np.zeros_like(a, dtype=bool)
    mask[:5, :] = True
    m1, m2 = manders(a, b, threshold_a=0.5, threshold_b=0.5, mask=mask)
    # within mask, a and b overlap fully
    assert m1 == pytest.approx(1.0)
    assert m2 == pytest.approx(1.0)


def test_manders_zero_intensity_returns_nan():
    a = np.zeros((10, 10))
    b = np.zeros((10, 10))
    m1, m2 = manders(a, b, threshold_a=0.5, threshold_b=0.5)
    assert np.isnan(m1) and np.isnan(m2)


def test_costes_returns_thresholds_in_range(rng):
    base = rng.random((128, 128))
    a = base + 0.05 * rng.random((128, 128))
    b = base + 0.05 * rng.random((128, 128))
    t_a, t_b = costes_threshold(a, b)
    assert a.min() <= t_a <= a.max()
    assert b.min() <= t_b <= b.max()


def test_costes_below_threshold_pcc_non_positive(rng):
    base = rng.random((128, 128))
    a = base + 0.1 * rng.random((128, 128))
    b = 0.7 * base + 0.3 * rng.random((128, 128))
    t_a, t_b = costes_threshold(a, b)
    below = (a <= t_a) | (b <= t_b)
    sub_a = a[below]
    sub_b = b[below]
    pcc = float(np.corrcoef(sub_a, sub_b)[0, 1])
    # the algorithm guarantees pcc <= 0 except when iteration runs
    # to the floor without ever reaching <= 0; allow a small slack
    assert pcc <= 0.05


def test_costes_anti_correlated_returns_max(rng):
    a = rng.random((64, 64))
    b = -a + 0.01 * rng.random((64, 64))
    t_a, t_b = costes_threshold(a, b)
    assert t_a == pytest.approx(a.max())
    assert t_b == pytest.approx(b.max())


def test_costes_constant_inputs_are_safe():
    a = np.zeros((10, 10))
    b = np.zeros((10, 10))
    t_a, t_b = costes_threshold(a, b)
    # no crash; thresholds are well-defined floats
    assert np.isfinite(t_a) and np.isfinite(t_b)


def test_icq_identical_is_half(rng):
    a = rng.random((64, 64))
    assert li_icq(a, a.copy()) == pytest.approx(0.5)


def test_icq_anti_correlated_is_minus_half(rng):
    a = rng.random((64, 64))
    assert li_icq(a, -a) == pytest.approx(-0.5)


def test_icq_independent_near_zero(rng):
    a = rng.random((128, 128))
    b = rng.random((128, 128))
    assert abs(li_icq(a, b)) < 0.05


def test_icq_3d_works(rng):
    a = rng.random((8, 16, 16))
    assert li_icq(a, a) == pytest.approx(0.5)


def test_icq_mask_excludes_pixels(rng):
    a = rng.random((32, 32))
    b = -a
    mask = np.zeros_like(a, dtype=bool)
    mask[16:] = True
    # within mask, b is exactly -a -> anti-correlated
    assert li_icq(a, b, mask=mask) == pytest.approx(-0.5)


def test_icq_constant_input_is_nan():
    a = np.zeros((10, 10))
    b = np.ones((10, 10))
    assert np.isnan(li_icq(a, b))


def test_icq_in_valid_range(rng):
    a = rng.random((128, 128))
    b = 0.5 * a + 0.5 * rng.random((128, 128))
    icq = li_icq(a, b)
    assert -0.5 <= icq <= 0.5


def test_overlap_identical_is_one(rng):
    a = rng.random((64, 64))
    r, k1, k2 = overlap(a, a.copy())
    assert r == pytest.approx(1.0)
    assert k1 == pytest.approx(1.0)
    assert k2 == pytest.approx(1.0)


def test_overlap_scaled_channel_splits_k1_k2():
    a = np.array([1.0, 2.0, 3.0, 4.0])
    # b = 2a: overlap r stays 1 (insensitive to brightness), but the
    # split coefficients reflect the 2x intensity difference.
    r, k1, k2 = overlap(a, 2 * a)
    assert r == pytest.approx(1.0)
    assert k1 == pytest.approx(2.0)
    assert k2 == pytest.approx(0.5)


def test_overlap_3d_works(rng):
    a = rng.random((8, 16, 16))
    r, k1, k2 = overlap(a, a)
    assert r == pytest.approx(1.0)


def test_overlap_mask_restricts_region(rng):
    a = rng.random((32, 32))
    b = np.zeros_like(a)
    b[16:] = 2 * a[16:]
    mask = np.zeros_like(a, dtype=bool)
    mask[16:] = True
    r, k1, k2 = overlap(a, b, mask=mask)
    assert r == pytest.approx(1.0)
    assert k1 == pytest.approx(2.0)
    assert k2 == pytest.approx(0.5)


def test_overlap_zero_channel_is_nan():
    a = np.zeros((10, 10))
    b = np.ones((10, 10))
    r, k1, k2 = overlap(a, b)
    # a is all-zero: sum(a*a)=0 and sum(a*b)=0, so r and k1 undefined.
    assert np.isnan(r)
    assert np.isnan(k1)


def test_overlap_empty_region_is_nan():
    a = np.ones((10, 10))
    b = np.ones((10, 10))
    mask = np.zeros_like(a, dtype=bool)
    r, k1, k2 = overlap(a, b, mask=mask)
    assert np.isnan(r) and np.isnan(k1) and np.isnan(k2)


def test_costes_regression_recovers_slope_intercept():
    a = np.linspace(0.0, 1.0, 256)
    b = 2.0 * a + 0.3
    slope, intercept = costes_regression(a, b)
    assert slope == pytest.approx(2.0)
    assert intercept == pytest.approx(0.3)


def test_costes_regression_degenerate_is_nan():
    a = np.ones((10, 10))
    b = np.ones((10, 10))
    slope, intercept = costes_regression(a, b)
    assert np.isnan(slope) and np.isnan(intercept)


def test_costes_regression_is_orthogonal_not_ols(rng):
    # Noisy, unequal-variance data so OLS and orthogonal disagree.
    a = rng.random(5000)
    b = 0.4 * a + 0.6 * rng.random(5000)
    slope, intercept = costes_regression(a, b)
    # closed-form orthogonal (total-least-squares) slope, as Coloc 2
    var_a, var_b = a.var(), b.var()
    cov = ((a - a.mean()) * (b - b.mean())).mean()
    m_orth = (var_b - var_a + np.sqrt((var_b - var_a) ** 2 + 4 * cov**2)) / (
        2 * cov
    )
    assert slope == pytest.approx(m_orth)
    assert intercept == pytest.approx(b.mean() - m_orth * a.mean())
    # and it is genuinely different from the OLS (np.polyfit) slope
    m_ols = np.polyfit(a, b, 1)[0]
    assert abs(slope - m_ols) > 1e-3


def test_costes_threshold_steep_slope_in_range(rng):
    # slope ~3 (|m| >= 1) exercises the channel-B stepping branch.
    a = rng.random((128, 128))
    b = 3 * a + 0.1 * rng.random((128, 128))
    t_a, t_b = costes_threshold(a, b)
    assert a.min() <= t_a <= a.max()
    assert b.min() <= t_b <= b.max()
