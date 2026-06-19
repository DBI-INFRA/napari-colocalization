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


# -- Pearson / Spearman -----------------------------------------------


def test_pearson_values(rng):
    a = rng.random((64, 64))
    assert pearson(a, a.copy())[0] == pytest.approx(1.0)
    assert pearson(a, -a)[0] == pytest.approx(-1.0)
    assert abs(pearson(a, rng.random((64, 64)))[0]) < 0.05


def test_pearson_respects_3d_and_mask(rng):
    a = rng.random((16, 32, 32))
    assert pearson(a, a)[0] == pytest.approx(1.0)  # 3D
    b = a[0].copy()
    b[:8] = rng.random((8, 32))  # differs only outside the mask
    mask = np.zeros_like(a[0], dtype=bool)
    mask[8:] = True
    assert pearson(a[0], b, mask=mask)[0] == pytest.approx(1.0)


def test_spearman_handles_monotonic_nonlinearity(rng):
    a = np.linspace(0.1, 10.0, 1000)
    b = a**3  # monotonic but very non-linear
    assert pearson(a, b)[0] < 0.95
    assert spearman(a, b)[0] == pytest.approx(1.0)


# -- Li ICQ -----------------------------------------------------------


def test_icq_values(rng):
    a = rng.random((64, 64))
    assert li_icq(a, a.copy()) == pytest.approx(0.5)  # co-varying
    assert li_icq(a, -a) == pytest.approx(-0.5)  # anti-varying
    assert abs(li_icq(a, rng.random((64, 64)))) < 0.05  # independent


def test_icq_respects_mask(rng):
    a = rng.random((32, 32))
    mask = np.zeros_like(a, dtype=bool)
    mask[16:] = True
    # within the mask b == -a, so the masked ICQ is fully anti-varying
    assert li_icq(a, -a, mask=mask) == pytest.approx(-0.5)


# -- Manders M1 / M2 --------------------------------------------------


def test_manders_coefficients():
    a = np.zeros((10, 10))
    b = np.zeros((10, 10))
    a[:, :] = 1.0  # all of A
    b[:5, :] = 1.0  # half of B overlaps A
    m1, m2 = manders(a, b, threshold_a=0.5, threshold_b=0.5)
    assert m1 == pytest.approx(0.5)  # half of A's signal sits under B
    assert m2 == pytest.approx(1.0)  # all of B's signal sits under A


def test_manders_respects_mask_and_3d():
    a = np.zeros((8, 8, 8))
    b = np.zeros((8, 8, 8))
    a[2:6, 2:6, 2:6] = 1.0
    b[2:6, 2:6, 2:6] = 1.0
    m1, m2 = manders(a, b, threshold_a=0.5, threshold_b=0.5)
    assert (m1, m2) == pytest.approx((1.0, 1.0))  # 3D full overlap


def test_manders_zero_intensity_is_nan():
    zeros = np.zeros((10, 10))
    m1, m2 = manders(zeros, zeros, threshold_a=0.5, threshold_b=0.5)
    assert np.isnan(m1) and np.isnan(m2)


# -- Overlap r / k1 / k2 ----------------------------------------------


def test_overlap_values(rng):
    a = rng.random((40, 40))
    assert overlap(a, a.copy()) == pytest.approx((1.0, 1.0, 1.0))
    # b = 2a: r is brightness-insensitive (1.0) but k1/k2 reflect the 2x
    r, k1, k2 = overlap(
        np.array([1.0, 2.0, 3.0, 4.0]), np.array([2, 4, 6, 8.0])
    )
    assert (r, k1, k2) == pytest.approx((1.0, 2.0, 0.5))


def test_overlap_mask_and_empty(rng):
    a = rng.random((20, 20))
    b = np.zeros_like(a)
    b[10:] = 2 * a[10:]
    mask = np.zeros_like(a, dtype=bool)
    mask[10:] = True
    assert overlap(a, b, mask=mask) == pytest.approx((1.0, 2.0, 0.5))
    # an empty region is undefined
    empty = np.zeros_like(a, dtype=bool)
    assert all(np.isnan(v) for v in overlap(a, b, mask=empty))


# -- Costes auto-threshold --------------------------------------------


def test_costes_threshold_separates_background(rng):
    base = rng.random((128, 128))
    a = base + 0.1 * rng.random((128, 128))
    b = 0.7 * base + 0.3 * rng.random((128, 128))
    t_a, t_b = costes_threshold(a, b)
    assert a.min() <= t_a <= a.max()
    assert b.min() <= t_b <= b.max()
    # below-threshold (background) pixels are ~uncorrelated
    below = (a <= t_a) | (b <= t_b)
    assert float(np.corrcoef(a[below], b[below])[0, 1]) <= 0.05


def test_costes_threshold_steep_slope_stays_in_range(rng):
    # slope ~3 (|m| >= 1) exercises the channel-B stepping branch
    a = rng.random((128, 128))
    b = 3 * a + 0.1 * rng.random((128, 128))
    t_a, t_b = costes_threshold(a, b)
    assert a.min() <= t_a <= a.max()
    assert b.min() <= t_b <= b.max()


def test_costes_threshold_anticorrelated_returns_max(rng):
    a = rng.random((64, 64))
    b = -a + 0.01 * rng.random((64, 64))  # negative slope -> no threshold
    assert costes_threshold(a, b) == pytest.approx((a.max(), b.max()))


def test_costes_regression_is_orthogonal_not_ols(rng):
    # noisy, unequal-variance data so OLS and orthogonal disagree
    a = rng.random(5000)
    b = 0.4 * a + 0.6 * rng.random(5000)
    slope, intercept = costes_regression(a, b)
    var_a, var_b = a.var(), b.var()
    cov = ((a - a.mean()) * (b - b.mean())).mean()
    m = (var_b - var_a + np.sqrt((var_b - var_a) ** 2 + 4 * cov**2)) / (
        2 * cov
    )
    assert slope == pytest.approx(m)  # Coloc 2's orthogonal slope
    assert intercept == pytest.approx(b.mean() - m * a.mean())
    assert abs(slope - np.polyfit(a, b, 1)[0]) > 1e-3  # not the OLS slope


# -- shared degeneracy ------------------------------------------------


@pytest.mark.parametrize(
    'func', [pearson, spearman, li_icq, costes_regression]
)
def test_degenerate_input_is_nan(func):
    # a constant channel has no variance -> the metric is undefined
    result = np.asarray(
        func(np.zeros((10, 10)), np.ones((10, 10))), dtype=float
    )
    assert np.all(np.isnan(result))
