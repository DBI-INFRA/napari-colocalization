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


def test_manders_coefficients_are_decided_independently():
    # tM1 divides by sum(A), so an empty B makes it 0.0 - a measured
    # "none of A co-occurs with B" - while tM2 (dividing by sum(B)) is
    # genuinely undefined. Blanking both would hide the first result.
    ones = np.ones((10, 10))
    zeros = np.zeros((10, 10))
    m1, m2 = manders(ones, zeros, threshold_a=0.5, threshold_b=0.5)
    assert m1 == pytest.approx(0.0)
    assert np.isnan(m2)
    m1, m2 = manders(zeros, ones, threshold_a=0.5, threshold_b=0.5)
    assert np.isnan(m1)
    assert m2 == pytest.approx(0.0)


def test_manders_undefined_threshold_blanks_only_its_own_coefficient():
    # threshold_b gates tM1 and threshold_a gates tM2, so an undefined
    # threshold on one channel leaves the other coefficient computable.
    a = np.ones((10, 10))
    b = np.zeros((10, 10))
    b[:5] = 1.0
    m1, m2 = manders(a, b, threshold_a=float('nan'), threshold_b=0.5)
    assert m1 == pytest.approx(0.5)
    assert np.isnan(m2)


def test_manders_negative_input_is_nan_not_raise():
    # skimage's manders_coloc_coeff raises on negative pixels, which
    # aborted the whole run for one background-subtracted channel.
    a = np.full((10, 10), -1.0)
    b = np.ones((10, 10))
    m1, m2 = manders(a, b, threshold_a=0.5, threshold_b=0.5)
    assert np.isnan(m1)  # summing a signed channel is meaningless...
    assert m2 == pytest.approx(0.0)  # ...but gating on it is fine


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


def test_costes_threshold_anticorrelated_is_nan(rng):
    # A negative slope means there is no threshold to find. Returning
    # max(a)/max(b) here (as Coloc 2 does) would make manders() report
    # exactly 0.0, which reads as a measured "no co-occurrence" rather
    # than "no threshold could be fitted" - so we return nan instead.
    a = rng.random((64, 64))
    b = -a + 0.01 * rng.random((64, 64))
    assert all(np.isnan(t) for t in costes_threshold(a, b))
    assert all(np.isnan(m) for m in manders(a, b, *costes_threshold(a, b)))


def test_costes_threshold_constant_channel_is_nan():
    constant = np.ones((32, 32))
    varied = np.arange(32 * 32, dtype=float).reshape(32, 32)
    assert all(np.isnan(t) for t in costes_threshold(constant, varied))


def test_overlap_is_dtype_independent():
    # skimage's manders_overlap_coeff squares in the input dtype, so an
    # integer image overflows and r escapes [0, 1]. We accumulate in
    # float64 instead: uint8/uint16 must agree with float64.
    rng = np.random.default_rng(0)
    a8 = rng.integers(0, 256, (64, 64)).astype(np.uint8)
    b8 = (a8 // 2 + 20).astype(np.uint8)
    expected = overlap(a8.astype(np.float64), b8.astype(np.float64))
    assert overlap(a8, b8) == pytest.approx(expected)
    assert overlap(a8.astype(np.uint16), b8.astype(np.uint16)) == (
        pytest.approx(expected)
    )
    assert 0.0 <= overlap(a8, b8)[0] <= 1.0


def test_overlap_negative_input_is_nan(rng):
    # Background subtraction can push intensities below zero; r is only
    # defined for one-signed data. nan, not a raise (which would abort
    # the whole run) and not an out-of-range number.
    a = rng.random((32, 32)) - 0.5
    assert all(np.isnan(v) for v in overlap(a, a.copy()))


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
