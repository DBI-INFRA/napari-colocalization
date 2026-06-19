import numpy as np
import pytest

from napari_colocalization._diagnostics import (
    costes_randomization,
    li_ica,
    van_steensel_ccf,
)


@pytest.fixture
def rng():
    return np.random.default_rng(seed=7)


# -- Van Steensel CCF --------------------------------------------------


def test_ccf_shape_and_peak_at_zero(rng):
    a = rng.random((64, 64))
    shifts, ccf = van_steensel_ccf(a, a.copy(), max_shift=10)
    assert shifts.shape == ccf.shape == (21,)
    assert shifts[0] == -10 and shifts[-1] == 10
    # identical channels: PCC is maximal (==1) at zero shift
    zero = np.where(shifts == 0)[0][0]
    assert ccf[zero] == pytest.approx(1.0)
    assert ccf[zero] == pytest.approx(np.nanmax(ccf))


def test_ccf_shift_recovers_offset(rng):
    a = rng.random((32, 64))
    # b is a shifted by +5 along the last axis. The CCF rolls b by
    # `shift`, so it realigns (peaks) when shift = -5 undoes the +5.
    b = np.roll(a, 5, axis=-1)
    shifts, ccf = van_steensel_ccf(a, b, max_shift=10, axis=-1)
    peak_shift = shifts[int(np.nanargmax(ccf))]
    assert peak_shift == -5


def test_ccf_shape_mismatch_raises():
    a = np.zeros((10, 10))
    b = np.zeros((10, 11))
    with pytest.raises(ValueError, match='shape mismatch'):
        van_steensel_ccf(a, b)


def test_ccf_bad_max_shift_raises():
    a = np.zeros((10, 10))
    with pytest.raises(ValueError, match='max_shift'):
        van_steensel_ccf(a, a, max_shift=0)


# -- Li ICA ------------------------------------------------------------


def test_li_ica_payload_shapes_and_icq(rng):
    a = rng.random((40, 40))
    out = li_ica(a, a.copy())
    n = a.size
    assert out['a'].shape == (n,)
    assert out['b'].shape == (n,)
    assert out['products'].shape == (n,)
    # identical channels co-vary -> products non-negative, ICQ ~ +0.5
    assert np.all(out['products'] >= -1e-12)
    assert out['icq'] == pytest.approx(0.5)


def test_li_ica_mask_restricts(rng):
    a = rng.random((20, 20))
    mask = np.zeros_like(a, dtype=bool)
    mask[10:] = True
    out = li_ica(a, a.copy(), mask=mask)
    assert out['a'].size == int(mask.sum())


def test_li_ica_too_few_pixels_raises():
    a = np.zeros((4, 4))
    mask = np.zeros_like(a, dtype=bool)
    mask[0, 0] = True
    with pytest.raises(ValueError, match='fewer than 2 pixels'):
        li_ica(a, a, mask=mask)


# -- Costes randomization ---------------------------------------------


def test_costes_randomization_correlated_is_significant(rng):
    base = rng.random((64, 64))
    a = base
    b = base.copy()
    out = costes_randomization(a, b, n_iter=50, block_size=8, seed=0)
    assert out['observed'] == pytest.approx(1.0)
    assert out['null'].shape == (50,)
    # scrambling destroys the colocalization, so observed sits far
    # above the null: smallest achievable p-value and a large z.
    assert out['p_value'] == pytest.approx(1 / 51)
    assert out['z_score'] > 3


def test_costes_randomization_independent_not_significant(rng):
    a = rng.random((64, 64))
    b = rng.random((64, 64))
    out = costes_randomization(a, b, n_iter=100, block_size=8, seed=1)
    # independent channels: observed PCC is just one draw from the
    # null, so it should not look extreme.
    assert out['p_value'] > 0.05


def test_costes_randomization_is_reproducible(rng):
    a = rng.random((48, 48))
    b = 0.5 * a + 0.5 * rng.random((48, 48))
    out1 = costes_randomization(a, b, n_iter=30, block_size=8, seed=42)
    out2 = costes_randomization(a, b, n_iter=30, block_size=8, seed=42)
    assert np.array_equal(out1['null'], out2['null'])


def test_costes_randomization_3d_runs(rng):
    base = rng.random((8, 16, 16))
    out = costes_randomization(
        base, base.copy(), n_iter=30, block_size=8, seed=0
    )
    assert out['observed'] == pytest.approx(1.0)
    assert out['null'].shape == (30,)
    # scrambling destroys the colocalization in 3D too
    assert out['p_value'] == pytest.approx(1 / 31)


def test_costes_randomization_block_larger_than_image_raises():
    a = np.zeros((8, 8))
    with pytest.raises(ValueError, match='larger than the image'):
        costes_randomization(a, a, block_size=16)


def test_scramble_example_preserves_values_and_shape(rng):
    from napari_colocalization._diagnostics import scramble_example

    arr = rng.random((16, 24))
    out = scramble_example(arr, block_size=8, seed=0)
    assert out.shape == arr.shape
    assert np.array_equal(np.sort(out.ravel()), np.sort(arr.ravel()))


def test_scramble_example_block_too_large_raises():
    from napari_colocalization._diagnostics import scramble_example

    with pytest.raises(ValueError, match='larger than the image'):
        scramble_example(np.zeros((8, 8)), block_size=16)


@pytest.mark.parametrize('shape', [(16, 24), (8, 16, 16)])
def test_scramble_blocks_preserves_values(rng, shape):
    from napari_colocalization._diagnostics import _scramble_blocks

    arr = rng.random(shape)
    scrambled = _scramble_blocks(
        arr, block_size=8, rng=np.random.default_rng(0)
    )
    assert scrambled.shape == arr.shape
    # block scramble only permutes blocks, so the multiset of pixel
    # values (and hence the histogram) is unchanged.
    assert np.array_equal(np.sort(scrambled.ravel()), np.sort(arr.ravel()))
