import numpy as np
import pytest

from napari_colocalization._diagnostics import (
    costes_randomization,
    li_ica,
    scramble_example,
    van_steensel_ccf,
)


@pytest.fixture
def rng():
    return np.random.default_rng(seed=7)


# -- Van Steensel CCF -------------------------------------------------


def test_ccf_peaks_at_alignment(rng):
    a = rng.random((32, 64))
    shifts, ccf = van_steensel_ccf(a, a.copy(), max_shift=10)
    assert shifts.shape == ccf.shape == (21,)
    zero = np.where(shifts == 0)[0][0]
    assert ccf[zero] == pytest.approx(1.0)  # identical -> peak at shift 0
    assert ccf[zero] == pytest.approx(np.nanmax(ccf))
    # b shifted by +5: the CCF realigns when it rolls b back by -5
    shifts, ccf = van_steensel_ccf(a, np.roll(a, 5, axis=-1), max_shift=10)
    assert shifts[int(np.nanargmax(ccf))] == -5


def test_ccf_invalid_args_raise():
    a = np.zeros((10, 10))
    with pytest.raises(ValueError, match='shape mismatch'):
        van_steensel_ccf(a, np.zeros((10, 11)))
    with pytest.raises(ValueError, match='max_shift'):
        van_steensel_ccf(a, a, max_shift=0)


# -- Li ICA -----------------------------------------------------------


def test_li_ica_payload(rng):
    a = rng.random((40, 40))
    out = li_ica(a, a.copy())
    n = a.size
    assert out['a'].shape == out['b'].shape == out['products'].shape == (n,)
    assert np.all(out['products'] >= -1e-12)  # identical -> co-varying
    assert out['icq'] == pytest.approx(0.5)
    mask = np.zeros_like(a, dtype=bool)
    mask[20:] = True
    assert li_ica(a, a.copy(), mask=mask)['a'].size == int(mask.sum())


def test_li_ica_too_few_pixels_raises():
    a = np.zeros((4, 4))
    mask = np.zeros_like(a, dtype=bool)
    mask[0, 0] = True
    with pytest.raises(ValueError, match='fewer than 2 pixels'):
        li_ica(a, a, mask=mask)


# -- Costes randomization ---------------------------------------------


def test_costes_randomization_significance(rng):
    base = rng.random((64, 64))
    sig = costes_randomization(
        base, base.copy(), n_iter=50, block_size=8, seed=0
    )
    assert sig['observed'] == pytest.approx(1.0)
    assert sig['null'].shape == (50,)
    # scrambling destroys colocalization -> observed sits far above the null
    assert sig['p_value'] == pytest.approx(1 / 51)
    assert sig['z_score'] > 3
    indep = costes_randomization(
        rng.random((64, 64)), rng.random((64, 64)), n_iter=100, seed=1
    )
    assert indep['p_value'] > 0.05  # independent -> not significant


def test_costes_randomization_reproducible_and_3d(rng):
    a = rng.random((48, 48))
    b = 0.5 * a + 0.5 * rng.random((48, 48))
    out1 = costes_randomization(a, b, n_iter=30, seed=42)
    out2 = costes_randomization(a, b, n_iter=30, seed=42)
    assert np.array_equal(out1['null'], out2['null'])  # seeded -> reproducible
    vol = rng.random((8, 16, 16))
    assert costes_randomization(vol, vol.copy(), n_iter=20, seed=0)[
        'observed'
    ] == pytest.approx(1.0)  # works in 3D


def test_costes_randomization_block_too_large_raises():
    with pytest.raises(ValueError, match='larger than the image'):
        costes_randomization(np.zeros((8, 8)), np.zeros((8, 8)), block_size=16)


# -- block scramble (via scramble_example) ----------------------------


@pytest.mark.parametrize('shape', [(16, 24), (8, 16, 16)])
def test_scramble_preserves_pixel_values(rng, shape):
    arr = rng.random(shape)
    out = scramble_example(arr, block_size=8, seed=0)
    assert out.shape == arr.shape
    # only whole blocks are permuted, so the value multiset is unchanged
    assert np.array_equal(np.sort(out.ravel()), np.sort(arr.ravel()))


def test_scramble_example_block_too_large_raises():
    with pytest.raises(ValueError, match='larger than the image'):
        scramble_example(np.zeros((8, 8)), block_size=16)
