import numpy as np
import pytest

from napari_colocalization._analysis import (
    COLUMNS,
    _all_to_all_row_count,
    _pairwise_row_count,
    analyse_all_to_all,
    analyse_pairwise,
)
from napari_colocalization._metrics import (
    costes_threshold,
    li_icq,
    manders,
    overlap,
    pearson,
    spearman,
)


@pytest.fixture
def rng():
    return np.random.default_rng(seed=0)


# -- pairwise ---------------------------------------------------------


def test_pairwise_all_metrics_populated(rng):
    a = rng.random((20, 20))
    rows = analyse_pairwise(
        a,
        a.copy(),
        threshold_method='manual',
        threshold_a=0.0,
        threshold_b=0.0,
    )
    assert len(rows) == 1
    row = rows[0]
    assert set(row) == set(COLUMNS)  # constant output schema
    assert row['region'] == 0
    assert row['n_pixels'] == 400
    # identical channels: every metric is at its "perfect" value
    assert row['pcc'] == pytest.approx(1.0)
    assert row['icq'] == pytest.approx(0.5)
    assert row['overlap'] == pytest.approx(1.0)
    assert row['tm1'] == pytest.approx(1.0)


def test_pairwise_unrequested_metrics_are_nan(rng):
    a = rng.random((10, 10))
    row = analyse_pairwise(a, a.copy(), metrics=('pcc',))[0]
    assert not np.isnan(row['pcc'])
    assert np.isnan(row['srcc'])
    assert np.isnan(row['overlap'])
    assert np.isnan(row['tm1'])


def test_pairwise_one_row_per_labelled_region():
    a = np.ones((10, 10))
    label_mask = np.zeros((10, 10), dtype=int)
    label_mask[:5], label_mask[5:] = 1, 2
    rows = analyse_pairwise(
        a,
        a.copy(),
        label_mask=label_mask,
        metrics=('mcc',),
        threshold_method='manual',
        threshold_a=0.5,
        threshold_b=0.5,
    )
    assert {r['region'] for r in rows} == {1, 2}
    assert all(r['n_pixels'] == 50 for r in rows)
    assert all(r['tm1'] == pytest.approx(1.0) for r in rows)


def _same(value, expected):
    """Equal within tolerance, treating NaN as matching only NaN."""
    if np.isnan(expected):
        return np.isnan(value)
    return value == pytest.approx(expected, rel=1e-12, abs=1e-12)


def test_pairwise_region_values_match_full_image_masks(rng):
    # Regions are analysed on a bounding-box crop; the metrics must
    # match what the equivalent full-size boolean mask gives.
    a = rng.random((24, 30))
    b = 0.4 * a + 0.6 * rng.random((24, 30))
    label_mask = np.zeros((24, 30), dtype=int)
    label_mask[0:6, 0:5] = 1  # touches two image borders
    label_mask[9:15, 12:20] = 2
    label_mask[18:24, 22:30] = 3  # touches the far borders

    rows = analyse_pairwise(
        a, b, label_mask=label_mask, threshold_method='costes'
    )
    assert [row['region'] for row in rows] == [1, 2, 3]

    for row in rows:
        mask = label_mask == row['region']
        assert row['n_pixels'] == int(mask.sum())
        pcc, _ = pearson(a, b, mask=mask)
        srcc, _ = spearman(a, b, mask=mask)
        r, k1, k2 = overlap(a, b, mask=mask)
        t_a, t_b = costes_threshold(a, b, mask=mask)
        m1, m2 = manders(a, b, t_a, t_b, mask=mask)
        expected = {
            'pcc': pcc,
            'srcc': srcc,
            'icq': li_icq(a, b, mask=mask),
            'overlap': r,
            'k1': k1,
            'k2': k2,
            'tm1': m1,
            'tm2': m2,
            'threshold_a': t_a,
            'threshold_b': t_b,
        }
        for key, want in expected.items():
            assert _same(row[key], want), key


def test_pairwise_handles_non_contiguous_label_values(rng):
    a = rng.random((12, 12))
    label_mask = np.zeros((12, 12), dtype=int)
    label_mask[1:4, 1:4] = 3
    label_mask[7:11, 6:10] = 7
    rows = analyse_pairwise(
        a, a.copy(), label_mask=label_mask, metrics=('pcc',)
    )
    assert [row['region'] for row in rows] == [3, 7]
    assert [row['n_pixels'] for row in rows] == [9, 16]
    assert all(row['pcc'] == pytest.approx(1.0) for row in rows)


def test_pairwise_works_in_3d(rng):
    a = rng.random((4, 8, 8))
    row = analyse_pairwise(
        a,
        a.copy(),
        metrics=('pcc', 'mcc'),
        threshold_method='manual',
        threshold_a=0.0,
        threshold_b=0.0,
    )[0]
    assert row['pcc'] == pytest.approx(1.0)
    assert row['tm1'] == pytest.approx(1.0)


def test_pairwise_invalid_arguments_raise():
    a = np.zeros((10, 10))
    a[2:8, 2:8] = 1.0
    with pytest.raises(ValueError, match='shape mismatch'):
        analyse_pairwise(a, np.zeros((10, 11)))
    with pytest.raises(ValueError, match='manual'):
        analyse_pairwise(a, a, metrics=('mcc',), threshold_method='manual')
    with pytest.raises(ValueError, match='unknown threshold_method'):
        analyse_pairwise(a, a, metrics=('mcc',), threshold_method='bogus')
    # a typo is caught even when no thresholded metric is requested
    with pytest.raises(ValueError, match='unknown threshold_method'):
        analyse_pairwise(a, a, metrics=('pcc',), threshold_method='ostu')


def test_auto_threshold_method():
    img = np.zeros((20, 20))
    img[5:15, 5:15] = 1.0
    row = analyse_pairwise(
        img, img.copy(), metrics=('mcc',), threshold_method='otsu'
    )[0]
    assert row['tm1'] == pytest.approx(1.0)
    assert 0.0 < row['threshold_a'] < 1.0
    # a constant channel has no Otsu threshold -> M1/M2 are NaN
    const = analyse_pairwise(
        np.ones((20, 20)), img, metrics=('mcc',), threshold_method='otsu'
    )[0]
    assert np.isnan(const['tm1'])


def test_region_warnings_flag_uncomputable_regions(rng):
    # a clean run produces no warnings
    warns = []
    analyse_pairwise(
        rng.random((16, 16)),
        rng.random((16, 16)),
        metrics=('pcc',),
        region_warnings=warns,
    )
    assert warns == []

    # one constant region (of two) is flagged, by region id and reason
    a = rng.random((10, 10))
    a[:5] = 1.0
    label_mask = np.zeros((10, 10), dtype=int)
    label_mask[:5], label_mask[5:] = 1, 2
    warns = []
    analyse_pairwise(
        a,
        a.copy(),
        label_mask=label_mask,
        metrics=('pcc',),
        region_warnings=warns,
    )
    assert len(warns) == 1
    assert 'region 1' in warns[0] and 'constant' in warns[0]

    # a single-pixel region is flagged too
    single = np.zeros((5, 5), dtype=int)
    single[0, 0] = 1
    warns = []
    analyse_pairwise(
        np.zeros((5, 5)),
        np.zeros((5, 5)),
        label_mask=single,
        metrics=('pcc',),
        region_warnings=warns,
    )
    assert 'fewer than 2 pixels' in warns[0]


def test_per_slice_gives_one_row_per_slice(rng):
    a = rng.random((5, 16, 16))
    rows = analyse_pairwise(a, a.copy(), metrics=('pcc',), slice_axis=0)
    assert sorted(int(r['slice']) for r in rows) == [0, 1, 2, 3, 4]
    assert all(r['pcc'] == pytest.approx(1.0) for r in rows)
    # a whole-volume run leaves the slice column as NaN
    whole = analyse_pairwise(a, a.copy(), metrics=('pcc',))
    assert len(whole) == 1 and np.isnan(whole[0]['slice'])


def test_per_slice_with_label_mask(rng):
    a = rng.random((3, 10, 10))
    label_mask = np.zeros((3, 10, 10), dtype=int)
    label_mask[:, :5], label_mask[:, 5:] = 1, 2
    rows = analyse_pairwise(
        a, a.copy(), label_mask=label_mask, metrics=('pcc',), slice_axis=0
    )
    assert len(rows) == 6  # 3 slices x 2 regions
    assert {int(r['slice']) for r in rows} == {0, 1, 2}
    assert {r['region'] for r in rows} == {1, 2}


# -- all-to-all -------------------------------------------------------


def test_all_to_all_covers_every_pair(rng):
    a = rng.random((10, 10))
    image = np.stack([a, a, a], axis=0)
    rows = analyse_all_to_all(
        image,
        channel_axis=0,
        metrics=('pcc', 'mcc'),
        threshold_method='manual',
        threshold_a=0.0,
        threshold_b=0.0,
        channel_names=['x', 'y', 'z'],
    )
    assert {(r['channel_a'], r['channel_b']) for r in rows} == {
        ('x', 'y'),
        ('x', 'z'),
        ('y', 'z'),
    }
    assert all(r['pcc'] == pytest.approx(1.0) for r in rows)


def test_all_to_all_with_label_mask():
    a = np.ones((6, 6))
    image = np.stack([a, a], axis=0)
    label_mask = np.zeros((6, 6), dtype=int)
    label_mask[:3], label_mask[3:] = 1, 2
    rows = analyse_all_to_all(
        image,
        channel_axis=0,
        label_mask=label_mask,
        metrics=('mcc',),
        threshold_method='manual',
        threshold_a=0.5,
        threshold_b=0.5,
    )
    assert len(rows) == 2  # 1 channel pair x 2 regions


def test_all_to_all_validates_channel_names_length():
    with pytest.raises(ValueError, match='channel_names'):
        analyse_all_to_all(
            np.zeros((3, 4, 4)), channel_axis=0, channel_names=['only_one']
        )


# -- row counts -------------------------------------------------------


def test_row_count_helpers_match_the_rows_produced(rng):
    # A region that is absent from a plane yields no row there, so the
    # per-slice count is not simply regions x slices.
    a = rng.random((3, 8, 8))
    label_mask = np.zeros((3, 8, 8), dtype=int)
    label_mask[:, :4] = 1
    label_mask[2, 4:] = 2
    for slice_axis in (None, 0):
        rows = analyse_pairwise(
            a,
            a.copy(),
            label_mask=label_mask,
            metrics=('pcc',),
            slice_axis=slice_axis,
        )
        assert _pairwise_row_count(a.shape, label_mask, slice_axis) == len(
            rows
        )

    image = np.stack([a[0], a[1], a[2]], axis=0)
    rows = analyse_all_to_all(
        image, channel_axis=0, label_mask=label_mask[0], metrics=('pcc',)
    )
    assert _all_to_all_row_count(3, label_mask[0]) == len(rows)
