import numpy as np
import pytest

from napari_colocalization._analysis import (
    COLUMNS,
    analyse_all_to_all,
    analyse_pairwise,
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
    assert row['m1'] == pytest.approx(1.0)


def test_pairwise_unrequested_metrics_are_nan(rng):
    a = rng.random((10, 10))
    row = analyse_pairwise(a, a.copy(), metrics=('pcc',))[0]
    assert not np.isnan(row['pcc'])
    assert np.isnan(row['srcc'])
    assert np.isnan(row['overlap'])
    assert np.isnan(row['m1'])


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
    assert all(r['m1'] == pytest.approx(1.0) for r in rows)


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
    assert row['m1'] == pytest.approx(1.0)


def test_pairwise_invalid_arguments_raise():
    a = np.zeros((10, 10))
    a[2:8, 2:8] = 1.0
    with pytest.raises(ValueError, match='shape mismatch'):
        analyse_pairwise(a, np.zeros((10, 11)))
    with pytest.raises(ValueError, match='manual'):
        analyse_pairwise(a, a, metrics=('mcc',), threshold_method='manual')
    with pytest.raises(ValueError, match='unknown threshold_method'):
        analyse_pairwise(a, a, metrics=('mcc',), threshold_method='bogus')


def test_auto_threshold_method():
    img = np.zeros((20, 20))
    img[5:15, 5:15] = 1.0
    row = analyse_pairwise(
        img, img.copy(), metrics=('mcc',), threshold_method='otsu'
    )[0]
    assert row['m1'] == pytest.approx(1.0)
    assert 0.0 < row['threshold_a'] < 1.0
    # a constant channel has no Otsu threshold -> M1/M2 are NaN
    const = analyse_pairwise(
        np.ones((20, 20)), img, metrics=('mcc',), threshold_method='otsu'
    )[0]
    assert np.isnan(const['m1'])


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
