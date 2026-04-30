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


def test_pairwise_identical_full_image_perfect():
    a = np.zeros((10, 10))
    a[2:8, 2:8] = 1.0
    rows = analyse_pairwise(
        a,
        a.copy(),
        threshold_method='manual',
        threshold_a=0.5,
        threshold_b=0.5,
    )
    assert len(rows) == 1
    row = rows[0]
    assert row['region'] == 0
    assert row['n_pixels'] == 100
    assert row['pcc'] == pytest.approx(1.0)
    assert row['m1'] == pytest.approx(1.0)
    assert row['m2'] == pytest.approx(1.0)


def test_pairwise_columns_match_schema():
    a = np.zeros((10, 10))
    a[2:8, 2:8] = 1.0
    rows = analyse_pairwise(
        a,
        a,
        threshold_method='manual',
        threshold_a=0.5,
        threshold_b=0.5,
    )
    assert set(rows[0].keys()) == set(COLUMNS)


def test_pairwise_per_region_counts():
    a = np.zeros((10, 10))
    a[:, :] = 1.0
    b = a.copy()
    label_mask = np.zeros((10, 10), dtype=int)
    label_mask[:5, :] = 1
    label_mask[5:, :] = 2
    rows = analyse_pairwise(
        a,
        b,
        label_mask=label_mask,
        metrics=('mcc',),
        threshold_method='manual',
        threshold_a=0.5,
        threshold_b=0.5,
    )
    assert len(rows) == 2
    assert {r['region'] for r in rows} == {1, 2}
    for r in rows:
        assert r['n_pixels'] == 50
        assert np.isnan(r['pcc'])
        assert r['m1'] == pytest.approx(1.0)


def test_pairwise_only_requested_metrics_populated():
    a = np.zeros((10, 10))
    a[2:8, 2:8] = 1.0
    rows = analyse_pairwise(a, a, metrics=('pcc',))
    row = rows[0]
    assert not np.isnan(row['pcc'])
    assert np.isnan(row['srcc'])
    assert np.isnan(row['icq'])
    assert np.isnan(row['m1'])
    assert np.isnan(row['m2'])


def test_pairwise_icq_identical_is_half(rng):
    a = rng.random((32, 32))
    rows = analyse_pairwise(a, a.copy(), metrics=('icq',))
    assert rows[0]['icq'] == pytest.approx(0.5)
    assert np.isnan(rows[0]['pcc'])


def test_pairwise_shape_mismatch_raises():
    a = np.zeros((10, 10))
    b = np.zeros((10, 11))
    with pytest.raises(ValueError, match='shape mismatch'):
        analyse_pairwise(a, b)


def test_pairwise_manual_without_thresholds_raises():
    a = np.zeros((10, 10))
    a[2:8, 2:8] = 1.0
    with pytest.raises(ValueError, match='manual'):
        analyse_pairwise(a, a, metrics=('mcc',), threshold_method='manual')


def test_pairwise_unknown_threshold_method_raises():
    a = np.zeros((10, 10))
    a[2:8, 2:8] = 1.0
    with pytest.raises(ValueError, match='unknown threshold_method'):
        analyse_pairwise(a, a, metrics=('mcc',), threshold_method='bogus')


def test_pairwise_3d_works():
    a = np.zeros((4, 8, 8))
    a[1:3, 2:6, 2:6] = 1.0
    rows = analyse_pairwise(
        a,
        a.copy(),
        metrics=('pcc', 'mcc'),
        threshold_method='manual',
        threshold_a=0.5,
        threshold_b=0.5,
    )
    assert rows[0]['pcc'] == pytest.approx(1.0)
    assert rows[0]['m1'] == pytest.approx(1.0)


def test_all_to_all_pair_count():
    image = np.stack(
        [
            np.full((8, 8), 1.0),
            np.full((8, 8), 1.0),
            np.full((8, 8), 1.0),
        ],
        axis=0,
    )
    rows = analyse_all_to_all(
        image,
        channel_axis=0,
        metrics=('mcc',),
        threshold_method='manual',
        threshold_a=0.5,
        threshold_b=0.5,
    )
    # 3 channels -> 3 pairs (0,1), (0,2), (1,2)
    assert len(rows) == 3


def test_all_to_all_channel_names_propagate():
    image = np.stack([np.ones((4, 4)), np.ones((4, 4))], axis=0)
    rows = analyse_all_to_all(
        image,
        channel_axis=0,
        metrics=('mcc',),
        threshold_method='manual',
        threshold_a=0.5,
        threshold_b=0.5,
        channel_names=['dna', 'tubulin'],
    )
    assert rows[0]['channel_a'] == 'dna'
    assert rows[0]['channel_b'] == 'tubulin'


def test_all_to_all_validates_channel_names_length():
    image = np.zeros((3, 4, 4))
    with pytest.raises(ValueError, match='channel_names'):
        analyse_all_to_all(image, channel_axis=0, channel_names=['only_one'])


def test_all_to_all_identical_pair_is_perfect(rng):
    a = rng.random((10, 10))
    image = np.stack([a, a, a], axis=-1)
    rows = analyse_all_to_all(
        image,
        channel_axis=-1,
        metrics=('pcc', 'mcc'),
        threshold_method='manual',
        threshold_a=0.0,
        threshold_b=0.0,
    )
    for row in rows:
        assert row['pcc'] == pytest.approx(1.0)
        assert row['m1'] == pytest.approx(1.0)
        assert row['m2'] == pytest.approx(1.0)


def test_all_to_all_propagates_label_mask():
    a = np.full((6, 6), 1.0)
    image = np.stack([a, a], axis=0)
    label_mask = np.zeros((6, 6), dtype=int)
    label_mask[:3, :] = 1
    label_mask[3:, :] = 2
    rows = analyse_all_to_all(
        image,
        channel_axis=0,
        label_mask=label_mask,
        metrics=('mcc',),
        threshold_method='manual',
        threshold_a=0.5,
        threshold_b=0.5,
    )
    # 1 pair x 2 regions = 2 rows
    assert len(rows) == 2
    for r in rows:
        assert r['n_pixels'] == 18
