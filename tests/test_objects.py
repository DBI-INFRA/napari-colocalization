import numpy as np
import pytest

from napari_colocalization._objects import (
    label_objects,
    nearest_neighbour_vectors,
    object_centroids,
    object_table,
)


@pytest.fixture
def rng():
    return np.random.default_rng(seed=3)


def _two_blobs():
    """A 20x20 image with two bright square blobs."""
    img = np.zeros((20, 20), dtype=float)
    img[2:6, 2:6] = 1.0
    img[12:16, 12:16] = 1.0
    return img


def test_label_objects_finds_two(rng):
    labels = label_objects(_two_blobs(), threshold_method='otsu')
    assert labels.max() == 2
    assert set(np.unique(labels)) == {0, 1, 2}


def test_label_objects_min_size_filters():
    img = np.zeros((20, 20), dtype=float)
    img[2:6, 2:6] = 1.0  # 16 px
    img[10, 10] = 1.0  # 1 px speck
    labels = label_objects(img, threshold_method='otsu', min_size=4)
    assert labels.max() == 1  # the speck is dropped


def test_label_objects_constant_is_empty():
    labels = label_objects(np.ones((10, 10)), threshold_method='otsu')
    assert labels.max() == 0


def test_object_centroids_locations():
    labels = label_objects(_two_blobs(), threshold_method='otsu')
    centroids = object_centroids(labels)
    assert centroids.shape == (2, 2)
    # blobs span indices 2-5 and 12-15 -> centroids at 3.5 and 13.5
    ordered = centroids[np.argsort(centroids[:, 0])]
    assert np.allclose(ordered, [[3.5, 3.5], [13.5, 13.5]])


def test_object_table_identical_all_coincident():
    labels = label_objects(_two_blobs(), threshold_method='otsu')
    rows, summary = object_table(labels, labels.copy(), 'a', 'b')
    # 2 objects per channel -> 4 rows, all coincident and overlapping
    assert len(rows) == 4
    assert all(r['coincident'] for r in rows)
    assert all(r['overlap'] for r in rows)
    assert summary['n_objects_a'] == 2
    assert summary['frac_coincident_a'] == pytest.approx(1.0)
    assert summary['frac_overlap_b'] == pytest.approx(1.0)


def test_object_table_disjoint_objects():
    a = np.zeros((20, 20), dtype=int)
    a[2:6, 2:6] = 1  # top-left object
    b = np.zeros((20, 20), dtype=int)
    b[12:16, 12:16] = 1  # bottom-right object, no overlap
    rows, summary = object_table(a, b, 'a', 'b')
    assert summary['n_objects_a'] == 1
    assert summary['n_objects_b'] == 1
    assert summary['coincident_a'] == 0
    assert summary['overlap_a'] == 0
    assert all(not r['coincident'] and not r['overlap'] for r in rows)


def test_object_table_overlap_without_coincidence():
    # A is a large ring-ish object; B sits in A's bounding box but A's
    # centroid lands on background -> overlaps but centroid not coincident.
    a = np.zeros((20, 20), dtype=int)
    a[4:16, 4:16] = 1
    a[8:12, 8:12] = 0  # hole at the centre (where the centroid is)
    b = np.zeros((20, 20), dtype=int)
    b[4:7, 4:7] = 1  # overlaps A's top-left, not A's centre
    rows, summary = object_table(a, b, 'a', 'b')
    a_row = next(r for r in rows if r['channel'] == 'a')
    assert a_row['overlap'] is True
    assert a_row['coincident'] is False


def test_object_table_shape_mismatch_raises():
    with pytest.raises(ValueError, match='shape mismatch'):
        object_table(np.zeros((10, 10), int), np.zeros((10, 11), int))


def test_nearest_neighbour_vectors_shape_and_direction():
    a = np.array([[0.0, 0.0], [10.0, 10.0]])
    b = np.array([[0.0, 1.0], [10.0, 9.0]])
    vectors = nearest_neighbour_vectors(a, b)
    assert vectors.shape == (2, 2, 2)
    # start points are the A centroids
    assert np.allclose(vectors[:, 0, :], a)
    # first A maps to nearest B (0, 1): direction (0, 1)
    assert np.allclose(vectors[0, 1, :], [0.0, 1.0])


def test_nearest_neighbour_vectors_empty():
    out = nearest_neighbour_vectors(np.empty((0, 2)), np.array([[1.0, 1.0]]))
    assert out.shape[0] == 0
