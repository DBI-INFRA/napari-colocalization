import numpy as np
import pytest

from napari_colocalization._objects import (
    label_objects,
    nearest_neighbour_distances,
    nearest_neighbour_vectors,
    object_centroids,
    object_table,
)


def _two_blobs():
    """A 20x20 image with two bright square blobs."""
    img = np.zeros((20, 20), dtype=float)
    img[2:6, 2:6] = 1.0
    img[12:16, 12:16] = 1.0
    return img


def test_label_objects():
    img = _two_blobs()
    assert label_objects(img, 'otsu').max() == 2  # two blobs
    speck = img.copy()
    speck[10, 10] = 1.0
    assert label_objects(speck, 'otsu', min_size=4).max() == 2  # speck dropped
    assert label_objects(np.ones((10, 10)), 'otsu').max() == 0  # no objects


def test_object_centroids():
    centroids = object_centroids(label_objects(_two_blobs(), 'otsu'))
    ordered = centroids[np.argsort(centroids[:, 0])]
    assert np.allclose(ordered, [[3.5, 3.5], [13.5, 13.5]])


def test_object_table_coincidence_and_overlap():
    labels = label_objects(_two_blobs(), 'otsu')
    rows, summary = object_table(labels, labels.copy(), 'a', 'b')
    assert len(rows) == 4  # 2 objects x 2 channels
    assert all(r['coincident'] and r['overlap'] for r in rows)
    assert summary['frac_coincident_a'] == pytest.approx(1.0)
    # disjoint objects coincide / overlap with nothing
    a = np.zeros((20, 20), int)
    a[2:6, 2:6] = 1
    b = np.zeros((20, 20), int)
    b[12:16, 12:16] = 1
    _, summary = object_table(a, b)
    assert summary['coincident_a'] == 0 and summary['overlap_a'] == 0


def test_object_overlap_without_coincidence():
    # a ring whose centroid lands in its hole: it overlaps B, but its
    # centroid is not inside any B object
    a = np.zeros((20, 20), int)
    a[4:16, 4:16] = 1
    a[8:12, 8:12] = 0
    b = np.zeros((20, 20), int)
    b[4:7, 4:7] = 1
    a_row = next(
        r for r in object_table(a, b, 'a', 'b')[0] if r['channel'] == 'a'
    )
    assert a_row['overlap'] and not a_row['coincident']


def test_object_table_shape_mismatch_raises():
    with pytest.raises(ValueError, match='shape mismatch'):
        object_table(np.zeros((10, 10), int), np.zeros((10, 11), int))


def test_nearest_neighbour_vectors():
    a = np.array([[0.0, 0.0], [10.0, 10.0]])
    b = np.array([[0.0, 1.0], [10.0, 9.0]])
    vectors = nearest_neighbour_vectors(a, b)
    assert vectors.shape == (2, 2, 2)
    assert np.allclose(vectors[:, 0, :], a)  # vectors start at A centroids
    assert np.allclose(vectors[0, 1, :], [0.0, 1.0])  # point to nearest B
    # no objects in one channel -> no links
    assert nearest_neighbour_vectors(np.empty((0, 2)), b).shape[0] == 0


def test_nearest_neighbour_distances():
    a = np.array([[0.0, 0.0], [10.0, 10.0]])
    b = np.array([[0.0, 1.0], [10.0, 9.0]])
    assert nearest_neighbour_distances(a, b) == pytest.approx([1.0, 1.0])
    # 3-4-5 triangle, to pin that this is a euclidean distance in the
    # full centroid space rather than a per-axis one
    assert nearest_neighbour_distances(
        np.array([[0.0, 0.0]]), np.array([[3.0, 4.0]])
    ) == pytest.approx([5.0])
    # nothing to measure to -> nan per object, not an empty array
    lonely = nearest_neighbour_distances(a, np.empty((0, 2)))
    assert lonely.shape == (2,) and np.all(np.isnan(lonely))
    assert nearest_neighbour_distances(np.empty((0, 2)), b).shape == (0,)


def test_object_table_reports_nn_distance():
    # Disjoint blobs at (3.5, 3.5) and (13.5, 13.5): neither coincides
    # with the other, but the distance says how far apart they are -
    # the graded reading the booleans can't give.
    a = np.zeros((20, 20), int)
    a[2:6, 2:6] = 1
    b = np.zeros((20, 20), int)
    b[12:16, 12:16] = 1
    rows, summary = object_table(a, b, 'a', 'b')
    expected = np.hypot(10.0, 10.0)
    assert all(
        row['nn_distance_px'] == pytest.approx(expected) for row in rows
    )
    assert summary['median_nn_distance_a'] == pytest.approx(expected)
    assert summary['median_nn_distance_b'] == pytest.approx(expected)


def test_object_table_nn_distance_nan_without_a_partner():
    a = np.zeros((20, 20), int)
    a[2:6, 2:6] = 1
    empty = np.zeros((20, 20), int)
    rows, summary = object_table(a, empty, 'a', 'b')
    assert len(rows) == 1
    assert np.isnan(rows[0]['nn_distance_px'])
    assert np.isnan(summary['median_nn_distance_a'])
    assert np.isnan(summary['median_nn_distance_b'])
