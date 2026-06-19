"""Pure-compute object-based colocalization.

Where `_metrics` works on pixel intensities, this works on
*segmented objects*: two labelled images (objects in channel A,
objects in channel B) are compared by

- **centre-particle coincidence** - does an object's centroid fall
  inside an object of the other channel?
- **object overlap** - do an object's pixels touch any object of the
  other channel?

It also exposes object centroids and nearest-neighbour links so the
widget can draw Points / Vectors overlays. As elsewhere there are no
napari/qtpy imports here, so everything operates on ndarrays and is
testable headlessly. Objects can be supplied directly (a Labels
layer) or obtained with `label_objects` (threshold + connected
components).
"""

import numpy as np
from scipy.spatial import cKDTree
from skimage import measure

from ._analysis import _AUTO_THRESHOLDS, _auto_threshold
from ._metrics import _flatten_with_mask

# Per-object result columns (channel + identity + the two measures).
OBJECT_COLUMNS = (
    'channel',
    'object',
    'n_pixels',
    'centroid',
    'coincident',
    'overlap',
)


def label_objects(image, threshold_method='otsu', mask=None, min_size=0):
    """Threshold an intensity image and label its connected components.

    Parameters
    ----------
    image : array_like
        Intensity image (2D or 3D).
    threshold_method : str, default 'otsu'
        Auto-threshold name from the ``skimage.filters.threshold_*``
        family ('otsu', 'li', 'triangle', 'yen', 'mean', 'isodata').
    mask : array_like of bool, optional
        Restrict object detection to this region.
    min_size : int, default 0
        Drop connected components smaller than this many pixels.

    Returns
    -------
    numpy.ndarray of int
        Integer label image (0 = background, 1..N = objects). All
        zeros when the channel has no defined threshold (constant).
    """
    image = np.asarray(image)
    flat, _ = _flatten_with_mask(image, image, mask)
    threshold = _auto_threshold(_AUTO_THRESHOLDS[threshold_method], flat)
    if not np.isfinite(threshold):
        return np.zeros(image.shape, dtype=int)
    binary = image > threshold
    if mask is not None:
        binary &= np.asarray(mask, dtype=bool)
    labels = measure.label(binary)
    if min_size > 0:
        # Drop components with fewer than min_size pixels, then relabel
        # so ids stay contiguous (version-robust; avoids the shifting
        # remove_small_objects signature).
        counts = np.bincount(labels.ravel())
        small = np.where(counts < min_size)[0]
        small = small[small != 0]
        if small.size:
            labels[np.isin(labels, small)] = 0
            labels = measure.label(labels > 0)
    return labels


def object_centroids(labels):
    """``(N, ndim)`` array of object centroids, in axis order."""
    props = measure.regionprops(np.asarray(labels))
    if not props:
        return np.empty((0, np.asarray(labels).ndim), dtype=float)
    return np.array([prop.centroid for prop in props], dtype=float)


def nearest_neighbour_vectors(centroids_a, centroids_b):
    """Vectors from each A centroid to its nearest B centroid.

    Returned as napari Vectors data ``(N, 2, ndim)`` - ``[start,
    direction]`` pairs. Empty when either set has no objects.
    """
    a = np.asarray(centroids_a, dtype=float)
    b = np.asarray(centroids_b, dtype=float)
    if a.size == 0 or b.size == 0:
        ndim = a.shape[1] if a.ndim == 2 and a.shape[1] else 0
        return np.empty((0, 2, ndim), dtype=float)
    _, idx = cKDTree(b).query(a)
    return np.stack([a, b[idx] - a], axis=1)


def _coincident_and_overlap(labels, other):
    """Per-object (coincident, overlap) flags against ``other``."""
    rows = []
    n_coincident = 0
    n_overlap = 0
    for prop in measure.regionprops(labels):
        idx = tuple(
            min(max(int(round(c)), 0), size - 1)
            for c, size in zip(prop.centroid, other.shape, strict=True)
        )
        coincident = bool(other[idx] != 0)
        overlap = bool(np.any(other[tuple(prop.coords.T)] != 0))
        n_coincident += coincident
        n_overlap += overlap
        rows.append(
            {
                'object': int(prop.label),
                'n_pixels': int(prop.area),
                'centroid': tuple(round(float(c), 1) for c in prop.centroid),
                'coincident': coincident,
                'overlap': overlap,
            }
        )
    return rows, n_coincident, n_overlap


def object_table(labels_a, labels_b, name_a='A', name_b='B'):
    """Object-based colocalization between two labelled images.

    Reports, for every object in each channel, whether its centroid
    is **coincident** with an object of the other channel and whether
    it **overlaps** one.

    Parameters
    ----------
    labels_a, labels_b : array_like of int
        Same-shape integer label images (0 = background).
    name_a, name_b : str
        Channel display names written into the ``'channel'`` column.

    Returns
    -------
    rows : list of dict
        One row per object, with the keys in `OBJECT_COLUMNS`.
    summary : dict
        Object counts and coincident/overlap counts and fractions per
        channel.

    Raises
    ------
    ValueError
        If the two label images differ in shape.
    """
    labels_a = np.asarray(labels_a)
    labels_b = np.asarray(labels_b)
    if labels_a.shape != labels_b.shape:
        raise ValueError(
            f'shape mismatch: {labels_a.shape} vs {labels_b.shape}'
        )

    rows = []
    stats = {}
    for labels, other, name in (
        (labels_a, labels_b, name_a),
        (labels_b, labels_a, name_b),
    ):
        obj_rows, n_coincident, n_overlap = _coincident_and_overlap(
            labels, other
        )
        for row in obj_rows:
            row['channel'] = name
        rows.extend(obj_rows)
        stats[name] = (len(obj_rows), n_coincident, n_overlap)

    def _frac(part, whole):
        return part / whole if whole else float('nan')

    n_a, coinc_a, ovl_a = stats[name_a]
    n_b, coinc_b, ovl_b = stats[name_b]
    summary = {
        'n_objects_a': n_a,
        'n_objects_b': n_b,
        'coincident_a': coinc_a,
        'coincident_b': coinc_b,
        'overlap_a': ovl_a,
        'overlap_b': ovl_b,
        'frac_coincident_a': _frac(coinc_a, n_a),
        'frac_coincident_b': _frac(coinc_b, n_b),
        'frac_overlap_a': _frac(ovl_a, n_a),
        'frac_overlap_b': _frac(ovl_b, n_b),
    }
    return rows, summary
