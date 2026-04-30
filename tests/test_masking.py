import types

import numpy as np
import pytest

from napari_colocalization._masking import (
    iter_regions,
    labels_to_label_mask,
    shapes_to_label_mask,
)


def _stub_shapes(rasterised):
    """Mock object exposing only the napari Shapes API we use."""

    def to_labels(labels_shape):
        assert tuple(labels_shape) == rasterised.shape
        return rasterised

    return types.SimpleNamespace(to_labels=to_labels)


def _stub_labels(data):
    return types.SimpleNamespace(data=data)


def test_shapes_to_label_mask_returns_int_array():
    rasterised = np.zeros((10, 10), dtype=int)
    rasterised[2:5, 2:5] = 1
    rasterised[6:9, 6:9] = 2
    layer = _stub_shapes(rasterised)
    out = shapes_to_label_mask(layer, (10, 10))
    np.testing.assert_array_equal(out, rasterised)
    assert out.dtype.kind == 'i'


def test_labels_to_label_mask_passes_through_matching_shape():
    data = np.zeros((8, 8), dtype=np.uint16)
    data[1:3, 1:3] = 5
    out = labels_to_label_mask(_stub_labels(data), (8, 8))
    np.testing.assert_array_equal(out, data)


def test_labels_to_label_mask_rejects_shape_mismatch():
    data = np.zeros((8, 8), dtype=int)
    with pytest.raises(ValueError, match='does not match'):
        labels_to_label_mask(_stub_labels(data), (10, 10))


def test_iter_regions_none_yields_whole_image():
    out = list(iter_regions(None))
    assert out == [(0, None)]


def test_iter_regions_yields_one_per_label():
    mask = np.zeros((6, 6), dtype=int)
    mask[0:2, 0:2] = 1
    mask[3:5, 3:5] = 2
    mask[5:6, 5:6] = 3
    regions = list(iter_regions(mask))
    assert [r[0] for r in regions] == [1, 2, 3]
    assert regions[0][1].sum() == 4
    assert regions[1][1].sum() == 4
    assert regions[2][1].sum() == 1


def test_iter_regions_skips_zero_label():
    mask = np.zeros((4, 4), dtype=int)
    mask[1:3, 1:3] = 7
    regions = list(iter_regions(mask))
    assert len(regions) == 1
    assert regions[0][0] == 7


def test_iter_regions_handles_3d_mask():
    mask = np.zeros((4, 4, 4), dtype=int)
    mask[1:3, 1:3, 1:3] = 1
    regions = list(iter_regions(mask))
    assert len(regions) == 1
    label, bool_mask = regions[0]
    assert label == 1
    assert bool_mask.shape == (4, 4, 4)
    assert int(bool_mask.sum()) == 8
