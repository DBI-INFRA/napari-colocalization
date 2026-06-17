import numpy as np

from napari_colocalization._metrics import manders, pearson
from napari_colocalization._sample_data import (
    make_sample_data,
    make_sample_data_3d,
    make_sample_data_coloc,
)


def test_sample_data_returns_two_image_tuples():
    layers = make_sample_data()
    assert len(layers) == 2
    for data, meta, layer_type in layers:
        assert layer_type == 'image'
        assert isinstance(meta, dict)
        assert isinstance(data, np.ndarray)
        assert data.ndim == 2


def test_sample_data_channels_have_matching_shape():
    layers = make_sample_data()
    a, b = layers[0][0], layers[1][0]
    assert a.shape == b.shape


def test_sample_data_pcc_is_in_partially_colocalised_band():
    layers = make_sample_data()
    a, b = layers[0][0], layers[1][0]
    pcc, _ = pearson(a, b)
    assert 0.4 < pcc < 0.95


def test_sample_data_manders_in_partial_band():
    layers = make_sample_data()
    a, b = layers[0][0], layers[1][0]
    threshold = 0.1
    m1, m2 = manders(a, b, threshold_a=threshold, threshold_b=threshold)
    assert 0.3 < m1 < 0.95
    assert 0.3 < m2 < 0.95


def test_sample_data_3d_returns_3d_volumes():
    layers = make_sample_data_3d()
    assert len(layers) == 2
    for data, meta, layer_type in layers:
        assert layer_type == 'image'
        assert isinstance(meta, dict)
        assert isinstance(data, np.ndarray)
        assert data.ndim == 3
    a, b = layers[0][0], layers[1][0]
    assert a.shape == b.shape


def test_sample_data_3d_pcc_is_in_partially_colocalised_band():
    layers = make_sample_data_3d()
    a, b = layers[0][0], layers[1][0]
    pcc, _ = pearson(a, b)
    assert 0.4 < pcc < 0.95

def test_make_sample_data_coloc_channels_are_colocalised():
    layers = make_sample_data_coloc()
    assert len(layers) == 2
    for data, meta, layer_type in layers:
        assert layer_type == 'image'
        assert isinstance(meta, dict)
        assert isinstance(data, np.ndarray)
    a, b = layers[0][0], layers[1][0]
    pcc, _ = pearson(a, b)
    assert pcc > 0.5
