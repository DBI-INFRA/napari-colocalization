import io
import zipfile

import numpy as np
import pytest

from napari_colocalization import _sample_data
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
    # Downloads from the Fiji sample server unless already cached.
    try:
        layers = make_sample_data_coloc()
    except OSError as exc:
        pytest.skip(f'sample server unreachable: {exc}')
    assert len(layers) == 2
    for data, meta, layer_type in layers:
        assert layer_type == 'image'
        assert isinstance(meta, dict)
        assert isinstance(data, np.ndarray)
    a, b = layers[0][0], layers[1][0]
    pcc, _ = pearson(a, b)
    assert pcc > 0.5


class _FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


@pytest.fixture
def fake_server(tmp_path, monkeypatch):
    """Serve ``payloads[url]`` from urlopen, caching into ``tmp_path``."""
    payloads = {}
    calls = []

    def urlopen(url, timeout=None):
        calls.append((url, timeout))
        payload = payloads[url]
        if isinstance(payload, Exception):
            raise payload
        return _FakeResponse(payload)

    monkeypatch.setattr(_sample_data, '_CACHE_DIR', tmp_path)
    monkeypatch.setattr(_sample_data.urllib.request, 'urlopen', urlopen)
    return payloads, calls


def test_cached_download_fetches_once_with_timeout(fake_server, tmp_path):
    payloads, calls = fake_server
    payloads['u'] = b'data'
    path = _sample_data._cached_download('u', 'f.bin')
    assert path.read_bytes() == b'data'
    assert _sample_data._cached_download('u', 'f.bin') == path
    assert len(calls) == 1
    assert calls[0][1] is not None  # never wait on the server forever


def test_cached_download_extracts_zip_member(fake_server):
    payloads, _ = fake_server
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w') as zf:
        zf.writestr('inner.tif', b'tiff')
    payloads['z'] = buffer.getvalue()
    path = _sample_data._cached_download('z', 'inner.tif', member='inner.tif')
    assert path.read_bytes() == b'tiff'


def test_interrupted_write_leaves_no_cache_entry(
    fake_server, tmp_path, monkeypatch
):
    payloads, _ = fake_server
    payloads['u'] = b'0123456789'

    def write_half_then_fail(self, data):
        with open(self, 'wb') as fh:
            fh.write(data[: len(data) // 2])
        raise OSError('disk full')

    monkeypatch.setattr(_sample_data.Path, 'write_bytes', write_half_then_fail)
    with pytest.raises(OSError, match='disk full'):
        _sample_data._cached_download('u', 'f.bin')
    # Nothing a later call could mistake for a finished download.
    assert list(tmp_path.iterdir()) == []
