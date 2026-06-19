"""Synthetic two-channel sample data for colocalization.

Synthetic samples (2D, 3D): gaussian blobs with channel B copying
60 % of channel A's blobs plus independent blobs and noise, giving
PCC ~ 0.7 and Manders M1/M2 ~ 0.6.

CBS006RBM: red and blue channels with ~50 % colocalization,
from the Colocalization Benchmark Source.
The TIFF is downloaded once and cached in ~/.cache/napari-colocalization/.

colocsample1bRGB: a confocal Z-stack (33 x 152 x 172) with red and
green dyes that strongly colocalise. Downloaded once from the Fiji
sample server and cached under ~/.cache/napari-colocalization/; this
is the image used by the ROI colocalization tutorial.
"""

import numpy as np


def _gaussian_blob_nd(shape, center, sigma, amplitude):
    """Render a single isotropic gaussian blob in any dimension."""
    grids = np.indices(shape, dtype=np.float32)
    sq = np.zeros(shape, dtype=np.float32)
    for axis_grid, c in zip(grids, center, strict=True):
        sq += (axis_grid - c) ** 2
    return amplitude * np.exp(-sq / (2.0 * sigma**2))


def _render_blobs(shape, centers, sigma, amplitudes):
    out = np.zeros(shape, dtype=np.float32)
    for center, amp in zip(centers, amplitudes, strict=True):
        out += _gaussian_blob_nd(shape, center, sigma, amp)
    return out


def _make_two_channels(shape, *, n_shared, n_b_only, sigma, margin, seed):
    """Generate (channel_a, channel_b) ndarrays for any spatial shape."""
    rng = np.random.default_rng(seed=seed)
    ndim = len(shape)

    def random_centers(n):
        return np.stack(
            [
                rng.integers(margin, shape[d] - margin, size=n)
                for d in range(ndim)
            ],
            axis=1,
        )

    shared_centers = random_centers(n_shared)
    shared_amps = rng.uniform(0.6, 1.0, size=n_shared).astype(np.float32)
    a = _render_blobs(shape, shared_centers, sigma, shared_amps)

    b_only_centers = random_centers(n_b_only)
    b_only_amps = rng.uniform(0.6, 1.0, size=n_b_only).astype(np.float32)
    keep = rng.choice(n_shared, size=int(0.6 * n_shared), replace=False)
    b = _render_blobs(
        shape, shared_centers[keep], sigma, shared_amps[keep]
    ) + _render_blobs(shape, b_only_centers, sigma, b_only_amps)

    noise_a = 0.02 * rng.standard_normal(shape).astype(np.float32)
    noise_b = 0.02 * rng.standard_normal(shape).astype(np.float32)
    a = np.clip(a + noise_a, 0.0, None)
    b = np.clip(b + noise_b, 0.0, None)
    return a, b


def make_sample_data():
    """2D sample: 256x256 channels of gaussian blobs."""
    a, b = _make_two_channels(
        (256, 256),
        n_shared=30,
        n_b_only=12,
        sigma=6.0,
        margin=16,
        seed=2026,
    )
    return [
        (a, {'name': 'channel_a', 'colormap': 'green'}, 'image'),
        (
            b,
            {
                'name': 'channel_b',
                'colormap': 'magenta',
                'blending': 'additive',
            },
            'image',
        ),
    ]


def make_sample_data_3d():
    """3D sample: 32x128x128 (Z, Y, X) volumes of gaussian blobs."""
    a, b = _make_two_channels(
        (32, 128, 128),
        n_shared=40,
        n_b_only=16,
        sigma=4.0,
        margin=8,
        seed=2026,
    )
    return [
        (a, {'name': 'channel_a_3d', 'colormap': 'green'}, 'image'),
        (
            b,
            {
                'name': 'channel_b_3d',
                'colormap': 'magenta',
                'blending': 'additive',
            },
            'image',
        ),
    ]


def make_sample_data_cbs006rbm():
    """2D sample from the Colocalization Benchmark Source (CBS006RBM).

    Red and blue channels with ~50 % colocalization.
    Downloaded once and cached under ~/.cache/napari-colocalization/.

    Source: https://colocalization-benchmark.com/
    """
    import io
    import urllib.request
    import zipfile
    from pathlib import Path

    from skimage.io import imread

    _URL = 'https://colocalization-benchmark.com/download/cbs006rbm.tiff.zip'
    _TIFF = 'CBS006RBM.tiff'

    cache_dir = Path.home() / '.cache' / 'napari-colocalization'
    cache_dir.mkdir(parents=True, exist_ok=True)
    tiff_path = cache_dir / _TIFF

    if not tiff_path.exists():
        with urllib.request.urlopen(_URL) as response:
            raw = response.read()
        with zipfile.ZipFile(io.BytesIO(raw)) as zf:
            tiff_path.write_bytes(zf.read(_TIFF))

    img = imread(str(tiff_path))  # (H, W, 3) uint8 RGB
    red = img[:, :, 0].astype(np.float32) / 255.0
    blue = img[:, :, 2].astype(np.float32) / 255.0

    return [
        (red, {'name': 'CBS006RBM_red', 'colormap': 'red'}, 'image'),
        (
            blue,
            {
                'name': 'CBS006RBM_blue',
                'colormap': 'blue',
                'blending': 'additive',
            },
            'image',
        ),
    ]


_COLOC_URL = 'https://samples.fiji.sc/colocsample1bRGB_BG.tif'
_COLOC_TIFF = 'colocsample1bRGB_BG.tif'


def coloc_sample_path():
    """Download (once) and return the cached ``colocsample1bRGB_BG.tif``.

    The TIFF is fetched from the Fiji sample server on first use and
    cached under ``~/.cache/napari-colocalization/``; subsequent calls
    reuse the cached copy.

    Returns
    -------
    path : pathlib.Path
        Filesystem path to the cached multi-channel TIFF.
    """
    import urllib.request
    from pathlib import Path

    cache_dir = Path.home() / '.cache' / 'napari-colocalization'
    cache_dir.mkdir(parents=True, exist_ok=True)
    tiff_path = cache_dir / _COLOC_TIFF

    if not tiff_path.exists():
        with urllib.request.urlopen(_COLOC_URL) as response:
            tiff_path.write_bytes(response.read())
    return tiff_path


def load_coloc_sample():
    """Load the colocalization sample as ``(red, green)`` arrays.

    Reads the confocal Z-stack (downloading and caching it on first use,
    see `coloc_sample_path`) and returns its red and green channels
    as separate ``float32`` volumes scaled to ``[0, 1]``. The (empty)
    blue channel is dropped. These two strongly colocalising dyes are the
    starting point for the ROI colocalization tutorial.

    Returns
    -------
    red, green : numpy.ndarray
        Same-shape ``(Z, Y, X)`` float32 arrays in ``[0, 1]``.
    """
    from skimage.io import imread

    img = imread(str(coloc_sample_path()))  # (Z, Y, X, 3) uint8 RGB
    red = img[..., 0].astype(np.float32) / 255.0
    green = img[..., 1].astype(np.float32) / 255.0
    return red, green


def make_sample_data_coloc():
    """2D-stack sample: red and green confocal dyes that colocalise.

    The ``colocsample1bRGB_BG.tif`` confocal Z-stack (33 x 152 x 172),
    downloaded once from the Fiji sample server and cached locally, split
    into its red and green channels. The two dyes strongly colocalise
    (whole-stack PCC ~ 0.75).
    """
    red, green = load_coloc_sample()
    return [
        (red, {'name': 'coloc_red', 'colormap': 'red'}, 'image'),
        (
            green,
            {
                'name': 'coloc_green',
                'colormap': 'green',
                'blending': 'additive',
            },
            'image',
        ),
    ]
