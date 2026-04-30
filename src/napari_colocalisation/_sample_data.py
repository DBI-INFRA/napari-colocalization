"""Synthetic two-channel sample data with structured colocalization.

The 2D and 3D samples follow the same recipe: gaussian blobs at
random positions, with channel B copying 60% of channel A's blobs
plus a few independent blobs and gaussian noise. This gives a
realistic "partially co-occurring" signal with PCC ~ 0.7 and
Manders M1, M2 ~ 0.6.
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
