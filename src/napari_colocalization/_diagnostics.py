"""Pure-compute colocalization *diagnostics*.

Unlike `_metrics`, whose functions return a scalar per region,
these produce a curve or a distribution - the payloads behind the
widget's Diagnostics tab. As in `_metrics` there are no
napari/qtpy imports here, so every function operates on ndarrays and
can be tested headlessly. The scalar *summaries* (peak shift, ICQ,
p-value) are returned alongside the arrays so the widget can drop them
into a label while the arrays go to the plot.

Failure policy matches the rest of the package: a degenerate *whole
input* (shape mismatch, too few pixels, an out-of-range parameter)
raises ``ValueError`` for the widget to surface as a napari warning,
rather than returning a silently meaningless result.
"""

import numpy as np

from ._metrics import _flatten_with_mask, li_icq, pearson


def van_steensel_ccf(a, b, mask=None, max_shift=20, axis=-1):
    """Van Steensel's cross-correlation function (CCF).

    Shifts ``b`` relative to ``a`` by every integer offset in
    ``[-max_shift, +max_shift]`` along ``axis`` and computes Pearson's
    coefficient at each shift (Van Steensel et al. 1996). Genuinely
    colocalised channels give a CCF that peaks at shift 0; mutually
    excluded channels give a trough at 0 with side peaks.

    Parameters
    ----------
    a, b : array_like
        Same-shape intensity arrays.
    mask : array_like of bool, optional
        Region of interest passed through to `pearson`.
    max_shift : int, default 20
        Largest pixel shift to evaluate in each direction.
    axis : int, default -1
        Axis along which ``b`` is shifted (wraps, ``numpy.roll``).

    Returns
    -------
    shifts : numpy.ndarray
        Integer offsets ``-max_shift … +max_shift``.
    ccf : numpy.ndarray
        Pearson coefficient at each shift; ``nan`` where the shifted
        pair is degenerate.

    Raises
    ------
    ValueError
        If ``a`` and ``b`` differ in shape or ``max_shift < 1``.
    """
    a = np.asarray(a)
    b = np.asarray(b)
    if a.shape != b.shape:
        raise ValueError(f'shape mismatch: {a.shape} vs {b.shape}')
    if max_shift < 1:
        raise ValueError('max_shift must be >= 1')
    shifts = np.arange(-max_shift, max_shift + 1)
    ccf = np.empty(shifts.size, dtype=float)
    for i, shift in enumerate(shifts):
        shifted = np.roll(b, int(shift), axis=axis)
        pcc, _ = pearson(a, shifted, mask=mask)
        ccf[i] = pcc
    return shifts, ccf


def li_ica(a, b, mask=None):
    """Li's Intensity Correlation Analysis (ICA) payload.

    Returns the per-pixel covariance products
    ``P_i = (a_i - mean a)(b_i - mean b)`` together with the paired
    intensities, so the widget can draw the two ICA scatter panels
    (intensity vs ``P``) for channels A and B. The scalar ICQ - the
    fraction of positive ``P`` re-centred to ``[-0.5, 0.5]`` - is
    included for the summary line.

    Parameters
    ----------
    a, b : array_like
        Same-shape intensity arrays.
    mask : array_like of bool, optional
        Region of interest.

    Returns
    -------
    dict
        ``{'a', 'b', 'products', 'icq'}``: the masked, flattened
        channel-A and channel-B intensities, their covariance
        products, and the ICQ scalar.

    Raises
    ------
    ValueError
        If ``a`` and ``b`` differ in shape, or the region has fewer
        than two pixels.
    """
    a = np.asarray(a)
    b = np.asarray(b)
    if a.shape != b.shape:
        raise ValueError(f'shape mismatch: {a.shape} vs {b.shape}')
    a_flat, b_flat = _flatten_with_mask(a, b, mask)
    if a_flat.size < 2:
        raise ValueError('region has fewer than 2 pixels for ICA')
    products = (a_flat - a_flat.mean()) * (b_flat - b_flat.mean())
    return {
        'a': a_flat,
        'b': b_flat,
        'products': products,
        'icq': li_icq(a, b, mask=mask),
    }


def _scramble_blocks(arr, block_size, rng):
    """Permute the block grid of an N-D array (Costes block scramble).

    ``arr`` is assumed already cropped to a whole number of blocks of
    side ``block_size`` in every dimension. Works for any
    dimensionality (2D and 3D in practice).
    """
    ndim = arr.ndim
    nb = [dim // block_size for dim in arr.shape]
    # Split every axis into (block-grid index, within-block index):
    # axes become nb0, s, nb1, s, ... - grid axes are even, within-
    # block axes odd.
    interleaved = arr.reshape([size for n in nb for size in (n, block_size)])
    grid_axes = list(range(0, 2 * ndim, 2))
    block_axes = list(range(1, 2 * ndim, 2))
    perm = grid_axes + block_axes
    moved = interleaved.transpose(perm)
    n_blocks = int(np.prod(nb))
    blocks = moved.reshape((n_blocks, *([block_size] * ndim)))
    shuffled = blocks[rng.permutation(n_blocks)]
    restored = shuffled.reshape((*nb, *([block_size] * ndim)))
    back = restored.transpose(np.argsort(perm))
    return back.reshape([n * block_size for n in nb])


def scramble_example(image, block_size=8, seed=None):
    """One block-scrambled copy of ``image`` for display.

    Produces a single realisation of the randomisation used by
    `costes_randomization`, so the user can see what a
    scrambled channel looks like. The image is cropped to a whole
    number of blocks (any remainder strip is left unchanged).

    Parameters
    ----------
    image : array_like
        2D or 3D intensity array to scramble.
    block_size : int, default 8
        Block side length.
    seed : int, optional
        Seed for reproducibility.

    Returns
    -------
    numpy.ndarray
        Same-shape copy with its block grid permuted.

    Raises
    ------
    ValueError
        If ``block_size`` is larger than the image in any dimension.
    """
    image = np.asarray(image)
    nb = [dim // block_size for dim in image.shape]
    if any(n < 1 for n in nb):
        raise ValueError('block_size is larger than the image')
    crop = tuple(slice(0, n * block_size) for n in nb)
    out = image.copy()
    out[crop] = _scramble_blocks(
        image[crop], block_size, np.random.default_rng(seed)
    )
    return out


def costes_randomization(a, b, mask=None, n_iter=200, block_size=8, seed=None):
    """Costes' randomization significance test for Pearson's PCC.

    Scrambles channel ``b`` in blocks ``n_iter`` times - destroying
    spatial co-occurrence while preserving each channel's intensity
    histogram and local texture - and recomputes the PCC each time to
    build a null distribution. The observed PCC is then placed against
    that null (Costes et al. 2004). A high observed PCC that sits far
    above the null (small p-value) is evidence of genuine
    colocalisation rather than chance overlap.

    Works for 2D or 3D images; the arrays are cropped to a whole
    number of blocks in each dimension before scrambling.

    Parameters
    ----------
    a, b : array_like
        Same-shape 2D or 3D intensity arrays.
    mask : array_like of bool, optional
        Region of interest; cropped alongside ``a``/``b``.
    n_iter : int, default 200
        Number of scrambles forming the null distribution.
    block_size : int, default 8
        Side length (px) of the scrambled blocks. Should approximate
        the point-spread-function width of the acquisition.
    seed : int, optional
        Seed for the random generator, for reproducibility.

    Returns
    -------
    dict
        ``{'observed', 'null', 'p_value', 'z_score'}``: the observed
        PCC over the cropped region, the array of ``n_iter`` null
        PCCs, the right-tailed p-value
        ``(#{null >= observed} + 1) / (n_iter + 1)``, and the z-score
        of the observed value against the null.

    Raises
    ------
    ValueError
        On shape mismatch, ``block_size < 1`` or ``n_iter < 1``, or a
        ``block_size`` larger than the image in any dimension.
    """
    a = np.asarray(a)
    b = np.asarray(b)
    if a.shape != b.shape:
        raise ValueError(f'shape mismatch: {a.shape} vs {b.shape}')
    if block_size < 1:
        raise ValueError('block_size must be >= 1')
    if n_iter < 1:
        raise ValueError('n_iter must be >= 1')
    nb = [dim // block_size for dim in a.shape]
    if any(n < 1 for n in nb):
        raise ValueError('block_size is larger than the image')

    crop = tuple(slice(0, n * block_size) for n in nb)
    a_c = a[crop]
    b_c = b[crop]
    mask_c = None if mask is None else np.asarray(mask)[crop]

    observed, _ = pearson(a_c, b_c, mask=mask_c)
    rng = np.random.default_rng(seed)
    null = np.empty(n_iter, dtype=float)
    for i in range(n_iter):
        scrambled = _scramble_blocks(b_c, block_size, rng)
        pcc, _ = pearson(a_c, scrambled, mask=mask_c)
        null[i] = pcc

    finite = null[np.isfinite(null)]
    if not np.isfinite(observed) or finite.size == 0:
        return {
            'observed': float(observed),
            'null': null,
            'p_value': float('nan'),
            'z_score': float('nan'),
        }
    p_value = (int(np.sum(finite >= observed)) + 1) / (finite.size + 1)
    std = float(finite.std())
    z_score = (
        (float(observed) - float(finite.mean())) / std
        if std > 0
        else float('nan')
    )
    return {
        'observed': float(observed),
        'null': null,
        'p_value': float(p_value),
        'z_score': float(z_score),
    }
