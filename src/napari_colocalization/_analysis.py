"""High-level orchestrators that produce a result row per region.

These accept ndarrays plus a label mask, walk every non-zero
region, compute the requested metrics, and return a list of dicts
suitable for direct insertion into a table or a CSV.
"""

import numpy as np

from ._masking import iter_regions
from ._metrics import costes_threshold, manders, pearson, spearman

ALL_METRICS = ('pcc', 'srcc', 'mcc')
COLUMNS = (
    'region',
    'channel_a',
    'channel_b',
    'n_pixels',
    'pcc',
    'pcc_pvalue',
    'srcc',
    'srcc_pvalue',
    'm1',
    'm2',
    'threshold_a',
    'threshold_b',
)


def _empty_row(region, channel_a, channel_b, n_pixels):
    nan = float('nan')
    return {
        'region': region,
        'channel_a': channel_a,
        'channel_b': channel_b,
        'n_pixels': n_pixels,
        'pcc': nan,
        'pcc_pvalue': nan,
        'srcc': nan,
        'srcc_pvalue': nan,
        'm1': nan,
        'm2': nan,
        'threshold_a': nan,
        'threshold_b': nan,
    }


def _resolve_thresholds(
    a, b, mask, threshold_method, threshold_a, threshold_b
):
    if threshold_method == 'manual':
        if threshold_a is None or threshold_b is None:
            raise ValueError(
                "threshold_method='manual' requires both "
                'threshold_a and threshold_b'
            )
        return float(threshold_a), float(threshold_b)
    if threshold_method == 'costes':
        return costes_threshold(a, b, mask=mask)
    raise ValueError(f"unknown threshold_method '{threshold_method}'")


def analyse_pairwise(
    a,
    b,
    *,
    label_mask=None,
    metrics=ALL_METRICS,
    threshold_method='costes',
    threshold_a=None,
    threshold_b=None,
    channel_a='A',
    channel_b='B',
):
    """Compute selected metrics for each region of two grayscale arrays.

    Walks every non-zero region of ``label_mask`` (or analyses the
    whole image when ``label_mask`` is ``None``) and computes the
    requested metrics within each region. Missing metrics are
    written as ``NaN`` so the output schema is constant.

    Parameters
    ----------
    a, b : numpy.ndarray
        Same-shape intensity arrays (2D or 3D).
    label_mask : numpy.ndarray of int, optional
        Integer label image (0 = background, 1..N = regions).
        ``None`` means analyse the whole image as one region.
    metrics : sequence of {'pcc', 'srcc', 'mcc'}, optional
        Which metrics to compute. Defaults to all three.
    threshold_method : {'costes', 'manual'}, default 'costes'
        Only used when ``'mcc'`` is in ``metrics``. ``'costes'``
        runs :func:`._metrics.costes_threshold` per region;
        ``'manual'`` uses the values supplied below.
    threshold_a, threshold_b : float, optional
        Required when ``threshold_method='manual'``.
    channel_a, channel_b : str, default 'A' and 'B'
        Display names for the two channels — written into the
        result rows so they round-trip into the table and CSV.

    Returns
    -------
    rows : list of dict
        One row per region. Each row has the keys listed in
        :data:`COLUMNS`.

    Raises
    ------
    ValueError
        If ``a`` and ``b`` have different shapes, or if
        ``threshold_method='manual'`` is given without both
        thresholds, or if ``threshold_method`` is unknown.

    Examples
    --------
    >>> import numpy as np
    >>> from napari_colocalization._analysis import analyse_pairwise
    >>> rng = np.random.default_rng(0)
    >>> a = rng.random((32, 32)); b = a.copy()
    >>> rows = analyse_pairwise(a, b, metrics=('pcc',))
    >>> rows[0]['pcc']
    1.0
    """
    a = np.asarray(a)
    b = np.asarray(b)
    if a.shape != b.shape:
        raise ValueError(f'shape mismatch: {a.shape} vs {b.shape}')
    metrics = tuple(metrics)
    rows = []
    for region_id, region_mask in iter_regions(label_mask):
        if region_mask is None:
            n_pixels = int(a.size)
        else:
            n_pixels = int(region_mask.sum())
        row = _empty_row(region_id, channel_a, channel_b, n_pixels)
        if 'pcc' in metrics:
            pcc, pval = pearson(a, b, mask=region_mask)
            row['pcc'] = pcc
            row['pcc_pvalue'] = pval
        if 'srcc' in metrics:
            rho, pval = spearman(a, b, mask=region_mask)
            row['srcc'] = rho
            row['srcc_pvalue'] = pval
        if 'mcc' in metrics:
            t_a, t_b = _resolve_thresholds(
                a,
                b,
                region_mask,
                threshold_method,
                threshold_a,
                threshold_b,
            )
            m1, m2 = manders(a, b, t_a, t_b, mask=region_mask)
            row['m1'] = m1
            row['m2'] = m2
            row['threshold_a'] = t_a
            row['threshold_b'] = t_b
        rows.append(row)
    return rows


def analyse_all_to_all(
    image,
    channel_axis,
    *,
    label_mask=None,
    metrics=ALL_METRICS,
    threshold_method='costes',
    threshold_a=None,
    threshold_b=None,
    channel_names=None,
):
    """Compute metrics for every channel pair (i, j), i < j.

    Iterates over each unordered pair of channels along
    ``channel_axis`` and dispatches to :func:`analyse_pairwise`.
    For ``N`` channels this produces ``N*(N-1)/2`` × R rows,
    where R is the number of non-zero regions in ``label_mask``
    (or 1 for the whole image).

    Parameters
    ----------
    image : numpy.ndarray
        N-dimensional array with one channel axis. The remaining
        axes are spatial (the plugin supports up to 3 spatial
        axes, i.e. 4D total).
    channel_axis : int
        Axis index along which channels are enumerated.
    label_mask : numpy.ndarray of int, optional
        Integer label image whose shape matches ``image`` with
        ``channel_axis`` removed.
    metrics : sequence of {'pcc', 'srcc', 'mcc'}, optional
        Forwarded to :func:`analyse_pairwise`.
    threshold_method : {'costes', 'manual'}, default 'costes'
        Forwarded to :func:`analyse_pairwise`. Manual thresholds
        apply identically to every channel pair.
    threshold_a, threshold_b : float, optional
        Forwarded to :func:`analyse_pairwise`.
    channel_names : sequence of str, optional
        One name per channel along ``channel_axis``. Defaults to
        ``['c0', 'c1', ...]``.

    Returns
    -------
    rows : list of dict
        Concatenation of the per-pair results from
        :func:`analyse_pairwise`.

    Raises
    ------
    ValueError
        If ``len(channel_names)`` does not match the channel
        count.
    """
    image = np.asarray(image)
    n_channels = image.shape[channel_axis]
    if channel_names is None:
        channel_names = [f'c{i}' for i in range(n_channels)]
    if len(channel_names) != n_channels:
        raise ValueError(
            f'len(channel_names)={len(channel_names)} does not '
            f'match image.shape[{channel_axis}]={n_channels}'
        )

    rows = []
    for i in range(n_channels):
        a = np.take(image, i, axis=channel_axis)
        for j in range(i + 1, n_channels):
            b = np.take(image, j, axis=channel_axis)
            rows.extend(
                analyse_pairwise(
                    a,
                    b,
                    label_mask=label_mask,
                    metrics=metrics,
                    threshold_method=threshold_method,
                    threshold_a=threshold_a,
                    threshold_b=threshold_b,
                    channel_a=channel_names[i],
                    channel_b=channel_names[j],
                )
            )
    return rows
