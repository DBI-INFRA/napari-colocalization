"""High-level orchestrators that produce a result row per region.

These accept ndarrays plus a label mask, walk every non-zero
region, compute the requested metrics, and return a list of dicts
suitable for direct insertion into a table or a CSV.

Failure policy: conditions that make the *whole* request invalid
(shape mismatch, unknown options, missing manual thresholds) raise
``ValueError`` so the caller can surface the reason. Per-region
degeneracies — a single ROI too small or with a constant/empty
channel — are *not* errors in a multi-region run; the affected
metrics stay ``NaN`` (a visible blank cell), and a human-readable
reason is appended to the optional ``region_warnings`` collector so
the caller can summarise how many regions were skipped and why.
"""

import numpy as np
from skimage import filters

from ._masking import iter_regions
from ._metrics import (
    _flatten_with_mask,
    costes_threshold,
    li_icq,
    manders,
    overlap,
    pearson,
    spearman,
)

# Which result columns carry each metric's value(s); used to detect
# when a requested metric came back NaN for a region.
_METRIC_VALUE_KEYS = {
    'pcc': ('pcc',),
    'srcc': ('srcc',),
    'icq': ('icq',),
    'overlap': ('overlap', 'k1', 'k2'),
    'mcc': ('m1', 'm2'),
}

ALL_METRICS = ('pcc', 'srcc', 'icq', 'overlap', 'mcc')
COLUMNS = (
    'region',
    'channel_a',
    'channel_b',
    'n_pixels',
    'pcc',
    'pcc_pvalue',
    'srcc',
    'srcc_pvalue',
    'icq',
    'overlap',
    'k1',
    'k2',
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
        'icq': nan,
        'overlap': nan,
        'k1': nan,
        'k2': nan,
        'm1': nan,
        'm2': nan,
        'threshold_a': nan,
        'threshold_b': nan,
    }


# Per-channel automatic thresholds (à la JACoP B / ImageJ Auto
# Threshold): each channel is thresholded independently from its own
# intensity histogram, giving the thresholded Manders tM1/tM2.
_AUTO_THRESHOLDS = {
    'otsu': filters.threshold_otsu,
    'li': filters.threshold_li,
    'triangle': filters.threshold_triangle,
    'yen': filters.threshold_yen,
    'mean': filters.threshold_mean,
    'isodata': filters.threshold_isodata,
}


def _auto_threshold(func, flat):
    """Apply a skimage threshold to one channel's masked pixels.

    Returns ``nan`` for a constant/empty channel, where the
    threshold is undefined (so :func:`._metrics.manders` reports
    NaN rather than a spurious 0).
    """
    if flat.size == 0 or np.ptp(flat) == 0:
        return float('nan')
    try:
        return float(func(flat))
    except (ValueError, RuntimeError):
        return float('nan')


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
    if threshold_method in _AUTO_THRESHOLDS:
        func = _AUTO_THRESHOLDS[threshold_method]
        a_flat, b_flat = _flatten_with_mask(a, b, mask)
        return _auto_threshold(func, a_flat), _auto_threshold(func, b_flat)
    raise ValueError(f"unknown threshold_method '{threshold_method}'")


def _row_has_uncomputed(row, metrics):
    """True if any requested metric is NaN in ``row``."""
    for metric in metrics:
        for key in _METRIC_VALUE_KEYS.get(metric, ()):
            value = row.get(key)
            if isinstance(value, float) and np.isnan(value):
                return True
    return False


def _describe_channel(flat, name):
    """Reason a constant/empty channel makes metrics undefined."""
    if flat.std() != 0:
        return None
    if flat.sum() == 0:
        return f"channel '{name}' has no signal"
    return f"channel '{name}' is constant"


def _skip_reason(a, b, region_mask, channel_a, channel_b):
    """Human-readable reason a region's metrics could not compute."""
    a_flat, b_flat = _flatten_with_mask(a, b, region_mask)
    if a_flat.size < 2:
        return 'fewer than 2 pixels'
    parts = [
        part
        for part in (
            _describe_channel(a_flat, channel_a),
            _describe_channel(b_flat, channel_b),
        )
        if part
    ]
    return '; '.join(parts) if parts else 'metric undefined for this region'


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
    region_warnings=None,
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
    metrics : sequence of {'pcc', 'srcc', 'icq', 'overlap', 'mcc'}, optional
        Which metrics to compute. Defaults to all. ``'overlap'``
        adds the threshold-free overlap coefficient and the split
        coefficients k1/k2.
    threshold_method : str, default 'costes'
        Only used when ``'mcc'`` is in ``metrics``. ``'costes'``
        runs :func:`._metrics.costes_threshold` per region;
        ``'manual'`` uses the values supplied below; or a per-channel
        auto-threshold name — one of ``'otsu'``, ``'li'``,
        ``'triangle'``, ``'yen'``, ``'mean'``, ``'isodata'`` (the
        ``skimage.filters.threshold_*`` family, thresholding each
        channel independently → Manders tM1/tM2).
    threshold_a, threshold_b : float, optional
        Required when ``threshold_method='manual'``.
    channel_a, channel_b : str, default 'A' and 'B'
        Display names for the two channels — written into the
        result rows so they round-trip into the table and CSV.
    region_warnings : list, optional
        If provided, a human-readable string is appended for each
        region where a *requested* metric could not be computed
        (too few pixels, or a constant/empty channel). The metric
        cell itself stays ``NaN``; this is the channel for telling
        the user why. ``None`` (default) discards the reasons.

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
        if 'icq' in metrics:
            row['icq'] = li_icq(a, b, mask=region_mask)
        if 'overlap' in metrics:
            r, k1, k2 = overlap(a, b, mask=region_mask)
            row['overlap'] = r
            row['k1'] = k1
            row['k2'] = k2
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
        if region_warnings is not None and _row_has_uncomputed(row, metrics):
            reason = _skip_reason(a, b, region_mask, channel_a, channel_b)
            region_warnings.append(
                f'region {region_id} ({channel_a} vs {channel_b}): {reason}'
            )
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
    region_warnings=None,
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
    metrics : sequence of {'pcc', 'srcc', 'icq', 'overlap', 'mcc'}, optional
        Forwarded to :func:`analyse_pairwise`.
    threshold_method : str, default 'costes'
        Forwarded to :func:`analyse_pairwise` ('costes', 'manual',
        or an auto-threshold name like 'otsu'). Manual thresholds
        apply identically to every channel pair.
    threshold_a, threshold_b : float, optional
        Forwarded to :func:`analyse_pairwise`.
    channel_names : sequence of str, optional
        One name per channel along ``channel_axis``. Defaults to
        ``['c0', 'c1', ...]``.
    region_warnings : list, optional
        Forwarded to :func:`analyse_pairwise`; collects per-region
        reasons across every channel pair.

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
                    region_warnings=region_warnings,
                )
            )
    return rows
