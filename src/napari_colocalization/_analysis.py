"""High-level orchestrators that produce a result row per region.

These accept ndarrays plus a label mask, walk every non-zero
region, compute the requested metrics, and return a list of dicts
suitable for direct insertion into a table or a CSV.

Failure policy: conditions that make the *whole* request invalid
(shape mismatch, unknown options, missing manual thresholds) raise
``ValueError`` so the caller can surface the reason. Per-region
degeneracies - a single ROI too small or with a constant/empty
channel - are *not* errors in a multi-region run; the affected
metrics stay ``NaN`` (a visible blank cell), and a human-readable
reason is appended to the optional ``region_warnings`` collector so
the caller can summarise how many regions were skipped and why.
"""

import numpy as np
from skimage import filters

from ._masking import iter_region_crops
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
    'mcc': ('tm1', 'tm2'),
}

ALL_METRICS = ('pcc', 'srcc', 'icq', 'overlap', 'mcc')
COLUMNS = (
    'region',
    'slice',
    'channel_a',
    'channel_b',
    'n_pixels',
    'pcc',
    'srcc',
    'icq',
    'overlap',
    'k1',
    'k2',
    'tm1',
    'tm2',
    'threshold_a',
    'threshold_b',
)


def _empty_row(region, channel_a, channel_b, n_pixels):
    nan = float('nan')
    return {
        'region': region,
        'slice': nan,
        'channel_a': channel_a,
        'channel_b': channel_b,
        'n_pixels': n_pixels,
        'pcc': nan,
        'srcc': nan,
        'icq': nan,
        'overlap': nan,
        'k1': nan,
        'k2': nan,
        'tm1': nan,
        'tm2': nan,
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
    threshold is undefined (so `manders` reports
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


def _skip_reason(a, b, region_mask, channel_a, channel_b, row, metrics):
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
    if parts:
        return '; '.join(parts)
    # Both channels carry signal, so the only remaining known cause is
    # a threshold that could not be fitted (e.g. Costes on a pair with
    # a non-positive regression slope). Naming it matters: without a
    # threshold tM1/tM2 are *unmeasured*, not measured as zero.
    if 'mcc' in metrics and not (
        np.isfinite(row['threshold_a']) and np.isfinite(row['threshold_b'])
    ):
        return (
            'no Manders threshold could be determined '
            '(try a different threshold method)'
        )
    return 'metric undefined for this region'


def _iter_region_inputs(a, b, label_mask):
    """Yield ``(label_id, a_crop, b_crop, mask_crop, n_pixels)``.

    Each region is cropped to its bounding box so the per-region
    cost scales with the region rather than with the image. The
    metrics only ever read ``a[mask]`` / ``b[mask]``, so cropping
    selects the same pixels in the same order. ``label_mask=None``
    means the whole image, passed through uncropped and unmasked.
    """
    if label_mask is None:
        yield 0, a, b, None, int(a.size)
        return
    for label, bbox, region_mask in iter_region_crops(label_mask):
        yield (
            label,
            a[bbox],
            b[bbox],
            region_mask,
            int(region_mask.sum()),
        )


def _count_regions(label_mask):
    """Number of non-zero regions in ``label_mask`` (1 for ``None``)."""
    if label_mask is None:
        return 1
    labels = np.unique(np.asarray(label_mask))
    return int(np.count_nonzero(labels))


def _pairwise_row_count(shape, label_mask=None, slice_axis=None):
    """How many rows `analyse_pairwise` will produce for these inputs.

    Counts labels instead of computing anything, so a caller can size
    a progress report before the work starts.
    """
    if slice_axis is None:
        return _count_regions(label_mask)
    n_slices = shape[slice_axis]
    if label_mask is None:
        return n_slices
    lm = np.asarray(label_mask)
    if lm.ndim != len(shape):
        return _count_regions(lm) * n_slices
    # A full-volume mask need not carry every region on every plane.
    return sum(
        _count_regions(np.take(lm, s, axis=slice_axis))
        for s in range(n_slices)
    )


def _all_to_all_row_count(n_channels, label_mask=None):
    """How many rows `analyse_all_to_all` will produce."""
    n_pairs = n_channels * (n_channels - 1) // 2
    return n_pairs * _count_regions(label_mask)


def _iter_pairwise_rows(
    a,
    b,
    *,
    label_mask,
    metrics,
    threshold_method,
    threshold_a,
    threshold_b,
    channel_a,
    channel_b,
    slice_axis,
    region_warnings,
):
    """Yield the result rows of `analyse_pairwise`, one at a time.

    Inputs are assumed already validated by the caller. Splitting the
    loop out lets the widget consume rows as they arrive, so a run can
    report progress and be cancelled between regions.
    """
    a = np.asarray(a)
    b = np.asarray(b)
    if slice_axis is not None:
        lm = None if label_mask is None else np.asarray(label_mask)
        for s in range(a.shape[slice_axis]):
            if lm is None:
                mask_s = None
            elif lm.ndim == a.ndim:
                mask_s = np.take(lm, s, axis=slice_axis)
            else:
                mask_s = lm
            slice_rows = _iter_pairwise_rows(
                np.take(a, s, axis=slice_axis),
                np.take(b, s, axis=slice_axis),
                label_mask=mask_s,
                metrics=metrics,
                threshold_method=threshold_method,
                threshold_a=threshold_a,
                threshold_b=threshold_b,
                channel_a=channel_a,
                channel_b=channel_b,
                slice_axis=None,
                region_warnings=region_warnings,
            )
            for row in slice_rows:
                row['slice'] = s
                yield row
        return

    for region in _iter_region_inputs(a, b, label_mask):
        region_id, a_r, b_r, region_mask, n_pixels = region
        row = _empty_row(region_id, channel_a, channel_b, n_pixels)
        # The p-values `pearson`/`spearman` return are discarded: they
        # assume independent samples, which spatially autocorrelated
        # pixels are not. The Costes randomization test in _diagnostics
        # is the significance test for these images.
        if 'pcc' in metrics:
            row['pcc'] = pearson(a_r, b_r, mask=region_mask)[0]
        if 'srcc' in metrics:
            row['srcc'] = spearman(a_r, b_r, mask=region_mask)[0]
        if 'icq' in metrics:
            row['icq'] = li_icq(a_r, b_r, mask=region_mask)
        if 'overlap' in metrics:
            r, k1, k2 = overlap(a_r, b_r, mask=region_mask)
            row['overlap'] = r
            row['k1'] = k1
            row['k2'] = k2
        if 'mcc' in metrics:
            t_a, t_b = _resolve_thresholds(
                a_r,
                b_r,
                region_mask,
                threshold_method,
                threshold_a,
                threshold_b,
            )
            m1, m2 = manders(a_r, b_r, t_a, t_b, mask=region_mask)
            row['tm1'] = m1
            row['tm2'] = m2
            row['threshold_a'] = t_a
            row['threshold_b'] = t_b
        if region_warnings is not None and _row_has_uncomputed(row, metrics):
            reason = _skip_reason(
                a_r, b_r, region_mask, channel_a, channel_b, row, metrics
            )
            region_warnings.append(
                f'region {region_id} ({channel_a} vs {channel_b}): {reason}'
            )
        yield row


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
    slice_axis=None,
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
        Validated always, but only used when ``'mcc'`` is in
        ``metrics``. ``'costes'``
        runs `costes_threshold` per region;
        ``'manual'`` uses the values supplied below; or a per-channel
        auto-threshold name - one of ``'otsu'``, ``'li'``,
        ``'triangle'``, ``'yen'``, ``'mean'``, ``'isodata'`` (the
        ``skimage.filters.threshold_*`` family, thresholding each
        channel independently → Manders tM1/tM2).
    threshold_a, threshold_b : float, optional
        Required when ``threshold_method='manual'``.
    channel_a, channel_b : str, default 'A' and 'B'
        Display names for the two channels - written into the
        result rows so they round-trip into the table and CSV.
    slice_axis : int, optional
        If given, analyse each plane along this axis **separately**
        (JACoP B's "consider Z slices separately") rather than the
        whole volume: every region is reported once per slice, with
        the plane index in the ``'slice'`` column. A ``label_mask``
        matching the full volume is sliced alongside; a
        slice-shaped mask is reused for every plane.
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
        `COLUMNS`.

    Raises
    ------
    ValueError
        If ``a`` and ``b`` have different shapes, if ``label_mask``
        does not match their shape (or, with ``slice_axis``, the
        per-slice shape), or if ``threshold_method='manual'`` is
        given without both thresholds, or if ``threshold_method``
        is unknown.

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
    if label_mask is not None:
        label_mask = np.asarray(label_mask)
        accepted = [a.shape]
        if slice_axis is not None:
            slice_shape = list(a.shape)
            del slice_shape[slice_axis]
            accepted.append(tuple(slice_shape))
        if label_mask.shape not in accepted:
            raise ValueError(
                f'label_mask shape {label_mask.shape} does not match '
                f'the image shape {a.shape}'
                + (
                    f' (or the per-slice shape {accepted[1]})'
                    if slice_axis is not None
                    else ''
                )
            )
    if threshold_method not in ('manual', 'costes', *_AUTO_THRESHOLDS):
        raise ValueError(f"unknown threshold_method '{threshold_method}'")
    return list(
        _iter_pairwise_rows(
            a,
            b,
            label_mask=label_mask,
            metrics=tuple(metrics),
            threshold_method=threshold_method,
            threshold_a=threshold_a,
            threshold_b=threshold_b,
            channel_a=channel_a,
            channel_b=channel_b,
            slice_axis=slice_axis,
            region_warnings=region_warnings,
        )
    )


def _iter_all_to_all_rows(
    image,
    channel_axis,
    *,
    label_mask,
    metrics,
    threshold_method,
    threshold_a,
    threshold_b,
    channel_names,
    region_warnings,
):
    """Yield the result rows of `analyse_all_to_all`, one at a time.

    The generator counterpart of `_iter_pairwise_rows` for every
    unordered channel pair; inputs are assumed already validated.
    """
    n_channels = image.shape[channel_axis]
    for i in range(n_channels):
        a = np.take(image, i, axis=channel_axis)
        for j in range(i + 1, n_channels):
            b = np.take(image, j, axis=channel_axis)
            yield from _iter_pairwise_rows(
                a,
                b,
                label_mask=label_mask,
                metrics=metrics,
                threshold_method=threshold_method,
                threshold_a=threshold_a,
                threshold_b=threshold_b,
                channel_a=channel_names[i],
                channel_b=channel_names[j],
                slice_axis=None,
                region_warnings=region_warnings,
            )


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
    ``channel_axis`` and dispatches to `analyse_pairwise`.
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
        Forwarded to `analyse_pairwise`.
    threshold_method : str, default 'costes'
        Forwarded to `analyse_pairwise` ('costes', 'manual',
        or an auto-threshold name like 'otsu'). Manual thresholds
        apply identically to every channel pair.
    threshold_a, threshold_b : float, optional
        Forwarded to `analyse_pairwise`.
    channel_names : sequence of str, optional
        One name per channel along ``channel_axis``. Defaults to
        ``['c0', 'c1', ...]``.
    region_warnings : list, optional
        Forwarded to `analyse_pairwise`; collects per-region
        reasons across every channel pair.

    Returns
    -------
    rows : list of dict
        Concatenation of the per-pair results from
        `analyse_pairwise`.

    Raises
    ------
    ValueError
        If ``len(channel_names)`` does not match the channel
        count, if ``label_mask`` does not match the image's
        spatial shape, or if ``threshold_method`` is unknown.
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

    if label_mask is not None:
        label_mask = np.asarray(label_mask)
        spatial_shape = list(image.shape)
        del spatial_shape[channel_axis]
        if label_mask.shape != tuple(spatial_shape):
            raise ValueError(
                f'label_mask shape {label_mask.shape} does not match '
                f'the spatial shape {tuple(spatial_shape)} of the image'
            )
    if threshold_method not in ('manual', 'costes', *_AUTO_THRESHOLDS):
        raise ValueError(f"unknown threshold_method '{threshold_method}'")
    return list(
        _iter_all_to_all_rows(
            image,
            channel_axis,
            label_mask=label_mask,
            metrics=tuple(metrics),
            threshold_method=threshold_method,
            threshold_a=threshold_a,
            threshold_b=threshold_b,
            channel_names=channel_names,
            region_warnings=region_warnings,
        )
    )
