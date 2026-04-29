"""Pure-numpy colocalisation metrics.

No napari / qtpy imports here — every function in this module
operates on ndarrays so it can be tested headlessly and reused
from scripts. PCC and Manders delegate to scikit-image; SRCC
uses scipy.stats; Costes auto-threshold is implemented locally
because scikit-image does not provide it.
"""

import numpy as np
from scipy import stats
from skimage import measure


def _flatten_with_mask(a, b, mask):
    a = np.asarray(a)
    b = np.asarray(b)
    if mask is None:
        return a.ravel(), b.ravel()
    mask = np.asarray(mask, dtype=bool)
    return a[mask], b[mask]


def pearson(a, b, mask=None):
    """Pearson correlation coefficient and two-tailed p-value.

    Wraps :func:`skimage.measure.pearson_corr_coeff`. Returns
    ``(nan, nan)`` for inputs with fewer than two samples or zero
    variance in either channel — both edge cases for which PCC is
    mathematically undefined.

    Parameters
    ----------
    a, b : array_like
        Same-shape intensity arrays of any dimensionality.
    mask : array_like of bool, optional
        Boolean array with the same shape as ``a``/``b``. Only
        pixels where ``mask`` is ``True`` are included in the
        calculation. ``None`` (the default) uses every pixel.

    Returns
    -------
    pcc : float
        The Pearson correlation coefficient in ``[-1, 1]``, or
        ``nan`` if the inputs are degenerate.
    p_value : float
        Two-tailed p-value for the null hypothesis that the
        coefficient is zero. ``nan`` if ``pcc`` is ``nan``.

    Examples
    --------
    >>> import numpy as np
    >>> from napari_colocalisation._metrics import pearson
    >>> rng = np.random.default_rng(0)
    >>> a = rng.random((64, 64))
    >>> pearson(a, a)[0]
    1.0
    """
    a = np.asarray(a)
    b = np.asarray(b)
    a_flat, b_flat = _flatten_with_mask(a, b, mask)
    if a_flat.size < 2 or a_flat.std() == 0 or b_flat.std() == 0:
        return float('nan'), float('nan')
    pcc, pval = measure.pearson_corr_coeff(a, b, mask=mask)
    return float(pcc), float(pval)


def spearman(a, b, mask=None):
    """Spearman rank correlation coefficient and p-value.

    Uses :func:`scipy.stats.spearmanr` on the masked, flattened
    intensities. Robust to monotonic non-linearity and outliers.

    Parameters
    ----------
    a, b : array_like
        Same-shape intensity arrays of any dimensionality.
    mask : array_like of bool, optional
        Boolean array with the same shape as ``a``/``b``. Only
        pixels where ``mask`` is ``True`` are included in the
        calculation.

    Returns
    -------
    rho : float
        Spearman's rank correlation coefficient in ``[-1, 1]``,
        or ``nan`` if the inputs are degenerate (fewer than two
        samples, or zero variance in either channel).
    p_value : float
        Two-tailed p-value for the rank correlation, or ``nan``.

    Examples
    --------
    >>> import numpy as np
    >>> from napari_colocalisation._metrics import spearman
    >>> a = np.linspace(0.1, 10.0, 1000)
    >>> spearman(a, a ** 3)[0]    # monotonic non-linear
    1.0
    """
    a_flat, b_flat = _flatten_with_mask(a, b, mask)
    if a_flat.size < 2 or a_flat.std() == 0 or b_flat.std() == 0:
        return float('nan'), float('nan')
    result = stats.spearmanr(a_flat, b_flat)
    return float(result.statistic), float(result.pvalue)


def manders(a, b, threshold_a, threshold_b, mask=None):
    """Manders' colocalisation coefficients M1 and M2.

    M1 is the fraction of the channel-A intensity that lies in
    pixels where channel B is above ``threshold_b``. M2 is the
    symmetric counterpart for channel B above ``threshold_a``.
    Asymmetry between the two is meaningful — it reflects the
    difference in how much of each channel co-occurs with the
    other.

    Wraps :func:`skimage.measure.manders_coloc_coeff`, which
    expects a binary mask for the second image; we threshold
    internally and then call it twice.

    Parameters
    ----------
    a, b : array_like
        Same-shape, non-negative intensity arrays.
    threshold_a, threshold_b : float
        Per-channel thresholds. Use :func:`costes_threshold` to
        derive them automatically, or pass values you have set
        manually.
    mask : array_like of bool, optional
        Boolean array selecting the analysed region. ``None``
        analyses every pixel.

    Returns
    -------
    m1, m2 : float
        Each in ``[0, 1]``, or ``nan`` if the analysed region
        contains no positive intensity in either channel.

    Examples
    --------
    >>> import numpy as np
    >>> from napari_colocalisation._metrics import manders
    >>> a = np.zeros((10, 10)); a[:, :] = 1.0
    >>> b = np.zeros((10, 10)); b[:5, :] = 1.0   # half of A overlaps B
    >>> m1, m2 = manders(a, b, threshold_a=0.5, threshold_b=0.5)
    >>> round(m1, 2), round(m2, 2)
    (0.5, 1.0)
    """
    a = np.asarray(a)
    b = np.asarray(b)
    a_above = a > threshold_a
    b_above = b > threshold_b
    a_flat, b_flat = _flatten_with_mask(a, b, mask)
    if a_flat.size == 0 or a_flat.sum() == 0 or b_flat.sum() == 0:
        return float('nan'), float('nan')
    m1 = measure.manders_coloc_coeff(a, b_above, mask=mask)
    m2 = measure.manders_coloc_coeff(b, a_above, mask=mask)
    return float(m1), float(m2)


def costes_threshold(a, b, mask=None, n_steps=256):
    """Costes' iterative auto-threshold for Manders' M1 / M2.

    Implements the algorithm of Costes et al. (2004):

    1. Fit a least-squares regression line ``b = m * a + c`` over
       the analysed region.
    2. Walk a candidate threshold ``T_a`` downward from
       ``max(a)``. At each step, set ``T_b = m * T_a + c`` so the
       threshold pair lies on the regression line.
    3. The "below-threshold" subset is every pixel with
       ``a <= T_a`` **or** ``b <= T_b``. Compute its Pearson
       correlation. Stop at the first ``T_a`` where that PCC
       drops to zero or below — at that point the below-threshold
       pixels are no longer correlated, i.e. background.

    Parameters
    ----------
    a, b : array_like
        Same-shape intensity arrays.
    mask : array_like of bool, optional
        Restrict the regression and search to this region.
    n_steps : int, default 256
        Number of candidate thresholds along ``[min(a), max(a)]``.

    Returns
    -------
    threshold_a, threshold_b : float
        Per-channel thresholds suitable to feed into
        :func:`manders`. Falls back to ``(max(a), max(b))`` when
        the regression slope is non-positive (no co-occurrence to
        threshold for) and to ``(min(a), min(b))`` when the
        iteration never reaches ``PCC <= 0``.

    Notes
    -----
    The Costes randomisation significance test (which scrambles
    pixel blocks to produce a p-value for PCC) is not implemented.

    References
    ----------
    Costes, S.V. et al. (2004). *Automatic and quantitative
    measurement of protein-protein colocalization in live cells.*
    Biophys. J. 86(6), 3993-4003.
    """
    a_flat, b_flat = _flatten_with_mask(a, b, mask)
    a_flat = a_flat.astype(np.float64, copy=False)
    b_flat = b_flat.astype(np.float64, copy=False)

    if a_flat.size < 2:
        a_max = float(a_flat.max()) if a_flat.size else 0.0
        b_max = float(b_flat.max()) if b_flat.size else 0.0
        return a_max, b_max

    if a_flat.std() == 0 or b_flat.std() == 0:
        return float(a_flat.max()), float(b_flat.max())

    slope, intercept = np.polyfit(a_flat, b_flat, 1)
    a_max = float(a_flat.max())
    a_min = float(a_flat.min())
    b_max = float(b_flat.max())

    if slope <= 0:
        return a_max, b_max
    if a_max == a_min:
        return a_max, float(slope * a_max + intercept)

    step = (a_max - a_min) / n_steps
    t_a = a_max
    last_t_a, last_t_b = a_max, float(slope * a_max + intercept)

    while t_a > a_min:
        t_b = float(slope * t_a + intercept)
        below = (a_flat <= t_a) | (b_flat <= t_b)
        n_below = int(below.sum())
        if n_below >= 2:
            a_sub = a_flat[below]
            b_sub = b_flat[below]
            if a_sub.std() > 0 and b_sub.std() > 0:
                pcc = float(np.corrcoef(a_sub, b_sub)[0, 1])
                if pcc <= 0:
                    return float(t_a), t_b
        last_t_a, last_t_b = float(t_a), t_b
        t_a -= step

    return last_t_a, last_t_b
