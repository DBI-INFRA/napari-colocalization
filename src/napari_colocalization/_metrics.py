"""Pure-numpy colocalization metrics.

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


def _degenerate(a_flat, b_flat):
    """True when a correlation is undefined: <2 samples or no variance."""
    return a_flat.size < 2 or a_flat.std() == 0 or b_flat.std() == 0


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
    >>> from napari_colocalization._metrics import pearson
    >>> rng = np.random.default_rng(0)
    >>> a = rng.random((64, 64))
    >>> pearson(a, a)[0]
    1.0
    """
    a = np.asarray(a)
    b = np.asarray(b)
    a_flat, b_flat = _flatten_with_mask(a, b, mask)
    if _degenerate(a_flat, b_flat):
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
    >>> from napari_colocalization._metrics import spearman
    >>> a = np.linspace(0.1, 10.0, 1000)
    >>> spearman(a, a ** 3)[0]    # monotonic non-linear
    1.0
    """
    a_flat, b_flat = _flatten_with_mask(a, b, mask)
    if _degenerate(a_flat, b_flat):
        return float('nan'), float('nan')
    result = stats.spearmanr(a_flat, b_flat)
    return float(result.statistic), float(result.pvalue)


def li_icq(a, b, mask=None):
    """Li's Intensity Correlation Quotient (ICQ).

    For each pixel ``i``, form the covariance contribution
    ``P_i = (a_i - mean(a)) * (b_i - mean(b))``. The ICQ is the
    fraction of pixels for which this product is positive,
    re-centred to lie on ``[-0.5, 0.5]``:

    .. math::
        \\mathrm{ICQ} = \\frac{|\\{i : P_i > 0\\}|}{N} - 0.5

    Following Li et al. (2004), values close to ``+0.5`` indicate
    dependent (co-varying) staining, ``0`` indicates random
    staining, and ``-0.5`` indicates segregated (anti-varying)
    staining.

    Parameters
    ----------
    a, b : array_like
        Same-shape intensity arrays of any dimensionality.
    mask : array_like of bool, optional
        Boolean array with the same shape as ``a`` / ``b``. Only
        pixels where ``mask`` is ``True`` are counted. ``None``
        (the default) uses every pixel.

    Returns
    -------
    icq : float
        Value in ``[-0.5, 0.5]``, or ``nan`` for inputs with
        fewer than two samples or zero variance in either channel
        (where the means coincide with every value and the sign
        of ``P_i`` is degenerate).

    Notes
    -----
    Pixels with ``P_i == 0`` (typically because one channel
    equals its mean exactly at that pixel) are excluded from the
    fraction's numerator and denominator, mirroring the
    convention adopted by ImageJ's Coloc 2 plugin.

    References
    ----------
    Li, Q. et al. (2004). *A Syntaxin 1, Galpha(o), and N-type
    Calcium Channel Complex at a Presynaptic Nerve Terminal:
    Analysis by Quantitative Immunocolocalization.*
    J. Neurosci. 24(16), 4070-4081.

    Examples
    --------
    >>> import numpy as np
    >>> from napari_colocalization._metrics import li_icq
    >>> rng = np.random.default_rng(0)
    >>> a = rng.random((128, 128))
    >>> round(li_icq(a, a), 2)         # perfectly co-varying
    0.5
    >>> round(li_icq(a, -a), 2)        # perfectly anti-varying
    -0.5
    """
    a_flat, b_flat = _flatten_with_mask(a, b, mask)
    if _degenerate(a_flat, b_flat):
        return float('nan')
    products = (a_flat - a_flat.mean()) * (b_flat - b_flat.mean())
    nonzero = products != 0
    n = int(nonzero.sum())
    if n == 0:
        return float('nan')
    positive = int((products > 0).sum())
    return float(positive / n - 0.5)


def manders(a, b, threshold_a, threshold_b, mask=None):
    """Manders' colocalization coefficients M1 and M2.

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
    >>> from napari_colocalization._metrics import manders
    >>> a = np.zeros((10, 10)); a[:, :] = 1.0
    >>> b = np.zeros((10, 10)); b[:5, :] = 1.0   # half of A overlaps B
    >>> m1, m2 = manders(a, b, threshold_a=0.5, threshold_b=0.5)
    >>> round(m1, 2), round(m2, 2)
    (0.5, 1.0)
    """
    a = np.asarray(a)
    b = np.asarray(b)
    if not (np.isfinite(threshold_a) and np.isfinite(threshold_b)):
        # An auto-threshold can be NaN for a constant/empty channel;
        # M1/M2 are then undefined rather than spuriously 0.
        return float('nan'), float('nan')
    a_above = a > threshold_a
    b_above = b > threshold_b
    a_flat, b_flat = _flatten_with_mask(a, b, mask)
    if a_flat.size == 0 or a_flat.sum() == 0 or b_flat.sum() == 0:
        return float('nan'), float('nan')
    m1 = measure.manders_coloc_coeff(a, b_above, mask=mask)
    m2 = measure.manders_coloc_coeff(b, a_above, mask=mask)
    return float(m1), float(m2)


def overlap(a, b, mask=None):
    """Manders' overlap coefficient ``r`` and split coefficients k1, k2.

    These threshold-free measures (Manders et al. 1992) quantify
    co-occurrence from the raw intensity products, complementing
    the threshold-dependent M1/M2 from :func:`manders`:

    .. math::
        r = \\frac{\\sum_i a_i b_i}
                  {\\sqrt{\\sum_i a_i^2 \\; \\sum_i b_i^2}}, \\quad
        k_1 = \\frac{\\sum_i a_i b_i}{\\sum_i a_i^2}, \\quad
        k_2 = \\frac{\\sum_i a_i b_i}{\\sum_i b_i^2}

    ``r`` lies in ``[0, 1]`` for non-negative intensities and is
    insensitive to a difference in mean brightness between the two
    channels. ``k1`` and ``k2`` split that co-occurrence per
    channel and are sensitive to such differences — their
    asymmetry is informative.

    ``r`` is delegated to
    :func:`skimage.measure.manders_overlap_coeff`. The split
    coefficients k1/k2 have no scikit-image equivalent
    (:func:`skimage.measure.manders_coloc_coeff` computes the
    threshold-gated M1/M2 used by :func:`manders`, a different
    quantity), so they are derived from the same intensity sums
    here.

    Parameters
    ----------
    a, b : array_like
        Same-shape, non-negative intensity arrays.
    mask : array_like of bool, optional
        Boolean array selecting the analysed region. ``None``
        (the default) uses every pixel.

    Returns
    -------
    r, k1, k2 : float
        The overlap coefficient and the two split coefficients.
        Any coefficient whose denominator is zero (an all-zero
        channel within the region) is returned as ``nan``; all
        three are ``nan`` for an empty region.

    References
    ----------
    Manders, E.M.M. et al. (1992). *Dynamics of three-dimensional
    replication patterns during the S-phase, analysed by double
    labelling of DNA and confocal microscopy.* J. Cell Sci. 103,
    857-862.

    Examples
    --------
    >>> import numpy as np
    >>> from napari_colocalization._metrics import overlap
    >>> a = np.array([1.0, 2.0, 3.0, 4.0])
    >>> overlap(a, a)                      # identical channels
    (1.0, 1.0, 1.0)
    >>> r, k1, k2 = overlap(a, 2 * a)      # b is twice a
    >>> round(r, 4), round(k1, 4), round(k2, 4)
    (1.0, 2.0, 0.5)
    """
    a_flat, b_flat = _flatten_with_mask(a, b, mask)
    a_flat = a_flat.astype(np.float64, copy=False)
    b_flat = b_flat.astype(np.float64, copy=False)
    if a_flat.size == 0:
        return float('nan'), float('nan'), float('nan')
    sum_aa = float(np.sum(a_flat * a_flat))
    sum_bb = float(np.sum(b_flat * b_flat))
    # The overlap coefficient r is delegated to scikit-image. k1/k2
    # have no scikit-image equivalent (manders_coloc_coeff computes
    # the threshold-gated M1/M2 — a different quantity), so they are
    # derived locally from the shared intensity sums. The size/sum
    # guards keep the "never raise, return nan" contract: an empty
    # region or a zero-variance channel would otherwise make
    # manders_overlap_coeff raise or emit a divide warning.
    if sum_aa > 0 and sum_bb > 0:
        r = float(measure.manders_overlap_coeff(a, b, mask=mask))
    else:
        r = float('nan')
    sum_ab = float(np.sum(a_flat * b_flat))
    k1 = sum_ab / sum_aa if sum_aa > 0 else float('nan')
    k2 = sum_ab / sum_bb if sum_bb > 0 else float('nan')
    return r, k1, k2


def costes_regression(a, b, mask=None):
    """Orthogonal-regression line ``b = slope * a + intercept``.

    This is the line :func:`costes_threshold` walks along; it is
    exposed separately so callers (e.g. the cytofluorogram) can
    draw the same line. To match Fiji's **Coloc 2** we use
    **orthogonal** (total-least-squares) regression rather than an
    ordinary least-squares fit — the relationship is symmetric
    (neither channel is the independent variable), and OLS would
    bias the slope when the predictor channel is noisy. The slope
    is the principal axis of the intensity covariance:

    .. math::
        m = \\frac{\\sigma_b^2 - \\sigma_a^2 +
            \\sqrt{(\\sigma_b^2 - \\sigma_a^2)^2 + 4\\sigma_{ab}^2}}
            {2\\sigma_{ab}}, \\quad c = \\bar b - m\\,\\bar a

    Parameters
    ----------
    a, b : array_like
        Same-shape intensity arrays.
    mask : array_like of bool, optional
        Restrict the regression to this region.

    Returns
    -------
    slope, intercept : float
        Regression coefficients, or ``(nan, nan)`` when the region
        has fewer than two samples, zero variance in either
        channel, or zero covariance (no line to fit).

    References
    ----------
    Costes, S.V. et al. (2004). *Automatic and quantitative
    measurement of protein-protein colocalization in live cells.*
    Biophys. J. 86(6), 3993-4003. Implementation matched to Fiji
    Coloc 2 ``AutoThresholdRegression``.
    """
    a_flat, b_flat = _flatten_with_mask(a, b, mask)
    a_flat = a_flat.astype(np.float64, copy=False)
    b_flat = b_flat.astype(np.float64, copy=False)
    if _degenerate(a_flat, b_flat):
        return float('nan'), float('nan')
    mean_a = a_flat.mean()
    mean_b = b_flat.mean()
    var_a = a_flat.var()
    var_b = b_flat.var()
    cov = ((a_flat - mean_a) * (b_flat - mean_b)).mean()
    if cov == 0:
        return float('nan'), float('nan')
    slope = (var_b - var_a + np.sqrt((var_b - var_a) ** 2 + 4 * cov**2)) / (
        2 * cov
    )
    intercept = mean_b - slope * mean_a
    return float(slope), float(intercept)


def _below_pearson(a_flat, b_flat, t_a, t_b):
    """Pearson r of pixels below threshold (``a < t_a`` OR ``b < t_b``).

    Matches Coloc 2's ``ThresholdMode.Below`` (strict ``<``, OR).
    Returns ``nan`` when the below-set is degenerate.
    """
    below = (a_flat < t_a) | (b_flat < t_b)
    if int(below.sum()) < 2:
        return float('nan')
    a_sub = a_flat[below]
    b_sub = b_flat[below]
    if a_sub.std() == 0 or b_sub.std() == 0:
        return float('nan')
    return float(np.corrcoef(a_sub, b_sub)[0, 1])


def costes_threshold(a, b, mask=None, max_iter=100):
    """Costes' iterative auto-threshold for Manders' M1 / M2.

    Follows Fiji **Coloc 2** (``AutoThresholdRegression``):

    1. Fit the :func:`costes_regression` orthogonal line
       ``b = m*a + c``.
    2. Step a candidate threshold by **bisection**. The threshold
       pair always lies on the line; the channel that is stepped
       is the one giving finer resolution — channel A when
       ``|m| < 1`` (then ``T_b = m*T_a + c``) else channel B (then
       ``T_a = (T_b - c)/m``).
    3. At each candidate, compute the Pearson correlation of the
       below-threshold pixels (``a < T_a`` **or** ``b < T_b``).
       Bisect downward while that correlation is positive and
       upward when it is non-positive (or undefined), converging on
       the threshold where the background pixels stop correlating.

    Parameters
    ----------
    a, b : array_like
        Same-shape intensity arrays.
    mask : array_like of bool, optional
        Restrict the regression and search to this region.
    max_iter : int, default 100
        Maximum bisection iterations (Coloc 2 default); the search
        also stops once the step falls below one intensity unit.

    Returns
    -------
    threshold_a, threshold_b : float
        Per-channel thresholds for :func:`manders`, clamped to the
        data range. Falls back to ``(max(a), max(b))`` when the
        regression slope is non-positive or undefined (no
        co-occurrence to threshold for).

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

    slope, intercept = costes_regression(a_flat, b_flat)
    a_max, a_min = float(a_flat.max()), float(a_flat.min())
    b_max, b_min = float(b_flat.max()), float(b_flat.min())

    if not np.isfinite(slope) or slope <= 0:
        return a_max, b_max

    # Step the channel that resolves the line best (Coloc 2: ch1 when
    # |m| < 1, else ch2), mapping the stepped value onto the other
    # channel via the orthogonal line.
    if abs(slope) < 1:

        def map_thresholds(value):
            t_a = min(max(value, a_min), a_max)
            return t_a, min(max(slope * t_a + intercept, b_min), b_max)

        threshold, upper, span = 0.5 * (a_max + a_min), a_max, a_max - a_min
    else:

        def map_thresholds(value):
            t_b = min(max(value, b_min), b_max)
            return min(max((t_b - intercept) / slope, a_min), a_max), t_b

        threshold, upper, span = 0.5 * (b_max + b_min), b_max, b_max - b_min

    # Bisection: positive below-r -> threshold too high, step down;
    # non-positive or undefined -> step up. Step halves each round.
    # Coloc 2 stops at a step of one integer intensity level; we use a
    # tolerance relative to the data span so it also works on
    # float-valued (e.g. 0..1 normalised) images.
    tol = span * 1e-4
    thr_diff = abs(upper - threshold)
    t_a, t_b = map_thresholds(threshold)
    for _ in range(max_iter):
        if thr_diff < tol:
            break
        t_a, t_b = map_thresholds(threshold)
        r = _below_pearson(a_flat, b_flat, t_a, t_b)
        thr_diff *= 0.5
        if np.isnan(r) or r < 0:
            threshold += thr_diff
        else:
            threshold -= thr_diff

    t_a, t_b = map_thresholds(threshold)
    return float(t_a), float(t_b)
