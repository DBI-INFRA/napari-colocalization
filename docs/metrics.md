# Metrics

Background on the four correlation metrics computed by the plugin, plus
the Costes auto-threshold used for Manders.

> Documentation index: [Home](index.md) · [Usage](usage.md) · **Metrics** · [Python API](api.md)

For a deeper treatment, see the
[ImageJ colocalization analysis page](https://imagej.net/imaging/colocalization-analysis)
that this plugin took its design cues from.

## Pearson (PCC)

The standard linear correlation coefficient between paired pixel
intensities, restricted to the analysed region.

$$
\mathrm{PCC} = \frac{\sum_i (a_i - \bar a)(b_i - \bar b)}
{\sqrt{\sum_i (a_i - \bar a)^2 \sum_i (b_i - \bar b)^2}}
$$

Range: −1 (perfect anti-correlation) to +1 (perfect correlation), with 0
meaning no linear relationship.

**When to use it.** As a quick first look at whether two channels co-vary
linearly. PCC is unaffected by uniform scaling or offsets, so it tolerates
differences in absolute brightness between channels.

**Watch out for.**
- Saturated pixels skew PCC towards 0; clip your acquisition appropriately.
- Strong background offsets reduce PCC; consider region-restricting to
  the foreground or use SRCC.
- A single bright outlier pair can dominate the result.

We delegate the computation to
[`skimage.measure.pearson_corr_coeff`](https://scikit-image.org/docs/stable/api/skimage.measure.html#skimage.measure.pearson_corr_coeff),
which returns the coefficient and a two-tailed p-value (both surfaced in
the results table).

## Spearman (SRCC)

The Pearson correlation of the **ranks** of the intensities, rather than
the intensities themselves.

**When to use it.** Whenever the relationship between channels is
monotonic but not necessarily linear (e.g. saturating sensor responses,
gamma-corrected images). SRCC is also robust to outliers - a single
bright pixel can't pull the rank away.

**Watch out for.**
- Like PCC, SRCC compares only paired pixels. If one channel is noisy
  background everywhere except where the other is bright, SRCC can be
  misleading. A region mask helps.

We use
[`scipy.stats.spearmanr`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.spearmanr.html)
on the masked, flattened intensities. Both the rank correlation
coefficient and the p-value are returned.

The plugin defaults to **Spearman only** because it gives a sensible
result on a wider range of microscopy data than PCC alone.

## Li ICQ

The Intensity Correlation Quotient introduced by Li et al. (2004).
For each pixel ``i`` form the *covariance contribution*
``P_i = (a_i - mean(a)) * (b_i - mean(b))``: this product is positive
when both channels co-vary at that pixel (both above or both below
their means) and negative when they anti-vary. ICQ is the fraction of
non-zero ``P_i`` that are positive, re-centred to lie on
``[-0.5, 0.5]``:

$$
\mathrm{ICQ} = \frac{|\{i : P_i > 0\}|}{N} - 0.5
$$

Range: −0.5 (perfectly segregated staining) → 0 (random) → +0.5
(perfectly dependent staining). Unlike PCC and Spearman, ICQ measures
*sign agreement* with respect to the channel means rather than
magnitude correlation, which makes it robust to many of the intensity
artefacts that affect PCC (offsets, soft saturation) while still being
sensitive to dependent staining.

**When to use it.** Reported alongside PCC/SRCC as a quick sanity check
on the *shape* of the dependence: an ICQ near 0 with a high PCC is a
warning that the PCC is being driven by a few extreme pixel pairs
rather than a population-wide co-variation.

**Watch out for.** ICQ does not have a standard p-value, only the
scalar; we report the value alone. Like the other correlation metrics
it is restricted to the analysed region (whole image, shape, or
label).

We compute it locally in `_metrics.li_icq` (no scikit-image
equivalent).

## Overlap coefficient (r, k1, k2)

The overlap coefficient of Manders et al. (1992) and its two split
components. Unlike M1/M2 below, these need **no threshold** - they are
computed directly from the raw intensity products over the region:

$$
r = \frac{\sum_i a_i b_i}{\sqrt{\sum_i a_i^2 \; \sum_i b_i^2}}
\qquad
k_1 = \frac{\sum_i a_i b_i}{\sum_i a_i^2}
\qquad
k_2 = \frac{\sum_i a_i b_i}{\sum_i b_i^2}
$$

Range of `r`: 0 to 1 for non-negative intensities. It is essentially the
Pearson coefficient computed *without* mean-subtraction, which makes it
insensitive to a difference in average brightness between the two
channels - but also means it cannot report anti-correlation.

`k1` and `k2` split that co-occurrence per channel: `k1` is dominated by
pixels where channel A is bright, `k2` by where B is bright. Their
asymmetry diagnoses which channel drives the overlap.

**When to use it.** As a brightness-insensitive companion to PCC,
particularly when the two channels were acquired at very different gain.
The split `k1`/`k2` help when one channel is much sparser than the other.

**Watch out for.**
- `r` hides the sign of any relationship; always read it next to PCC or
  ICQ, never alone.
- Like M1/M2 it assumes non-negative intensities; subtract background
  first if your data has a strong offset.
- Any coefficient whose denominator is zero (an all-zero channel within
  the region) is reported as a blank/`NaN` cell.

Computed locally in `_metrics.overlap`; the three values populate the
`overlap`, `k1` and `k2` columns.

## Manders (MCC)

Two coefficients, M1 and M2, that ask: *"what fraction of the intensity
in one channel is co-located with above-threshold signal in the other
channel?"*

$$
M_1 = \frac{\sum_i a_i \cdot \mathbb{1}[b_i > T_b]}{\sum_i a_i}
\qquad
M_2 = \frac{\sum_i b_i \cdot \mathbb{1}[a_i > T_a]}{\sum_i b_i}
$$

Range: 0 to 1 each. They are **not** percentages - interpret them as
"the fraction of channel A's signal that overlaps with channel B's
above-threshold pixels", and similarly for M2. Asymmetry between M1 and
M2 is meaningful and expected when one channel is sparser than the other.

**When to use it.** When you care about *co-occurrence* of signal rather
than the *correlation* of intensities. Two channels can have low PCC but
high M1 and M2 if their bright pixels reliably overlap but with widely
varying intensities.

**Watch out for.**
- Manders is sensitive to the choice of threshold. Use Costes (below)
  if you don't have a principled way to set it manually.
- High background offsets confuse Costes - pre-process or set thresholds
  manually.

We delegate to
[`skimage.measure.manders_coloc_coeff`](https://scikit-image.org/docs/stable/api/skimage.measure.html#skimage.measure.manders_coloc_coeff)
after applying the chosen thresholds.

### Costes' auto-threshold

Manders' coefficients depend on a per-channel threshold separating
"signal" from "background". Picking the threshold by eye is subjective;
the iterative method introduced by Costes et al. (2004) gives a
reproducible answer that is widely cited in the colocalization
literature.

Algorithm (as implemented in `_metrics.costes_threshold`, matched to
Fiji's **Coloc 2** `AutoThresholdRegression`):

1. Fit an **orthogonal** (total-least-squares) regression line
   `b = m·a + c` over the masked pixels - *not* an ordinary least-squares
   fit. Orthogonal regression treats the two channels symmetrically
   (neither is the independent variable) and avoids the slope bias OLS
   shows when the predictor channel is noisy.
2. Move a candidate threshold along that line by **bisection**, keeping
   the pair on the line (`T_b = m·T_a + c`). The channel that is stepped
   is the one giving finer resolution - channel A when `|m| < 1`, else
   channel B.
3. At each candidate, compute the Pearson correlation of the
   below-threshold pixels (`a < T_a` **or** `b < T_b`). Bisect downward
   while that correlation is positive and upward when it is non-positive,
   converging on the threshold where the background pixels stop
   correlating.

Falls back to `(max(a), max(b))` when the regression slope is
non-positive or undefined (no linear co-occurrence to threshold for).

The selected thresholds appear in the `threshold_a` / `threshold_b`
columns of the results table and as red reference lines on the scatter
plot. The orthogonal regression line is overlaid as a cyan dashed line
(via `_metrics.costes_regression`) so you can see the axis the
auto-threshold was found along.

> The Costes randomisation significance test is available on the
> **Diagnostics** tab (see below).

### Other auto-thresholds (Otsu, Li, …)

Besides Costes, the threshold method can be any of the standard
histogram thresholds - **Otsu, Li, Triangle, Yen, Mean, IsoData** (the
`skimage.filters.threshold_*` family, as ImageJ's Auto Threshold offers
in JACoP B). Each channel is thresholded **independently** from its own
intensity histogram, giving the *thresholded* Manders coefficients
(tM1/tM2). These are a good choice when the two channels have clearly
bimodal (signal vs background) histograms; Costes is preferable when the
relationship between the channels is the thing you want the threshold to
respect. A channel with no contrast (constant within the region) has no
defined auto-threshold, so its M1/M2 are reported as blank/`NaN`.

## Diagnostics

The metrics above each collapse colocalization to a number per region.
The **Diagnostics** tab instead produces a *plot* for a single channel
pair (optionally restricted to a region), to inspect the *shape* of the
relationship. Each is computed in the napari-free `_diagnostics` module.

### Costes randomization (significance test)

Answers "could this PCC have arisen by chance?". Channel B is scrambled
in blocks `n_iter` times - destroying spatial co-occurrence while
preserving each channel's intensity histogram and local texture - and
the PCC is recomputed each time to build a null distribution. The
observed PCC is plotted against that null, with a right-tailed p-value
`(#{null ≥ observed} + 1) / (n_iter + 1)` and a z-score. A genuine
colocalization sits far to the right of the null. The block size should
approximate the point-spread-function width. Works on 2D and 3D images.

This is the Costes randomization the v1 metrics page noted as missing.

### Van Steensel cross-correlation function (CCF)

Shifts channel B relative to A by ±`max_shift` pixels and plots the PCC
at each shift. A peak **at** shift 0 indicates colocalization; a central
**trough** with side peaks indicates mutual exclusion (the channels are
anti-located by roughly the shift of the side peaks). The summary
reports the peak Pearson r and the shift it occurs at.

### Li ICA

The scatter behind the [Li ICQ](#li-icq) scalar: for each channel, the
per-pixel covariance product `(A−Ā)(B−B̄)` is plotted against that
channel's intensity. Dependent staining produces a characteristic
rightward (positive-product) skew that grows with intensity; random or
segregated staining stays centred on zero. The ICQ value is shown
alongside.

## Choosing a metric

| Question you're asking | Use |
|---|---|
| "Are the two channels linearly correlated?" | Pearson |
| "Is the relationship monotonic but not necessarily linear?" | Spearman |
| "Do the channels co-vary in the *same direction* relative to their means, ignoring magnitude?" | Li ICQ |
| "How much do the signals overlap, ignoring brightness differences and without a threshold?" | Overlap r (k1/k2) |
| "What fraction of A's signal sits where B is bright?" | Manders M1 |
| "...and vice versa?" | Manders M2 |
| "Just give me a robust default" | Spearman |

PCC and SRCC measure *correlation*; MCC measures *co-occurrence*. They
answer different questions and disagree often - which is precisely why
reporting more than one is good practice.

## References

- Bolte, S. & Cordelières, F.P. (2006). *A guided tour into subcellular
  colocalization analysis in light microscopy.* J. Microsc. 224(3),
  213-232.
- Costes, S.V. et al. (2004). *Automatic and quantitative measurement of
  protein-protein colocalization in live cells.* Biophys. J. 86(6),
  3993-4003.
- Dunn, K.W., Kamocka, M.M., McDonald, J.H. (2011). *A practical guide to
  evaluating colocalization in biological microscopy.* Am. J. Physiol.
  Cell Physiol. 300(4), C723-C742.
- Li, Q. et al. (2004). *A Syntaxin 1, Galpha(o), and N-type Calcium
  Channel Complex at a Presynaptic Nerve Terminal: Analysis by
  Quantitative Immunocolocalization.* J. Neurosci. 24(16), 4070-4081.
- [ImageJ colocalization analysis](https://imagej.net/imaging/colocalization-analysis)
- [ImageJ Coloc 2 plugin](https://imagej.net/plugins/coloc-2)
