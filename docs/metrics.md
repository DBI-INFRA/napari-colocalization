# Metrics

Background on the three correlation metrics computed by the plugin, plus
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
gamma-corrected images). SRCC is also robust to outliers — a single
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

## Manders (MCC)

Two coefficients, M1 and M2, that ask: *"what fraction of the intensity
in one channel is co-located with above-threshold signal in the other
channel?"*

$$
M_1 = \frac{\sum_i a_i \cdot \mathbb{1}[b_i > T_b]}{\sum_i a_i}
\qquad
M_2 = \frac{\sum_i b_i \cdot \mathbb{1}[a_i > T_a]}{\sum_i b_i}
$$

Range: 0 to 1 each. They are **not** percentages — interpret them as
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
- High background offsets confuse Costes — pre-process or set thresholds
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

Algorithm (as implemented in `_metrics.costes_threshold`):

1. Fit a least-squares regression line `b = m·a + c` on the masked pixel
   pairs.
2. Walk a candidate threshold `T_a` downward from `max(a)`. At each step,
   set `T_b = m·T_a + c` so the threshold pair lies on the regression
   line.
3. The "below-threshold" set is every pixel with `a ≤ T_a` **or**
   `b ≤ T_b`. Compute its Pearson correlation.
4. Stop at the first `T_a` where that PCC drops to zero or below — the
   below-threshold pixels are now uncorrelated, i.e. background. Return
   `(T_a, T_b)`.

Falls back to `(max(a), max(b))` if the regression slope is non-positive
(no linear co-occurrence to threshold for) and to `(min(a), min(b))` if
the iteration runs to completion without ever hitting PCC ≤ 0.

The selected thresholds appear in the `threshold_a` / `threshold_b`
columns of the results table and as red reference lines on the scatter
plot.

> The Costes randomisation test (statistical significance of PCC by
> shuffled pixel blocks) is **not** implemented in v1.

## Choosing a metric

| Question you're asking | Use |
|---|---|
| "Are the two channels linearly correlated?" | Pearson |
| "Is the relationship monotonic but not necessarily linear?" | Spearman |
| "What fraction of A's signal sits where B is bright?" | Manders M1 |
| "...and vice versa?" | Manders M2 |
| "Just give me a robust default" | Spearman |

PCC and SRCC measure *correlation*; MCC measures *co-occurrence*. They
answer different questions and disagree often — which is precisely why
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
- [ImageJ colocalization analysis](https://imagej.net/imaging/colocalization-analysis)
- [ImageJ Coloc 2 plugin](https://imagej.net/plugins/coloc-2)
