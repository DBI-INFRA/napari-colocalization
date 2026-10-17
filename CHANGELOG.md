# Changelog

All notable changes to this project are documented here. The format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the
project uses [semantic versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

### Changed

- Requires Python 3.11 or newer, matching napari 0.9.
- `napari` is now a declared dependency; the widget modules import it
  at runtime.

### Fixed

- Object and diagnostic CSV exports no longer take their provenance
  from a later run that was cancelled or failed, and object exports
  name the region layer.
- Coloc masks, centroid/link overlays and scrambled examples take the
  source image's scale and translate, so they line up with anisotropic
  or offset images. A region layer placed differently from the image
  is rejected instead of silently selecting the wrong pixels.
- Many-region runs no longer hold a full-size mask per result row, and
  revisiting a row reuses its regression line and the fixed-axis
  maxima instead of recomputing them on the UI thread.
- Downloaded samples time out after 30 s instead of hanging the viewer,
  and a failed write no longer leaves a truncated file in the cache.
- Multiscale Labels layers are rejected up front on the Object-based
  tab, like multiscale images, instead of failing mid-run.
- Adding a scrambled example and building object overlays no longer
  block the viewer on large images.

## 0.3.1

### Added

- Analysis runs are cancellable: while a run is in flight the **Run**
  button reads **Cancel**, and clicking it discards the run and keeps the
  previous results.
- A progress bar under each tab's Run button tracks regions,
  randomization iterations or CCF shifts, falling back to a busy
  indicator for the object-segmentation steps.
- Per-column header tooltips on the results table and the object results
  table.
- A note under the results table explaining why no per-pixel p-values are
  reported, with an **Open Diagnostics** button that switches tabs and
  carries the channel and region selection across so the Costes
  randomization test is ready to run.
- The embedded plots follow napari's active theme instead of a
  hard-coded dark palette.
- Tooltip on the Costes **Block size** spinbox describing the
  point-spread-function guideline.
- The compute API is importable from the package root:
  `from napari_colocalization import pearson, analyse_pairwise, ...`.
  None of the re-exported modules pulls in napari or Qt.

### Changed

- **Default metrics are now Pearson + Manders** (previously Spearman
  only) - one metric from each family. Checking Manders reveals the
  threshold group.
- **Van Steensel CCF no longer wraps around the image edges.** Each shift
  now correlates only the overlapping region instead of rolling one edge
  onto the other, so values in the tails of the curve differ from
  earlier releases; `max_shift` must be smaller than the shifted axis.
- Costes bisection stops at one intensity level for integer images,
  matching Coloc 2, and keeps the span-relative tolerance for float
  data.
- Shape and mask mismatches raise `ValueError` across the metric,
  threshold and analysis functions instead of silently correlating
  unrelated pixels.
- `analyse_pairwise` validates `threshold_method` unconditionally, so a
  typo is caught even when Manders is not requested.
- The dock widget is split into one module per tab over a shared
  helpers module; `ColocalizationWidget`'s import path and constructor
  are unchanged.

### Removed

- **The `pcc_pvalue` and `srcc_pvalue` columns** are gone from the
  results table, the CSV export and `COLUMNS`. This changes the CSV
  schema. Those p-values assume independent samples, which spatially
  autocorrelated pixels are not; use the Costes randomization test on
  the Diagnostics tab for significance. See
  [docs/metrics.md](docs/metrics.md#why-there-are-no-per-pixel-p-values).

### Fixed

- Lazy (dask/zarr-backed) layers load on the analysis thread instead
  of inside the Run click handler, so the viewer stays responsive
  while the data is read.
- Images containing NaN pixels (registration or flat-field borders)
  no longer break the cytofluorogram; non-finite pixels are excluded
  from the plot.
- Selecting the same layer for both channels - an autocorrelation
  control - is no longer overridden when the layer list changes; only
  the initial auto-assigned collision is nudged apart.
- Result tables sort numerically instead of lexicographically, so
  `n_pixels` 9999 no longer follows 10000 and `1e-05` no longer sorts
  above `0.5`. NaN sorts below every value.
- **Add coloc mask layer** on a per-Z-slice row writes only that plane,
  embedded in a full-size volume so it stays aligned with the image.
- Multiscale layers are rejected with a clear message, and layer data is
  no longer materialized when the image-stack combo changes.
- CSV and figure write failures surface as warnings instead of failing
  silently.
- The CCF max shift is validated up front, like the Costes block size.

### Performance

- Per-region metrics run on each region's bounding-box crop rather than
  on the full image, an order-of-magnitude speed-up for masks with many
  regions. Results are unchanged.
- The plot context and the all-to-all channel arrays are taken as views
  rather than copies; an all-to-all run no longer duplicates every
  channel.

## 0.2.0

### Fixed

- **Manders overlap coefficient `r` overflowed on integer images.** It
  was computed via `skimage.measure.manders_overlap_coeff`, which
  squares and multiplies in the *input* dtype, so 8- and 16-bit images
  wrapped around. A uint8 pair returned `r = 1.945`, outside the
  documented `[0, 1]` range, while `k1`/`k2` beside it were correct.
  All three are now accumulated in float64 and agree across dtypes.
  **Any `overlap` value computed by `0.1.x` on an integer image was
  wrong.**
- **A failed Costes auto-threshold was reported as `tM1 = tM2 = 0.00`.**
  When no threshold could be fitted (a non-positive regression slope, a
  constant channel, fewer than two pixels), the thresholds fell back to
  `(max(a), max(b))`, which made every pixel fall below threshold and
  produced a hard zero with no warning. Such rows are now blank (`NaN`)
  and say why.
- **`tM1` and `tM2` were blanked together.** They have different
  denominators, so a region where B was empty had a well-defined
  `tM1 = 0` that was being hidden. Each coefficient is now decided on
  its own inputs. A blank cell always means "not measurable", never
  "measured as zero".
- **A background-subtracted channel aborted the whole run.** Negative
  pixels made scikit-image's Manders functions raise, which killed
  every region and every other metric with it, including PCC, SRCC and
  ICQ, all well defined on signed data. The co-occurrence metrics now
  report blank for signed input and the run completes.

### Added

- Object-based tab: **Export CSV...**.
- Diagnostics tab: **Export values...**.
- Diagnostics tab: progress bar and a working **Cancel**.
- `nn_distance_px` on the object table and its CSV.
- The manual T_a/T_b spinboxes are seeded from the selected layer's own
  intensity range, with a step and decimal count to match. They
  previously stepped by 0.01 over +/-1e9 with no hint of the channel's
  units, so reaching a threshold of 17000 on a 16-bit image meant
  holding the arrow key down.

### Changed

- The mode radios stack instead of sharing a row: "All-to-all (one
  layer + channel axis)" was being truncated mid-label at dock width.
- The metric checkboxes flow into two columns. Five on one row needed
  ~467 px, more than a docked panel gets, and the config scroll area
  has no horizontal scrollbar - so the overflow was unreachable rather
  than merely tight.

### Removed

- The object results summary line. It repeated what the per-object rows
  already show, and joined two per-channel sentences with a run of
  spaces that a word-wrapping label renders as a large gap. The table
  gets that height instead; `object_table` still returns its summary
  dict for API callers.

---

Releases before 0.2.0 predate this changelog; see `git tag -n` for
their one-line notes.
