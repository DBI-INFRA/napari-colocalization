# Changelog

All notable changes to this project are documented here. The format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the
project uses [semantic versioning](https://semver.org/spec/v2.0.0.html).

## 0.3.0

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
  matching Fiji Coloc 2, and keeps the span-relative tolerance for float
  data.
- Shape and mask mismatches raise `ValueError` across the metric,
  threshold and analysis functions instead of silently correlating
  unrelated pixels.
- `manders` and `overlap` reject negative intensities with an explicit
  message.
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

---

Releases before 0.3.0 predate this changelog.
