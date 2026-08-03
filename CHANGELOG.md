# Changelog

All notable changes to this project are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versions follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0]

### Fixed

- **Manders overlap coefficient `r` overflowed on integer images.** It was
  computed via `skimage.measure.manders_overlap_coeff`, which squares and
  multiplies in the *input* dtype, so 8- and 16-bit images wrapped around.
  A uint8 pair returned `r = 1.945`, outside the documented `[0, 1]` range,
  while `k1`/`k2` beside it were correct. All three are now accumulated in
  float64 and agree across dtypes. **Any `overlap` value computed by `0.1.x`
  on an integer image was wrong.**
- **A failed Costes auto-threshold was reported as `tM1 = tM2 = 0.00`.**
  When no threshold could be fitted (a non-positive regression slope, a
  constant channel, fewer than two pixels), the thresholds fell back to
  `(max(a), max(b))`, which made every pixel fall below threshold and
  produced a hard zero with no warning. "No threshold could be fitted" was
  indistinguishable from the measured result "nothing co-occurs", and it
  reached the exported CSV as a number nobody measured. Such rows are now
  blank (`NaN`) and say why.
- **`tM1` and `tM2` were blanked together.** They have different
  denominators (`tM1` divides by the total of channel A and only *gates* on
  B), so a region where B was empty had a well-defined `tM1 = 0` that was
  being hidden. Each coefficient is now decided on its own inputs, so one
  can carry a value while the other is blank. A blank cell always means "not
  measurable", never "measured as zero".
- **A background-subtracted channel aborted the whole run.** Negative pixels
  made scikit-image's Manders functions raise, which killed every region and
  every other metric with it, including PCC, SRCC and ICQ, all well defined
  on signed data. The co-occurrence metrics now report blank for signed
  input and the run completes.

### Added

- **Object-based tab: `Export CSV…`.**
- **Diagnostics tab: `Export values…`.**
- **Diagnostics tab: progress bar and a working `Cancel`.**
- **`nn_distance_px` on the object table and its CSV**

---

Releases before `0.2.0` predate this file; see `git tag -n` for their
one-line notes.

[0.2.0]: https://github.com/DBI-INFRA/napari-colocalization/compare/v0.1.8...v0.2.0
