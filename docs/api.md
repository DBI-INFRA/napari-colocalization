# Python API

The pure-compute layer of `napari-colocalization` is independent of napari
and Qt - you can import the metric, masking, and analysis functions from
a script or notebook and use them on plain ndarrays.

> Documentation index: [Home](index.md) · [Usage](usage.md) · [Metrics](metrics.md) · **Python API**

The function reference below is generated directly from the source
docstrings. The narrative and examples on this page are written by hand;
the `:::` blocks are filled in automatically by mkdocstrings at build
time.

## The public API

Everything documented on this page is re-exported from the package
root - that is the surface to import from:

```python
from napari_colocalization import analyse_pairwise, pearson, manders
```

| Group | Names |
| --- | --- |
| Metrics | `pearson`, `spearman`, `li_icq`, `manders`, `overlap`, `costes_threshold`, `costes_regression` |
| Masking | `shapes_to_label_mask`, `labels_to_label_mask`, `iter_regions` |
| Analysis | `analyse_pairwise`, `analyse_all_to_all`, `COLUMNS` |
| Diagnostics | `costes_randomization`, `van_steensel_ccf`, `li_ica`, `scramble_example` |
| Object-based | `label_objects`, `object_table`, `object_centroids`, `nearest_neighbour_vectors`, `OBJECT_COLUMNS` |
| Sample data | `make_sample_data`, `make_sample_data_3d`, `make_sample_data_cbs006rbm`, `make_sample_data_coloc` |
| Widget | `ColocalizationWidget` |

Internally those live in underscore-prefixed modules (a convention
from the napari plugin template), which is also where the docstrings
below are read from:

```
napari_colocalization
├── _metrics.py     pearson, spearman, li_icq, manders, overlap,
│                   costes_threshold, costes_regression
├── _masking.py     shapes_to_label_mask, labels_to_label_mask, iter_regions
├── _analysis.py    analyse_pairwise, analyse_all_to_all, COLUMNS
├── _diagnostics.py costes_randomization, van_steensel_ccf, li_ica,
│                   scramble_example
├── _objects.py     label_objects, object_table, object_centroids,
│                   nearest_neighbour_vectors, OBJECT_COLUMNS
├── _sample_data.py make_sample_data, make_sample_data_3d,
│                   make_sample_data_cbs006rbm, make_sample_data_coloc
├── _plot.py        embedded matplotlib canvases (Qt-only)
├── _widget.py      ColocalizationWidget - the dock widget (Qt-only)
├── _pixel_tab.py       the three analysis tabs and the Qt/layer
├── _diagnostics_tab.py helpers they share (Qt-only)
├── _object_tab.py
└── _widget_shared.py
```

The module layout may be rearranged between releases; the names in
the table above are the stable ones.

## Metrics - `napari_colocalization._metrics`

::: napari_colocalization._metrics
    options:
      members:
        - pearson
        - spearman
        - li_icq
        - manders
        - overlap
        - costes_threshold
        - costes_regression
      show_root_heading: false
      heading_level: 3

## Masking - `napari_colocalization._masking`

::: napari_colocalization._masking
    options:
      members:
        - shapes_to_label_mask
        - labels_to_label_mask
        - iter_regions
      show_root_heading: false
      heading_level: 3

## Analysis - `napari_colocalization._analysis`

`COLUMNS` is the canonical column order shared by the table, the CSV
export, and the row dicts:

```python
('region', 'slice', 'channel_a', 'channel_b', 'n_pixels',
 'pcc', 'srcc', 'icq',
 'overlap', 'k1', 'k2',
 'tm1', 'tm2', 'threshold_a', 'threshold_b')
```

(`slice` is the plane index for per-Z-slice runs, and ``NaN`` otherwise.)

`pearson` and `spearman` return a `(coefficient, p_value)` pair, but the
p-value is not part of the result rows: it assumes independent samples,
which image pixels are not
(see [metrics.md](metrics.md#why-there-are-no-per-pixel-p-values)). Use
`costes_randomization` for significance.

::: napari_colocalization._analysis
    options:
      members:
        - analyse_pairwise
        - analyse_all_to_all
      show_root_heading: false
      heading_level: 3

## Diagnostics - `napari_colocalization._diagnostics`

Curve/distribution producers behind the Diagnostics tab. Unlike the
metrics, a degenerate whole input raises `ValueError` rather than
returning `NaN`.

::: napari_colocalization._diagnostics
    options:
      members:
        - costes_randomization
        - van_steensel_ccf
        - li_ica
        - scramble_example
      show_root_heading: false
      heading_level: 3

## Object-based - `napari_colocalization._objects`

Object-level colocalization: from two labelled images, report
centre-particle coincidence and object overlap per object, plus
centroids and nearest-neighbour links for viewer overlays. Objects can
be supplied directly or obtained with `label_objects` (threshold +
connected components).

::: napari_colocalization._objects
    options:
      members:
        - label_objects
        - object_table
        - object_centroids
        - nearest_neighbour_vectors
      show_root_heading: false
      heading_level: 3

## Sample data - `napari_colocalization._sample_data`

`make_sample_data_cbs006rbm` downloads the CBS006RBM benchmark image
from the [Colocalization Benchmark Source](https://colocalization-benchmark.com)
on first use and caches it under `~/.cache/napari-colocalization/`.

::: napari_colocalization._sample_data
    options:
      members:
        - make_sample_data
        - make_sample_data_3d
        - make_sample_data_cbs006rbm
        - make_sample_data_coloc
      show_root_heading: false
      heading_level: 3

## Putting it together: scripted analysis

```python
import numpy as np
import pandas as pd
from napari_colocalization import analyse_pairwise, COLUMNS

a = np.load('channel_a.npy')
b = np.load('channel_b.npy')
mask = np.load('cell_labels.npy')   # 0 = bg, 1..N = cells

rows = analyse_pairwise(
    a, b,
    label_mask=mask,
    metrics=('pcc', 'srcc', 'mcc'),
    threshold_method='costes',
    channel_a='dna', channel_b='tubulin',
)

df = pd.DataFrame(rows, columns=COLUMNS)
df.to_csv('colocalization_per_cell.csv', index=False)
print(df.describe())
```

## Stability

- The function signatures, return shapes and `COLUMNS` tuple above are
  intended to be stable across point releases.
- Internal helpers prefixed with `_` (private functions inside each
  module) may change without notice.
- `ColocalizationWidget` (`_widget.py`) is the GUI surface and not part
  of the scripting API; for scripts, drive `analyse_*` directly.
