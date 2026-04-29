<p align="center">
  <!-- TODO: replace with final logo -->
  <img src="docs/img/logo.png" alt="napari-colocalisation logo" width="160"/>
</p>

# napari-colocalisation

[![License MIT](https://img.shields.io/pypi/l/napari-colocalisation.svg?color=green)](https://github.com/DBI-INFRA/napari-colocalisation/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/napari-colocalisation.svg?color=green)](https://pypi.org/project/napari-colocalisation)
[![Python Version](https://img.shields.io/pypi/pyversions/napari-colocalisation.svg?color=green)](https://python.org)
[![tests](https://github.com/DBI-INFRA/napari-colocalisation/workflows/tests/badge.svg)](https://github.com/DBI-INFRA/napari-colocalisation/actions)
[![codecov](https://codecov.io/gh/DBI-INFRA/napari-colocalisation/branch/main/graph/badge.svg)](https://codecov.io/gh/DBI-INFRA/napari-colocalisation)
[![napari hub](https://img.shields.io/endpoint?url=https://api.napari-hub.org/shields/napari-colocalisation)](https://napari-hub.org/plugins/napari-colocalisation)
[![npe2](https://img.shields.io/badge/plugin-npe2-blue?link=https://napari.org/stable/plugins/index.html)](https://napari.org/stable/plugins/index.html)

Interactive intensity-colocalisation analysis for [napari](https://napari.org).
Pick two channels (or one multi-channel image), optionally restrict the
analysis to a region drawn as shapes or labels, choose your metric, and get a
results table plus an intensity-vs-intensity scatter plot — all without leaving
napari.

<p align="center">
  <img src="docs/img/widget_overview.png" alt="napari-colocalisation widget" width="780"/>
</p>

## Features

- **Three correlation metrics**: Pearson (PCC), Spearman rank (SRCC), and
  Manders' coefficients M1/M2 (MCC).
- **Pairwise or all-to-all** mode: analyse two grayscale layers, or every
  channel pair within a single multi-channel layer.
- **2D and 3D** support natively (no time-series for now).
- **Region-restricted analysis** via a Shapes or Labels layer — each non-zero
  region is reported on its own row.
- **Manders thresholds**: choose **Costes auto** (iterative regression-based)
  or **Manual**.
- **Interactive results**: in-widget table, scatter plot of the selected row,
  multi-row selection that highlights all matching shapes/labels in the viewer.
- **CSV export** of the current table.

## Installation

```bash
pip install napari-colocalisation
```

If napari isn't already installed, install both at once:

```bash
pip install "napari-colocalisation[all]"
```

For the latest development version:

```bash
pip install git+https://github.com/DBI-INFRA/napari-colocalisation.git
```

## Quick start

1. Launch napari.
2. Load sample data: **File → Open Sample → napari-colocalisation →
   Colocalisation sample (2D)** (a 3D sample is also provided).

   <p align="center">
     <img src="docs/img/quickstart_sample.png" alt="Open Sample menu" width="520"/>
   </p>

3. Open the widget: **Plugins → Colocalisation Analysis**. Two image layers
   `channel_a` and `channel_b` are auto-selected for pairwise mode.

4. Click **Run**. The results table populates with a single row (the whole
   image), and the scatter plot below shows the intensity pairs with the
   metric values overlaid.

   <p align="center">
     <img src="docs/img/quickstart_run.png" alt="Run result" width="780"/>
   </p>

<p align="center">
  <img src="docs/img/widget_overview.png" alt="napari-colocalisation widget" width="780"/>
</p>

5. *(Optional)* Add a Shapes layer, draw a few rectangles or polygons, set
   **Region** to *Shapes* and pick the layer. Re-run — the table now has one
   row per shape, and clicking a row highlights the matching shape in the
   viewer.

<p align="center">
  <img src="docs/img/widget_shapes.png" alt="napari-colocalisation widget" width="780"/>
</p>

See [docs/usage.md](docs/usage.md) for the full walkthrough.

## Documentation

- **[Usage guide](docs/usage.md)** — every control in the widget, in order.
- **[Metrics](docs/metrics.md)** — what PCC, SRCC and MCC mean, when to use
  which, and how the Costes auto-threshold works.
- **[Python API](docs/api.md)** — calling the pure-compute layer
  (`pearson`, `spearman`, `manders`, `costes_threshold`,
  `analyse_pairwise`, `analyse_all_to_all`) from scripts or notebooks.

## Related projects

- [Coloc 2](https://imagej.net/plugins/coloc-2) — the reference ImageJ
  colocalisation plugin; this plugin follows it in spirit.
- [scikit-image colocalization metrics](https://scikit-image.org/docs/stable/auto_examples/applications/plot_colocalization_metrics.html)
  — the underlying implementations of PCC and Manders.

## Contributing

Contributions are welcome. Run the test suite with:

```bash
pip install -e . --group dev
python -m pytest tests/ -v
```

Pre-commit hooks (ruff lint + format, napari-plugin-checks) ship with the
repo:

```bash
pre-commit install
pre-commit run --all-files
```

Please keep test coverage at or above the current level when submitting a PR.

## License

Distributed under the terms of the [MIT](http://opensource.org/licenses/MIT)
licence; `napari-colocalisation` is free and open-source software.

## Issues

Found a bug or have a feature request? Please
[open an issue](https://github.com/DBI-INFRA/napari-colocalisation/issues).
