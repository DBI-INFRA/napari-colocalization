"""Regenerate the README/docs screenshots.

Drives a real napari viewer (desktop display required) into known
states and saves PNGs into ``docs/img/``.

Skipped intentionally:

- ``logo.png`` — static art; no automation needed.
- ``quickstart_sample.png`` — the open File menu is fragile to
  script reliably; redo by hand if the menu structure changes.

Usage::

    python scripts/update_screenshots.py
"""

import sys
import time
from pathlib import Path

import napari
import numpy as np
from qtpy.QtWidgets import QApplication

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))

from napari_colocalization._sample_data import make_sample_data  # noqa: E402
from napari_colocalization._widget import ColocalizationWidget  # noqa: E402

OUT = ROOT / 'docs' / 'img'
WIN_W, WIN_H = 1280, 800


def _spin(qapp, seconds=0.3):
    """Pump the Qt event loop for ``seconds`` so widgets repaint."""
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        qapp.processEvents()
        time.sleep(0.01)


def _wait_for_results(qapp, widget, timeout=30.0):
    """Poll until the analysis worker delivers results, then settle UI."""
    deadline = time.monotonic() + timeout
    while not widget._results and time.monotonic() < deadline:
        qapp.processEvents()
        time.sleep(0.02)
    if not widget._results:
        raise RuntimeError('analysis timed out before producing results')
    _spin(qapp, 0.4)


def _select_region_layer(widget, layer):
    """Set the widget's Region combo to ``layer`` (matches by identity)."""
    combo = widget._region_combo
    for i in range(combo.count()):
        if combo.itemData(i) is layer:
            combo.setCurrentIndex(i)
            return
    raise RuntimeError(f'layer {layer.name!r} not in region combo')


def _load_sample(viewer):
    for data, kw, _ in make_sample_data():
        viewer.add_image(data, **kw)


def _new_viewer():
    viewer = napari.Viewer(show=True)
    viewer.window.resize(WIN_W, WIN_H)
    qapp = QApplication.instance()
    _spin(qapp, 0.4)
    return qapp, viewer


def _grab(viewer, path):
    viewer.window.screenshot(path=str(path), canvas_only=False, flash=False)
    print(f'wrote {path.relative_to(ROOT)}')


# --- per-shot setup --------------------------------------------------------


def shot_usage_widget_initial(qapp, viewer, widget):
    _load_sample(viewer)
    _spin(qapp)
    _grab(viewer, OUT / 'usage_widget_initial.png')


def shot_widget_overview(qapp, viewer, widget):
    _load_sample(viewer)
    _spin(qapp)
    widget._on_run_clicked()
    _wait_for_results(qapp, widget)
    _grab(viewer, OUT / 'widget_overview.png')


def shot_quickstart_run(qapp, viewer, widget):
    _load_sample(viewer)
    _spin(qapp)
    widget._on_run_clicked()
    _wait_for_results(qapp, widget)
    _grab(viewer, OUT / 'quickstart_run.png')


def shot_widget_shapes(qapp, viewer, widget):
    _load_sample(viewer)
    _spin(qapp)
    rectangles = [
        np.array([[40, 30], [40, 110], [120, 110], [120, 30]]),
        np.array([[140, 130], [140, 220], [220, 220], [220, 130]]),
        np.array([[50, 160], [50, 230], [110, 230], [110, 160]]),
    ]
    shapes_layer = viewer.add_shapes(
        rectangles,
        shape_type='rectangle',
        name='regions',
        edge_color='yellow',
        face_color='transparent',
        edge_width=2,
    )
    _spin(qapp)
    _select_region_layer(widget, shapes_layer)
    _spin(qapp, 0.1)
    widget._on_run_clicked()
    _wait_for_results(qapp, widget)
    _grab(viewer, OUT / 'widget_shapes.png')


SHOTS = [
    ('usage_widget_initial', shot_usage_widget_initial),
    ('widget_overview', shot_widget_overview),
    ('quickstart_run', shot_quickstart_run),
    ('widget_shapes', shot_widget_shapes),
]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for name, fn in SHOTS:
        print(f'-- {name}')
        # Fresh viewer per shot so state never leaks between shots.
        qapp, viewer = _new_viewer()
        widget = ColocalizationWidget(viewer)
        viewer.window.add_dock_widget(
            widget,
            area='right',
            name='Colocalization Analysis',
        )
        _spin(qapp, 0.3)
        try:
            fn(qapp, viewer, widget)
        finally:
            viewer.close()
            _spin(qapp, 0.2)


if __name__ == '__main__':
    main()
