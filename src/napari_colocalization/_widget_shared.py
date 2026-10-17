"""Building blocks shared by the plugin's tab widgets.

Cell formatting, the sortable table item, the figure-export dialog,
and the small Qt/layer-combo helpers that the Pixel-based,
Diagnostics and Object-based tabs all build on. Keeping them here
lets the tab modules stay independent of one another - none of them
imports another tab.
"""

import csv
from datetime import datetime
from typing import Any, NamedTuple

import numpy as np
from napari.utils.notifications import show_info, show_warning
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLayout,
    QProgressBar,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from ._masking import labels_to_label_mask, shapes_to_label_mask
from ._objects import OBJECT_COLUMNS

try:
    from ._version import version as _PLUGIN_VERSION
except ImportError:  # running from a source tree without a build
    _PLUGIN_VERSION = 'unknown'

# Qt.UserRole already carries the ctx-index mapping of the intensity
# table, so numeric sort keys need a role of their own.
_SORT_ROLE = Qt.UserRole + 1

# Every progress report crosses threads as a queued signal, so a
# many-region run reports a sample of its units rather than all of them.
_MAX_PROGRESS_REPORTS = 200

# Header tooltips for the Pixel-based results table, keyed by the
# column names of `_analysis.COLUMNS`.
PIXEL_COLUMN_TOOLTIPS = {
    'region': (
        'Region id: the 0-based shape index for a Shapes region, or the '
        'label value for a Labels region. 0 means the whole image.'
    ),
    'slice': (
        'Z-plane index, for a per-Z-slice run. Blank when the whole '
        'volume was analysed at once.'
    ),
    'channel_a': 'Name of the first channel of the pair.',
    'channel_b': 'Name of the second channel of the pair.',
    'n_pixels': 'Number of pixels analysed in this region.',
    'pcc': (
        'Pearson correlation coefficient of the paired pixel '
        'intensities, -1 to 1. Measures how linearly the two channels '
        'co-vary.'
    ),
    'srcc': (
        'Spearman rank correlation, -1 to 1. Pearson on intensity '
        'ranks, so it is robust to outliers and to non-linear but '
        'monotonic relationships.'
    ),
    'icq': (
        "Li's intensity correlation quotient: the fraction of pixels "
        'where both channels deviate from their means in the same '
        'direction, from -0.5 (segregated) through 0 (random) to +0.5 '
        '(dependent).'
    ),
    'overlap': (
        'Manders overlap coefficient r, 0 to 1: threshold-free '
        'co-occurrence, insensitive to a brightness difference between '
        'the channels but blind to the sign of the relationship.'
    ),
    'k1': (
        'Manders split coefficient k1: the overlap dominated by pixels '
        'where channel A is bright.'
    ),
    'k2': (
        'Manders split coefficient k2: the overlap dominated by pixels '
        'where channel B is bright.'
    ),
    'tm1': (
        'Thresholded Manders coefficient tM1, 0 to 1: the fraction of '
        "channel A's intensity that sits in pixels where channel B is "
        'above its threshold.'
    ),
    'tm2': (
        'Thresholded Manders coefficient tM2, 0 to 1: the fraction of '
        "channel B's intensity that sits in pixels where channel A is "
        'above its threshold.'
    ),
    'threshold_a': 'Channel-A intensity threshold used for tM2.',
    'threshold_b': 'Channel-B intensity threshold used for tM1.',
}

# Header tooltips for the Object-based results table, keyed by the
# column names of `_objects.OBJECT_COLUMNS`.
OBJECT_COLUMN_TOOLTIPS = {
    'channel': 'Channel the object was segmented from.',
    'object': 'Label value of the object within its channel.',
    'n_pixels': 'Size of the object in pixels.',
    'centroid': 'Centroid coordinates of the object, in axis order.',
    'coincident': (
        "Whether the object's centroid falls inside an object of the "
        'other channel.'
    ),
    'overlap': (
        "Whether any of the object's pixels touch an object of the "
        'other channel.'
    ),
    'nn_distance_px': (
        "Distance in pixels from this object's centroid to the nearest "
        'centroid in the other channel. Blank when that channel has no '
        'objects. Not scaled to physical units.'
    ),
}


def _progress_step(total):
    """Report every Nth unit of ``total`` work, at most ~200 times."""
    return max(1, -(-int(total) // _MAX_PROGRESS_REPORTS))


def _shape_without_axis(shape, axis):
    return tuple(s for i, s in enumerate(shape) if i != axis)


def _slice_index(row):
    """A result row's Z index, or ``None`` if it is not per-slice."""
    value = row.get('slice')
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    return int(value)


def _axis_view(arr, index, axis):
    """``arr`` at ``index`` along ``axis``, as a view.

    ``np.take`` would copy the whole plane/channel; every caller only
    reads the result, so basic indexing is enough.
    """
    arr = np.asarray(arr)
    key = [slice(None)] * arr.ndim
    key[axis] = index
    return arr[tuple(key)]


def _display_region(region_source, label):
    """The region label as the user sees it in the viewer.

    napari's Shapes layer numbers shapes from 0 in its hover status,
    but ``Shapes.to_labels`` rasterises them from 1 - so a shapes run
    subtracts one for display. Labels regions carry their own label
    values and are shown as-is. Result rows always store the
    canonical (rasterised) label; this is the only place the two
    conventions are reconciled for presentation.
    """
    if region_source == 'shapes' and label > 0:
        return label - 1
    return label


def _reject_multiscale(*layers):
    """True (after warning) if any of ``layers`` is multiscale.

    Multiscale data is a pyramid of arrays, not one array, so none of
    the compute layers can consume it.
    """
    if any(getattr(layer, 'multiscale', False) for layer in layers):
        show_warning(
            'Multiscale images are not supported; extract a single '
            'resolution level first.'
        )
        return True
    return False


def _base_provenance():
    """Provenance fields every export shares."""
    return {
        'plugin_version': _PLUGIN_VERSION,
        'analysed_at': datetime.now()
        .astimezone()
        .isoformat(timespec='seconds'),
    }


def _layer_name(layer):
    return '' if layer is None else layer.name


def _layer_geometry(layer, drop_axis=None):
    """``scale``/``translate`` kwargs placing a derived layer on ``layer``.

    Everything the plugin adds to the viewer is computed in the source
    layer's data coordinates; without these an anisotropic or offset
    image gets misaligned overlays. ``drop_axis`` removes a channel
    axis the derived data no longer has.
    """
    scale = list(layer.scale)
    translate = list(layer.translate)
    if drop_axis is not None:
        del scale[drop_axis]
        del translate[drop_axis]
    return {'scale': tuple(scale), 'translate': tuple(translate)}


def _check_aligned(region_layer, geometry):
    """Raise ``ValueError`` if ``region_layer`` sits elsewhere in the world.

    Regions are rasterised in the image's data coordinates, which only
    matches what the user drew when both layers share scale and
    translate. Leading axes a 2D region layer lacks are not compared.
    """
    n = min(region_layer.ndim, len(geometry['scale']))
    for key in ('scale', 'translate'):
        ours = np.asarray(getattr(region_layer, key), dtype=float)[-n:]
        theirs = np.asarray(geometry[key], dtype=float)[-n:]
        if not np.allclose(ours, theirs):
            raise ValueError(
                f"Region layer '{region_layer.name}' has a different "
                f'{key} from the image, so its regions would not line '
                'up with the pixels analysed.'
            )


def write_csv_rows(path, rows, columns, provenance=None):
    """Write ``rows`` as CSV, with ``provenance`` repeated on every row.

    Provenance travels as extra *columns* rather than a ``#``-commented
    header so the file still opens unchanged in Excel and in
    ``pandas.read_csv`` without a ``comment=`` flag. The repetition
    costs a few bytes per row and buys a file that says how it was
    produced - the numbers alone cannot be reproduced from the table.

    ``rows`` may be a generator - the Li ICA export is one row per
    pixel, so nothing materialises the whole table. Returns the number
    of data rows written.
    """
    provenance = dict(provenance or {})
    fieldnames = list(columns) + list(provenance)
    written = 0
    with open(path, 'w', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {**{key: row.get(key) for key in columns}, **provenance}
            )
            written += 1
    return written


def _object_rows_for_csv(rows):
    """Split each object row's centroid tuple into per-axis columns.

    A ``"(12.0, 34.5)"`` cell is awkward to use downstream, so the
    centroid becomes ``centroid_0``/``centroid_1``[/``centroid_2``] in
    the position the single column occupied.
    """
    ndim = max((len(row['centroid']) for row in rows), default=0)
    axis_columns = tuple(f'centroid_{i}' for i in range(ndim))
    columns = []
    for column in OBJECT_COLUMNS:
        if column == 'centroid':
            columns.extend(axis_columns)
        else:
            columns.append(column)
    flat = []
    for row in rows:
        out = {key: value for key, value in row.items() if key != 'centroid'}
        out.update(dict(zip(axis_columns, row['centroid'], strict=False)))
        flat.append(out)
    return flat, tuple(columns)


def _layer_value_range(layer):
    """``(low, high)`` intensity range of an image layer.

    Prefers napari's own ``contrast_limits_range``, which it has
    already computed, over scanning the array again - the array may be
    a large or lazily-loaded stack. Only an in-memory array is ever
    scanned; this runs on the GUI thread.
    """
    for attr in ('contrast_limits_range', 'contrast_limits'):
        limits = getattr(layer, attr, None)
        if limits is not None and len(limits) == 2:
            low, high = float(limits[0]), float(limits[1])
            if np.isfinite(low) and np.isfinite(high) and high > low:
                return low, high
    data = layer.data
    if not isinstance(data, np.ndarray) or data.size == 0:
        return 0.0, 1.0
    low, high = float(np.nanmin(data)), float(np.nanmax(data))
    return (low, high) if high > low else (low, low + 1.0)


def _decimals_for_span(span):
    """Digits worth showing for a range of this width.

    A 16-bit channel spans ~65535, where six decimals is noise; a
    normalised 0-1 channel needs them.
    """
    if span >= 1000:
        return 0
    if span >= 10:
        return 2
    return 6


def _format_cell(value):
    if isinstance(value, float):
        if np.isnan(value):
            return ''
        return f'{value:.4g}'
    return str(value)


def _format_object_cell(value):
    if isinstance(value, bool):
        return 'yes' if value else 'no'
    if isinstance(value, tuple):
        return '(' + ', '.join(f'{v:g}' for v in value) + ')'
    # bool is checked first: it is a subclass of int, not of float.
    if isinstance(value, float):
        return '' if np.isnan(value) else f'{value:.4g}'
    return str(value)


def _layer_color(layer):
    """An image layer's colormap colour (top of ramp) as a hex string.

    Returned as ``#rrggbb`` (unambiguously a single colour for
    ``add_points``); falls back to white when there's no usable
    colormap.
    """
    colormap = getattr(layer, 'colormap', None)
    if colormap is None:
        return 'white'
    try:
        rgb = np.asarray(colormap.map([1.0])).ravel()[:3]
    except (AttributeError, ValueError, TypeError):
        return 'white'
    r, g, b = (int(round(float(c) * 255)) for c in rgb)
    return f'#{r:02x}{g:02x}{b:02x}'


class _SortableTableItem(QTableWidgetItem):
    """Table item sorting on a numeric key instead of its text.

    Qt compares the displayed string, which would order n_pixels
    10000 before 9999 and '1e-05' above '0.5'. Cells without a
    numeric key (channel names, centroids) keep text comparison;
    NaN sorts below every real value so blank cells group together.
    """

    def __lt__(self, other):
        mine = self.data(_SORT_ROLE)
        theirs = other.data(_SORT_ROLE)
        if mine is None or theirs is None:
            return super().__lt__(other)
        if np.isnan(mine):
            return not np.isnan(theirs)
        if np.isnan(theirs):
            return False
        return mine < theirs


class FigureExportDialog(QDialog):
    """Modal dialog asking the user for figure size (inches) and DPI."""

    def __init__(self, parent, width_in, height_in, dpi):
        super().__init__(parent)
        self.setWindowTitle('Export figure')
        self._width = QDoubleSpinBox()
        self._width.setRange(1.0, 50.0)
        self._width.setDecimals(2)
        self._width.setSingleStep(0.5)
        self._width.setSuffix(' in')
        self._width.setValue(width_in)
        self._height = QDoubleSpinBox()
        self._height.setRange(1.0, 50.0)
        self._height.setDecimals(2)
        self._height.setSingleStep(0.5)
        self._height.setSuffix(' in')
        self._height.setValue(height_in)
        self._dpi = QSpinBox()
        self._dpi.setRange(50, 1200)
        self._dpi.setSingleStep(50)
        self._dpi.setValue(dpi)
        form = QFormLayout()
        form.addRow('Width:', self._width)
        form.addRow('Height:', self._height)
        form.addRow('DPI:', self._dpi)
        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout = QVBoxLayout()
        layout.addLayout(form)
        layout.addWidget(buttons)
        self.setLayout(layout)

    def width_in(self):
        return float(self._width.value())

    def height_in(self):
        return float(self._height.value())

    def dpi(self):
        return int(self._dpi.value())


# -- Qt layout builders -------------------------------------------------


def _make_group(title, *items, vertical=True):
    """Wrap *items* (widgets or sub-layouts) in a titled box."""
    group = QGroupBox(title)
    layout = QVBoxLayout() if vertical else QHBoxLayout()
    for item in items:
        if isinstance(item, QLayout):
            layout.addLayout(item)
        else:
            layout.addWidget(item)
    group.setLayout(layout)
    return group


def _make_grid_group(title, *items, columns=2):
    """Wrap ``items`` in a titled box, flowing across ``columns``.

    A single row of long labels overflows a docked panel, and the
    config scroll area has no horizontal scrollbar - so the overflow is
    unreachable, not merely tight. Flowing keeps every label readable
    at dock width without the height of one per row.
    """
    group = QGroupBox(title)
    layout = QGridLayout()
    for i, item in enumerate(items):
        layout.addWidget(item, i // columns, i % columns)
    group.setLayout(layout)
    return group


def _hbox(*widgets):
    layout = QHBoxLayout()
    for widget in widgets:
        layout.addWidget(widget)
    return layout


def _make_table(columns, tooltips=None):
    """A read-only, row-selectable, sortable results table.

    ``tooltips`` optionally maps a column name to the explanation
    shown when the user hovers that column's header.
    """
    table = QTableWidget(0, len(columns))
    table.setHorizontalHeaderLabels(list(columns))
    for c, name in enumerate(columns):
        tooltip = (tooltips or {}).get(name)
        if tooltip:
            table.horizontalHeaderItem(c).setToolTip(tooltip)
    table.setSelectionBehavior(QTableWidget.SelectRows)
    table.setEditTriggers(QTableWidget.NoEditTriggers)
    table.setSortingEnabled(True)
    return table


def _new_region_combo():
    combo = QComboBox()
    combo.addItem('None', None)
    return combo


def _fill_table(table, rows, columns, formatter, *, user_role=False):
    """Populate ``table`` from ``rows`` using ``formatter`` per cell.

    ``user_role`` stores each item's source row index in
    ``Qt.UserRole`` (the intensity table needs it to map a selected
    row back to its plot context, even after sorting).
    """
    table.setSortingEnabled(False)
    table.setRowCount(len(rows))
    for r, row in enumerate(rows):
        for c, key in enumerate(columns):
            value = row.get(key)
            item = _SortableTableItem(formatter(value))
            if isinstance(value, (bool, int, float, np.number, np.bool_)):
                item.setData(_SORT_ROLE, float(value))
            if user_role:
                item.setData(Qt.UserRole, r)
            table.setItem(r, c, item)
    table.setSortingEnabled(True)
    table.resizeColumnsToContents()


# -- workers / export ---------------------------------------------------


class _RunControl:
    """A tab's Run button, its progress bar and its running worker.

    Workers are generator functions: napari can only interrupt one
    between yields, so those yields double as the progress reports
    driving the bar. While a worker is alive the Run button reads
    'Cancel' and clicking it asks the worker to quit, discarding
    whatever it had produced - the button is never disabled, so a long
    run always has a way out.
    """

    def __init__(self, button, on_start):
        self.button = button
        self.progress = QProgressBar()
        self.progress.setTextVisible(False)
        self.progress.setMaximumHeight(10)
        self.progress.setVisible(False)
        self._run_text = button.text()
        self._on_start = on_start
        self._worker = None
        button.clicked.connect(self._on_clicked)

    def is_running(self):
        return self._worker is not None

    def start(self, worker, on_ready, on_error):
        """Take over the button and start ``worker``."""
        self._worker = worker
        # Busy indicator until (and unless) the worker reports a total.
        self.progress.setRange(0, 0)
        self.progress.setVisible(True)
        self.button.setText('Cancel')
        worker.yielded.connect(self._on_yielded)
        worker.returned.connect(on_ready)
        worker.errored.connect(on_error)
        worker.aborted.connect(self._on_aborted)
        worker.finished.connect(self._on_finished)
        worker.start()

    def _on_clicked(self):
        if self._worker is None:
            self._on_start()
        else:
            self._worker.quit()

    def _on_yielded(self, report):
        # Coarse steps yield nothing to report and only keep the busy
        # indicator running; a (done, total) pair drives a real bar.
        if not isinstance(report, tuple):
            return
        done, total = report
        maximum = max(int(total), 1)
        if self.progress.maximum() != maximum:
            self.progress.setRange(0, maximum)
        self.progress.setValue(min(int(done), maximum))

    def _on_aborted(self):
        show_info('Analysis cancelled.')

    def _on_finished(self):
        self._worker = None
        self.button.setText(self._run_text)
        self.progress.setVisible(False)
        self.progress.setRange(0, 1)
        self.progress.setValue(0)


def _export_canvas_figure(parent, canvas, default_name):
    """Prompt for size/DPI and path, then save the canvas figure."""
    figure = canvas._figure
    width, height = (float(v) for v in figure.get_size_inches())
    dlg = FigureExportDialog(parent, width, height, int(figure.get_dpi()))
    if dlg.exec_() != QDialog.Accepted:
        return
    path, _ = QFileDialog.getSaveFileName(
        parent,
        'Save figure',
        default_name,
        'PNG (*.png);;PDF (*.pdf);;SVG (*.svg);;TIFF (*.tif *.tiff)',
    )
    if not path:
        return
    # ValueError covers matplotlib rejecting the path itself, e.g. an
    # extension it has no writer for.
    try:
        canvas.save_figure(path, dlg.width_in(), dlg.height_in(), dlg.dpi())
    except (OSError, ValueError) as exc:
        show_warning(f'Could not write {path}: {exc}')
        return
    show_info(f'Wrote {path}')


# -- layer / region combos ----------------------------------------------


def _set_combo_choices(combo, layers):
    previous = combo.value
    combo.choices = layers
    if previous in layers:
        combo.value = previous
    else:
        # magicgui defaulted the value; remember it was not the
        # user's choice so _nudge_distinct may still move it.
        combo._auto_value = combo.value


def _nudge_distinct(combo_a, combo_b, candidates):
    """Point B at a layer different from A when they collide.

    Only auto-assigned collisions are nudged: once B holds a value
    the user picked themselves (anything other than the last value
    this machinery assigned), a collision is deliberate - e.g. A vs A
    as an autocorrelation control - and is left alone.
    """
    if len(candidates) < 2 or combo_a.value is not combo_b.value:
        return
    if combo_b.value is not getattr(combo_b, '_auto_value', None):
        return
    for layer in candidates:
        if layer is not combo_a.value:
            combo_b.value = layer
            combo_b._auto_value = layer
            break


def _refresh_region_combo(region_combo, candidates):
    """Rebuild a region combo from ``candidates``, keeping the choice."""
    previous = region_combo.currentData()
    region_combo.blockSignals(True)
    try:
        region_combo.clear()
        region_combo.addItem('None', None)
        for layer in candidates:
            region_combo.addItem(layer.name, layer)
        if previous is not None:
            _select_region_layer(region_combo, previous)
    finally:
        region_combo.blockSignals(False)


def _select_region_layer(region_combo, layer):
    """Select the entry holding ``layer``; leave the combo alone if absent."""
    for i in range(region_combo.count()):
        if region_combo.itemData(i) is layer:
            region_combo.setCurrentIndex(i)
            return


def _resolve_region(region_combo, spatial_shape, geometry=None):
    """The combo's region as a label mask, plus the layer it came from.

    ``geometry`` (from `_layer_geometry`) is the image's placement; a
    region layer placed differently raises ``ValueError``.
    """
    from napari.layers import Shapes

    layer = region_combo.currentData()
    if layer is None:
        return None, None
    if geometry is not None:
        _check_aligned(layer, geometry)
    if isinstance(layer, Shapes):
        return shapes_to_label_mask(layer, spatial_shape), layer
    return labels_to_label_mask(layer, spatial_shape), layer


class _LayerPair(NamedTuple):
    """A validated layer pair, data still lazy, plus its region."""

    a: Any
    b: Any
    label_mask: Any
    region_layer: Any
    geometry: dict


def _gather_pair(layer_a, layer_b, region_combo=None, what='image layers'):
    """Validate two selected layers and resolve the region against them.

    The checks every tab runs before a pairwise worker: both chosen,
    neither multiscale, same shape, region aligned. Layer data is left
    lazy - only ``.shape`` is read - so the worker materializes it off
    the GUI thread. Returns a `_LayerPair`, or ``None`` after warning.
    """
    if layer_a is None or layer_b is None:
        show_warning(f'Select both {what}.')
        return None
    if _reject_multiscale(layer_a, layer_b):
        return None
    a, b = layer_a.data, layer_b.data
    if a.shape != b.shape:
        show_warning(f'Shape mismatch: {a.shape} vs {b.shape}.')
        return None
    geometry = _layer_geometry(layer_a)
    label_mask = region_layer = None
    if region_combo is not None:
        try:
            label_mask, region_layer = _resolve_region(
                region_combo, a.shape, geometry
            )
        except ValueError as exc:
            show_warning(str(exc))
            return None
    return _LayerPair(a, b, label_mask, region_layer, geometry)


def _region_source_for(layer):
    from napari.layers import Shapes

    if layer is None:
        return 'none'
    return 'shapes' if isinstance(layer, Shapes) else 'labels'
