"""Top-level dock widget for the colocalization plugin.

A single QWidget that wires the pure-compute layers (_metrics,
_masking, _analysis, _diagnostics) to napari layers via magicgui
combos and runs the work on a background thread. A "Pixel-based"
tab with the per-region metric table + cytofluorogram and
CSV/figure export, a "Diagnostics" tab that renders one single-pair
diagnostic plot at a time (Costes randomization, Van Steensel CCF,
or Li ICA), and an "Object-based" tab for object-level analysis.
"""

import contextlib
import csv
from datetime import datetime
from typing import TYPE_CHECKING

import numpy as np
from magicgui.widgets import create_widget
from napari.qt.threading import thread_worker
from napari.utils.notifications import show_info, show_warning
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLayout,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ._analysis import (
    COLUMNS,
    analyse_all_to_all,
    analyse_pairwise,
)
from ._diagnostics import (
    costes_randomization_steps,
    li_ica,
    scramble_example,
    van_steensel_ccf_steps,
)
from ._masking import labels_to_label_mask, shapes_to_label_mask
from ._metrics import costes_regression
from ._objects import (
    OBJECT_COLUMNS,
    label_objects,
    nearest_neighbour_vectors,
    object_centroids,
    object_table,
)
from ._plot import DiagnosticCanvas, ScatterCanvas

if TYPE_CHECKING:
    import napari

try:  # pragma: no cover - matches the guard in __init__
    from ._version import version as _PLUGIN_VERSION
except ImportError:  # pragma: no cover
    _PLUGIN_VERSION = 'unknown'


def _shape_without_axis(shape, axis):
    return tuple(s for i, s in enumerate(shape) if i != axis)


def _write_csv(path, rows, columns, provenance=None):
    """Write ``rows`` as CSV, with ``provenance`` repeated on every row.

    Provenance travels as extra *columns* rather than a ``#``-commented
    header so the file still opens unchanged in Excel and in
    ``pandas.read_csv`` without a ``comment=`` flag. The repetition costs
    a few bytes per row and buys a file that says how it was produced -
    the numbers alone can't be reproduced from the table.

    ``rows`` may be a generator - the Li ICA export is one row per pixel,
    so nothing materialises the whole table. Returns the number of data
    rows written.
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


def _layer_value_range(layer):
    """``(low, high)`` intensity range of an image layer.

    Prefers napari's own ``contrast_limits_range``, which it has
    already computed, over scanning the array again - the array may be
    a large or lazily-loaded stack.
    """
    for attr in ('contrast_limits_range', 'contrast_limits'):
        limits = getattr(layer, attr, None)
        if limits is not None and len(limits) == 2:
            low, high = float(limits[0]), float(limits[1])
            if np.isfinite(low) and np.isfinite(high) and high > low:
                return low, high
    data = np.asarray(layer.data)
    if data.size == 0:
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


class ColocalizationWidget(QWidget):
    """Dock widget: configuration + Run + results table + scatter."""

    def __init__(self, viewer: 'napari.viewer.Viewer'):
        super().__init__()
        self._viewer = viewer
        self._results = []
        self._plot_context = []
        self._region_layer = None
        self._region_source = 'none'
        self._threshold_method = 'costes'
        # How the last run of each tab was configured, written into that
        # tab's CSV so a saved file records its own provenance.
        self._provenance = {}
        self._object_results = []
        self._object_provenance = {}
        self._diag_result = None
        self._diag_provenance = {}
        self._active_diag_worker = None
        # Full-resolution arrays / mask of the last run, kept so the
        # "Add coloc mask" output layer can be built from a selected row.
        self._channel_arrays = {}
        self._label_mask = None
        # Centroid/link layers added by the last object run, so a new
        # run can clear them first. Centroid point size defaults to a
        # value scaled to the image (``None`` = auto); once the user
        # manually resizes, that value is locked and carried over.
        self._object_overlay_layers = []
        self._centroid_size = None
        self._centroid_size_auto = None

        # Analysis families live on separate tabs: the multi-region
        # intensity table on one, the single-pair diagnostic plots on
        # the other. Inputs are not shared - diagnostics are always
        # pairwise, so each tab carries the channel selectors it needs.
        self._tabs = QTabWidget()
        self._tabs.addTab(self._build_intensity_tab(), 'Pixel-based')
        self._tabs.addTab(self._build_diagnostics_tab(), 'Diagnostics')
        self._tabs.addTab(self._build_object_tab(), 'Object-based')

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(self._tabs)

        self._on_mode_changed()
        self._on_metrics_changed()
        self._on_threshold_changed()
        self._on_diag_method_changed()
        self._on_obj_source_changed()
        self._connect_layer_events()

    # -- layout builders -------------------------------------------------

    def _build_intensity_tab(self):
        # Configuration block - its own scroll area so a tall set
        # of options doesn't squeeze the results panel below.
        config_inner = QWidget()
        config_layout = QVBoxLayout(config_inner)
        config_layout.addWidget(self._build_mode_group())
        config_layout.addWidget(self._build_pairwise_group())
        config_layout.addWidget(self._build_all_to_all_group())
        config_layout.addWidget(self._build_region_group())
        config_layout.addWidget(self._build_metrics_group())
        config_layout.addWidget(self._build_threshold_group())
        config_layout.addStretch(1)

        config_scroll = QScrollArea()
        config_scroll.setWidgetResizable(True)
        config_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        config_scroll.setWidget(config_inner)

        # Results block - Run, table, scatter, export. Wrapped in
        # its own scroll area so it scrolls independently when the
        # dock is short.
        results_inner = QWidget()
        results_layout = QVBoxLayout(results_inner)
        results_layout.addWidget(self._build_run_row())
        results_layout.addWidget(self._build_results_group(), stretch=1)
        results_layout.addWidget(self._build_export_row())
        results_inner.setMinimumHeight(360)

        results_scroll = QScrollArea()
        results_scroll.setWidgetResizable(True)
        results_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        results_scroll.setWidget(results_inner)

        # User-draggable divider between config and results.
        self._main_splitter = QSplitter(Qt.Vertical)
        self._main_splitter.addWidget(config_scroll)
        self._main_splitter.addWidget(results_scroll)
        self._main_splitter.setStretchFactor(0, 1)
        self._main_splitter.setStretchFactor(1, 2)
        self._main_splitter.setSizes([400, 500])

        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._main_splitter)
        return tab

    @staticmethod
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

    @staticmethod
    def _hbox(*widgets):
        layout = QHBoxLayout()
        for widget in widgets:
            layout.addWidget(widget)
        return layout

    @staticmethod
    def _make_table(columns):
        """A read-only, row-selectable, sortable results table."""
        table = QTableWidget(0, len(columns))
        table.setHorizontalHeaderLabels(list(columns))
        table.setSelectionBehavior(QTableWidget.SelectRows)
        table.setEditTriggers(QTableWidget.NoEditTriggers)
        table.setSortingEnabled(True)
        return table

    @staticmethod
    def _new_region_combo():
        combo = QComboBox()
        combo.addItem('None', None)
        return combo

    def _fill_table(self, table, rows, columns, formatter, *, user_role=False):
        """Populate ``table`` from ``rows`` using ``formatter`` per cell.

        ``user_role`` stores each item's source row index in
        ``Qt.UserRole`` (the intensity table needs it to map a selected
        row back to its plot context, even after sorting).
        """
        table.setSortingEnabled(False)
        table.setRowCount(len(rows))
        for r, row in enumerate(rows):
            for c, key in enumerate(columns):
                item = QTableWidgetItem(formatter(row.get(key)))
                if user_role:
                    item.setData(Qt.UserRole, r)
                table.setItem(r, c, item)
        table.setSortingEnabled(True)
        table.resizeColumnsToContents()

    def _build_mode_group(self):
        self._mode_pairwise = QRadioButton('Pairwise (two layers)')
        self._mode_all = QRadioButton('All-to-all (one layer + channel axis)')
        self._mode_pairwise.setChecked(True)
        self._mode_pairwise.toggled.connect(self._on_mode_changed)
        return self._make_group(
            'Mode', self._mode_pairwise, self._mode_all, vertical=False
        )

    def _build_pairwise_group(self):
        self._image_a_combo = create_widget(
            label='Image A', annotation='napari.layers.Image'
        )
        self._image_b_combo = create_widget(
            label='Image B', annotation='napari.layers.Image'
        )
        for combo in (self._image_a_combo, self._image_b_combo):
            combo.changed.connect(
                lambda _value=None: self._sync_threshold_spins()
            )
        # Per-Z-slice option (JACoP B "consider Z slices separately"):
        # only meaningful for 3D+ images; one result row per slice.
        self._per_slice_check = QCheckBox('Per Z-slice')
        self._per_slice_check.setToolTip(
            'Analyse each plane along the Z axis separately '
            '(one result row per slice). Requires a 3D image.'
        )
        self._slice_axis_spin = QSpinBox()
        self._slice_axis_spin.setMinimum(0)
        self._slice_axis_spin.setMaximum(3)
        self._pairwise_group = self._make_group(
            'Channels (pairwise)',
            self._image_a_combo.native,
            self._image_b_combo.native,
            self._hbox(
                self._per_slice_check,
                QLabel('Z axis'),
                self._slice_axis_spin,
            ),
        )
        return self._pairwise_group

    def _build_all_to_all_group(self):
        self._stack_combo = create_widget(
            label='Image stack', annotation='napari.layers.Image'
        )
        self._stack_combo.changed.connect(self._on_stack_changed)
        self._channel_axis_spin = QSpinBox()
        self._channel_axis_spin.setMinimum(0)
        self._channel_axis_spin.setMaximum(0)
        self._all_group = self._make_group(
            'Channels (all-to-all)',
            self._stack_combo.native,
            self._hbox(QLabel('Channel axis'), self._channel_axis_spin),
        )
        return self._all_group

    def _build_region_group(self):
        # Populated by _refresh_region_combo from the viewer's Shapes +
        # Labels layers, keeping the leading None entry.
        self._region_combo = self._new_region_combo()
        return self._make_group('Region (optional)', self._region_combo)

    def _build_metrics_group(self):
        self._cb_pcc = QCheckBox('Pearson')
        self._cb_srcc = QCheckBox('Spearman')
        self._cb_icq = QCheckBox('Li ICQ')
        self._cb_overlap = QCheckBox('Overlap (r, k1, k2)')
        self._cb_overlap.setToolTip(
            'Manders overlap coefficient r and split coefficients '
            'k1/k2 - threshold-free co-occurrence measures.'
        )
        self._cb_mcc = QCheckBox('Manders')
        for cb, checked in (
            (self._cb_pcc, False),
            (self._cb_srcc, True),
            (self._cb_icq, False),
            (self._cb_overlap, False),
            (self._cb_mcc, False),
        ):
            cb.setChecked(checked)
            cb.toggled.connect(self._on_metrics_changed)
        return self._make_group(
            'Colocalization metrics',
            self._cb_pcc,
            self._cb_srcc,
            self._cb_icq,
            self._cb_overlap,
            self._cb_mcc,
            vertical=False,
        )

    def _build_threshold_group(self):
        self._threshold_combo = QComboBox()
        for label, key in (
            ('Costes (auto)', 'costes'),
            ('Otsu', 'otsu'),
            ('Li', 'li'),
            ('Triangle', 'triangle'),
            ('Yen', 'yen'),
            ('Mean', 'mean'),
            ('IsoData', 'isodata'),
            ('Manual', 'manual'),
        ):
            self._threshold_combo.addItem(label, key)
        self._threshold_combo.currentIndexChanged.connect(
            self._on_threshold_changed
        )
        self._th_a_spin = QDoubleSpinBox()
        self._th_b_spin = QDoubleSpinBox()
        # Range/step/decimals are re-seeded from the selected layers by
        # _sync_threshold_spins; these are only placeholders for the
        # moment before any layer exists.
        for spin in (self._th_a_spin, self._th_b_spin):
            spin.setDecimals(6)
            spin.setRange(0.0, 1.0)
            spin.setSingleStep(0.01)
        # Manual T_a/T_b row, shown only when the method is 'manual'.
        self._manual_row = QWidget()
        manual_layout = QHBoxLayout(self._manual_row)
        manual_layout.setContentsMargins(0, 0, 0, 0)
        for widget in (
            QLabel('T_a'),
            self._th_a_spin,
            QLabel('T_b'),
            self._th_b_spin,
        ):
            manual_layout.addWidget(widget)
        self._threshold_group = self._make_group(
            'Manders threshold',
            self._hbox(QLabel('Method'), self._threshold_combo),
            self._manual_row,
        )
        return self._threshold_group

    def _build_run_row(self):
        self._run_button = QPushButton('Run')
        self._run_button.clicked.connect(self._on_run_clicked)
        return self._run_button

    def _build_results_group(self):
        self._results_group = QGroupBox('Results')
        self._table = self._make_table(COLUMNS)
        # Qt defaults the sort indicator to descending on column 0;
        # ascending matches users' expectation of "region 0, 1, 2".
        self._table.sortByColumn(0, Qt.AscendingOrder)
        self._table.itemSelectionChanged.connect(self._on_row_selected)
        self._scatter = ScatterCanvas()
        # Splitter so the user can re-balance table vs scatter; the
        # initial 60/40 split favours the table for many-region runs.
        self._splitter = QSplitter(Qt.Vertical)
        self._splitter.addWidget(self._table)
        self._splitter.addWidget(self._scatter)
        self._splitter.setStretchFactor(0, 3)
        self._splitter.setStretchFactor(1, 2)
        self._splitter.setSizes([300, 200])
        # Footer line summarising any regions whose metrics could not
        # be computed; hidden until a run produces such a case.
        self._summary_label = QLabel('')
        self._summary_label.setWordWrap(True)
        self._summary_label.setStyleSheet('color: goldenrod;')
        self._summary_label.setVisible(False)
        # Lock the cytofluorogram to [0, channel max] so the scatter is
        # comparable across regions/slices/images (JACoP B plot bounds).
        self._fixed_axes_check = QCheckBox('Fixed plot axes (0–max)')
        self._fixed_axes_check.setToolTip(
            'Scale the cytofluorogram to [0, channel max] instead of '
            'auto-fitting each selection, for comparable plots.'
        )
        self._fixed_axes_check.toggled.connect(
            lambda _checked: self._on_row_selected()
        )
        layout = QVBoxLayout()
        layout.addWidget(self._splitter)
        layout.addWidget(self._fixed_axes_check)
        layout.addWidget(self._summary_label)
        self._results_group.setLayout(layout)
        self._results_group.setVisible(False)
        return self._results_group

    def _build_export_row(self):
        self._export_button = QPushButton('Export CSV…')
        self._export_button.clicked.connect(self._on_export_clicked)
        self._export_figure_button = QPushButton('Export figure…')
        self._export_figure_button.clicked.connect(
            self._on_export_figure_clicked
        )
        self._add_mask_button = QPushButton('Add coloc mask layer')
        self._add_mask_button.setToolTip(
            'Add a Labels layer of the colocalized pixels (both channels '
            'above their Manders thresholds) for the selected row.'
        )
        self._add_mask_button.clicked.connect(self._on_add_mask_clicked)
        self._export_row = QWidget()
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(self._export_button)
        row.addWidget(self._export_figure_button)
        row.addWidget(self._add_mask_button)
        self._export_row.setLayout(row)
        self._export_row.setVisible(False)
        return self._export_row

    # -- UI state callbacks ---------------------------------------------

    def _on_mode_changed(self):
        pairwise = self._mode_pairwise.isChecked()
        self._pairwise_group.setVisible(pairwise)
        self._all_group.setVisible(not pairwise)
        # The mode decides which layer each threshold spinbox tracks.
        self._sync_threshold_spins()

    def _on_metrics_changed(self):
        self._threshold_group.setVisible(self._cb_mcc.isChecked())

    def _on_threshold_changed(self):
        self._manual_row.setVisible(
            self._threshold_combo.currentData() == 'manual'
        )
        self._sync_threshold_spins()

    def _threshold_spin_sources(self):
        """``((layer, spin), …)`` the manual thresholds apply to.

        In all-to-all mode one pair of thresholds is used for *every*
        channel pair, so both spinboxes take the whole stack's range.
        """
        if self._mode_pairwise.isChecked():
            return (
                (self._image_a_combo.value, self._th_a_spin),
                (self._image_b_combo.value, self._th_b_spin),
            )
        stack = self._stack_combo.value
        return ((stack, self._th_a_spin), (stack, self._th_b_spin))

    def _sync_threshold_spins(self):
        """Match the manual T_a/T_b spinboxes to the selected channels.

        Without this the boxes step by 0.01 over +/-1e9 with no hint of
        the channel's units, so reaching a threshold of 17000 on a
        16-bit image means holding the arrow key. Re-seeding on every
        layer change also discards a value left over from a channel
        with a completely different range, where it meant nothing.
        """
        for layer, spin in self._threshold_spin_sources():
            if layer is None:
                continue
            low, high = _layer_value_range(layer)
            span = high - low
            spin.setDecimals(_decimals_for_span(span))
            spin.setRange(low, high)
            spin.setSingleStep(span / 100.0)
            spin.setValue(low + span / 2.0)
            spin.setToolTip(
                f'{layer.name}: {low:g} to {high:g} (step {span / 100.0:g})'
            )

    def _on_stack_changed(self, layer):
        self._sync_threshold_spins()
        if layer is None:
            self._channel_axis_spin.setMaximum(0)
            return
        ndim = int(np.asarray(layer.data).ndim)
        self._channel_axis_spin.setMaximum(max(ndim - 1, 0))
        shape = tuple(np.asarray(layer.data).shape)
        if shape:
            smallest = int(np.argmin(shape))
            if shape[smallest] <= 8:
                self._channel_axis_spin.setValue(smallest)

    # -- layer combo refresh --------------------------------------------

    def _layers_of(self, layer_type):
        return [
            layer
            for layer in self._viewer.layers
            if isinstance(layer, layer_type)
        ]

    def _connect_layer_events(self):
        self._viewer.layers.events.inserted.connect(self._on_layer_inserted)
        self._viewer.layers.events.removed.connect(self._on_layer_removed)
        for layer in self._viewer.layers:
            self._subscribe_layer_name(layer)
        self._refresh_layer_combos()

    def _on_layer_inserted(self, event):
        self._subscribe_layer_name(event.value)
        self._refresh_layer_combos()

    def _on_layer_removed(self, event):
        self._unsubscribe_layer_name(event.value)
        self._refresh_layer_combos()

    def _subscribe_layer_name(self, layer):
        with contextlib.suppress(AttributeError):
            layer.events.name.connect(self._refresh_layer_combos)

    def _unsubscribe_layer_name(self, layer):
        with contextlib.suppress(
            AttributeError, TypeError, ValueError, KeyError
        ):
            layer.events.name.disconnect(self._refresh_layer_combos)

    def _refresh_layer_combos(self, _event=None):
        from napari.layers import Image, Labels

        images = self._layers_of(Image)
        for combo in (
            self._image_a_combo,
            self._image_b_combo,
            self._stack_combo,
            self._diag_image_a_combo,
            self._diag_image_b_combo,
            self._obj_image_a_combo,
            self._obj_image_b_combo,
        ):
            self._set_combo_choices(combo, images)
        labels_layers = self._layers_of(Labels)
        for combo in (self._obj_labels_a_combo, self._obj_labels_b_combo):
            self._set_combo_choices(combo, labels_layers)
        self._refresh_region_combo(self._region_combo)
        self._refresh_region_combo(self._diag_region_combo)
        self._refresh_region_combo(self._obj_region_combo)

        # Default A/B pairs to two distinct layers: magicgui otherwise
        # populates both with the same first layer, which is never a
        # meaningful colocalization input.
        for combo_a, combo_b, candidates in (
            (self._image_a_combo, self._image_b_combo, images),
            (self._diag_image_a_combo, self._diag_image_b_combo, images),
            (self._obj_image_a_combo, self._obj_image_b_combo, images),
            (
                self._obj_labels_a_combo,
                self._obj_labels_b_combo,
                labels_layers,
            ),
        ):
            self._nudge_distinct(combo_a, combo_b, candidates)

    @staticmethod
    def _nudge_distinct(combo_a, combo_b, candidates):
        """Point B at a layer different from A when they collide."""
        if len(candidates) >= 2 and combo_a.value is combo_b.value:
            for layer in candidates:
                if layer is not combo_a.value:
                    combo_b.value = layer
                    break

    def _refresh_region_combo(self, region_combo):
        from napari.layers import Labels, Shapes

        candidates = [
            layer
            for layer in self._viewer.layers
            if isinstance(layer, (Shapes, Labels))
        ]
        previous = region_combo.currentData()
        region_combo.blockSignals(True)
        try:
            region_combo.clear()
            region_combo.addItem('None', None)
            for layer in candidates:
                region_combo.addItem(layer.name, layer)
            if previous is not None:
                for i in range(region_combo.count()):
                    if region_combo.itemData(i) is previous:
                        region_combo.setCurrentIndex(i)
                        break
        finally:
            region_combo.blockSignals(False)

    @staticmethod
    def _set_combo_choices(combo, layers):
        previous = combo.value
        combo.choices = layers
        if previous in layers:
            combo.value = previous

    # -- params --------------------------------------------------------

    def _selected_metrics(self):
        out = []
        if self._cb_pcc.isChecked():
            out.append('pcc')
        if self._cb_srcc.isChecked():
            out.append('srcc')
        if self._cb_icq.isChecked():
            out.append('icq')
        if self._cb_overlap.isChecked():
            out.append('overlap')
        if self._cb_mcc.isChecked():
            out.append('mcc')
        return tuple(out)

    def _resolve_region(self, spatial_shape, combo=None):
        from napari.layers import Shapes

        if combo is None:
            combo = self._region_combo
        layer = combo.currentData()
        if layer is None:
            return None, None
        if isinstance(layer, Shapes):
            return shapes_to_label_mask(layer, spatial_shape), layer
        return labels_to_label_mask(layer, spatial_shape), layer

    def _region_source_for(self, layer):
        from napari.layers import Shapes

        if layer is None:
            return 'none'
        return 'shapes' if isinstance(layer, Shapes) else 'labels'

    def gather_params(self):
        """Build the parameter dict for the current form state.

        Returns ``None`` (and surfaces a notification) if any
        required input is missing or invalid.
        """
        metrics = self._selected_metrics()
        if not metrics:
            show_warning('Pick at least one metric.')
            return None
        common = {
            'metrics': metrics,
            'threshold_method': self._threshold_combo.currentData(),
            'threshold_a': float(self._th_a_spin.value()),
            'threshold_b': float(self._th_b_spin.value()),
            'region_source': self._region_source_for(
                self._region_combo.currentData()
            ),
        }
        if self._mode_pairwise.isChecked():
            return self._pairwise_params(common)
        return self._all_to_all_params(common)

    def _pairwise_params(self, common):
        layer_a = self._image_a_combo.value
        layer_b = self._image_b_combo.value
        if layer_a is None or layer_b is None:
            show_warning('Select both image layers.')
            return None
        a = np.asarray(layer_a.data)
        b = np.asarray(layer_b.data)
        if a.shape != b.shape:
            show_warning(f'Shape mismatch: {a.shape} vs {b.shape}.')
            return None
        try:
            label_mask, region_layer = self._resolve_region(a.shape)
        except ValueError as exc:
            show_warning(str(exc))
            return None
        slice_axis = None
        if self._per_slice_check.isChecked():
            slice_axis = int(self._slice_axis_spin.value())
            if a.ndim < 3:
                show_warning('Per Z-slice requires a 3D image.')
                return None
            if slice_axis >= a.ndim:
                show_warning(f'Z axis {slice_axis} >= ndim {a.ndim}.')
                return None
        return {
            **common,
            'mode': 'pairwise',
            'a': a,
            'b': b,
            'label_mask': label_mask,
            'region_layer': region_layer,
            'channel_a': layer_a.name,
            'channel_b': layer_b.name,
            'slice_axis': slice_axis,
        }

    def _all_to_all_params(self, common):
        layer = self._stack_combo.value
        if layer is None:
            show_warning('Select an image stack.')
            return None
        image = np.asarray(layer.data)
        axis = int(self._channel_axis_spin.value())
        if axis >= image.ndim:
            show_warning(f'Channel axis {axis} >= ndim {image.ndim}.')
            return None
        spatial_shape = _shape_without_axis(image.shape, axis)
        try:
            label_mask, region_layer = self._resolve_region(spatial_shape)
        except ValueError as exc:
            show_warning(str(exc))
            return None
        return {
            **common,
            'mode': 'all_to_all',
            'image': image,
            'channel_axis': axis,
            'label_mask': label_mask,
            'region_layer': region_layer,
            'channel_names': [
                f'{layer.name}_{i}' for i in range(image.shape[axis])
            ],
        }

    # -- run -----------------------------------------------------------

    @staticmethod
    def _run_provenance(params):
        """How this analysis was configured, for the exported CSV.

        The metric columns alone don't say which threshold method
        produced ``threshold_a``/``threshold_b``, which region layer was
        used, or which version computed them - all of which a reader
        needs to reproduce or trust the numbers.
        """
        metrics = params['metrics']
        slice_axis = params.get('slice_axis')
        return {
            **_base_provenance(),
            'mode': params['mode'],
            'metrics': ' '.join(metrics),
            # Only meaningful when Manders was requested; blank
            # otherwise, so nobody reads it as having been applied.
            'threshold_method': (
                params['threshold_method'] if 'mcc' in metrics else ''
            ),
            'region_layer': _layer_name(params.get('region_layer')),
            'slice_axis': '' if slice_axis is None else slice_axis,
        }

    def _run_in_background(
        self, worker, button, on_ready, on_error, *, cancel=None, progress=None
    ):
        """Disable ``button``, wire the worker's signals, and start it.

        ``cancel`` and ``progress`` are only passed for work that can
        run long enough to be worth interrupting - a generator worker
        yielding ``(done, total)``. The other tabs finish in well under
        a second (measured), so they stay on the plain path.
        """
        button.setEnabled(False)
        worker.returned.connect(on_ready)
        worker.errored.connect(on_error)
        if progress is not None:
            progress.setValue(0)
            progress.setVisible(True)
            worker.yielded.connect(
                lambda step: self._update_progress(progress, step)
            )
        if cancel is not None:
            cancel.setEnabled(True)
        worker.finished.connect(
            lambda: self._reset_run_controls(button, cancel, progress)
        )
        worker.start()

    @staticmethod
    def _reset_run_controls(button, cancel, progress):
        button.setEnabled(True)
        if cancel is not None:
            cancel.setEnabled(False)
        if progress is not None:
            progress.setVisible(False)

    @staticmethod
    def _update_progress(bar, step):
        """Drive a 0-100 bar from a worker's ``(done, total)`` yield.

        The bar is in percent rather than raw iterations so a 100 000
        -iteration run repaints at most 100 times; the worker still
        yields every iteration, which is what keeps Cancel responsive.
        """
        try:
            done, total = step
        except (TypeError, ValueError):
            return
        percent = int(100 * done / total) if total else 0
        if percent != bar.value():
            bar.setValue(percent)

    def _on_run_clicked(self):
        params = self.gather_params()
        if params is None:
            return
        self._region_layer = params.get('region_layer')
        self._region_source = params.get('region_source', 'none')
        self._threshold_method = params.get('threshold_method', 'costes')
        # Snapshot how this run was configured *now*, not at export
        # time - by then the form (or a layer name) may have moved on.
        self._provenance = self._run_provenance(params)
        self._run_in_background(
            self._run_worker(params),
            self._run_button,
            self._on_results_ready,
            self._on_worker_error,
        )

    @staticmethod
    @thread_worker
    def _run_worker(params):
        region_warnings = []
        if params['mode'] == 'pairwise':
            rows = analyse_pairwise(
                params['a'],
                params['b'],
                label_mask=params['label_mask'],
                metrics=params['metrics'],
                threshold_method=params['threshold_method'],
                threshold_a=params['threshold_a'],
                threshold_b=params['threshold_b'],
                channel_a=params['channel_a'],
                channel_b=params['channel_b'],
                slice_axis=params.get('slice_axis'),
                region_warnings=region_warnings,
            )
            channel_arrays = {
                params['channel_a']: params['a'],
                params['channel_b']: params['b'],
            }
        else:
            rows = analyse_all_to_all(
                params['image'],
                channel_axis=params['channel_axis'],
                label_mask=params['label_mask'],
                metrics=params['metrics'],
                threshold_method=params['threshold_method'],
                threshold_a=params['threshold_a'],
                threshold_b=params['threshold_b'],
                channel_names=params['channel_names'],
                region_warnings=region_warnings,
            )
            channel_arrays = {
                name: np.take(params['image'], i, axis=params['channel_axis'])
                for i, name in enumerate(params['channel_names'])
            }
        return (
            rows,
            channel_arrays,
            params['label_mask'],
            params.get('region_source', 'none'),
            region_warnings,
            params.get('slice_axis'),
        )

    def _on_results_ready(self, payload):
        (
            rows,
            channel_arrays,
            label_mask,
            region_source,
            region_warnings,
            slice_axis,
        ) = payload
        # napari shapes hover shows 0-based indices, but Shapes.to_labels
        # rasterises non-zero labels starting at 1. Re-align so the
        # table/scatter/CSV match what the user sees in the Shapes layer.
        for row in rows:
            row['region_label'] = row['region']
            if region_source == 'shapes' and row['region'] > 0:
                row['region'] = row['region'] - 1
        self._results = rows
        self._channel_arrays = channel_arrays
        self._label_mask = label_mask
        self._populate_table(rows)
        self._plot_context = self._build_plot_context(
            rows, channel_arrays, label_mask, slice_axis
        )
        self._results_group.setVisible(bool(rows))
        self._export_row.setVisible(bool(rows))
        if self._plot_context:
            self._table.clearSelection()
            self._table.selectRow(0)
            self._on_row_selected()
        else:
            self._scatter.clear()
        self._report_region_warnings(region_warnings, len(rows))

    def _report_region_warnings(self, region_warnings, n_rows):
        """Summarise regions whose metrics could not be computed.

        The metric cells are already blank (NaN); this tells the
        user how many rows that affected and why, both inline under
        the table and as a napari warning notification.
        """
        if not region_warnings:
            self._summary_label.setText('')
            self._summary_label.setVisible(False)
            return
        n = len(region_warnings)
        summary = (
            f'{n} of {n_rows} row(s) had metrics that could not be '
            'computed (shown as blank cells).'
        )
        self._summary_label.setText(summary)
        self._summary_label.setVisible(True)
        # The full per-region reasons go to the notification; cap the
        # detail so a many-region run doesn't produce a wall of text.
        detail = '\n'.join(region_warnings[:10])
        if n > 10:
            detail += f'\n… and {n - 10} more'
        show_warning(f'{summary}\n{detail}')

    def _on_worker_error(self, exc):
        show_warning(f'Analysis failed: {exc}')

    # -- table / plot --------------------------------------------------

    def _populate_table(self, rows):
        self._fill_table(
            self._table, rows, COLUMNS, _format_cell, user_role=True
        )

    def _build_plot_context(
        self, rows, channel_arrays, label_mask, slice_axis=None
    ):
        context = []
        for row in rows:
            a = channel_arrays.get(row['channel_a'])
            b = channel_arrays.get(row['channel_b'])
            lm = label_mask
            # For a per-Z-slice row, scatter/regression should reflect
            # that plane only - take the slice of the channel arrays
            # (and of a full-volume label mask).
            s = row.get('slice')
            if (
                slice_axis is not None
                and s is not None
                and not (isinstance(s, float) and np.isnan(s))
            ):
                si = int(s)
                a = np.take(a, si, axis=slice_axis)
                b = np.take(b, si, axis=slice_axis)
                if lm is not None and np.asarray(lm).ndim > a.ndim:
                    lm = np.take(label_mask, si, axis=slice_axis)
            mask = None
            mask_label = row.get('region_label', row['region'])
            if lm is not None and mask_label != 0:
                mask = lm == mask_label
            context.append(
                {
                    'a': a,
                    'b': b,
                    'mask': mask,
                    'row': row,
                }
            )
        return context

    def _on_row_selected(self):
        selected_ctx = self._selected_ctx_indices()
        if not selected_ctx:
            self._scatter.clear()
            self._clear_region_highlight()
            return
        primary_ctx = self._primary_ctx_index(selected_ctx)
        self._render_scatter(primary_ctx)
        self._highlight_regions(self._mask_labels_from_ctx(selected_ctx))

    def _selected_ctx_indices(self):
        """Unique ctx indices from currently selected items, in row order."""
        seen = set()
        ordered = []
        for item in self._table.selectedItems():
            ctx_index = item.data(Qt.UserRole)
            if ctx_index is None or ctx_index in seen:
                continue
            if ctx_index >= len(self._plot_context):
                continue
            seen.add(ctx_index)
            ordered.append(ctx_index)
        return ordered

    def _primary_ctx_index(self, selected_ctx):
        """Most recently activated row's ctx, falling back to first."""
        current_row = self._table.currentRow()
        primary_item = (
            self._table.item(current_row, 0) if current_row >= 0 else None
        )
        if primary_item is not None:
            ctx_index = primary_item.data(Qt.UserRole)
            if (
                ctx_index is not None
                and 0 <= ctx_index < len(self._plot_context)
                and ctx_index in selected_ctx
            ):
                return ctx_index
        return selected_ctx[0]

    def _mask_labels_from_ctx(self, ctx_indices):
        labels = []
        for ctx_index in ctx_indices:
            row = self._plot_context[ctx_index]['row']
            mask_label = row.get('region_label', row['region'])
            if mask_label > 0:
                labels.append(int(mask_label))
        return labels

    def _render_scatter(self, ctx_index):
        ctx = self._plot_context[ctx_index]
        if ctx['a'] is None or ctx['b'] is None:
            self._scatter.clear()
            return
        row = ctx['row']
        # The channel pair is on the axis labels now; the title only
        # carries the row identity (region, and slice when per-slice).
        title = f'region {row["region"]}'
        slice_index = row.get('slice')
        if slice_index is not None and not (
            isinstance(slice_index, float) and np.isnan(slice_index)
        ):
            title += f', slice {int(slice_index)}'
        annotation_lines = []
        for key, label in (
            ('pcc', 'Pearson'),
            ('srcc', 'Spearman'),
            ('icq', 'ICQ'),
            ('overlap', 'Overlap r'),
            ('k1', 'k1'),
            ('k2', 'k2'),
            ('tm1', 'tM1'),
            ('tm2', 'tM2'),
        ):
            value = row.get(key)
            if value is not None and not (
                isinstance(value, float) and np.isnan(value)
            ):
                annotation_lines.append(f'{label} = {value:.4g}')
        slope, intercept = self._costes_line_for(ctx, row)
        xlim, ylim = self._fixed_axes_for(row)
        self._scatter.update_plot(
            ctx['a'],
            ctx['b'],
            mask=ctx['mask'],
            threshold_a=row.get('threshold_a'),
            threshold_b=row.get('threshold_b'),
            slope=slope,
            intercept=intercept,
            xlim=xlim,
            ylim=ylim,
            xlabel=f'{row["channel_a"]} intensity',
            ylabel=f'{row["channel_b"]} intensity',
            title=title,
            annotation='\n'.join(annotation_lines),
        )

    def _fixed_axes_for(self, row):
        """Common [0, max] axis bounds when 'Fixed plot axes' is on.

        Uses the full channel arrays (not the selected subset) so the
        scale is identical for every region/slice of the run.
        """
        if not self._fixed_axes_check.isChecked():
            return None, None
        full_a = self._channel_arrays.get(row['channel_a'])
        full_b = self._channel_arrays.get(row['channel_b'])
        if full_a is None or full_b is None:
            return None, None
        return (
            (0.0, float(np.nanmax(full_a))),
            (0.0, float(np.nanmax(full_b))),
        )

    def _costes_line_for(self, ctx, row):
        """Regression slope/intercept to draw, or ``(None, None)``.

        Only returned when the run used the Costes method and the
        row carries finite thresholds (i.e. Manders was computed),
        so the line matches the threshold pair already shown.
        """
        if self._threshold_method != 'costes':
            return None, None
        t_a = row.get('threshold_a')
        t_b = row.get('threshold_b')
        if not (
            isinstance(t_a, float)
            and isinstance(t_b, float)
            and np.isfinite(t_a)
            and np.isfinite(t_b)
        ):
            return None, None
        return costes_regression(ctx['a'], ctx['b'], mask=ctx['mask'])

    def _highlight_regions(self, mask_labels):
        layer = self._region_layer
        if layer is None or not mask_labels:
            self._clear_region_highlight()
            return
        try:
            if self._region_source == 'shapes':
                indices = {
                    int(label) - 1
                    for label in mask_labels
                    if 0 <= int(label) - 1 < len(layer.data)
                }
                layer.selected_data = indices
                layer.refresh()
            elif self._region_source == 'labels':
                # napari Labels can only emphasise one label at a time;
                # with multiple selections, drop the focus filter so all
                # labels remain visible in the viewer.
                if len(mask_labels) == 1:
                    layer.selected_label = int(mask_labels[0])
                    layer.show_selected_label = True
                else:
                    layer.show_selected_label = False
        except (AttributeError, ValueError):
            pass

    def _clear_region_highlight(self):
        layer = self._region_layer
        if layer is None:
            return
        try:
            if self._region_source == 'shapes':
                layer.selected_data = set()
                layer.refresh()
            elif self._region_source == 'labels':
                layer.show_selected_label = False
        except (AttributeError, ValueError):
            pass

    # -- export --------------------------------------------------------

    def _on_export_clicked(self):
        if not self._results:
            show_warning('No results to export.')
            return
        path, _ = QFileDialog.getSaveFileName(
            self, 'Save results CSV', 'colocalization.csv', 'CSV (*.csv)'
        )
        if not path:
            return
        self.write_csv(path, self._results, COLUMNS, self._provenance)
        show_info(f'Wrote {len(self._results)} row(s) to {path}')

    def _on_export_figure_clicked(self):
        if not self._results:
            show_warning('No figure to export.')
            return
        self._export_canvas_figure(self._scatter, 'colocalization.png')

    def _export_canvas_figure(self, canvas, default_name):
        """Prompt for size/DPI and path, then save the canvas figure."""
        figure = canvas._figure
        width, height = (float(v) for v in figure.get_size_inches())
        dlg = FigureExportDialog(self, width, height, int(figure.get_dpi()))
        if dlg.exec_() != QDialog.Accepted:
            return
        path, _ = QFileDialog.getSaveFileName(
            self,
            'Save figure',
            default_name,
            'PNG (*.png);;PDF (*.pdf);;SVG (*.svg);;TIFF (*.tif *.tiff)',
        )
        if not path:
            return
        canvas.save_figure(path, dlg.width_in(), dlg.height_in(), dlg.dpi())
        show_info(f'Wrote {path}')

    def _on_add_mask_clicked(self):
        ctx_indices = self._selected_ctx_indices()
        if not ctx_indices:
            show_warning('Select a result row first.')
            return
        row = self._plot_context[ctx_indices[0]]['row']
        t_a = row.get('threshold_a')
        t_b = row.get('threshold_b')
        if not (np.isfinite(t_a) and np.isfinite(t_b)):
            show_warning(
                'Selected row has no Manders thresholds - '
                'run with the Manders metric first.'
            )
            return
        a = self._channel_arrays.get(row['channel_a'])
        b = self._channel_arrays.get(row['channel_b'])
        if a is None or b is None:
            show_warning('Channel data is unavailable for this row.')
            return
        # Colocalized pixels: above threshold in both channels (the
        # thresholds the row's M1/M2 were computed against), optionally
        # restricted to the row's region.
        coloc = (np.asarray(a) > t_a) & (np.asarray(b) > t_b)
        mask_label = row.get('region_label', row['region'])
        if self._label_mask is not None and mask_label != 0:
            coloc = coloc & (self._label_mask == mask_label)
        self._viewer.add_labels(
            coloc.astype('uint8'),
            name=f'coloc {row["channel_a"]} & {row["channel_b"]}',
        )

    @staticmethod
    def write_csv(path, rows, columns=COLUMNS, provenance=None):
        _write_csv(path, rows, columns, provenance)

    # == Diagnostics tab ===============================================

    def _build_diagnostics_tab(self):
        inner = QWidget()
        layout = QVBoxLayout(inner)
        layout.addWidget(self._build_diag_channels_group())
        layout.addWidget(self._build_diag_region_group())
        layout.addWidget(self._build_diag_method_group())
        layout.addWidget(self._build_diag_params())
        layout.addWidget(self._build_diag_run_row())
        layout.addWidget(self._build_diag_results_group(), stretch=1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setWidget(inner)

        tab = QWidget()
        tab_layout = QVBoxLayout(tab)
        tab_layout.setContentsMargins(0, 0, 0, 0)
        tab_layout.addWidget(scroll)
        return tab

    def _build_diag_channels_group(self):
        self._diag_image_a_combo = create_widget(
            label='Image A', annotation='napari.layers.Image'
        )
        self._diag_image_b_combo = create_widget(
            label='Image B', annotation='napari.layers.Image'
        )
        return self._make_group(
            'Channels (pairwise)',
            self._diag_image_a_combo.native,
            self._diag_image_b_combo.native,
        )

    def _build_diag_region_group(self):
        self._diag_region_combo = self._new_region_combo()
        return self._make_group('Region (optional)', self._diag_region_combo)

    def _build_diag_method_group(self):
        self._diag_method_combo = QComboBox()
        self._diag_method_combo.addItem('Costes randomization', 'costes')
        self._diag_method_combo.addItem('Van Steensel CCF', 'ccf')
        self._diag_method_combo.addItem('Li ICA', 'ica')
        self._diag_method_combo.currentIndexChanged.connect(
            self._on_diag_method_changed
        )
        return self._make_group('Diagnostic', self._diag_method_combo)

    def _build_diag_params(self):
        self._costes_niter = QSpinBox()
        self._costes_niter.setRange(10, 100000)
        self._costes_niter.setValue(200)
        self._costes_block = QSpinBox()
        self._costes_block.setRange(1, 512)
        self._costes_block.setValue(8)
        self._diag_costes_group = self._make_group(
            'Costes parameters',
            self._hbox(QLabel('Iterations'), self._costes_niter),
            self._hbox(QLabel('Block size (px)'), self._costes_block),
        )

        self._ccf_max_shift = QSpinBox()
        self._ccf_max_shift.setRange(1, 500)
        self._ccf_max_shift.setValue(20)
        self._diag_ccf_group = self._make_group(
            'CCF parameters',
            self._hbox(QLabel('Max shift (px)'), self._ccf_max_shift),
        )

        self._diag_ica_group = self._make_group(
            'Li ICA',
            QLabel('No parameters - plots intensity vs covariance product.'),
        )

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        for group in (
            self._diag_costes_group,
            self._diag_ccf_group,
            self._diag_ica_group,
        ):
            layout.addWidget(group)
        return container

    def _build_diag_run_row(self):
        self._diag_run_button = QPushButton('Run diagnostic')
        self._diag_run_button.clicked.connect(self._on_diag_run_clicked)
        # Costes randomization is the one unbounded knob in the plugin
        # (Iterations goes to 100 000), so this tab gets a real Cancel.
        self._diag_cancel_button = QPushButton('Cancel')
        self._diag_cancel_button.setToolTip(
            'Stop the running diagnostic at the next iteration.'
        )
        self._diag_cancel_button.setEnabled(False)
        self._diag_cancel_button.clicked.connect(self._on_diag_cancel_clicked)
        self._diag_export_button = QPushButton('Export figure…')
        self._diag_export_button.clicked.connect(self._on_diag_export_clicked)
        # The figure alone isn't re-analysable; this saves the numbers
        # behind it (null distribution, CCF curve, or ICA points).
        self._diag_export_values_button = QPushButton('Export values…')
        self._diag_export_values_button.setToolTip(
            'Save the numbers behind the plot as CSV.'
        )
        self._diag_export_values_button.setEnabled(False)
        self._diag_export_values_button.clicked.connect(
            self._on_diag_export_values_clicked
        )
        # Scrambled-example output is a Costes-randomization concept.
        self._diag_scramble_button = QPushButton('Add scrambled example')
        self._diag_scramble_button.setToolTip(
            'Add one block-scrambled copy of Image B as an Image layer '
            '(the randomization the Costes test builds its null from).'
        )
        self._diag_scramble_button.clicked.connect(
            self._on_diag_scramble_clicked
        )
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._diag_run_button)
        layout.addWidget(self._diag_cancel_button)
        layout.addWidget(self._diag_export_button)
        layout.addWidget(self._diag_export_values_button)
        layout.addWidget(self._diag_scramble_button)
        return row

    def _build_diag_results_group(self):
        self._diagnostic_canvas = DiagnosticCanvas()
        self._diag_summary_label = QLabel('')
        self._diag_summary_label.setWordWrap(True)
        self._diag_progress = QProgressBar()
        self._diag_progress.setRange(0, 100)
        self._diag_progress.setVisible(False)
        group = QGroupBox('Diagnostic result')
        layout = QVBoxLayout()
        layout.addWidget(self._diagnostic_canvas, stretch=1)
        layout.addWidget(self._diag_progress)
        layout.addWidget(self._diag_summary_label)
        group.setLayout(layout)
        return group

    def _on_diag_method_changed(self):
        method = self._diag_method_combo.currentData()
        self._diag_costes_group.setVisible(method == 'costes')
        self._diag_ccf_group.setVisible(method == 'ccf')
        self._diag_ica_group.setVisible(method == 'ica')
        self._diag_scramble_button.setVisible(method == 'costes')

    def _gather_diag_params(self):
        layer_a = self._diag_image_a_combo.value
        layer_b = self._diag_image_b_combo.value
        if layer_a is None or layer_b is None:
            show_warning('Select both image layers for the diagnostic.')
            return None
        a = np.asarray(layer_a.data)
        b = np.asarray(layer_b.data)
        if a.shape != b.shape:
            show_warning(f'Shape mismatch: {a.shape} vs {b.shape}.')
            return None
        try:
            label_mask, region_layer = self._resolve_region(
                a.shape, combo=self._diag_region_combo
            )
        except ValueError as exc:
            show_warning(str(exc))
            return None
        # Diagnostics are single-ROI: collapse any multi-region label
        # mask to one boolean "analyse here" mask.
        mask = None if label_mask is None else (label_mask > 0)
        method = self._diag_method_combo.currentData()
        block_size = int(self._costes_block.value())
        # Validate preconditions synchronously: a worker exception would
        # both log a traceback and trigger the warning, surfacing twice.
        if method == 'costes' and any(
            dim // block_size < 1 for dim in a.shape
        ):
            show_warning('Costes block size is larger than the image.')
            return None
        if mask is not None and int(mask.sum()) < 2:
            show_warning('The selected region has fewer than 2 pixels.')
            return None
        return {
            'method': method,
            'a': a,
            'b': b,
            'mask': mask,
            'channel_a': layer_a.name,
            'channel_b': layer_b.name,
            'region_layer': region_layer,
            'n_iter': int(self._costes_niter.value()),
            'block_size': block_size,
            'max_shift': int(self._ccf_max_shift.value()),
        }

    @staticmethod
    def _diag_provenance_for(params):
        """How the diagnostic was configured, for the exported CSV."""
        method = params['method']
        return {
            **_base_provenance(),
            'diagnostic': method,
            'channel_a': params['channel_a'],
            'channel_b': params['channel_b'],
            'region_layer': _layer_name(params.get('region_layer')),
            # Each diagnostic has its own knobs; blank where not used.
            'n_iter': params['n_iter'] if method == 'costes' else '',
            'block_size': params['block_size'] if method == 'costes' else '',
            'max_shift': params['max_shift'] if method == 'ccf' else '',
        }

    def _on_diag_run_clicked(self):
        params = self._gather_diag_params()
        if params is None:
            return
        self._diag_provenance = self._diag_provenance_for(params)
        # Kept so Cancel has something to ask to stop.
        self._active_diag_worker = self._diag_worker(params)
        self._active_diag_worker.aborted.connect(self._on_diag_aborted)
        self._run_in_background(
            self._active_diag_worker,
            self._diag_run_button,
            self._on_diag_results_ready,
            self._on_diag_worker_error,
            cancel=self._diag_cancel_button,
            progress=self._diag_progress,
        )

    def _on_diag_cancel_clicked(self):
        worker = self._active_diag_worker
        if worker is None:
            return
        # quit() only *requests* the stop; the worker checks it after
        # the iteration in flight, so the button reports intent.
        worker.quit()
        self._diag_cancel_button.setEnabled(False)
        self._diag_cancel_button.setText('Cancelling…')

    def _on_diag_aborted(self):
        self._diag_cancel_button.setText('Cancel')
        self._diag_summary_label.setText('Diagnostic cancelled.')
        self._diagnostic_canvas.clear('Cancelled - run a diagnostic again')
        show_info('Diagnostic cancelled.')

    @staticmethod
    @thread_worker
    def _diag_worker(params):
        """Generator worker: ``yield from`` propagates progress upward.

        Yielding per iteration is what lets napari's worker notice an
        abort request; the sub-generators carry the algorithms.
        """
        method = params['method']
        a, b, mask = params['a'], params['b'], params['mask']
        if method == 'costes':
            result = yield from costes_randomization_steps(
                a,
                b,
                mask=mask,
                n_iter=params['n_iter'],
                block_size=params['block_size'],
            )
        elif method == 'ccf':
            shifts, ccf = yield from van_steensel_ccf_steps(
                a, b, mask=mask, max_shift=params['max_shift']
            )
            result = {'shifts': shifts, 'ccf': ccf}
        else:
            # Single pass, no meaningful chunk boundary to report.
            result = li_ica(a, b, mask=mask)
        return method, result, params['channel_a'], params['channel_b']

    def _on_diag_results_ready(self, payload):
        method, result, name_a, name_b = payload
        self._diag_result = (method, result)
        self._diag_export_values_button.setEnabled(True)
        title = f'{name_a} vs {name_b}'
        if method == 'costes':
            self._diagnostic_canvas.plot_costes(
                result['observed'],
                result['null'],
                result['p_value'],
                result['z_score'],
                title=title,
            )
            self._diag_summary_label.setText(
                f'Observed PCC = {result["observed"]:.4g}    '
                f'p = {result["p_value"]:.4g}    '
                f'z = {result["z_score"]:.3g}'
            )
        elif method == 'ccf':
            shifts, ccf = result['shifts'], result['ccf']
            self._diagnostic_canvas.plot_ccf(shifts, ccf, title=title)
            if np.any(np.isfinite(ccf)):
                peak = int(np.nanargmax(ccf))
                self._diag_summary_label.setText(
                    f'Peak Pearson r = {ccf[peak]:.4g} '
                    f'at shift {int(shifts[peak])} px'
                )
            else:
                self._diag_summary_label.setText(
                    'CCF undefined for this input.'
                )
        else:
            self._diagnostic_canvas.plot_ica(
                result['a'],
                result['b'],
                result['products'],
                names=(name_a, name_b),
                title=title,
            )
            self._diag_summary_label.setText(f'ICQ = {result["icq"]:.4g}')

    def _on_diag_worker_error(self, exc):
        show_warning(f'Diagnostic failed: {exc}')

    def _on_diag_scramble_clicked(self):
        layer_b = self._diag_image_b_combo.value
        if layer_b is None:
            show_warning('Select Image B first.')
            return
        try:
            scrambled = scramble_example(
                np.asarray(layer_b.data),
                block_size=int(self._costes_block.value()),
            )
        except ValueError as exc:
            show_warning(str(exc))
            return
        self._viewer.add_image(scrambled, name=f'{layer_b.name} (scrambled)')

    def _on_diag_export_clicked(self):
        self._export_canvas_figure(self._diagnostic_canvas, 'diagnostic.png')

    @staticmethod
    def _diag_rows_for_csv(method, result):
        """``(rows, columns, extra_provenance)`` for one diagnostic.

        The per-run scalars (p-value, ICQ, …) ride along as provenance
        columns so the summary line and the raw values stay in one file.
        """
        if method == 'costes':
            rows = (
                {'iteration': i, 'null_pcc': float(value)}
                for i, value in enumerate(result['null'])
            )
            return (
                rows,
                ('iteration', 'null_pcc'),
                {
                    'observed_pcc': result['observed'],
                    'p_value': result['p_value'],
                    'z_score': result['z_score'],
                },
            )
        if method == 'ccf':
            rows = (
                {'shift_px': int(shift), 'pearson_r': float(value)}
                for shift, value in zip(
                    result['shifts'], result['ccf'], strict=True
                )
            )
            return rows, ('shift_px', 'pearson_r'), {}
        # Li ICA: one row per pixel, so keep it lazy.
        rows = (
            {'a': float(x), 'b': float(y), 'product': float(p)}
            for x, y, p in zip(
                result['a'], result['b'], result['products'], strict=True
            )
        )
        return rows, ('a', 'b', 'product'), {'icq': result['icq']}

    def _on_diag_export_values_clicked(self):
        if self._diag_result is None:
            show_warning('Run a diagnostic first.')
            return
        method, result = self._diag_result
        path, _ = QFileDialog.getSaveFileName(
            self,
            'Save diagnostic values CSV',
            f'{method}.csv',
            'CSV (*.csv)',
        )
        if not path:
            return
        rows, columns, extra = self._diag_rows_for_csv(method, result)
        written = _write_csv(
            path, rows, columns, {**self._diag_provenance, **extra}
        )
        show_info(f'Wrote {written} row(s) to {path}')

    # == Object-based tab ==============================================

    def _build_object_tab(self):
        inner = QWidget()
        layout = QVBoxLayout(inner)
        layout.addWidget(self._build_obj_source_group())
        layout.addWidget(self._build_obj_threshold_group())
        layout.addWidget(self._build_obj_labels_group())
        layout.addWidget(self._build_obj_region_group())
        layout.addWidget(self._build_obj_overlay_group())
        layout.addWidget(self._build_obj_run_row())
        layout.addWidget(self._build_obj_results_group(), stretch=1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setWidget(inner)

        tab = QWidget()
        tab_layout = QVBoxLayout(tab)
        tab_layout.setContentsMargins(0, 0, 0, 0)
        tab_layout.addWidget(scroll)
        return tab

    def _build_obj_source_group(self):
        self._obj_source_combo = QComboBox()
        self._obj_source_combo.addItem('Threshold images', 'threshold')
        self._obj_source_combo.addItem('Labels layers', 'labels')
        self._obj_source_combo.currentIndexChanged.connect(
            self._on_obj_source_changed
        )
        return self._make_group('Objects from', self._obj_source_combo)

    def _build_obj_threshold_group(self):
        self._obj_image_a_combo = create_widget(
            label='Image A', annotation='napari.layers.Image'
        )
        self._obj_image_b_combo = create_widget(
            label='Image B', annotation='napari.layers.Image'
        )
        self._obj_method_combo = QComboBox()
        for label, key in (
            ('Otsu', 'otsu'),
            ('Li', 'li'),
            ('Triangle', 'triangle'),
            ('Yen', 'yen'),
            ('Mean', 'mean'),
            ('IsoData', 'isodata'),
        ):
            self._obj_method_combo.addItem(label, key)
        self._obj_min_size = QSpinBox()
        self._obj_min_size.setRange(0, 1_000_000)
        self._obj_threshold_group = self._make_group(
            'Threshold → objects',
            self._obj_image_a_combo.native,
            self._obj_image_b_combo.native,
            self._hbox(QLabel('Threshold'), self._obj_method_combo),
            self._hbox(QLabel('Min object size (px)'), self._obj_min_size),
        )
        return self._obj_threshold_group

    def _build_obj_labels_group(self):
        self._obj_labels_a_combo = create_widget(
            label='Labels A', annotation='napari.layers.Labels'
        )
        self._obj_labels_b_combo = create_widget(
            label='Labels B', annotation='napari.layers.Labels'
        )
        self._obj_labels_group = self._make_group(
            'Object labels',
            self._obj_labels_a_combo.native,
            self._obj_labels_b_combo.native,
        )
        return self._obj_labels_group

    def _build_obj_region_group(self):
        self._obj_region_combo = self._new_region_combo()
        self._obj_region_group = self._make_group(
            'Region (optional)', self._obj_region_combo
        )
        return self._obj_region_group

    def _build_obj_overlay_group(self):
        self._obj_points_check = QCheckBox('Show centroids (Points)')
        self._obj_points_check.setChecked(True)
        self._obj_links_check = QCheckBox(
            'Show nearest-neighbour links (Vectors)'
        )
        self._obj_links_check.setChecked(True)
        return self._make_group(
            'Overlays', self._obj_points_check, self._obj_links_check
        )

    def _build_obj_run_row(self):
        self._obj_run_button = QPushButton('Run object analysis')
        self._obj_run_button.clicked.connect(self._on_object_run_clicked)
        # Disabled rather than hidden until there are results: the
        # affordance stays visible, and the row doesn't reflow on Run.
        self._obj_export_button = QPushButton('Export CSV…')
        self._obj_export_button.setToolTip(
            'Save the per-object table (one row per object, with the '
            'centroid split into per-axis columns).'
        )
        self._obj_export_button.setEnabled(False)
        self._obj_export_button.clicked.connect(self._on_object_export_clicked)
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._obj_run_button)
        layout.addWidget(self._obj_export_button)
        return row

    def _build_obj_results_group(self):
        self._object_table = self._make_table(OBJECT_COLUMNS)
        self._obj_summary_label = QLabel('')
        self._obj_summary_label.setWordWrap(True)
        group = QGroupBox('Object results')
        layout = QVBoxLayout()
        layout.addWidget(self._object_table, stretch=1)
        layout.addWidget(self._obj_summary_label)
        group.setLayout(layout)
        return group

    def _on_obj_source_changed(self):
        threshold = self._obj_source_combo.currentData() == 'threshold'
        self._obj_threshold_group.setVisible(threshold)
        self._obj_region_group.setVisible(threshold)
        self._obj_labels_group.setVisible(not threshold)

    def _gather_object_params(self):
        if self._obj_source_combo.currentData() == 'threshold':
            layer_a = self._obj_image_a_combo.value
            layer_b = self._obj_image_b_combo.value
            if layer_a is None or layer_b is None:
                show_warning('Select both image layers.')
                return None
            a = np.asarray(layer_a.data)
            b = np.asarray(layer_b.data)
            if a.shape != b.shape:
                show_warning(f'Shape mismatch: {a.shape} vs {b.shape}.')
                return None
            try:
                label_mask, region_layer = self._resolve_region(
                    a.shape, combo=self._obj_region_combo
                )
            except ValueError as exc:
                show_warning(str(exc))
                return None
            return {
                'source': 'threshold',
                'a': a,
                'b': b,
                'mask': None if label_mask is None else (label_mask > 0),
                'method': self._obj_method_combo.currentData(),
                'min_size': int(self._obj_min_size.value()),
                'name_a': layer_a.name,
                'name_b': layer_b.name,
                'region_layer': region_layer,
                'color_a': _layer_color(layer_a),
                'color_b': _layer_color(layer_b),
            }
        layer_a = self._obj_labels_a_combo.value
        layer_b = self._obj_labels_b_combo.value
        if layer_a is None or layer_b is None:
            show_warning('Select both Labels layers.')
            return None
        labels_a = np.asarray(layer_a.data)
        labels_b = np.asarray(layer_b.data)
        if labels_a.shape != labels_b.shape:
            show_warning(
                f'Shape mismatch: {labels_a.shape} vs {labels_b.shape}.'
            )
            return None
        return {
            'source': 'labels',
            'labels_a': labels_a,
            'labels_b': labels_b,
            'name_a': layer_a.name,
            'name_b': layer_b.name,
            'region_layer': None,
            # Labels layers have no single intensity colour; use defaults.
            'color_a': 'cyan',
            'color_b': 'magenta',
        }

    @staticmethod
    def _object_provenance_for(params):
        """How the object run was configured, for the exported CSV."""
        from_threshold = params['source'] == 'threshold'
        return {
            **_base_provenance(),
            'objects_from': params['source'],
            # Only the threshold source segments anything; with Labels
            # layers the segmentation happened upstream.
            'threshold_method': params['method'] if from_threshold else '',
            'min_object_size': params['min_size'] if from_threshold else '',
            'region_layer': _layer_name(params.get('region_layer')),
        }

    def _on_object_run_clicked(self):
        params = self._gather_object_params()
        if params is None:
            return
        self._object_provenance = self._object_provenance_for(params)
        self._clear_object_overlays()
        self._run_in_background(
            self._object_worker(params),
            self._obj_run_button,
            self._on_object_results_ready,
            self._on_object_worker_error,
        )

    @staticmethod
    @thread_worker
    def _object_worker(params):
        if params['source'] == 'threshold':
            labels_a = label_objects(
                params['a'],
                threshold_method=params['method'],
                mask=params['mask'],
                min_size=params['min_size'],
            )
            labels_b = label_objects(
                params['b'],
                threshold_method=params['method'],
                mask=params['mask'],
                min_size=params['min_size'],
            )
        else:
            labels_a = params['labels_a']
            labels_b = params['labels_b']
        rows, summary = object_table(
            labels_a, labels_b, params['name_a'], params['name_b']
        )
        return (
            rows,
            summary,
            labels_a,
            labels_b,
            params['name_a'],
            params['name_b'],
            params['color_a'],
            params['color_b'],
        )

    def _on_object_results_ready(self, payload):
        rows, summary, labels_a, labels_b, name_a, name_b, color_a, color_b = (
            payload
        )
        self._object_results = rows
        self._populate_object_table(rows)
        self._obj_export_button.setEnabled(bool(rows))
        self._obj_summary_label.setText(
            self._object_summary_text(summary, name_a, name_b)
        )
        if not (
            self._obj_points_check.isChecked()
            or self._obj_links_check.isChecked()
        ):
            return
        centroids_a = object_centroids(labels_a)
        centroids_b = object_centroids(labels_b)
        if self._centroid_size is not None:
            size = self._centroid_size
        else:
            size = self._auto_centroid_size(labels_a.shape)
            self._centroid_size_auto = size
        if self._obj_points_check.isChecked():
            if centroids_a.size:
                self._add_overlay_layer(
                    self._viewer.add_points(
                        centroids_a,
                        name=f'{name_a} centroids',
                        size=size,
                        face_color=color_a,
                    )
                )
            if centroids_b.size:
                self._add_overlay_layer(
                    self._viewer.add_points(
                        centroids_b,
                        name=f'{name_b} centroids',
                        size=size,
                        face_color=color_b,
                    )
                )
        if self._obj_links_check.isChecked():
            vectors = nearest_neighbour_vectors(centroids_a, centroids_b)
            if vectors.shape[0]:
                self._add_overlay_layer(
                    self._viewer.add_vectors(
                        vectors,
                        name=f'{name_a} → {name_b} links',
                        edge_width=0.5,
                    )
                )

    def _add_overlay_layer(self, layer):
        self._object_overlay_layers.append(layer)

    def _clear_object_overlays(self):
        """Remove centroid/link layers added by the previous run.

        If the user has manually resized a centroid layer (its size no
        longer matches our auto value) that size is locked and carried
        over; otherwise we stay in auto mode and rescale to the next
        image. Vectors have no ``current_size`` and are skipped.
        """
        for layer in self._object_overlay_layers:
            size = getattr(layer, 'current_size', None)
            if size is not None:
                size = float(size)
                if self._centroid_size_auto is None or not np.isclose(
                    size, self._centroid_size_auto
                ):
                    self._centroid_size = size
            with contextlib.suppress(ValueError, KeyError):
                self._viewer.layers.remove(layer)
        self._object_overlay_layers = []

    @staticmethod
    def _auto_centroid_size(shape):
        """Centroid point size scaled to the image (floored at 2 px)."""
        extent = max(shape) if len(shape) else 1
        return float(max(2.0, round(extent / 200)))

    def _on_object_export_clicked(self):
        if not self._object_results:
            show_warning('No object results to export.')
            return
        path, _ = QFileDialog.getSaveFileName(
            self, 'Save object results CSV', 'objects.csv', 'CSV (*.csv)'
        )
        if not path:
            return
        rows, columns = _object_rows_for_csv(self._object_results)
        _write_csv(path, rows, columns, self._object_provenance)
        show_info(f'Wrote {len(rows)} object(s) to {path}')

    def _on_object_worker_error(self, exc):
        show_warning(f'Object analysis failed: {exc}')

    def _populate_object_table(self, rows):
        self._fill_table(
            self._object_table, rows, OBJECT_COLUMNS, _format_object_cell
        )

    @staticmethod
    def _object_summary_text(summary, name_a, name_b):
        def _channel(name, suffix):
            median = summary[f'median_nn_distance_{suffix}']
            text = (
                f'{name}: {summary[f"n_objects_{suffix}"]} objects '
                f'({summary[f"coincident_{suffix}"]} coincident, '
                f'{summary[f"overlap_{suffix}"]} overlapping'
            )
            if np.isfinite(median):
                text += f', median NN {median:.4g} px'
            return text + ')'

        return f'{_channel(name_a, "a")}    {_channel(name_b, "b")}'
