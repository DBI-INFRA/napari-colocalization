"""The Pixel-based tab of the colocalization dock.

Whole-image or per-region intensity correlation/co-occurrence: the
mode, channel, region, metric and threshold controls, the per-region
result table with its cytofluorogram, CSV/figure export and the
colocalized-pixel mask layer.
"""

import csv

import numpy as np
from magicgui.widgets import create_widget
from napari.qt.threading import thread_worker
from napari.utils.notifications import show_info, show_warning
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from ._analysis import COLUMNS, analyse_all_to_all, analyse_pairwise
from ._metrics import costes_regression
from ._plot import ScatterCanvas
from ._widget_shared import (
    _axis_view,
    _display_region,
    _export_canvas_figure,
    _fill_table,
    _format_cell,
    _hbox,
    _make_group,
    _make_table,
    _new_region_combo,
    _nudge_distinct,
    _refresh_region_combo,
    _region_source_for,
    _reject_multiscale,
    _resolve_region,
    _run_in_background,
    _set_combo_choices,
    _shape_without_axis,
    _slice_index,
)


class PixelTab(QWidget):
    """Configuration + Run + results table + cytofluorogram."""

    def __init__(self, viewer, on_open_diagnostics):
        super().__init__()
        self._viewer = viewer
        # Called with the (image A, image B, region layer) of the last
        # pairwise run, or None; the container owns the tab switch so
        # this tab never reaches into the Diagnostics tab.
        self._on_open_diagnostics = on_open_diagnostics
        self._results = []
        self._plot_context = []
        self._region_layer = None
        self._region_source = 'none'
        self._threshold_method = 'costes'
        self._run_mode = None
        # Full-resolution arrays / mask of the last run, kept so the
        # "Add coloc mask" output layer can be built from a selected row.
        self._channel_arrays = {}
        self._label_mask = None
        self._slice_axis = None

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

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._main_splitter)

        self._on_mode_changed()
        self._on_metrics_changed()
        self._on_threshold_changed()

    # -- layout builders -------------------------------------------------

    def _build_mode_group(self):
        self._mode_pairwise = QRadioButton('Pairwise (two layers)')
        self._mode_all = QRadioButton('All-to-all (one layer + channel axis)')
        self._mode_pairwise.setChecked(True)
        self._mode_pairwise.toggled.connect(self._on_mode_changed)
        return _make_group(
            'Mode', self._mode_pairwise, self._mode_all, vertical=False
        )

    def _build_pairwise_group(self):
        self._image_a_combo = create_widget(
            label='Image A', annotation='napari.layers.Image'
        )
        self._image_b_combo = create_widget(
            label='Image B', annotation='napari.layers.Image'
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
        self._pairwise_group = _make_group(
            'Channels (pairwise)',
            self._image_a_combo.native,
            self._image_b_combo.native,
            _hbox(
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
        self._all_group = _make_group(
            'Channels (all-to-all)',
            self._stack_combo.native,
            _hbox(QLabel('Channel axis'), self._channel_axis_spin),
        )
        return self._all_group

    def _build_region_group(self):
        # Populated by refresh_layers from the viewer's Shapes +
        # Labels layers, keeping the leading None entry.
        self._region_combo = _new_region_combo()
        return _make_group('Region (optional)', self._region_combo)

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
        # Defaults: one metric from each family - PCC for correlation,
        # Manders for co-occurrence.
        for cb, checked in (
            (self._cb_pcc, True),
            (self._cb_srcc, False),
            (self._cb_icq, False),
            (self._cb_overlap, False),
            (self._cb_mcc, True),
        ):
            cb.setChecked(checked)
            cb.toggled.connect(self._on_metrics_changed)
        return _make_group(
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
        for spin in (self._th_a_spin, self._th_b_spin):
            spin.setDecimals(6)
            spin.setRange(-1e9, 1e9)
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
        self._threshold_group = _make_group(
            'Manders threshold',
            _hbox(QLabel('Method'), self._threshold_combo),
            self._manual_row,
        )
        return self._threshold_group

    def _build_run_row(self):
        self._run_button = QPushButton('Run')
        self._run_button.clicked.connect(self._on_run_clicked)
        return self._run_button

    def _build_results_group(self):
        self._results_group = QGroupBox('Results')
        self._table = _make_table(COLUMNS)
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
        self._significance_row = self._build_significance_row()
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
        layout.addWidget(self._significance_row)
        self._results_group.setLayout(layout)
        self._results_group.setVisible(False)
        return self._results_group

    def _build_significance_row(self):
        """Footer pointing from the coefficients to the actual test.

        A large correlation coefficient is not evidence that the
        channels colocalize; the Costes randomization test on the
        Diagnostics tab is what answers that.
        """
        self._significance_label = QLabel(
            'Correlation strength is not significance - validate with '
            'the Costes randomization test.'
        )
        self._significance_label.setWordWrap(True)
        self._open_diagnostics_button = QPushButton('Open Diagnostics')
        self._open_diagnostics_button.setToolTip(
            'Switch to the Diagnostics tab, carrying over the channels '
            'and region of the last pairwise run.'
        )
        self._open_diagnostics_button.clicked.connect(
            self._on_open_diagnostics_clicked
        )
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._significance_label, stretch=1)
        layout.addWidget(self._open_diagnostics_button)
        return row

    def _on_open_diagnostics_clicked(self):
        # Diagnostics are always pairwise, so only a pairwise run has
        # selections to carry over; an all-to-all run just switches tab.
        inputs = None
        if self._run_mode == 'pairwise':
            inputs = (
                self._image_a_combo.value,
                self._image_b_combo.value,
                self._region_combo.currentData(),
            )
        self._on_open_diagnostics(inputs)

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

    def _on_metrics_changed(self):
        self._threshold_group.setVisible(self._cb_mcc.isChecked())

    def _on_threshold_changed(self):
        self._manual_row.setVisible(
            self._threshold_combo.currentData() == 'manual'
        )

    def _on_stack_changed(self, layer):
        if layer is None:
            self._channel_axis_spin.setMaximum(0)
            return
        # Read the geometry off the layer rather than the data: a
        # lazy array must not be materialized just to change a combo.
        self._channel_axis_spin.setMaximum(max(int(layer.ndim) - 1, 0))
        if getattr(layer, 'multiscale', False):
            return
        shape = getattr(layer.data, 'shape', None)
        if shape:
            smallest = int(np.argmin(shape))
            if shape[smallest] <= 8:
                self._channel_axis_spin.setValue(smallest)

    def refresh_layers(self, images, labels_layers, region_layers):
        for combo in (self._image_a_combo, self._image_b_combo):
            _set_combo_choices(combo, images)
        _set_combo_choices(self._stack_combo, images)
        _refresh_region_combo(self._region_combo, region_layers)
        _nudge_distinct(self._image_a_combo, self._image_b_combo, images)

    def apply_theme(self, background, foreground):
        self._scatter.set_theme(background, foreground)
        # Re-render so the plot contents pick up the new colours too.
        if self._plot_context:
            self._on_row_selected()
        else:
            self._scatter.clear()

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
            'region_source': _region_source_for(
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
        if _reject_multiscale(layer_a, layer_b):
            return None
        a = np.asarray(layer_a.data)
        b = np.asarray(layer_b.data)
        if a.shape != b.shape:
            show_warning(f'Shape mismatch: {a.shape} vs {b.shape}.')
            return None
        try:
            label_mask, region_layer = _resolve_region(
                self._region_combo, a.shape
            )
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
        if _reject_multiscale(layer):
            return None
        image = np.asarray(layer.data)
        axis = int(self._channel_axis_spin.value())
        if axis >= image.ndim:
            show_warning(f'Channel axis {axis} >= ndim {image.ndim}.')
            return None
        spatial_shape = _shape_without_axis(image.shape, axis)
        try:
            label_mask, region_layer = _resolve_region(
                self._region_combo, spatial_shape
            )
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

    def _on_run_clicked(self):
        params = self.gather_params()
        if params is None:
            return
        self._region_layer = params.get('region_layer')
        self._region_source = params.get('region_source', 'none')
        self._threshold_method = params.get('threshold_method', 'costes')
        self._run_mode = params['mode']
        _run_in_background(
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
                name: _axis_view(params['image'], i, params['channel_axis'])
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
        self._results = rows
        self._region_source = region_source
        self._channel_arrays = channel_arrays
        self._label_mask = label_mask
        self._slice_axis = slice_axis
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

    def _display_rows(self, rows):
        """Rows with ``region`` mapped to the viewer's numbering."""
        return [
            {
                **row,
                'region': _display_region(self._region_source, row['region']),
            }
            for row in rows
        ]

    def _populate_table(self, rows):
        _fill_table(
            self._table,
            self._display_rows(rows),
            COLUMNS,
            _format_cell,
            user_role=True,
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
            si = _slice_index(row)
            if slice_axis is not None and si is not None:
                a = _axis_view(a, si, slice_axis)
                b = _axis_view(b, si, slice_axis)
                if lm is not None and np.asarray(lm).ndim > a.ndim:
                    lm = _axis_view(label_mask, si, slice_axis)
            mask = None
            if lm is not None and row['region'] != 0:
                mask = lm == row['region']
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
        self._highlight_regions(self._region_labels_from_ctx(selected_ctx))

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

    def _region_labels_from_ctx(self, ctx_indices):
        labels = []
        for ctx_index in ctx_indices:
            label = self._plot_context[ctx_index]['row']['region']
            if label > 0:
                labels.append(int(label))
        return labels

    def _render_scatter(self, ctx_index):
        ctx = self._plot_context[ctx_index]
        if ctx['a'] is None or ctx['b'] is None:
            self._scatter.clear()
            return
        row = ctx['row']
        # The channel pair is on the axis labels now; the title only
        # carries the row identity (region, and slice when per-slice).
        region = _display_region(self._region_source, row['region'])
        title = f'region {region}'
        slice_index = _slice_index(row)
        if slice_index is not None:
            title += f', slice {slice_index}'
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

    def _highlight_regions(self, region_labels):
        layer = self._region_layer
        if layer is None or not region_labels:
            self._clear_region_highlight()
            return
        try:
            if self._region_source == 'shapes':
                # Shapes.to_labels rasterises shape i as label i + 1;
                # the layer's own selection is indexed from 0.
                indices = {
                    int(label) - 1
                    for label in region_labels
                    if 0 <= int(label) - 1 < len(layer.data)
                }
                layer.selected_data = indices
                layer.refresh()
            elif self._region_source == 'labels':
                # napari Labels can only emphasise one label at a time;
                # with multiple selections, drop the focus filter so all
                # labels remain visible in the viewer.
                if len(region_labels) == 1:
                    layer.selected_label = int(region_labels[0])
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
        try:
            self.write_csv(path, self._results)
        except OSError as exc:
            show_warning(f'Could not write {path}: {exc}')
            return
        show_info(f'Wrote {path}')

    def _on_export_figure_clicked(self):
        if not self._results:
            show_warning('No figure to export.')
            return
        _export_canvas_figure(self, self._scatter, 'colocalization.png')

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
        a = np.asarray(a)
        b = np.asarray(b)
        label_mask = self._label_mask
        # A per-slice row's thresholds only describe its own plane, so
        # the mask is computed there and embedded in a full-size volume
        # to stay aligned with the image layer.
        si = _slice_index(row) if self._slice_axis is not None else None
        plane_a, plane_b = a, b
        if si is not None:
            plane_a = _axis_view(a, si, self._slice_axis)
            plane_b = _axis_view(b, si, self._slice_axis)
            if (
                label_mask is not None
                and np.asarray(label_mask).ndim > plane_a.ndim
            ):
                label_mask = _axis_view(label_mask, si, self._slice_axis)
        # Colocalized pixels: above threshold in both channels (the
        # thresholds the row's M1/M2 were computed against), optionally
        # restricted to the row's region.
        coloc = (plane_a > t_a) & (plane_b > t_b)
        if label_mask is not None and row['region'] != 0:
            coloc = coloc & (label_mask == row['region'])
        data = coloc.astype('uint8')
        if si is not None:
            volume = np.zeros(a.shape, dtype='uint8')
            index = [slice(None)] * a.ndim
            index[self._slice_axis] = si
            volume[tuple(index)] = data
            data = volume
        self._viewer.add_labels(
            data,
            name=f'coloc {row["channel_a"]} & {row["channel_b"]}',
        )

    def write_csv(self, path, rows):
        """Write ``rows`` to ``path`` with the table's column order."""
        with open(path, 'w', newline='') as fh:
            writer = csv.DictWriter(fh, fieldnames=list(COLUMNS))
            writer.writeheader()
            for row in self._display_rows(rows):
                writer.writerow({k: row.get(k) for k in COLUMNS})
