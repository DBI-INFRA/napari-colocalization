"""Top-level dock widget for the colocalisation plugin.

A single QWidget that wires the pure-compute layer (_metrics,
_masking, _analysis) to napari layers via magicgui combos, runs
the analysis on a background thread, and shows results in a
table + embedded scatter plot. CSV export is provided for the
current table. The whole widget is wrapped in a vertical scroll
area so it stays usable in narrow docks.
"""

import contextlib
import csv
from typing import TYPE_CHECKING

import numpy as np
from magicgui.widgets import create_widget
from napari.qt.threading import thread_worker
from napari.utils.notifications import show_info
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLayout,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ._analysis import (
    COLUMNS,
    analyse_all_to_all,
    analyse_pairwise,
)
from ._masking import labels_to_label_mask, shapes_to_label_mask
from ._plot import ScatterCanvas

if TYPE_CHECKING:
    import napari


def _shape_without_axis(shape, axis):
    return tuple(s for i, s in enumerate(shape) if i != axis)


def _format_cell(value):
    if isinstance(value, float):
        if np.isnan(value):
            return ''
        return f'{value:.4g}'
    return str(value)


class ColocalisationWidget(QWidget):
    """Dock widget: configuration + Run + results table + scatter."""

    def __init__(self, viewer: 'napari.viewer.Viewer'):
        super().__init__()
        self._viewer = viewer
        self._results = []
        self._plot_context = []
        self._region_layer = None
        self._region_source = 'none'

        # Configuration block — its own scroll area so a tall set
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

        # Results block — Run, table, scatter, export. Wrapped in
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

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(self._main_splitter)

        self._on_mode_changed()
        self._on_metrics_changed()
        self._connect_layer_events()

    # -- layout builders -------------------------------------------------

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
        self._pairwise_group = self._make_group(
            'Channels (pairwise)',
            self._image_a_combo.native,
            self._image_b_combo.native,
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
        self._region_combo = QComboBox()
        # Populated dynamically by _refresh_region_combo from the
        # viewer's Shapes + Labels layers, with a leading None entry.
        self._region_combo.addItem('None', None)
        return self._make_group('Region (optional)', self._region_combo)

    def _build_metrics_group(self):
        self._cb_pcc = QCheckBox('Pearson')
        self._cb_srcc = QCheckBox('Spearman')
        self._cb_mcc = QCheckBox('Manders')
        for cb, checked in (
            (self._cb_pcc, False),
            (self._cb_srcc, True),
            (self._cb_mcc, False),
        ):
            cb.setChecked(checked)
            cb.toggled.connect(self._on_metrics_changed)
        return self._make_group(
            'Correlation metrics',
            self._cb_pcc,
            self._cb_srcc,
            self._cb_mcc,
            vertical=False,
        )

    def _build_threshold_group(self):
        self._th_costes = QRadioButton('Costes (auto)')
        self._th_manual = QRadioButton('Manual')
        self._th_costes.setChecked(True)
        button_group = QButtonGroup(self)
        button_group.addButton(self._th_costes)
        button_group.addButton(self._th_manual)
        self._th_costes.toggled.connect(self._on_threshold_changed)
        self._th_a_spin = QDoubleSpinBox()
        self._th_b_spin = QDoubleSpinBox()
        for spin in (self._th_a_spin, self._th_b_spin):
            spin.setDecimals(6)
            spin.setRange(-1e9, 1e9)
            spin.setSingleStep(0.01)
        self._threshold_group = self._make_group(
            'Manders threshold',
            self._hbox(self._th_costes, self._th_manual),
            self._hbox(
                QLabel('T_a'),
                self._th_a_spin,
                QLabel('T_b'),
                self._th_b_spin,
            ),
        )
        return self._threshold_group

    def _build_run_row(self):
        self._run_button = QPushButton('Run')
        self._run_button.clicked.connect(self._on_run_clicked)
        return self._run_button

    def _build_results_group(self):
        self._results_group = QGroupBox('Results')
        self._table = QTableWidget(0, len(COLUMNS))
        self._table.setHorizontalHeaderLabels(list(COLUMNS))
        self._table.setSelectionBehavior(QTableWidget.SelectRows)
        self._table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._table.setSortingEnabled(True)
        # Qt's default sort indicator is descending on column 0;
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
        layout = QVBoxLayout()
        layout.addWidget(self._splitter)
        self._results_group.setLayout(layout)
        self._results_group.setVisible(False)
        return self._results_group

    def _build_export_row(self):
        self._export_button = QPushButton('Export CSV…')
        self._export_button.clicked.connect(self._on_export_clicked)
        self._export_button.setVisible(False)
        return self._export_button

    # -- UI state callbacks ---------------------------------------------

    def _on_mode_changed(self):
        pairwise = self._mode_pairwise.isChecked()
        self._pairwise_group.setVisible(pairwise)
        self._all_group.setVisible(not pairwise)

    def _on_metrics_changed(self):
        self._threshold_group.setVisible(self._cb_mcc.isChecked())

    def _on_threshold_changed(self):
        manual = self._th_manual.isChecked()
        self._th_a_spin.setEnabled(manual)
        self._th_b_spin.setEnabled(manual)

    def _on_stack_changed(self, layer):
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
        from napari.layers import Image

        images = self._layers_of(Image)
        for combo in (
            self._image_a_combo,
            self._image_b_combo,
            self._stack_combo,
        ):
            self._set_combo_choices(combo, images)
        self._refresh_region_combo()

        # Pairwise default: when magicgui populates A and B with the
        # same first image (or a new layer collapses them), nudge B
        # to a different image so the user can run pairwise without
        # changing the defaults.
        if (
            len(images) >= 2
            and self._image_a_combo.value is self._image_b_combo.value
        ):
            for layer in images:
                if layer is not self._image_a_combo.value:
                    self._image_b_combo.value = layer
                    break

    def _refresh_region_combo(self):
        from napari.layers import Labels, Shapes

        candidates = [
            layer
            for layer in self._viewer.layers
            if isinstance(layer, (Shapes, Labels))
        ]
        previous = self._region_combo.currentData()
        self._region_combo.blockSignals(True)
        try:
            self._region_combo.clear()
            self._region_combo.addItem('None', None)
            for layer in candidates:
                self._region_combo.addItem(layer.name, layer)
            if previous is not None:
                for i in range(self._region_combo.count()):
                    if self._region_combo.itemData(i) is previous:
                        self._region_combo.setCurrentIndex(i)
                        break
        finally:
            self._region_combo.blockSignals(False)

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
        if self._cb_mcc.isChecked():
            out.append('mcc')
        return tuple(out)

    def _resolve_region(self, spatial_shape):
        from napari.layers import Shapes

        layer = self._region_combo.currentData()
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
            show_info('Pick at least one metric.')
            return None
        common = {
            'metrics': metrics,
            'threshold_method': (
                'manual'
                if (self._th_manual.isChecked() and 'mcc' in metrics)
                else 'costes'
            ),
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
            show_info('Select both image layers.')
            return None
        a = np.asarray(layer_a.data)
        b = np.asarray(layer_b.data)
        if a.shape != b.shape:
            show_info(f'Shape mismatch: {a.shape} vs {b.shape}.')
            return None
        try:
            label_mask, region_layer = self._resolve_region(a.shape)
        except ValueError as exc:
            show_info(str(exc))
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
        }

    def _all_to_all_params(self, common):
        layer = self._stack_combo.value
        if layer is None:
            show_info('Select an image stack.')
            return None
        image = np.asarray(layer.data)
        axis = int(self._channel_axis_spin.value())
        if axis >= image.ndim:
            show_info(f'Channel axis {axis} >= ndim {image.ndim}.')
            return None
        spatial_shape = _shape_without_axis(image.shape, axis)
        try:
            label_mask, region_layer = self._resolve_region(spatial_shape)
        except ValueError as exc:
            show_info(str(exc))
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
        self._run_button.setEnabled(False)
        worker = self._run_worker(params)
        worker.returned.connect(self._on_results_ready)
        worker.errored.connect(self._on_worker_error)
        worker.finished.connect(lambda: self._run_button.setEnabled(True))
        worker.start()

    @staticmethod
    @thread_worker
    def _run_worker(params):
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
        )

    def _on_results_ready(self, payload):
        rows, channel_arrays, label_mask, region_source = payload
        # napari shapes hover shows 0-based indices, but Shapes.to_labels
        # rasterises non-zero labels starting at 1. Re-align so the
        # table/scatter/CSV match what the user sees in the Shapes layer.
        for row in rows:
            row['region_label'] = row['region']
            if region_source == 'shapes' and row['region'] > 0:
                row['region'] = row['region'] - 1
        self._results = rows
        self._populate_table(rows)
        self._plot_context = self._build_plot_context(
            rows, channel_arrays, label_mask
        )
        self._results_group.setVisible(bool(rows))
        self._export_button.setVisible(bool(rows))
        if self._plot_context:
            self._table.clearSelection()
            self._table.selectRow(0)
            self._on_row_selected()
        else:
            self._scatter.clear()
        # show_info(f'Computed {len(rows)} row(s).')

    def _on_worker_error(self, exc):
        show_info(f'Analysis failed: {exc}')

    # -- table / plot --------------------------------------------------

    def _populate_table(self, rows):
        self._table.setSortingEnabled(False)
        self._table.setRowCount(len(rows))
        for r, row in enumerate(rows):
            for c, key in enumerate(COLUMNS):
                item = QTableWidgetItem(_format_cell(row.get(key)))
                item.setData(Qt.UserRole, r)
                self._table.setItem(r, c, item)
        self._table.setSortingEnabled(True)
        self._table.resizeColumnsToContents()

    def _build_plot_context(self, rows, channel_arrays, label_mask):
        context = []
        for row in rows:
            a = channel_arrays.get(row['channel_a'])
            b = channel_arrays.get(row['channel_b'])
            mask = None
            mask_label = row.get('region_label', row['region'])
            if label_mask is not None and mask_label != 0:
                mask = label_mask == mask_label
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
        title = (
            f'region {row["region"]}: {row["channel_a"]} vs {row["channel_b"]}'
        )
        annotation_lines = []
        for key, label in (
            ('pcc', 'Pearson'),
            ('srcc', 'Spearman'),
            ('m1', 'M1'),
            ('m2', 'M2'),
        ):
            value = row.get(key)
            if value is not None and not (
                isinstance(value, float) and np.isnan(value)
            ):
                annotation_lines.append(f'{label} = {value:.4g}')
        self._scatter.update_plot(
            ctx['a'],
            ctx['b'],
            mask=ctx['mask'],
            threshold_a=row.get('threshold_a'),
            threshold_b=row.get('threshold_b'),
            title=title,
            annotation='\n'.join(annotation_lines),
        )

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
            show_info('No results to export.')
            return
        path, _ = QFileDialog.getSaveFileName(
            self, 'Save results CSV', 'colocalisation.csv', 'CSV (*.csv)'
        )
        if not path:
            return
        self.write_csv(path, self._results)
        show_info(f'Wrote {path}')

    @staticmethod
    def write_csv(path, rows):
        with open(path, 'w', newline='') as fh:
            writer = csv.DictWriter(fh, fieldnames=list(COLUMNS))
            writer.writeheader()
            for row in rows:
                writer.writerow({k: row.get(k) for k in COLUMNS})
