"""The Object-based tab of the colocalization dock.

Segments both channels into objects (threshold + connected
components, or two Labels layers), reports centre-particle
coincidence and object overlap per object, and adds the centroid /
nearest-neighbour overlays to the viewer.
"""

import contextlib

import numpy as np
from magicgui.widgets import create_widget
from napari.qt.threading import thread_worker
from napari.utils.notifications import show_info, show_warning
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QGroupBox,
    QLabel,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ._objects import (
    OBJECT_COLUMNS,
    label_objects,
    nearest_neighbour_vectors,
    object_centroids,
    object_table,
)
from ._widget_shared import (
    OBJECT_COLUMN_TOOLTIPS,
    _base_provenance,
    _fill_table,
    _format_object_cell,
    _gather_pair,
    _hbox,
    _layer_color,
    _layer_name,
    _make_group,
    _make_table,
    _new_region_combo,
    _nudge_distinct,
    _object_rows_for_csv,
    _refresh_region_combo,
    _RunControl,
    _set_combo_choices,
    write_csv_rows,
)


class ObjectTab(QWidget):
    """Object segmentation + per-object coincidence/overlap table."""

    def __init__(self, viewer):
        super().__init__()
        self._viewer = viewer
        # Centroid/link layers added by the last run, so a new run can
        # clear them first. Centroid point size defaults to a value
        # scaled to the image (``None`` = auto); once the user manually
        # resizes, that value is locked and carried over.
        self._object_overlay_layers = []
        self._centroid_size = None
        self._centroid_size_auto = None

        inner = QWidget()
        layout = QVBoxLayout(inner)
        layout.addWidget(self._build_source_group())
        layout.addWidget(self._build_threshold_group())
        layout.addWidget(self._build_labels_group())
        layout.addWidget(self._build_region_group())
        layout.addWidget(self._build_overlay_group())
        layout.addWidget(self._build_run_row())
        layout.addWidget(self._build_results_group(), stretch=1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setWidget(inner)

        tab_layout = QVBoxLayout(self)
        tab_layout.setContentsMargins(0, 0, 0, 0)
        tab_layout.addWidget(scroll)

        self._object_rows = []
        self._object_provenance = {}
        self._on_obj_source_changed()

    # -- layout builders -------------------------------------------------

    def _build_source_group(self):
        self._obj_source_combo = QComboBox()
        self._obj_source_combo.addItem('Threshold images', 'threshold')
        self._obj_source_combo.addItem('Labels layers', 'labels')
        self._obj_source_combo.currentIndexChanged.connect(
            self._on_obj_source_changed
        )
        return _make_group('Objects from', self._obj_source_combo)

    def _build_threshold_group(self):
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
        self._obj_threshold_group = _make_group(
            'Threshold → objects',
            self._obj_image_a_combo.native,
            self._obj_image_b_combo.native,
            _hbox(QLabel('Threshold'), self._obj_method_combo),
            _hbox(QLabel('Min object size (px)'), self._obj_min_size),
        )
        return self._obj_threshold_group

    def _build_labels_group(self):
        self._obj_labels_a_combo = create_widget(
            label='Labels A', annotation='napari.layers.Labels'
        )
        self._obj_labels_b_combo = create_widget(
            label='Labels B', annotation='napari.layers.Labels'
        )
        self._obj_labels_group = _make_group(
            'Object labels',
            self._obj_labels_a_combo.native,
            self._obj_labels_b_combo.native,
        )
        return self._obj_labels_group

    def _build_region_group(self):
        self._obj_region_combo = _new_region_combo()
        self._obj_region_group = _make_group(
            'Region (optional)', self._obj_region_combo
        )
        return self._obj_region_group

    def _build_overlay_group(self):
        self._obj_points_check = QCheckBox('Show centroids (Points)')
        self._obj_points_check.setChecked(True)
        self._obj_links_check = QCheckBox(
            'Show nearest-neighbour links (Vectors)'
        )
        self._obj_links_check.setChecked(True)
        return _make_group(
            'Overlays', self._obj_points_check, self._obj_links_check
        )

    def _build_run_row(self):
        self._obj_run_button = QPushButton('Run object analysis')
        self._obj_run_control = _RunControl(
            self._obj_run_button, self._on_object_run_clicked
        )
        row = QWidget()
        layout = QVBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._obj_run_button)
        layout.addWidget(self._obj_run_control.progress)
        return row

    def _build_results_group(self):
        # The per-object rows carry the counts a summary would repeat,
        # and the table sorts and exports; the label only duplicated it.
        self._object_table = _make_table(
            OBJECT_COLUMNS, OBJECT_COLUMN_TOOLTIPS
        )
        self._obj_export_button = QPushButton('Export CSV…')
        self._obj_export_button.setEnabled(False)
        self._obj_export_button.clicked.connect(self._on_object_export_clicked)
        group = QGroupBox('Object results')
        layout = QVBoxLayout()
        layout.addWidget(self._object_table, stretch=1)
        layout.addWidget(self._obj_export_button)
        group.setLayout(layout)
        return group

    def _on_obj_source_changed(self):
        threshold = self._obj_source_combo.currentData() == 'threshold'
        self._obj_threshold_group.setVisible(threshold)
        self._obj_region_group.setVisible(threshold)
        self._obj_labels_group.setVisible(not threshold)

    def refresh_layers(self, images, labels_layers, region_layers):
        for combo in (self._obj_image_a_combo, self._obj_image_b_combo):
            _set_combo_choices(combo, images)
        for combo in (self._obj_labels_a_combo, self._obj_labels_b_combo):
            _set_combo_choices(combo, labels_layers)
        _refresh_region_combo(self._obj_region_combo, region_layers)
        _nudge_distinct(
            self._obj_image_a_combo, self._obj_image_b_combo, images
        )
        _nudge_distinct(
            self._obj_labels_a_combo, self._obj_labels_b_combo, labels_layers
        )

    # -- run -----------------------------------------------------------

    def _gather_object_params(self):
        if self._obj_source_combo.currentData() == 'threshold':
            layer_a = self._obj_image_a_combo.value
            layer_b = self._obj_image_b_combo.value
            pair = _gather_pair(layer_a, layer_b, self._obj_region_combo)
            if pair is None:
                return None
            label_mask = pair.label_mask
            return {
                'source': 'threshold',
                'a': pair.a,
                'b': pair.b,
                'mask': None if label_mask is None else (label_mask > 0),
                'region_layer': pair.region_layer,
                'method': self._obj_method_combo.currentData(),
                'min_size': int(self._obj_min_size.value()),
                'name_a': layer_a.name,
                'name_b': layer_b.name,
                'color_a': _layer_color(layer_a),
                'color_b': _layer_color(layer_b),
                'geometry': pair.geometry,
            }
        layer_a = self._obj_labels_a_combo.value
        layer_b = self._obj_labels_b_combo.value
        pair = _gather_pair(layer_a, layer_b, what='Labels layers')
        if pair is None:
            return None
        return {
            'source': 'labels',
            'labels_a': pair.a,
            'labels_b': pair.b,
            'name_a': layer_a.name,
            'name_b': layer_b.name,
            # Labels layers have no single intensity colour; use defaults.
            'color_a': 'cyan',
            'color_b': 'magenta',
            'geometry': pair.geometry,
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

    def _on_object_export_clicked(self):
        if not self._object_rows:
            show_warning('No object results to export.')
            return
        path, _ = QFileDialog.getSaveFileName(
            self, 'Save object results CSV', 'objects.csv', 'CSV (*.csv)'
        )
        if not path:
            return
        rows, columns = _object_rows_for_csv(self._object_rows)
        try:
            write_csv_rows(path, rows, columns, self._object_provenance)
        except OSError as exc:
            show_warning(f'Could not write {path}: {exc}')
            return
        show_info(f'Wrote {path}')

    def _on_object_run_clicked(self):
        params = self._gather_object_params()
        if params is None:
            return
        # Provenance rides with the results so a cancelled or failed
        # run cannot relabel the previous run's export.
        params['provenance'] = self._object_provenance_for(params)
        self._obj_run_control.start(
            self._object_worker(params),
            self._on_object_results_ready,
            self._on_object_worker_error,
        )

    @staticmethod
    @thread_worker
    def _object_worker(params):
        # The bare yields between steps are the only points at which
        # the worker can honour a cancel request; the steps themselves
        # have no progress to report, so the bar stays a busy
        # indicator.
        # Lazy (dask/zarr) layer data loads here, on the worker
        # thread, so the click handler never blocks on it.
        if params['source'] == 'threshold':
            yield
            labels_a = label_objects(
                np.asarray(params['a']),
                threshold_method=params['method'],
                mask=params['mask'],
                min_size=params['min_size'],
            )
            yield
            labels_b = label_objects(
                np.asarray(params['b']),
                threshold_method=params['method'],
                mask=params['mask'],
                min_size=params['min_size'],
            )
        else:
            labels_a = np.asarray(params['labels_a'])
            labels_b = np.asarray(params['labels_b'])
        yield
        # object_table's summary dict is still there for API callers;
        # the tab shows the per-object rows only.
        rows, _summary = object_table(
            labels_a, labels_b, params['name_a'], params['name_b']
        )
        return {
            'rows': rows,
            'labels_a': labels_a,
            'labels_b': labels_b,
            **{
                key: params[key]
                for key in (
                    'name_a',
                    'name_b',
                    'color_a',
                    'color_b',
                    'geometry',
                    'provenance',
                )
            },
        }

    def _on_object_results_ready(self, payload):
        rows = payload['rows']
        labels_a = payload['labels_a']
        labels_b = payload['labels_b']
        name_a, name_b = payload['name_a'], payload['name_b']
        geometry = payload['geometry']
        self._object_provenance = payload['provenance']
        # Deferred until the run completes so a cancelled run leaves
        # the previous overlays in the viewer.
        self._clear_object_overlays()
        self._object_rows = rows
        self._obj_export_button.setEnabled(bool(rows))
        self._populate_object_table(rows)
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
                        face_color=payload['color_a'],
                        **geometry,
                    )
                )
            if centroids_b.size:
                self._add_overlay_layer(
                    self._viewer.add_points(
                        centroids_b,
                        name=f'{name_b} centroids',
                        size=size,
                        face_color=payload['color_b'],
                        **geometry,
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
                        **geometry,
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

    def _on_object_worker_error(self, exc):
        show_warning(f'Object analysis failed: {exc}')

    def _populate_object_table(self, rows):
        _fill_table(
            self._object_table, rows, OBJECT_COLUMNS, _format_object_cell
        )
