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
from napari.utils.notifications import show_warning
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
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
    _fill_table,
    _format_object_cell,
    _hbox,
    _layer_color,
    _make_group,
    _make_table,
    _new_region_combo,
    _nudge_distinct,
    _refresh_region_combo,
    _reject_multiscale,
    _resolve_region,
    _RunControl,
    _set_combo_choices,
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
        self._object_table = _make_table(
            OBJECT_COLUMNS, OBJECT_COLUMN_TOOLTIPS
        )
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
                label_mask, _ = _resolve_region(
                    self._obj_region_combo, a.shape
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
            # Labels layers have no single intensity colour; use defaults.
            'color_a': 'cyan',
            'color_b': 'magenta',
        }

    def _on_object_run_clicked(self):
        params = self._gather_object_params()
        if params is None:
            return
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
        if params['source'] == 'threshold':
            yield
            labels_a = label_objects(
                params['a'],
                threshold_method=params['method'],
                mask=params['mask'],
                min_size=params['min_size'],
            )
            yield
            labels_b = label_objects(
                params['b'],
                threshold_method=params['method'],
                mask=params['mask'],
                min_size=params['min_size'],
            )
        else:
            labels_a = params['labels_a']
            labels_b = params['labels_b']
        yield
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
        # Deferred until the run completes so a cancelled run leaves
        # the previous overlays in the viewer.
        self._clear_object_overlays()
        self._populate_object_table(rows)
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

    def _on_object_worker_error(self, exc):
        show_warning(f'Object analysis failed: {exc}')

    def _populate_object_table(self, rows):
        _fill_table(
            self._object_table, rows, OBJECT_COLUMNS, _format_object_cell
        )

    @staticmethod
    def _object_summary_text(summary, name_a, name_b):
        return (
            f'{name_a}: {summary["n_objects_a"]} objects '
            f'({summary["coincident_a"]} coincident, '
            f'{summary["overlap_a"]} overlapping)    '
            f'{name_b}: {summary["n_objects_b"]} objects '
            f'({summary["coincident_b"]} coincident, '
            f'{summary["overlap_b"]} overlapping)'
        )
