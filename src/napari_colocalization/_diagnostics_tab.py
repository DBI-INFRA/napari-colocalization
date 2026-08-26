"""The Diagnostics tab of the colocalization dock.

Renders one single-pair diagnostic at a time - the Costes
randomization null distribution, the Van Steensel cross-correlation
function, or the Li ICA scatters - plus the block-scrambled example
layer the Costes null is built from.
"""

import numpy as np
from magicgui.widgets import create_widget
from napari.qt.threading import thread_worker
from napari.utils.notifications import show_warning
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ._diagnostics import (
    costes_randomization,
    li_ica,
    scramble_example,
    van_steensel_ccf,
)
from ._plot import DiagnosticCanvas
from ._widget_shared import (
    _export_canvas_figure,
    _hbox,
    _make_group,
    _new_region_combo,
    _nudge_distinct,
    _refresh_region_combo,
    _reject_multiscale,
    _resolve_region,
    _run_in_background,
    _select_region_layer,
    _set_combo_choices,
)


class DiagnosticsTab(QWidget):
    """Channel pair + method + parameters, and one diagnostic plot."""

    def __init__(self, viewer):
        super().__init__()
        self._viewer = viewer

        inner = QWidget()
        layout = QVBoxLayout(inner)
        layout.addWidget(self._build_channels_group())
        layout.addWidget(self._build_region_group())
        layout.addWidget(self._build_method_group())
        layout.addWidget(self._build_params())
        layout.addWidget(self._build_run_row())
        layout.addWidget(self._build_results_group(), stretch=1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setWidget(inner)

        tab_layout = QVBoxLayout(self)
        tab_layout.setContentsMargins(0, 0, 0, 0)
        tab_layout.addWidget(scroll)

        self._on_diag_method_changed()

    # -- layout builders -------------------------------------------------

    def _build_channels_group(self):
        self._diag_image_a_combo = create_widget(
            label='Image A', annotation='napari.layers.Image'
        )
        self._diag_image_b_combo = create_widget(
            label='Image B', annotation='napari.layers.Image'
        )
        return _make_group(
            'Channels (pairwise)',
            self._diag_image_a_combo.native,
            self._diag_image_b_combo.native,
        )

    def _build_region_group(self):
        self._diag_region_combo = _new_region_combo()
        return _make_group('Region (optional)', self._diag_region_combo)

    def _build_method_group(self):
        self._diag_method_combo = QComboBox()
        self._diag_method_combo.addItem('Costes randomization', 'costes')
        self._diag_method_combo.addItem('Van Steensel CCF', 'ccf')
        self._diag_method_combo.addItem('Li ICA', 'ica')
        self._diag_method_combo.currentIndexChanged.connect(
            self._on_diag_method_changed
        )
        return _make_group('Diagnostic', self._diag_method_combo)

    def _build_params(self):
        self._costes_niter = QSpinBox()
        self._costes_niter.setRange(10, 100000)
        self._costes_niter.setValue(200)
        self._costes_block = QSpinBox()
        self._costes_block.setRange(1, 512)
        self._costes_block.setValue(8)
        self._costes_block.setToolTip(
            'Side length of the blocks channel B is scrambled in. Set it '
            'to about the point-spread-function width of the acquisition '
            '(the smallest resolvable feature, in pixels): larger blocks '
            'keep the local texture intact while destroying '
            'co-occurrence, smaller ones break up the texture too and '
            'inflate significance.'
        )
        self._diag_costes_group = _make_group(
            'Costes parameters',
            _hbox(QLabel('Iterations'), self._costes_niter),
            _hbox(QLabel('Block size (px)'), self._costes_block),
        )

        self._ccf_max_shift = QSpinBox()
        self._ccf_max_shift.setRange(1, 500)
        self._ccf_max_shift.setValue(20)
        self._diag_ccf_group = _make_group(
            'CCF parameters',
            _hbox(QLabel('Max shift (px)'), self._ccf_max_shift),
        )

        self._diag_ica_group = _make_group(
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

    def _build_run_row(self):
        self._diag_run_button = QPushButton('Run diagnostic')
        self._diag_run_button.clicked.connect(self._on_diag_run_clicked)
        self._diag_export_button = QPushButton('Export figure…')
        self._diag_export_button.clicked.connect(self._on_diag_export_clicked)
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
        layout.addWidget(self._diag_export_button)
        layout.addWidget(self._diag_scramble_button)
        return row

    def _build_results_group(self):
        self._diagnostic_canvas = DiagnosticCanvas()
        self._diag_summary_label = QLabel('')
        self._diag_summary_label.setWordWrap(True)
        group = QGroupBox('Diagnostic result')
        layout = QVBoxLayout()
        layout.addWidget(self._diagnostic_canvas, stretch=1)
        layout.addWidget(self._diag_summary_label)
        group.setLayout(layout)
        return group

    def _on_diag_method_changed(self):
        method = self._diag_method_combo.currentData()
        self._diag_costes_group.setVisible(method == 'costes')
        self._diag_ccf_group.setVisible(method == 'ccf')
        self._diag_ica_group.setVisible(method == 'ica')
        self._diag_scramble_button.setVisible(method == 'costes')

    def refresh_layers(self, images, labels_layers, region_layers):
        for combo in (self._diag_image_a_combo, self._diag_image_b_combo):
            _set_combo_choices(combo, images)
        _refresh_region_combo(self._diag_region_combo, region_layers)
        _nudge_distinct(
            self._diag_image_a_combo, self._diag_image_b_combo, images
        )

    def apply_theme(self, background, foreground):
        self._diagnostic_canvas.set_theme(background, foreground)

    def set_inputs(self, layer_a, layer_b, region_layer):
        """Carry another tab's channel/region selection over."""
        self._diag_image_a_combo.value = layer_a
        self._diag_image_b_combo.value = layer_b
        _select_region_layer(self._diag_region_combo, region_layer)

    # -- run -----------------------------------------------------------

    def _gather_diag_params(self):
        layer_a = self._diag_image_a_combo.value
        layer_b = self._diag_image_b_combo.value
        if layer_a is None or layer_b is None:
            show_warning('Select both image layers for the diagnostic.')
            return None
        if _reject_multiscale(layer_a, layer_b):
            return None
        a = np.asarray(layer_a.data)
        b = np.asarray(layer_b.data)
        if a.shape != b.shape:
            show_warning(f'Shape mismatch: {a.shape} vs {b.shape}.')
            return None
        try:
            label_mask, _ = _resolve_region(self._diag_region_combo, a.shape)
        except ValueError as exc:
            show_warning(str(exc))
            return None
        # Diagnostics are single-ROI: collapse any multi-region label
        # mask to one boolean "analyse here" mask.
        mask = None if label_mask is None else (label_mask > 0)
        method = self._diag_method_combo.currentData()
        block_size = int(self._costes_block.value())
        max_shift = int(self._ccf_max_shift.value())
        # Validate preconditions synchronously: a worker exception would
        # both log a traceback and trigger the warning, surfacing twice.
        if method == 'costes' and any(
            dim // block_size < 1 for dim in a.shape
        ):
            show_warning('Costes block size is larger than the image.')
            return None
        if method == 'ccf' and max_shift >= a.shape[-1]:
            show_warning(
                f'CCF max shift must be smaller than the shifted axis '
                f'({a.shape[-1]} px).'
            )
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
            'n_iter': int(self._costes_niter.value()),
            'block_size': block_size,
            'max_shift': max_shift,
        }

    def _on_diag_run_clicked(self):
        params = self._gather_diag_params()
        if params is None:
            return
        _run_in_background(
            self._diag_worker(params),
            self._diag_run_button,
            self._on_diag_results_ready,
            self._on_diag_worker_error,
        )

    @staticmethod
    @thread_worker
    def _diag_worker(params):
        method = params['method']
        a, b, mask = params['a'], params['b'], params['mask']
        if method == 'costes':
            result = costes_randomization(
                a,
                b,
                mask=mask,
                n_iter=params['n_iter'],
                block_size=params['block_size'],
            )
        elif method == 'ccf':
            shifts, ccf = van_steensel_ccf(
                a, b, mask=mask, max_shift=params['max_shift']
            )
            result = {'shifts': shifts, 'ccf': ccf}
        else:
            result = li_ica(a, b, mask=mask)
        return method, result, params['channel_a'], params['channel_b']

    def _on_diag_results_ready(self, payload):
        method, result, name_a, name_b = payload
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
        _export_canvas_figure(self, self._diagnostic_canvas, 'diagnostic.png')
