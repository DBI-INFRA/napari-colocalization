import csv

import numpy as np
import pytest
from qtpy.QtCore import Qt

from napari_colocalization import ColocalizationWidget
from napari_colocalization._analysis import COLUMNS


@pytest.fixture
def rng():
    return np.random.default_rng(seed=1)


@pytest.fixture
def widget(make_napari_viewer, qtbot):
    """A widget on a fresh viewer, registered with qtbot.

    qtbot owns the lifecycle and closes it (and its matplotlib
    canvases) deterministically at teardown, so a deferred canvas
    draw can't fire on a deleted Qt object in a later test.
    """
    w = ColocalizationWidget(make_napari_viewer())
    qtbot.addWidget(w)
    return w


# -- helpers ----------------------------------------------------------


def _select(combo, data):
    """Select a combo entry by its stored data value."""
    combo.setCurrentIndex(combo.findData(data))


def _select_region(widget, layer):
    combo = widget._pixel_tab._region_combo
    for i in range(combo.count()):
        if combo.itemData(i) is layer:
            combo.setCurrentIndex(i)
            return
    raise AssertionError(f'region layer {layer!r} not in combo')


def _region_options(widget):
    combo = widget._pixel_tab._region_combo
    return [combo.itemText(i) for i in range(combo.count())]


def _add_pair(widget, a, b=None):
    """Add image layers A and B and select them on the Pixel-based tab."""
    viewer = widget._viewer
    layer_a = viewer.add_image(a, name='a')
    layer_b = viewer.add_image(a.copy() if b is None else b, name='b')
    widget._pixel_tab._image_a_combo.value = layer_a
    widget._pixel_tab._image_b_combo.value = layer_b
    return layer_a, layer_b


def _run(widget, qtbot):
    widget._pixel_tab._on_run_clicked()
    qtbot.waitUntil(
        lambda: widget._pixel_tab._table.rowCount() > 0, timeout=10000
    )


def _cell(table, row, column):
    headers = [
        table.horizontalHeaderItem(c).text()
        for c in range(table.columnCount())
    ]
    return table.item(row, headers.index(column)).text()


def _column(table, column):
    return [_cell(table, r, column) for r in range(table.rowCount())]


# -- construction & UI toggles ----------------------------------------


def test_default_metrics_are_pearson_and_manders(widget):
    assert widget._pixel_tab._cb_pcc.isChecked()
    assert widget._pixel_tab._cb_mcc.isChecked()
    assert not any(
        cb.isChecked()
        for cb in (
            widget._pixel_tab._cb_srcc,
            widget._pixel_tab._cb_icq,
            widget._pixel_tab._cb_overlap,
        )
    )
    # Manders on means its threshold group is visible from the start.
    assert not widget._pixel_tab._threshold_group.isHidden()


def test_mode_toggle_swaps_pairwise_and_all_to_all(widget):
    assert not widget._pixel_tab._pairwise_group.isHidden()
    assert widget._pixel_tab._all_group.isHidden()
    widget._pixel_tab._mode_all.setChecked(True)
    assert widget._pixel_tab._pairwise_group.isHidden()
    assert not widget._pixel_tab._all_group.isHidden()


def test_threshold_controls_visibility(widget):
    # The threshold group only appears with Manders; the manual T_a/T_b
    # row only appears for the Manual method.
    widget._pixel_tab._cb_mcc.setChecked(False)
    assert widget._pixel_tab._threshold_group.isHidden()
    widget._pixel_tab._cb_mcc.setChecked(True)
    assert not widget._pixel_tab._threshold_group.isHidden()
    assert widget._pixel_tab._manual_row.isHidden()  # default is Costes
    _select(widget._pixel_tab._threshold_combo, 'manual')
    assert not widget._pixel_tab._manual_row.isHidden()
    _select(widget._pixel_tab._threshold_combo, 'otsu')
    assert widget._pixel_tab._manual_row.isHidden()


def test_ab_selectors_default_to_distinct_layers(widget):
    viewer = widget._viewer
    a = np.zeros((16, 16), dtype=np.float32)
    viewer.add_image(a, name='a')
    viewer.add_image(a.copy(), name='b')
    for combo_a, combo_b in (
        (widget._pixel_tab._image_a_combo, widget._pixel_tab._image_b_combo),
        (
            widget._diagnostics_tab._diag_image_a_combo,
            widget._diagnostics_tab._diag_image_b_combo,
        ),
        (
            widget._object_tab._obj_image_a_combo,
            widget._object_tab._obj_image_b_combo,
        ),
    ):
        assert combo_a.value is not combo_b.value


def test_region_combo_tracks_shapes_and_labels(widget):
    viewer = widget._viewer
    viewer.add_image(np.zeros((10, 10), np.float32), name='img')  # excluded
    shapes = viewer.add_shapes(name='roi')
    labels = viewer.add_labels(np.zeros((10, 10), np.int32), name='lbl')
    assert _region_options(widget)[0] == 'None'
    assert {'roi', 'lbl'} <= set(_region_options(widget))
    assert 'img' not in _region_options(widget)
    shapes.name = 'renamed'
    assert 'renamed' in _region_options(widget)
    viewer.layers.remove(labels)
    assert 'lbl' not in _region_options(widget)


# -- intensity runs ---------------------------------------------------


def test_pairwise_run_reports_metric_values(widget, qtbot, rng):
    # identical channels -> Pearson and overlap are both 1.
    _add_pair(widget, rng.random((32, 32)).astype(np.float32))
    widget._pixel_tab._cb_pcc.setChecked(True)
    widget._pixel_tab._cb_overlap.setChecked(True)
    _run(widget, qtbot)
    assert widget._pixel_tab._table.rowCount() == 1
    assert float(_cell(widget._pixel_tab._table, 0, 'pcc')) == pytest.approx(
        1.0
    )
    assert float(
        _cell(widget._pixel_tab._table, 0, 'overlap')
    ) == pytest.approx(1.0)
    # results panel is revealed once there are rows
    assert not widget._pixel_tab._results_group.isHidden()


def test_significance_row_links_to_diagnostics(widget, qtbot, rng):
    layer_a, layer_b = _add_pair(
        widget, rng.random((16, 16)).astype(np.float32)
    )
    labels = widget._viewer.add_labels(
        np.ones((16, 16), np.int32), name='regions'
    )
    _select_region(widget, labels)
    widget._pixel_tab._cb_mcc.setChecked(
        False
    )  # skip Costes, keep the run quick
    _run(widget, qtbot)
    assert not widget._pixel_tab._significance_row.isHidden()

    widget._pixel_tab._open_diagnostics_button.click()
    assert widget._tabs.currentWidget() is widget._diagnostics_tab
    assert widget._diagnostics_tab._diag_image_a_combo.value is layer_a
    assert widget._diagnostics_tab._diag_image_b_combo.value is layer_b
    assert widget._diagnostics_tab._diag_region_combo.currentData() is labels


def test_open_diagnostics_survives_deleted_layers(widget, qtbot, rng):
    # A selection that no longer exists must be skipped, not assigned:
    # magicgui raises on values outside the current choices.
    _add_pair(widget, rng.random((16, 16)).astype(np.float32))
    widget._pixel_tab._cb_mcc.setChecked(False)
    _run(widget, qtbot)
    widget._viewer.layers.clear()
    widget._pixel_tab._open_diagnostics_button.click()
    assert widget._tabs.currentWidget() is widget._diagnostics_tab


def test_deliberate_same_channel_selection_survives_layer_events(widget, rng):
    layer_a, _ = _add_pair(widget, rng.random((8, 8)).astype(np.float32))
    # A vs A is a legitimate autocorrelation control; a later layer
    # event must not nudge B away from it.
    widget._pixel_tab._image_b_combo.value = layer_a
    widget._viewer.add_points(name='pts')
    assert widget._pixel_tab._image_b_combo.value is layer_a


def test_all_to_all_run_covers_every_pair(widget, qtbot, rng):
    a = rng.random((32, 32)).astype(np.float32)
    layer = widget._viewer.add_image(np.stack([a, a, a]), name='stack')
    widget._pixel_tab._mode_all.setChecked(True)
    widget._pixel_tab._stack_combo.value = layer
    widget._pixel_tab._channel_axis_spin.setValue(0)
    widget._pixel_tab._cb_mcc.setChecked(False)
    _run(widget, qtbot)
    assert (
        widget._pixel_tab._table.rowCount() == 3
    )  # 3 unordered channel pairs


def test_region_mask_gives_one_row_per_region(widget, qtbot, rng):
    _add_pair(widget, rng.random((20, 20)).astype(np.float32))
    label = np.zeros((20, 20), np.int32)
    label[:10], label[10:] = 1, 2
    layer = widget._viewer.add_labels(label, name='regions')
    _select_region(widget, layer)
    _run(widget, qtbot)
    assert widget._pixel_tab._table.rowCount() == 2
    # Rows share the run's label mask instead of each holding a
    # full-size boolean copy of their region.
    context = widget._pixel_tab._plot_context
    assert context[0]['label_mask'] is context[1]['label_mask']


def test_table_sorts_numerically_not_lexicographically(widget):
    rows = [
        {'region': 0, 'n_pixels': 2, 'pcc': 0.5},
        {'region': 1, 'n_pixels': 10, 'pcc': 1e-05},
        {'region': 2, 'n_pixels': 9, 'pcc': float('nan')},
    ]
    widget._pixel_tab._populate_table(rows)
    table = widget._pixel_tab._table
    table.sortItems(COLUMNS.index('n_pixels'), Qt.AscendingOrder)
    assert _column(table, 'n_pixels') == ['2', '9', '10']
    # NaN (blank) sorts below every real value.
    table.sortItems(COLUMNS.index('pcc'), Qt.AscendingOrder)
    assert _column(table, 'pcc') == ['', '1e-05', '0.5']


def test_per_slice_run_gives_one_row_per_slice(widget, qtbot, rng):
    _add_pair(widget, rng.random((4, 16, 16)).astype(np.float32))
    widget._pixel_tab._per_slice_check.setChecked(True)
    widget._pixel_tab._slice_axis_spin.setValue(0)
    _run(widget, qtbot)
    assert widget._pixel_tab._table.rowCount() == 4
    widget._pixel_tab._table.selectRow(2)  # slice-aware scatter must not raise


def test_per_slice_requires_3d(widget, rng):
    _add_pair(widget, rng.random((16, 16)).astype(np.float32))  # 2D
    widget._pixel_tab._per_slice_check.setChecked(True)
    assert widget._pixel_tab.gather_params() is None


def test_multiscale_image_is_rejected(widget):
    arr = np.zeros((32, 32), dtype=np.float32)
    layer = widget._viewer.add_image(
        [arr, arr[::2, ::2]], multiscale=True, name='ms'
    )
    widget._pixel_tab._image_a_combo.value = layer
    widget._pixel_tab._image_b_combo.value = layer
    assert widget._pixel_tab._image_a_combo.value is layer
    assert widget._pixel_tab.gather_params() is None


def test_multiscale_labels_are_rejected(widget):
    arr = np.zeros((32, 32), dtype=np.int32)
    layer = widget._viewer.add_labels(
        [arr, arr[::2, ::2]], multiscale=True, name='ms'
    )
    tab = widget._object_tab
    _select(tab._obj_source_combo, 'labels')
    tab._obj_labels_a_combo.value = layer
    tab._obj_labels_b_combo.value = layer
    assert tab._gather_object_params() is None


def test_auto_threshold_manders(widget, qtbot, rng):
    # Otsu cleanly separates a two-level image, so the bright square
    # fully colocalizes with itself (M1 == 1).
    img = np.zeros((32, 32), dtype=np.float32)
    img[8:24, 8:24] = 1.0
    _add_pair(widget, img)
    widget._pixel_tab._cb_mcc.setChecked(True)
    _select(widget._pixel_tab._threshold_combo, 'otsu')
    _run(widget, qtbot)
    assert float(_cell(widget._pixel_tab._table, 0, 'tm1')) == pytest.approx(
        1.0
    )


def test_uncomputable_region_is_summarised(widget, qtbot, rng):
    a = rng.random((20, 20)).astype(np.float32)
    a[:10] = 1.0  # constant region -> PCC undefined there
    _add_pair(widget, a)
    label = np.zeros((20, 20), np.int32)
    label[:10], label[10:] = 1, 2
    _select_region(widget, widget._viewer.add_labels(label, name='r'))
    widget._pixel_tab._cb_srcc.setChecked(False)
    widget._pixel_tab._cb_pcc.setChecked(True)
    _run(widget, qtbot)
    assert 'could not be computed' in widget._pixel_tab._summary_label.text()


def test_fixed_axes_lock_scatter_bounds(widget, qtbot, rng):
    a = (rng.random((32, 32)) * 5.0).astype(np.float32)
    _add_pair(widget, a)
    _run(widget, qtbot)
    widget._pixel_tab._fixed_axes_check.setChecked(True)
    widget._pixel_tab._table.selectRow(0)
    xlim = widget._pixel_tab._scatter._ax.get_xlim()
    assert xlim[0] == pytest.approx(0.0)
    assert xlim[1] == pytest.approx(float(a.max()))


def test_csv_export(widget, qtbot, rng, tmp_path):
    _add_pair(widget, rng.random((16, 16)).astype(np.float32))
    _run(widget, qtbot)
    out = tmp_path / 'results.csv'
    widget._pixel_tab.write_csv(str(out), widget._pixel_tab._results)
    with open(out, newline='') as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 1
    assert 'pcc' in rows[0]


def test_csv_export_records_provenance(widget, qtbot, rng, tmp_path):
    # The metric columns alone don't say how the run was configured, so
    # a saved file has to carry that with it.
    a = rng.random((16, 16)).astype(np.float32)
    _add_pair(widget, a)
    shapes = widget._viewer.add_shapes(
        [np.array([[1.0, 1.0], [1.0, 8.0], [8.0, 8.0], [8.0, 1.0]])],
        shape_type='polygon',
        name='my_rois',
    )
    _select_region(widget, shapes)
    widget._pixel_tab._cb_mcc.setChecked(True)
    _select(widget._pixel_tab._threshold_combo, 'otsu')
    _run(widget, qtbot)

    tab = widget._pixel_tab
    out = tmp_path / 'results.csv'
    tab.write_csv(str(out), tab._results, provenance=tab._provenance)
    with open(out, newline='') as fh:
        row = next(iter(csv.DictReader(fh)))
    assert row['threshold_method'] == 'otsu'
    assert row['region_layer'] == 'my_rois'
    assert row['mode'] == 'pairwise'
    assert 'mcc' in row['metrics']
    assert row['plugin_version']
    assert row['analysed_at']
    # Data columns survive alongside the provenance ones.
    assert 'pcc' in row


def test_provenance_blanks_threshold_when_manders_not_requested(widget):
    provenance = widget._pixel_tab._run_provenance(
        {
            'mode': 'pairwise',
            'metrics': ('pcc',),
            'threshold_method': 'costes',
            'region_layer': None,
            'slice_axis': None,
        }
    )
    # Costes is the combo's default even when Manders is off; recording
    # it would imply a threshold was applied.
    assert provenance['threshold_method'] == ''
    assert provenance['region_layer'] == ''
    assert provenance['slice_axis'] == ''


def test_row_selection_highlights_regions(widget, qtbot, rng):
    _add_pair(widget, rng.random((20, 20)).astype(np.float32))
    shapes = widget._viewer.add_shapes(
        [
            np.array([[1.0, 1.0], [1.0, 8.0], [8.0, 8.0], [8.0, 1.0]]),
            np.array([[12.0, 12.0], [12.0, 18.0], [18.0, 18.0], [18.0, 12.0]]),
        ],
        shape_type='polygon',
        name='regions',
    )
    _select_region(widget, shapes)
    widget._pixel_tab._on_run_clicked()
    qtbot.waitUntil(
        lambda: widget._pixel_tab._table.rowCount() >= 2, timeout=10000
    )

    widget._pixel_tab._table.selectRow(0)
    assert 0 in shapes.selected_data
    widget._pixel_tab._table.selectAll()
    assert set(shapes.selected_data) == {0, 1}
    widget._pixel_tab._table.clearSelection()
    assert len(shapes.selected_data) == 0


def test_add_coloc_mask_layer(widget, qtbot, rng):
    from napari.layers import Labels

    img = np.zeros((32, 32), dtype=np.float32)
    img[8:24, 8:24] = 1.0
    _add_pair(widget, img)
    widget._pixel_tab._cb_mcc.setChecked(True)
    _select(widget._pixel_tab._threshold_combo, 'otsu')
    _run(widget, qtbot)
    widget._pixel_tab._on_add_mask_clicked()
    masks = [
        layer
        for layer in widget._viewer.layers
        if isinstance(layer, Labels) and 'coloc' in layer.name
    ]
    assert len(masks) == 1
    assert int(masks[0].data.sum()) == 16 * 16  # the bright square


def test_per_slice_coloc_mask_covers_only_that_plane(widget, qtbot):
    from napari.layers import Labels

    img = np.zeros((3, 16, 16), dtype=np.float32)
    img[:, 4:12, 4:12] = 1.0
    _add_pair(widget, img)
    widget._pixel_tab._per_slice_check.setChecked(True)
    widget._pixel_tab._slice_axis_spin.setValue(0)
    widget._pixel_tab._cb_mcc.setChecked(True)
    _select(widget._pixel_tab._threshold_combo, 'otsu')
    _run(widget, qtbot)

    slices = _column(widget._pixel_tab._table, 'slice')
    widget._pixel_tab._table.selectRow(slices.index('1'))
    widget._pixel_tab._on_add_mask_clicked()
    mask = next(
        layer
        for layer in widget._viewer.layers
        if isinstance(layer, Labels) and 'coloc' in layer.name
    )
    assert mask.data.shape == img.shape
    assert int(mask.data[1].sum()) == 8 * 8
    assert int(mask.data[0].sum()) == 0
    assert int(mask.data[2].sum()) == 0


def test_added_layers_follow_source_geometry(widget, qtbot):
    from napari.layers import Labels, Points, Vectors

    img = np.zeros((3, 16, 16), dtype=np.float32)
    img[:, 4:12, 4:12] = 1.0
    geometry = {'scale': (3.0, 0.5, 0.5), 'translate': (0.0, 10.0, 20.0)}
    viewer = widget._viewer
    layer_a = viewer.add_image(img, name='a', **geometry)
    layer_b = viewer.add_image(img.copy(), name='b', **geometry)
    widget._pixel_tab._image_a_combo.value = layer_a
    widget._pixel_tab._image_b_combo.value = layer_b
    _select(widget._pixel_tab._threshold_combo, 'otsu')
    _run(widget, qtbot)
    widget._pixel_tab._table.selectRow(0)
    widget._pixel_tab._on_add_mask_clicked()

    obj = widget._object_tab
    obj._obj_image_a_combo.value = layer_a
    obj._obj_image_b_combo.value = layer_b
    obj._on_object_run_clicked()
    qtbot.waitUntil(lambda: obj._object_table.rowCount() > 0, timeout=10000)

    widget._diagnostics_tab._diag_image_b_combo.value = layer_b
    widget._diagnostics_tab._costes_block.setValue(2)
    widget._diagnostics_tab._on_diag_scramble_clicked()
    qtbot.waitUntil(
        lambda: any('scrambled' in layer.name for layer in viewer.layers),
        timeout=10000,
    )

    added = [
        layer
        for layer in viewer.layers
        if isinstance(layer, (Labels, Points, Vectors))
        or 'scrambled' in layer.name
    ]
    assert len(added) == 5  # mask, 2 centroid layers, links, scrambled
    for layer in added:
        np.testing.assert_allclose(layer.scale, geometry['scale'])
        np.testing.assert_allclose(layer.translate, geometry['translate'])


def test_misaligned_region_layer_is_rejected(widget, monkeypatch):
    import napari_colocalization._widget_shared as shared

    warnings = []
    monkeypatch.setattr(shared, 'show_warning', warnings.append)
    img = np.ones((16, 16), dtype=np.float32)
    _add_pair(widget, img)
    roi = widget._viewer.add_shapes(
        [np.array([[0, 0], [0, 8], [8, 8], [8, 0]])], name='roi', scale=(2, 2)
    )
    _select_region(widget, roi)
    assert widget._pixel_tab.gather_params() is None
    assert 'scale' in warnings[-1]


# -- diagnostics tab --------------------------------------------------


def test_diag_param_groups_follow_method(widget):
    assert (
        not widget._diagnostics_tab._diag_costes_group.isHidden()
    )  # Costes default
    _select(widget._diagnostics_tab._diag_method_combo, 'ccf')
    assert not widget._diagnostics_tab._diag_ccf_group.isHidden()
    assert widget._diagnostics_tab._diag_costes_group.isHidden()
    _select(widget._diagnostics_tab._diag_method_combo, 'ica')
    assert not widget._diagnostics_tab._diag_ica_group.isHidden()


@pytest.mark.parametrize(
    ('method', 'expected'),
    [
        ('costes', 'Observed PCC'),
        ('ccf', 'Peak Pearson r'),
        ('ica', 'ICQ'),
    ],
)
def test_diagnostic_run_summarises(widget, qtbot, rng, method, expected):
    a = rng.random((32, 32)).astype(np.float32)
    viewer = widget._viewer
    widget._diagnostics_tab._diag_image_a_combo.value = viewer.add_image(
        a, name='a'
    )
    widget._diagnostics_tab._diag_image_b_combo.value = viewer.add_image(
        a.copy(), name='b'
    )
    _select(widget._diagnostics_tab._diag_method_combo, method)
    widget._diagnostics_tab._costes_niter.setValue(
        20
    )  # keep the costes test quick
    widget._diagnostics_tab._ccf_max_shift.setValue(5)
    widget._diagnostics_tab._on_diag_run_clicked()
    qtbot.waitUntil(
        lambda: widget._diagnostics_tab._diag_summary_label.text() != '',
        timeout=15000,
    )
    assert expected in widget._diagnostics_tab._diag_summary_label.text()


def test_diag_costes_validation(widget, rng):
    viewer = widget._viewer
    a = rng.random((8, 16, 16)).astype(np.float32)  # 3D
    widget._diagnostics_tab._diag_image_a_combo.value = viewer.add_image(
        a, name='a'
    )
    widget._diagnostics_tab._diag_image_b_combo.value = viewer.add_image(
        a.copy(), name='b'
    )
    _select(widget._diagnostics_tab._diag_method_combo, 'costes')
    # Costes supports 3D...
    widget._diagnostics_tab._costes_block.setValue(8)
    assert widget._diagnostics_tab._gather_diag_params() is not None
    # ...but a block larger than the image is rejected up front.
    widget._diagnostics_tab._costes_block.setValue(64)
    assert widget._diagnostics_tab._gather_diag_params() is None


def test_add_scrambled_example_layer(widget, qtbot, rng):
    viewer = widget._viewer
    a = rng.random((32, 32)).astype(np.float32)
    viewer.add_image(a, name='a')
    widget._diagnostics_tab._diag_image_b_combo.value = viewer.add_image(
        a.copy(), name='b'
    )
    n_before = len(viewer.layers)
    widget._diagnostics_tab._on_diag_scramble_clicked()
    qtbot.waitUntil(lambda: len(viewer.layers) == n_before + 1, timeout=10000)
    assert any('scrambled' in layer.name for layer in viewer.layers)


# -- object-based tab -------------------------------------------------


def test_object_csv_export(widget, qtbot, tmp_path):
    from napari_colocalization._widget_shared import (
        _object_rows_for_csv,
        write_csv_rows,
    )

    img = np.zeros((20, 20), dtype=np.float32)
    img[2:6, 2:6] = 1.0
    img[12:16, 12:16] = 1.0
    viewer = widget._viewer
    tab = widget._object_tab
    tab._obj_image_a_combo.value = viewer.add_image(img, name='a')
    tab._obj_image_b_combo.value = viewer.add_image(img.copy(), name='b')
    # Nothing to export until a run has produced rows.
    assert not tab._obj_export_button.isEnabled()
    tab._on_object_run_clicked()
    qtbot.waitUntil(lambda: tab._object_table.rowCount() > 0, timeout=10000)
    assert tab._obj_export_button.isEnabled()

    out = tmp_path / 'objects.csv'
    rows, columns = _object_rows_for_csv(tab._object_rows)
    write_csv_rows(str(out), rows, columns, tab._object_provenance)
    with open(out, newline='') as fh:
        written = list(csv.DictReader(fh))
    assert len(written) == 4  # 2 objects x 2 channels
    # The centroid tuple is split per axis rather than written as a
    # single "(y, x)" string that downstream tools can't parse.
    assert 'centroid' not in written[0]
    assert float(written[0]['centroid_0']) == pytest.approx(3.5)
    assert float(written[0]['centroid_1']) == pytest.approx(3.5)
    assert written[0]['coincident'] == 'True'
    assert written[0]['threshold_method'] == 'otsu'
    assert written[0]['objects_from'] == 'threshold'
    assert written[0]['plugin_version']


def test_object_provenance_names_region_and_survives_unfinished_run(
    widget, qtbot, monkeypatch
):
    img = np.zeros((20, 20), dtype=np.float32)
    img[2:6, 2:6] = 1.0
    viewer = widget._viewer
    tab = widget._object_tab
    tab._obj_image_a_combo.value = viewer.add_image(img, name='a')
    tab._obj_image_b_combo.value = viewer.add_image(img.copy(), name='b')
    roi = viewer.add_shapes(
        [np.array([[0, 0], [0, 10], [10, 10], [10, 0]])], name='roi'
    )
    combo = tab._obj_region_combo
    combo.setCurrentIndex(
        [combo.itemData(i) for i in range(combo.count())].index(roi)
    )
    tab._on_object_run_clicked()
    qtbot.waitUntil(lambda: tab._object_table.rowCount() > 0, timeout=10000)
    assert tab._object_provenance['region_layer'] == 'roi'

    # A run that never delivers (cancelled, errored) must not relabel
    # the export of the results still on screen.
    _select(tab._obj_method_combo, 'li')
    monkeypatch.setattr(tab._obj_run_control, 'start', lambda *a: None)
    tab._on_object_run_clicked()
    assert tab._object_provenance['threshold_method'] == 'otsu'


def test_object_provenance_blanks_threshold_for_labels_source(widget):
    provenance = widget._object_tab._object_provenance_for(
        {'source': 'labels', 'region_layer': None}
    )
    # With Labels layers the segmentation happened upstream, so naming a
    # threshold method here would be a fiction.
    assert provenance['objects_from'] == 'labels'
    assert provenance['threshold_method'] == ''
    assert provenance['min_object_size'] == ''


@pytest.mark.parametrize(
    ('method', 'columns', 'n_rows'),
    [
        ('costes', ('iteration', 'null_pcc'), 20),
        ('ccf', ('shift_px', 'pearson_r'), 11),
        ('ica', ('a', 'b', 'product'), 32 * 32),
    ],
)
def test_diagnostic_value_export(
    widget, qtbot, rng, tmp_path, method, columns, n_rows
):
    from napari_colocalization._widget_shared import write_csv_rows

    a = rng.random((32, 32)).astype(np.float32)
    viewer = widget._viewer
    tab = widget._diagnostics_tab
    tab._diag_image_a_combo.value = viewer.add_image(a, name='a')
    tab._diag_image_b_combo.value = viewer.add_image(a.copy(), name='b')
    _select(tab._diag_method_combo, method)
    tab._costes_niter.setValue(20)
    tab._ccf_max_shift.setValue(5)  # -> 11 shifts
    # The figure is exportable but the numbers behind it were not.
    assert not tab._diag_export_values_button.isEnabled()
    tab._on_diag_run_clicked()
    qtbot.waitUntil(
        lambda: tab._diag_summary_label.text() != '', timeout=15000
    )
    assert tab._diag_export_values_button.isEnabled()

    out = tmp_path / f'{method}.csv'
    rows, cols, extra = tab._diag_rows_for_csv(*tab._diag_result)
    assert cols == columns
    written = write_csv_rows(
        str(out), rows, cols, {**tab._diag_provenance, **extra}
    )
    assert written == n_rows
    with open(out, newline='') as fh:
        first = next(iter(csv.DictReader(fh)))
    assert first['diagnostic'] == method
    assert first['channel_a'] == 'a'
    assert all(column in first for column in columns)


def test_diagnostic_provenance_survives_unfinished_run(
    widget, qtbot, rng, monkeypatch
):
    a = rng.random((32, 32)).astype(np.float32)
    viewer = widget._viewer
    tab = widget._diagnostics_tab
    tab._diag_image_a_combo.value = viewer.add_image(a, name='a')
    tab._diag_image_b_combo.value = viewer.add_image(a.copy(), name='b')
    _select(tab._diag_method_combo, 'ccf')
    tab._ccf_max_shift.setValue(5)
    tab._on_diag_run_clicked()
    qtbot.waitUntil(
        lambda: tab._diag_summary_label.text() != '', timeout=15000
    )
    _select(tab._diag_method_combo, 'ica')
    monkeypatch.setattr(tab._diag_run_control, 'start', lambda *a: None)
    tab._on_diag_run_clicked()
    assert tab._diag_provenance['diagnostic'] == 'ccf'
    assert tab._diag_result[0] == 'ccf'


def test_manual_threshold_spins_track_the_selected_layers(widget):
    # Seeded from the layer's own range: a 16-bit channel must not be
    # left stepping by 0.01 over +/-1e9 with no hint of its units.
    viewer = widget._viewer
    tab = widget._pixel_tab
    a = (np.arange(256 * 256, dtype=np.uint16) % 4096).reshape(256, 256)
    b = np.linspace(0.0, 1.0, 256 * 256, dtype=np.float32).reshape(256, 256)
    tab._image_a_combo.value = viewer.add_image(a, name='sixteen_bit')
    tab._image_b_combo.value = viewer.add_image(b, name='normalised')
    _select(tab._threshold_combo, 'manual')

    assert tab._th_a_spin.maximum() == pytest.approx(4095.0, rel=1e-3)
    assert tab._th_a_spin.singleStep() > 1.0  # usable on a 16-bit range
    assert tab._th_a_spin.value() == pytest.approx(2047.5, rel=1e-2)
    assert 'sixteen_bit' in tab._th_a_spin.toolTip()

    # The 0-1 channel keeps fine resolution rather than inheriting the
    # coarse step that suits its 16-bit neighbour.
    assert tab._th_b_spin.maximum() == pytest.approx(1.0)
    assert tab._th_b_spin.singleStep() < 0.1
    assert tab._th_b_spin.decimals() >= 6


def test_object_table_shows_nn_distance(widget, qtbot):
    img_a = np.zeros((20, 20), dtype=np.float32)
    img_a[2:6, 2:6] = 1.0
    img_b = np.zeros((20, 20), dtype=np.float32)
    img_b[12:16, 12:16] = 1.0
    viewer = widget._viewer
    tab = widget._object_tab
    tab._obj_image_a_combo.value = viewer.add_image(img_a, name='a')
    tab._obj_image_b_combo.value = viewer.add_image(img_b, name='b')
    tab._on_object_run_clicked()
    qtbot.waitUntil(lambda: tab._object_table.rowCount() > 0, timeout=10000)
    # Rendered as a short number, not a raw repr.
    assert _cell(tab._object_table, 0, 'nn_distance_px') == '14.14'


def test_object_source_toggle(widget):
    assert (
        not widget._object_tab._obj_threshold_group.isHidden()
    )  # threshold default
    assert widget._object_tab._obj_labels_group.isHidden()
    _select(widget._object_tab._obj_source_combo, 'labels')
    assert widget._object_tab._obj_threshold_group.isHidden()
    assert not widget._object_tab._obj_labels_group.isHidden()


def test_object_run_from_threshold_with_overlays(widget, qtbot):
    from napari.layers import Points, Vectors

    img = np.zeros((20, 20), dtype=np.float32)
    img[2:6, 2:6] = 1.0
    img[12:16, 12:16] = 1.0
    viewer = widget._viewer
    widget._object_tab._obj_image_a_combo.value = viewer.add_image(
        img, name='a'
    )
    widget._object_tab._obj_image_b_combo.value = viewer.add_image(
        img.copy(), name='b'
    )
    widget._object_tab._on_object_run_clicked()
    qtbot.waitUntil(
        lambda: widget._object_tab._object_table.rowCount() > 0, timeout=10000
    )
    assert (
        widget._object_tab._object_table.rowCount() == 4
    )  # 2 objects x 2 channels
    assert _cell(widget._object_tab._object_table, 0, 'coincident') == 'yes'
    assert any(isinstance(layer, Points) for layer in viewer.layers)
    assert any(isinstance(layer, Vectors) for layer in viewer.layers)


def test_object_run_from_labels_layers(widget, qtbot):
    viewer = widget._viewer
    labels = np.zeros((20, 20), dtype=np.int32)
    labels[2:6, 2:6] = 1
    _select(widget._object_tab._obj_source_combo, 'labels')
    widget._object_tab._obj_labels_a_combo.value = viewer.add_labels(
        labels, name='oa'
    )
    widget._object_tab._obj_labels_b_combo.value = viewer.add_labels(
        labels.copy(), name='ob'
    )
    widget._object_tab._obj_points_check.setChecked(False)
    widget._object_tab._obj_links_check.setChecked(False)
    widget._object_tab._on_object_run_clicked()
    qtbot.waitUntil(
        lambda: widget._object_tab._object_table.rowCount() > 0, timeout=10000
    )
    assert (
        widget._object_tab._object_table.rowCount() == 2
    )  # 1 object x 2 channels


def test_object_rerun_clears_previous_overlays(widget, qtbot):
    img = np.zeros((20, 20), dtype=np.float32)
    img[2:6, 2:6] = 1.0
    img[12:16, 12:16] = 1.0
    viewer = widget._viewer
    widget._object_tab._obj_image_a_combo.value = viewer.add_image(
        img, name='a'
    )
    widget._object_tab._obj_image_b_combo.value = viewer.add_image(
        img.copy(), name='b'
    )

    widget._object_tab._on_object_run_clicked()
    qtbot.waitUntil(
        lambda: len(widget._object_tab._object_overlay_layers) > 0,
        timeout=10000,
    )
    first = list(widget._object_tab._object_overlay_layers)
    n_layers = len(viewer.layers)

    widget._object_tab._on_object_run_clicked()
    qtbot.waitUntil(
        lambda: bool(widget._object_tab._object_overlay_layers)
        and widget._object_tab._object_overlay_layers[0] not in first,
        timeout=10000,
    )
    assert all(layer not in viewer.layers for layer in first)
    assert len(viewer.layers) == n_layers  # overlays replaced, not stacked


# -- progress, cancellation and header tooltips ------------------------


def test_pixel_run_reports_progress_and_resets(widget, qtbot, rng):
    _add_pair(widget, rng.random((16, 16)).astype(np.float32))
    label = np.zeros((16, 16), np.int32)
    label[:8], label[8:] = 1, 2
    _select_region(widget, widget._viewer.add_labels(label, name='r'))
    widget._pixel_tab._cb_mcc.setChecked(False)
    control = widget._pixel_tab._run_control
    reported = []
    control.progress.valueChanged.connect(reported.append)

    _run(widget, qtbot)
    qtbot.waitUntil(lambda: not control.is_running(), timeout=10000)

    assert max(reported) == 2  # one report per region
    assert control.progress.isHidden()
    assert widget._pixel_tab._run_button.text() == 'Run'


def test_cancelling_a_run_discards_it(widget, qtbot):
    import time

    from napari.qt.threading import thread_worker

    @thread_worker
    def _never_ends():
        while True:
            time.sleep(0.01)
            yield 0, 1

    control = widget._pixel_tab._run_control
    aborted = []
    worker = _never_ends()
    worker.aborted.connect(lambda: aborted.append(True))
    control.start(worker, lambda _payload: None, lambda _exc: None)
    assert control.is_running()
    assert widget._pixel_tab._run_button.text() == 'Cancel'

    widget._pixel_tab._run_button.click()  # the button is now Cancel
    qtbot.waitUntil(lambda: not control.is_running(), timeout=10000)

    assert aborted
    assert widget._pixel_tab._run_button.text() == 'Run'
    assert control.progress.isHidden()
    assert widget._pixel_tab._table.rowCount() == 0  # no partial results


def test_column_headers_carry_tooltips(widget):
    def _tooltips(table):
        return {
            table.horizontalHeaderItem(c).text(): table.horizontalHeaderItem(
                c
            ).toolTip()
            for c in range(table.columnCount())
        }

    pixel = _tooltips(widget._pixel_tab._table)
    assert all(pixel.values())
    assert 'Pearson' in pixel['pcc']
    assert 'threshold' in pixel['tm1']

    objects = _tooltips(widget._object_tab._object_table)
    assert all(objects.values())
    assert 'centroid' in objects['coincident']
