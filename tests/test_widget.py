import csv

import numpy as np
import pytest

from napari_colocalization import ColocalizationWidget


@pytest.fixture
def rng():
    return np.random.default_rng(seed=1)


def test_widget_instantiates(make_napari_viewer):
    viewer = make_napari_viewer()
    assert ColocalizationWidget(viewer) is not None


def test_default_metric_is_spearman_only(make_napari_viewer):
    viewer = make_napari_viewer()
    widget = ColocalizationWidget(viewer)
    assert widget._cb_pcc.isChecked() is False
    assert widget._cb_srcc.isChecked() is True
    assert widget._cb_icq.isChecked() is False
    assert widget._cb_overlap.isChecked() is False
    assert widget._cb_mcc.isChecked() is False


def test_run_icq_populates_table(make_napari_viewer, qtbot, rng):
    viewer = make_napari_viewer()
    a = rng.random((32, 32)).astype(np.float32)
    layer_a = viewer.add_image(a, name='a')
    layer_b = viewer.add_image(a.copy(), name='b')
    widget = ColocalizationWidget(viewer)
    widget._image_a_combo.value = layer_a
    widget._image_b_combo.value = layer_b
    widget._cb_srcc.setChecked(False)
    widget._cb_icq.setChecked(True)

    widget._on_run_clicked()
    qtbot.waitUntil(lambda: widget._table.rowCount() > 0, timeout=10000)
    headers = [
        widget._table.horizontalHeaderItem(c).text()
        for c in range(widget._table.columnCount())
    ]
    icq_col = headers.index('icq')
    icq_value = float(widget._table.item(0, icq_col).text())
    assert icq_value == pytest.approx(0.5)


def test_run_overlap_populates_columns(make_napari_viewer, qtbot, rng):
    viewer = make_napari_viewer()
    a = rng.random((32, 32)).astype(np.float32)
    layer_a = viewer.add_image(a, name='a')
    layer_b = viewer.add_image((2 * a), name='b')
    widget = ColocalizationWidget(viewer)
    widget._image_a_combo.value = layer_a
    widget._image_b_combo.value = layer_b
    widget._cb_srcc.setChecked(False)
    widget._cb_overlap.setChecked(True)
    assert 'overlap' in widget._selected_metrics()

    widget._on_run_clicked()
    qtbot.waitUntil(lambda: widget._table.rowCount() > 0, timeout=10000)
    headers = [
        widget._table.horizontalHeaderItem(c).text()
        for c in range(widget._table.columnCount())
    ]
    for name in ('overlap', 'k1', 'k2'):
        assert name in headers
    overlap_value = float(
        widget._table.item(0, headers.index('overlap')).text()
    )
    assert overlap_value == pytest.approx(1.0)


def test_pairwise_defaults_pick_distinct_layers(make_napari_viewer, rng):
    viewer = make_napari_viewer()
    a = rng.random((16, 16)).astype(np.float32)
    layer_a = viewer.add_image(a, name='a')
    layer_b = viewer.add_image(a.copy(), name='b')
    widget = ColocalizationWidget(viewer)
    assert widget._image_a_combo.value is layer_a
    assert widget._image_b_combo.value is layer_b


def test_mode_toggle_shows_correct_inputs(make_napari_viewer):
    viewer = make_napari_viewer()
    widget = ColocalizationWidget(viewer)
    assert widget._pairwise_group.isHidden() is False
    assert widget._all_group.isHidden() is True
    widget._mode_all.setChecked(True)
    assert widget._pairwise_group.isHidden() is True
    assert widget._all_group.isHidden() is False


def test_threshold_section_visibility_follows_mcc(make_napari_viewer):
    viewer = make_napari_viewer()
    widget = ColocalizationWidget(viewer)
    assert widget._threshold_group.isHidden() is True
    widget._cb_mcc.setChecked(True)
    assert widget._threshold_group.isHidden() is False
    widget._cb_mcc.setChecked(False)
    assert widget._threshold_group.isHidden() is True


def _select_region_layer(widget, layer):
    for i in range(widget._region_combo.count()):
        if widget._region_combo.itemData(i) is layer:
            widget._region_combo.setCurrentIndex(i)
            return
    raise AssertionError(f'layer {layer!r} not in region combo')


def _region_options(widget):
    return [
        widget._region_combo.itemText(i)
        for i in range(widget._region_combo.count())
    ]


def test_region_combo_lists_only_shapes_and_labels(make_napari_viewer, rng):
    viewer = make_napari_viewer()
    a = rng.random((10, 10)).astype(np.float32)
    viewer.add_image(a, name='img')  # should NOT appear
    viewer.add_shapes(name='roi_shapes')
    viewer.add_labels(np.zeros((10, 10), dtype=np.int32), name='roi_labels')
    widget = ColocalizationWidget(viewer)
    options = _region_options(widget)
    assert options[0] == 'None'
    assert 'roi_shapes' in options
    assert 'roi_labels' in options
    assert 'img' not in options


def test_region_combo_updates_on_layer_added_and_removed(
    make_napari_viewer,
):
    viewer = make_napari_viewer()
    widget = ColocalizationWidget(viewer)
    assert _region_options(widget) == ['None']

    shapes = viewer.add_shapes(name='roi')
    assert 'roi' in _region_options(widget)

    viewer.layers.remove(shapes)
    assert _region_options(widget) == ['None']


def test_region_combo_updates_on_rename(make_napari_viewer):
    viewer = make_napari_viewer()
    shapes = viewer.add_shapes(name='before')
    widget = ColocalizationWidget(viewer)
    assert 'before' in _region_options(widget)
    shapes.name = 'after'
    assert 'after' in _region_options(widget)
    assert 'before' not in _region_options(widget)


def test_run_pairwise_populates_table(make_napari_viewer, qtbot, rng):
    viewer = make_napari_viewer()
    a = rng.random((32, 32)).astype(np.float32)
    layer_a = viewer.add_image(a, name='a')
    layer_b = viewer.add_image(a.copy(), name='b')
    widget = ColocalizationWidget(viewer)
    widget._image_a_combo.value = layer_a
    widget._image_b_combo.value = layer_b
    widget._cb_pcc.setChecked(True)
    widget._cb_mcc.setChecked(True)
    _select_combo_data(widget._threshold_combo, 'manual')
    widget._th_a_spin.setValue(0.0)
    widget._th_b_spin.setValue(0.0)

    widget._on_run_clicked()

    qtbot.waitUntil(lambda: widget._table.rowCount() > 0, timeout=10000)
    assert widget._table.rowCount() == 1
    headers = [
        widget._table.horizontalHeaderItem(c).text()
        for c in range(widget._table.columnCount())
    ]
    pcc_col = headers.index('pcc')
    pcc_value = float(widget._table.item(0, pcc_col).text())
    assert pcc_value == pytest.approx(1.0)


def test_run_per_slice_populates_rows(make_napari_viewer, qtbot, rng):
    viewer = make_napari_viewer()
    a = rng.random((4, 16, 16)).astype(np.float32)
    layer_a = viewer.add_image(a, name='a')
    layer_b = viewer.add_image(a.copy(), name='b')
    widget = ColocalizationWidget(viewer)
    widget._image_a_combo.value = layer_a
    widget._image_b_combo.value = layer_b
    widget._cb_mcc.setChecked(False)
    widget._per_slice_check.setChecked(True)
    widget._slice_axis_spin.setValue(0)

    widget._on_run_clicked()
    qtbot.waitUntil(lambda: widget._table.rowCount() > 0, timeout=10000)
    assert widget._table.rowCount() == 4  # one row per Z slice
    headers = [
        widget._table.horizontalHeaderItem(c).text()
        for c in range(widget._table.columnCount())
    ]
    assert 'slice' in headers
    # selecting a per-slice row should not raise (slice-aware scatter)
    widget._table.selectRow(2)


def test_per_slice_on_2d_rejected(make_napari_viewer, rng):
    viewer = make_napari_viewer()
    a = rng.random((16, 16)).astype(np.float32)  # 2D
    layer_a = viewer.add_image(a, name='a')
    layer_b = viewer.add_image(a.copy(), name='b')
    widget = ColocalizationWidget(viewer)
    widget._image_a_combo.value = layer_a
    widget._image_b_combo.value = layer_b
    widget._per_slice_check.setChecked(True)
    assert widget.gather_params() is None


def test_run_with_otsu_threshold(make_napari_viewer, qtbot, rng):
    viewer = make_napari_viewer()
    a = np.zeros((32, 32), dtype=np.float32)
    a[8:24, 8:24] = 1.0
    layer_a = viewer.add_image(a, name='a')
    layer_b = viewer.add_image(a.copy(), name='b')
    widget = ColocalizationWidget(viewer)
    widget._image_a_combo.value = layer_a
    widget._image_b_combo.value = layer_b
    widget._cb_mcc.setChecked(True)
    _select_combo_data(widget._threshold_combo, 'otsu')
    # an auto-threshold method hides the manual T_a/T_b row
    assert widget._manual_row.isHidden() is True

    widget._on_run_clicked()
    qtbot.waitUntil(lambda: widget._table.rowCount() > 0, timeout=10000)
    headers = [
        widget._table.horizontalHeaderItem(c).text()
        for c in range(widget._table.columnCount())
    ]
    m1 = float(widget._table.item(0, headers.index('m1')).text())
    assert m1 == pytest.approx(1.0)


def test_manual_row_visibility_follows_method(make_napari_viewer):
    viewer = make_napari_viewer()
    widget = ColocalizationWidget(viewer)
    # default 'costes' -> manual row hidden
    assert widget._manual_row.isHidden() is True
    _select_combo_data(widget._threshold_combo, 'manual')
    assert widget._manual_row.isHidden() is False
    _select_combo_data(widget._threshold_combo, 'li')
    assert widget._manual_row.isHidden() is True


def test_run_all_to_all_populates_table(make_napari_viewer, qtbot, rng):
    viewer = make_napari_viewer()
    a = rng.random((32, 32)).astype(np.float32)
    stack = np.stack([a, a, a], axis=0)
    layer = viewer.add_image(stack, name='stack')
    widget = ColocalizationWidget(viewer)
    widget._mode_all.setChecked(True)
    widget._stack_combo.value = layer
    widget._channel_axis_spin.setValue(0)
    widget._cb_mcc.setChecked(False)

    widget._on_run_clicked()

    qtbot.waitUntil(lambda: widget._table.rowCount() > 0, timeout=10000)
    assert widget._table.rowCount() == 3


def test_export_csv_writes_file(make_napari_viewer, qtbot, rng, tmp_path):
    viewer = make_napari_viewer()
    a = rng.random((16, 16)).astype(np.float32)
    layer_a = viewer.add_image(a, name='a')
    layer_b = viewer.add_image(a.copy(), name='b')
    widget = ColocalizationWidget(viewer)
    widget._image_a_combo.value = layer_a
    widget._image_b_combo.value = layer_b
    widget._cb_mcc.setChecked(False)

    widget._on_run_clicked()
    qtbot.waitUntil(lambda: widget._table.rowCount() > 0, timeout=10000)

    out_path = tmp_path / 'results.csv'
    widget.write_csv(str(out_path), widget._results)
    assert out_path.exists()
    with open(out_path, newline='') as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
    assert len(rows) == 1
    assert 'pcc' in rows[0]


def test_run_with_label_mask_yields_one_row_per_region(
    make_napari_viewer, qtbot, rng
):
    viewer = make_napari_viewer()
    a = rng.random((20, 20)).astype(np.float32)
    layer_a = viewer.add_image(a, name='a')
    layer_b = viewer.add_image(a.copy(), name='b')
    label = np.zeros((20, 20), dtype=np.int32)
    label[:10, :] = 1
    label[10:, :] = 2
    label_layer = viewer.add_labels(label, name='regions')
    widget = ColocalizationWidget(viewer)
    widget._image_a_combo.value = layer_a
    widget._image_b_combo.value = layer_b
    _select_region_layer(widget, label_layer)
    widget._cb_mcc.setChecked(False)

    widget._on_run_clicked()
    qtbot.waitUntil(lambda: widget._table.rowCount() > 0, timeout=10000)
    assert widget._table.rowCount() == 2


def test_degenerate_region_sets_summary(make_napari_viewer, qtbot, rng):
    viewer = make_napari_viewer()
    a = rng.random((20, 20)).astype(np.float32)
    a[:10, :] = 1.0  # top region constant -> PCC undefined there
    layer_a = viewer.add_image(a, name='a')
    layer_b = viewer.add_image(a.copy(), name='b')
    label = np.zeros((20, 20), dtype=np.int32)
    label[:10, :] = 1
    label[10:, :] = 2
    label_layer = viewer.add_labels(label, name='regions')
    widget = ColocalizationWidget(viewer)
    widget._image_a_combo.value = layer_a
    widget._image_b_combo.value = layer_b
    _select_region_layer(widget, label_layer)
    widget._cb_srcc.setChecked(False)
    widget._cb_pcc.setChecked(True)
    widget._cb_mcc.setChecked(False)

    widget._on_run_clicked()
    qtbot.waitUntil(lambda: widget._table.rowCount() > 0, timeout=10000)
    assert 'could not be computed' in widget._summary_label.text()


def test_clean_run_leaves_summary_empty(make_napari_viewer, qtbot, rng):
    viewer = make_napari_viewer()
    a = rng.random((16, 16)).astype(np.float32)
    layer_a = viewer.add_image(a, name='a')
    layer_b = viewer.add_image(a.copy(), name='b')
    widget = ColocalizationWidget(viewer)
    widget._image_a_combo.value = layer_a
    widget._image_b_combo.value = layer_b
    widget._cb_mcc.setChecked(False)

    widget._on_run_clicked()
    qtbot.waitUntil(lambda: widget._table.rowCount() > 0, timeout=10000)
    assert widget._summary_label.text() == ''


def test_results_hidden_until_run(make_napari_viewer, qtbot, rng):
    viewer = make_napari_viewer()
    a = rng.random((16, 16)).astype(np.float32)
    layer_a = viewer.add_image(a, name='a')
    layer_b = viewer.add_image(a.copy(), name='b')
    widget = ColocalizationWidget(viewer)
    assert widget._results_group.isHidden() is True

    widget._image_a_combo.value = layer_a
    widget._image_b_combo.value = layer_b
    widget._cb_mcc.setChecked(False)
    widget._on_run_clicked()

    qtbot.waitUntil(
        lambda: not widget._results_group.isHidden(), timeout=10000
    )
    assert widget._export_button.isHidden() is False


def _select_combo_data(combo, data):
    combo.setCurrentIndex(combo.findData(data))


def test_diag_param_groups_follow_method(make_napari_viewer):
    viewer = make_napari_viewer()
    widget = ColocalizationWidget(viewer)
    # Costes is the default selection.
    assert widget._diag_costes_group.isHidden() is False
    assert widget._diag_ccf_group.isHidden() is True
    _select_combo_data(widget._diag_method_combo, 'ccf')
    assert widget._diag_ccf_group.isHidden() is False
    assert widget._diag_costes_group.isHidden() is True
    _select_combo_data(widget._diag_method_combo, 'ica')
    assert widget._diag_ica_group.isHidden() is False


def test_diag_combos_list_images(make_napari_viewer, rng):
    viewer = make_napari_viewer()
    a = rng.random((16, 16)).astype(np.float32)
    layer_a = viewer.add_image(a, name='a')
    layer_b = viewer.add_image(a.copy(), name='b')
    widget = ColocalizationWidget(viewer)
    # diagnostics pair defaults to two distinct image layers
    assert widget._diag_image_a_combo.value is layer_a
    assert widget._diag_image_b_combo.value is layer_b


def test_diag_ccf_runs_and_summarises(make_napari_viewer, qtbot, rng):
    viewer = make_napari_viewer()
    a = rng.random((32, 32)).astype(np.float32)
    layer_a = viewer.add_image(a, name='a')
    layer_b = viewer.add_image(a.copy(), name='b')
    widget = ColocalizationWidget(viewer)
    widget._diag_image_a_combo.value = layer_a
    widget._diag_image_b_combo.value = layer_b
    _select_combo_data(widget._diag_method_combo, 'ccf')
    widget._ccf_max_shift.setValue(5)

    widget._on_diag_run_clicked()
    qtbot.waitUntil(
        lambda: widget._diag_summary_label.text() != '', timeout=10000
    )
    assert 'Peak Pearson r' in widget._diag_summary_label.text()


def test_diag_costes_runs_and_summarises(make_napari_viewer, qtbot, rng):
    viewer = make_napari_viewer()
    a = rng.random((32, 32)).astype(np.float32)
    layer_a = viewer.add_image(a, name='a')
    layer_b = viewer.add_image(a.copy(), name='b')
    widget = ColocalizationWidget(viewer)
    widget._diag_image_a_combo.value = layer_a
    widget._diag_image_b_combo.value = layer_b
    _select_combo_data(widget._diag_method_combo, 'costes')
    widget._costes_niter.setValue(20)
    widget._costes_block.setValue(8)

    widget._on_diag_run_clicked()
    qtbot.waitUntil(
        lambda: widget._diag_summary_label.text() != '', timeout=15000
    )
    assert 'Observed PCC' in widget._diag_summary_label.text()


def test_diag_ica_runs_and_summarises(make_napari_viewer, qtbot, rng):
    viewer = make_napari_viewer()
    a = rng.random((24, 24)).astype(np.float32)
    layer_a = viewer.add_image(a, name='a')
    layer_b = viewer.add_image(a.copy(), name='b')
    widget = ColocalizationWidget(viewer)
    widget._diag_image_a_combo.value = layer_a
    widget._diag_image_b_combo.value = layer_b
    _select_combo_data(widget._diag_method_combo, 'ica')

    widget._on_diag_run_clicked()
    qtbot.waitUntil(
        lambda: widget._diag_summary_label.text() != '', timeout=10000
    )
    assert 'ICQ' in widget._diag_summary_label.text()


def test_diag_costes_accepts_3d(make_napari_viewer, rng):
    viewer = make_napari_viewer()
    a = rng.random((8, 16, 16)).astype(np.float32)  # 3D
    layer_a = viewer.add_image(a, name='a')
    layer_b = viewer.add_image(a.copy(), name='b')
    widget = ColocalizationWidget(viewer)
    widget._diag_image_a_combo.value = layer_a
    widget._diag_image_b_combo.value = layer_b
    _select_combo_data(widget._diag_method_combo, 'costes')
    widget._costes_block.setValue(8)
    # Costes now supports 3D, so it is accepted (worker would run).
    assert widget._gather_diag_params() is not None


def test_diag_costes_block_too_large_rejected(make_napari_viewer, rng):
    viewer = make_napari_viewer()
    a = rng.random((16, 16)).astype(np.float32)
    layer_a = viewer.add_image(a, name='a')
    layer_b = viewer.add_image(a.copy(), name='b')
    widget = ColocalizationWidget(viewer)
    widget._diag_image_a_combo.value = layer_a
    widget._diag_image_b_combo.value = layer_b
    _select_combo_data(widget._diag_method_combo, 'costes')
    widget._costes_block.setValue(32)  # larger than the 16 px image
    # Anticipated precondition: rejected up front, no worker dispatched.
    assert widget._gather_diag_params() is None


def test_diag_ccf_accepts_3d(make_napari_viewer, rng):
    viewer = make_napari_viewer()
    a = rng.random((4, 16, 16)).astype(np.float32)  # 3D
    layer_a = viewer.add_image(a, name='a')
    layer_b = viewer.add_image(a.copy(), name='b')
    widget = ColocalizationWidget(viewer)
    widget._diag_image_a_combo.value = layer_a
    widget._diag_image_b_combo.value = layer_b
    _select_combo_data(widget._diag_method_combo, 'ccf')
    # CCF works in 3D (shift along the last axis), so it is not blocked.
    assert widget._gather_diag_params() is not None


def _build_two_shape_widget(make_napari_viewer, qtbot, rng):
    viewer = make_napari_viewer()
    a = rng.random((20, 20)).astype(np.float32)
    layer_a = viewer.add_image(a, name='a')
    layer_b = viewer.add_image(a.copy(), name='b')
    shapes = viewer.add_shapes(
        [
            np.array([[1.0, 1.0], [1.0, 8.0], [8.0, 8.0], [8.0, 1.0]]),
            np.array([[12.0, 12.0], [12.0, 18.0], [18.0, 18.0], [18.0, 12.0]]),
        ],
        shape_type='polygon',
        name='regions',
    )
    widget = ColocalizationWidget(viewer)
    widget._image_a_combo.value = layer_a
    widget._image_b_combo.value = layer_b
    _select_region_layer(widget, shapes)

    widget._on_run_clicked()
    qtbot.waitUntil(lambda: widget._table.rowCount() >= 2, timeout=10000)
    return widget, shapes


def test_row_selection_highlights_shape(make_napari_viewer, qtbot, rng):
    widget, shapes = _build_two_shape_widget(make_napari_viewer, qtbot, rng)
    widget._table.selectRow(0)
    assert 0 in shapes.selected_data


def test_multi_row_selection_highlights_all_shapes(
    make_napari_viewer, qtbot, rng
):
    widget, shapes = _build_two_shape_widget(make_napari_viewer, qtbot, rng)
    widget._table.selectAll()
    assert set(shapes.selected_data) == {0, 1}


def test_clearing_table_selection_clears_highlight(
    make_napari_viewer, qtbot, rng
):
    widget, shapes = _build_two_shape_widget(make_napari_viewer, qtbot, rng)
    widget._table.selectRow(0)
    assert 0 in shapes.selected_data
    widget._table.clearSelection()
    assert len(shapes.selected_data) == 0
