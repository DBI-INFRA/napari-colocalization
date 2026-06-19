import csv

import numpy as np
import pytest

from napari_colocalization import ColocalizationWidget


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
    combo = widget._region_combo
    for i in range(combo.count()):
        if combo.itemData(i) is layer:
            combo.setCurrentIndex(i)
            return
    raise AssertionError(f'region layer {layer!r} not in combo')


def _region_options(widget):
    combo = widget._region_combo
    return [combo.itemText(i) for i in range(combo.count())]


def _add_pair(widget, a, b=None):
    """Add image layers A and B and select them on the Intensity tab."""
    viewer = widget._viewer
    layer_a = viewer.add_image(a, name='a')
    layer_b = viewer.add_image(a.copy() if b is None else b, name='b')
    widget._image_a_combo.value = layer_a
    widget._image_b_combo.value = layer_b
    return layer_a, layer_b


def _run(widget, qtbot):
    widget._on_run_clicked()
    qtbot.waitUntil(lambda: widget._table.rowCount() > 0, timeout=10000)


def _cell(table, row, column):
    headers = [
        table.horizontalHeaderItem(c).text()
        for c in range(table.columnCount())
    ]
    return table.item(row, headers.index(column)).text()


# -- construction & UI toggles ----------------------------------------


def test_default_metric_is_spearman_only(widget):
    assert widget._cb_srcc.isChecked()
    assert not any(
        cb.isChecked()
        for cb in (
            widget._cb_pcc,
            widget._cb_icq,
            widget._cb_overlap,
            widget._cb_mcc,
        )
    )


def test_mode_toggle_swaps_pairwise_and_all_to_all(widget):
    assert not widget._pairwise_group.isHidden()
    assert widget._all_group.isHidden()
    widget._mode_all.setChecked(True)
    assert widget._pairwise_group.isHidden()
    assert not widget._all_group.isHidden()


def test_threshold_controls_visibility(widget):
    # The threshold group only appears with Manders; the manual T_a/T_b
    # row only appears for the Manual method.
    assert widget._threshold_group.isHidden()
    widget._cb_mcc.setChecked(True)
    assert not widget._threshold_group.isHidden()
    assert widget._manual_row.isHidden()  # default is Costes
    _select(widget._threshold_combo, 'manual')
    assert not widget._manual_row.isHidden()
    _select(widget._threshold_combo, 'otsu')
    assert widget._manual_row.isHidden()


def test_ab_selectors_default_to_distinct_layers(widget):
    viewer = widget._viewer
    a = np.zeros((16, 16), dtype=np.float32)
    viewer.add_image(a, name='a')
    viewer.add_image(a.copy(), name='b')
    for combo_a, combo_b in (
        (widget._image_a_combo, widget._image_b_combo),
        (widget._diag_image_a_combo, widget._diag_image_b_combo),
        (widget._obj_image_a_combo, widget._obj_image_b_combo),
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
    widget._cb_pcc.setChecked(True)
    widget._cb_overlap.setChecked(True)
    _run(widget, qtbot)
    assert widget._table.rowCount() == 1
    assert float(_cell(widget._table, 0, 'pcc')) == pytest.approx(1.0)
    assert float(_cell(widget._table, 0, 'overlap')) == pytest.approx(1.0)
    # results panel is revealed once there are rows
    assert not widget._results_group.isHidden()


def test_all_to_all_run_covers_every_pair(widget, qtbot, rng):
    a = rng.random((32, 32)).astype(np.float32)
    layer = widget._viewer.add_image(np.stack([a, a, a]), name='stack')
    widget._mode_all.setChecked(True)
    widget._stack_combo.value = layer
    widget._channel_axis_spin.setValue(0)
    widget._cb_mcc.setChecked(False)
    _run(widget, qtbot)
    assert widget._table.rowCount() == 3  # 3 unordered channel pairs


def test_region_mask_gives_one_row_per_region(widget, qtbot, rng):
    _add_pair(widget, rng.random((20, 20)).astype(np.float32))
    label = np.zeros((20, 20), np.int32)
    label[:10], label[10:] = 1, 2
    layer = widget._viewer.add_labels(label, name='regions')
    _select_region(widget, layer)
    _run(widget, qtbot)
    assert widget._table.rowCount() == 2


def test_per_slice_run_gives_one_row_per_slice(widget, qtbot, rng):
    _add_pair(widget, rng.random((4, 16, 16)).astype(np.float32))
    widget._per_slice_check.setChecked(True)
    widget._slice_axis_spin.setValue(0)
    _run(widget, qtbot)
    assert widget._table.rowCount() == 4
    widget._table.selectRow(2)  # slice-aware scatter must not raise


def test_per_slice_requires_3d(widget, rng):
    _add_pair(widget, rng.random((16, 16)).astype(np.float32))  # 2D
    widget._per_slice_check.setChecked(True)
    assert widget.gather_params() is None


def test_auto_threshold_manders(widget, qtbot, rng):
    # Otsu cleanly separates a two-level image, so the bright square
    # fully colocalizes with itself (M1 == 1).
    img = np.zeros((32, 32), dtype=np.float32)
    img[8:24, 8:24] = 1.0
    _add_pair(widget, img)
    widget._cb_mcc.setChecked(True)
    _select(widget._threshold_combo, 'otsu')
    _run(widget, qtbot)
    assert float(_cell(widget._table, 0, 'm1')) == pytest.approx(1.0)


def test_uncomputable_region_is_summarised(widget, qtbot, rng):
    a = rng.random((20, 20)).astype(np.float32)
    a[:10] = 1.0  # constant region -> PCC undefined there
    _add_pair(widget, a)
    label = np.zeros((20, 20), np.int32)
    label[:10], label[10:] = 1, 2
    _select_region(widget, widget._viewer.add_labels(label, name='r'))
    widget._cb_srcc.setChecked(False)
    widget._cb_pcc.setChecked(True)
    _run(widget, qtbot)
    assert 'could not be computed' in widget._summary_label.text()


def test_fixed_axes_lock_scatter_bounds(widget, qtbot, rng):
    a = (rng.random((32, 32)) * 5.0).astype(np.float32)
    _add_pair(widget, a)
    _run(widget, qtbot)
    widget._fixed_axes_check.setChecked(True)
    widget._table.selectRow(0)
    xlim = widget._scatter._ax.get_xlim()
    assert xlim[0] == pytest.approx(0.0)
    assert xlim[1] == pytest.approx(float(a.max()))


def test_csv_export(widget, qtbot, rng, tmp_path):
    _add_pair(widget, rng.random((16, 16)).astype(np.float32))
    _run(widget, qtbot)
    out = tmp_path / 'results.csv'
    widget.write_csv(str(out), widget._results)
    with open(out, newline='') as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 1
    assert 'pcc' in rows[0]


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
    widget._on_run_clicked()
    qtbot.waitUntil(lambda: widget._table.rowCount() >= 2, timeout=10000)

    widget._table.selectRow(0)
    assert 0 in shapes.selected_data
    widget._table.selectAll()
    assert set(shapes.selected_data) == {0, 1}
    widget._table.clearSelection()
    assert len(shapes.selected_data) == 0


def test_add_coloc_mask_layer(widget, qtbot, rng):
    from napari.layers import Labels

    img = np.zeros((32, 32), dtype=np.float32)
    img[8:24, 8:24] = 1.0
    _add_pair(widget, img)
    widget._cb_mcc.setChecked(True)
    _select(widget._threshold_combo, 'otsu')
    _run(widget, qtbot)
    widget._on_add_mask_clicked()
    masks = [
        layer
        for layer in widget._viewer.layers
        if isinstance(layer, Labels) and 'coloc' in layer.name
    ]
    assert len(masks) == 1
    assert int(masks[0].data.sum()) == 16 * 16  # the bright square


# -- diagnostics tab --------------------------------------------------


def test_diag_param_groups_follow_method(widget):
    assert not widget._diag_costes_group.isHidden()  # Costes default
    _select(widget._diag_method_combo, 'ccf')
    assert not widget._diag_ccf_group.isHidden()
    assert widget._diag_costes_group.isHidden()
    _select(widget._diag_method_combo, 'ica')
    assert not widget._diag_ica_group.isHidden()


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
    widget._diag_image_a_combo.value = viewer.add_image(a, name='a')
    widget._diag_image_b_combo.value = viewer.add_image(a.copy(), name='b')
    _select(widget._diag_method_combo, method)
    widget._costes_niter.setValue(20)  # keep the costes test quick
    widget._ccf_max_shift.setValue(5)
    widget._on_diag_run_clicked()
    qtbot.waitUntil(
        lambda: widget._diag_summary_label.text() != '', timeout=15000
    )
    assert expected in widget._diag_summary_label.text()


def test_diag_costes_validation(widget, rng):
    viewer = widget._viewer
    a = rng.random((8, 16, 16)).astype(np.float32)  # 3D
    widget._diag_image_a_combo.value = viewer.add_image(a, name='a')
    widget._diag_image_b_combo.value = viewer.add_image(a.copy(), name='b')
    _select(widget._diag_method_combo, 'costes')
    # Costes supports 3D...
    widget._costes_block.setValue(8)
    assert widget._gather_diag_params() is not None
    # ...but a block larger than the image is rejected up front.
    widget._costes_block.setValue(64)
    assert widget._gather_diag_params() is None


def test_add_scrambled_example_layer(widget, rng):
    viewer = widget._viewer
    a = rng.random((32, 32)).astype(np.float32)
    viewer.add_image(a, name='a')
    widget._diag_image_b_combo.value = viewer.add_image(a.copy(), name='b')
    n_before = len(viewer.layers)
    widget._on_diag_scramble_clicked()
    assert len(viewer.layers) == n_before + 1
    assert any('scrambled' in layer.name for layer in viewer.layers)


# -- object-based tab -------------------------------------------------


def test_object_source_toggle(widget):
    assert not widget._obj_threshold_group.isHidden()  # threshold default
    assert widget._obj_labels_group.isHidden()
    _select(widget._obj_source_combo, 'labels')
    assert widget._obj_threshold_group.isHidden()
    assert not widget._obj_labels_group.isHidden()


def test_object_run_from_threshold_with_overlays(widget, qtbot):
    from napari.layers import Points, Vectors

    img = np.zeros((20, 20), dtype=np.float32)
    img[2:6, 2:6] = 1.0
    img[12:16, 12:16] = 1.0
    viewer = widget._viewer
    widget._obj_image_a_combo.value = viewer.add_image(img, name='a')
    widget._obj_image_b_combo.value = viewer.add_image(img.copy(), name='b')
    widget._on_object_run_clicked()
    qtbot.waitUntil(lambda: widget._object_table.rowCount() > 0, timeout=10000)
    assert widget._object_table.rowCount() == 4  # 2 objects x 2 channels
    assert 'coincident' in widget._obj_summary_label.text()
    assert any(isinstance(layer, Points) for layer in viewer.layers)
    assert any(isinstance(layer, Vectors) for layer in viewer.layers)


def test_object_run_from_labels_layers(widget, qtbot):
    viewer = widget._viewer
    labels = np.zeros((20, 20), dtype=np.int32)
    labels[2:6, 2:6] = 1
    _select(widget._obj_source_combo, 'labels')
    widget._obj_labels_a_combo.value = viewer.add_labels(labels, name='oa')
    widget._obj_labels_b_combo.value = viewer.add_labels(
        labels.copy(), name='ob'
    )
    widget._obj_points_check.setChecked(False)
    widget._obj_links_check.setChecked(False)
    widget._on_object_run_clicked()
    qtbot.waitUntil(lambda: widget._object_table.rowCount() > 0, timeout=10000)
    assert widget._object_table.rowCount() == 2  # 1 object x 2 channels


def test_object_rerun_clears_previous_overlays(widget, qtbot):
    img = np.zeros((20, 20), dtype=np.float32)
    img[2:6, 2:6] = 1.0
    img[12:16, 12:16] = 1.0
    viewer = widget._viewer
    widget._obj_image_a_combo.value = viewer.add_image(img, name='a')
    widget._obj_image_b_combo.value = viewer.add_image(img.copy(), name='b')

    widget._on_object_run_clicked()
    qtbot.waitUntil(
        lambda: len(widget._object_overlay_layers) > 0, timeout=10000
    )
    first = list(widget._object_overlay_layers)
    n_layers = len(viewer.layers)

    widget._on_object_run_clicked()
    qtbot.waitUntil(
        lambda: bool(widget._object_overlay_layers)
        and widget._object_overlay_layers[0] not in first,
        timeout=10000,
    )
    assert all(layer not in viewer.layers for layer in first)
    assert len(viewer.layers) == n_layers  # overlays replaced, not stacked
