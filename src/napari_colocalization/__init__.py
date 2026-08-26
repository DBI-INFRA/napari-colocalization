try:
    from ._version import version as __version__
except ImportError:
    __version__ = 'unknown'

from ._analysis import COLUMNS, analyse_all_to_all, analyse_pairwise
from ._diagnostics import (
    costes_randomization,
    li_ica,
    scramble_example,
    van_steensel_ccf,
)
from ._masking import (
    iter_regions,
    labels_to_label_mask,
    shapes_to_label_mask,
)
from ._metrics import (
    costes_regression,
    costes_threshold,
    li_icq,
    manders,
    overlap,
    pearson,
    spearman,
)
from ._objects import (
    OBJECT_COLUMNS,
    label_objects,
    nearest_neighbour_vectors,
    object_centroids,
    object_table,
)
from ._sample_data import (
    make_sample_data,
    make_sample_data_3d,
    make_sample_data_cbs006rbm,
    make_sample_data_coloc,
)


def __getattr__(name):
    # The widget is the one root export that needs napari and Qt.
    # Importing it lazily (PEP 562) keeps the compute API usable in
    # environments without them; npe2 resolves the manifest's
    # python_name via attribute access, so plugin discovery still
    # works.
    if name == 'ColocalizationWidget':
        from ._widget import ColocalizationWidget

        return ColocalizationWidget
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')


__all__ = (
    'COLUMNS',
    'OBJECT_COLUMNS',
    'ColocalizationWidget',
    'analyse_all_to_all',
    'analyse_pairwise',
    'costes_randomization',
    'costes_regression',
    'costes_threshold',
    'iter_regions',
    'label_objects',
    'labels_to_label_mask',
    'li_ica',
    'li_icq',
    'make_sample_data',
    'make_sample_data_3d',
    'make_sample_data_cbs006rbm',
    'make_sample_data_coloc',
    'manders',
    'nearest_neighbour_vectors',
    'object_centroids',
    'object_table',
    'overlap',
    'pearson',
    'scramble_example',
    'shapes_to_label_mask',
    'spearman',
    'van_steensel_ccf',
)
