"""The package root must import without napari or Qt installed.

The compute API is advertised as usable from plain script
environments, but the development environment always has
``napari[qt]`` - so the import is checked in a subprocess with both
modules blocked.
"""

import subprocess
import sys


def test_package_root_imports_without_qt_or_napari():
    code = (
        'import sys; '
        "sys.modules['qtpy'] = None; "
        "sys.modules['napari'] = None; "
        'import napari_colocalization; '
        'napari_colocalization.pearson'
    )
    subprocess.run([sys.executable, '-c', code], check=True)


def test_widget_is_reachable_from_package_root():
    # npe2 resolves the manifest's python_name by attribute access on
    # the package, which the lazy __getattr__ must serve.
    import napari_colocalization

    assert napari_colocalization.ColocalizationWidget is not None
