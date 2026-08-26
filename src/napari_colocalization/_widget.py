"""Top-level dock widget for the colocalization plugin.

Holds the three analysis tabs - "Pixel-based" (per-region metric
table + cytofluorogram), "Diagnostics" (one single-pair diagnostic
plot at a time) and "Object-based" - and owns everything they share
with the viewer: the layer-list subscription that keeps their combos
in sync, the tab switch the Pixel-based tab asks for, and the plot
colours derived from napari's active theme.
"""

import contextlib
from typing import TYPE_CHECKING

from qtpy.QtWidgets import (
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ._diagnostics_tab import DiagnosticsTab
from ._object_tab import ObjectTab
from ._pixel_tab import PixelTab

if TYPE_CHECKING:
    import napari


# Plot colours used until (or unless) a napari theme can be read.
_DEFAULT_PLOT_BACKGROUND = 'black'
_DEFAULT_PLOT_FOREGROUND = 'white'


class ColocalizationWidget(QWidget):
    """Dock widget: the analysis tabs plus their viewer wiring."""

    def __init__(self, viewer: 'napari.viewer.Viewer'):
        super().__init__()
        self._viewer = viewer

        # Analysis families live on separate tabs: the multi-region
        # intensity table on one, the single-pair diagnostic plots on
        # the other. Inputs are not shared - diagnostics are always
        # pairwise, so each tab carries the channel selectors it needs.
        self._tabs = QTabWidget()
        self._pixel_tab = PixelTab(viewer, self._open_diagnostics)
        self._diagnostics_tab = DiagnosticsTab(viewer)
        self._object_tab = ObjectTab(viewer)
        self._tabs.addTab(self._pixel_tab, 'Pixel-based')
        self._tabs.addTab(self._diagnostics_tab, 'Diagnostics')
        self._tabs.addTab(self._object_tab, 'Object-based')

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(self._tabs)

        self._connect_layer_events()
        self._connect_theme_events()

    def _analysis_tabs(self):
        return (self._pixel_tab, self._diagnostics_tab, self._object_tab)

    # -- cross-tab navigation -------------------------------------------

    def _open_diagnostics(self, inputs):
        """Show the Diagnostics tab, optionally carrying inputs over.

        The Pixel-based tab hands over its channel/region selection
        (or ``None``); mediating here keeps the tabs unaware of each
        other.
        """
        if inputs is not None:
            self._diagnostics_tab.set_inputs(*inputs)
        self._tabs.setCurrentIndex(self._tabs.indexOf(self._diagnostics_tab))

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
        from napari.layers import Image, Labels, Shapes

        images = self._layers_of(Image)
        labels_layers = self._layers_of(Labels)
        region_layers = self._layers_of((Shapes, Labels))
        for tab in self._analysis_tabs():
            tab.refresh_layers(images, labels_layers, region_layers)

    # -- theme ----------------------------------------------------------

    def _connect_theme_events(self):
        with contextlib.suppress(AttributeError):
            self._viewer.events.theme.connect(self._apply_theme)
        self._apply_theme()

    def _apply_theme(self, _event=None):
        background, foreground = self._theme_colors()
        for tab in self._analysis_tabs():
            with contextlib.suppress(AttributeError):
                tab.apply_theme(background, foreground)

    def _theme_colors(self):
        """Plot (background, foreground) for the viewer's theme.

        Falls back to the dark-theme look if the theme cannot be
        resolved - theme names are user-extensible and the registry
        API is not part of napari's stable surface.
        """
        try:
            from napari.utils.theme import get_theme

            theme = get_theme(self._viewer.theme)
            return str(theme.canvas), str(theme.text)
        except (ImportError, AttributeError, KeyError, ValueError, TypeError):
            return _DEFAULT_PLOT_BACKGROUND, _DEFAULT_PLOT_FOREGROUND
