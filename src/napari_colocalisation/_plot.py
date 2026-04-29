"""Embedded matplotlib scatter plot for the results dock.

The canvas plots channel-A intensity against channel-B intensity
for the currently selected (region, channel-pair) and overlays
horizontal/vertical lines at the Manders thresholds when
available. Subsampled to keep redraws responsive on large images.
"""

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from qtpy.QtWidgets import QSizePolicy

MAX_POINTS = 5000


class ScatterCanvas(FigureCanvasQTAgg):
    """Two-axis scatter of paired intensities."""

    def __init__(self):
        # constrained_layout reflows axis labels and titles on
        # every resize event, so the plot does not get cropped
        # when the dock widget is narrow.
        self._figure = Figure(figsize=(4, 3), constrained_layout=True)
        super().__init__(self._figure)
        self._ax = self._figure.add_subplot(111)
        # Expanding policies let the canvas grow into the layout
        # cell that owns it (stretch=1 in the parent QVBoxLayout).
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumHeight(200)
        self.clear()

    def clear(self):
        self._ax.clear()
        self._ax.set_xlabel('Channel A intensity')
        self._ax.set_ylabel('Channel B intensity')
        self._ax.text(
            0.5,
            0.5,
            'Run analysis to populate',
            ha='center',
            va='center',
            transform=self._ax.transAxes,
            color='grey',
        )
        self.draw_idle()

    def update_plot(
        self,
        a,
        b,
        *,
        mask=None,
        threshold_a=None,
        threshold_b=None,
        title='',
        annotation='',
    ):
        """Re-render the scatter for the given channel pair / region.

        Parameters
        ----------
        a, b : ndarray
            Same-shape intensity arrays.
        mask : ndarray or None
            Boolean array selecting which pixels to plot.
        threshold_a, threshold_b : float or None
            If both finite, draw red v/h lines at those values.
        title : str
            Axes title (typically the region + channel-pair label).
        annotation : str
            Multi-line text drawn in the upper-left corner — used
            to display metric values for the selected row.
        """
        if mask is None:
            xs = np.asarray(a).ravel()
            ys = np.asarray(b).ravel()
        else:
            xs = np.asarray(a)[mask]
            ys = np.asarray(b)[mask]

        if xs.size > MAX_POINTS:
            idx = np.random.default_rng(seed=0).choice(
                xs.size, size=MAX_POINTS, replace=False
            )
            xs = xs[idx]
            ys = ys[idx]

        self._ax.clear()
        self._ax.set_xlabel('Channel A intensity')
        self._ax.set_ylabel('Channel B intensity')
        self._ax.scatter(xs, ys, s=4, alpha=0.4)
        self._ax.set_title(title)

        if (
            threshold_a is not None
            and threshold_b is not None
            and np.isfinite(threshold_a)
            and np.isfinite(threshold_b)
        ):
            self._ax.axvline(threshold_a, color='red', linewidth=1)
            self._ax.axhline(threshold_b, color='red', linewidth=1)

        if annotation:
            self._ax.text(
                0.02,
                0.98,
                annotation,
                transform=self._ax.transAxes,
                va='top',
                ha='left',
                fontsize=8,
                bbox={
                    'facecolor': 'white',
                    'alpha': 0.7,
                    'edgecolor': 'none',
                },
            )

        self.draw_idle()
