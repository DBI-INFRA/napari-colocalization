"""Embedded matplotlib canvases for the results dock.

``ScatterCanvas`` plots channel-A intensity against channel-B
intensity for the currently selected (region, channel-pair) and
overlays the Manders thresholds and Costes regression line. Uses
hexbin to stay responsive on large images. ``DiagnosticCanvas``
backs the Diagnostics tab, rendering the Costes randomization null
distribution, the Van Steensel CCF curve, or the Li ICA scatters.
"""

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from qtpy.QtWidgets import QSizePolicy


def _style_axes_dark(ax):
    """White-on-black styling matching napari's dark theme."""
    ax.set_facecolor('black')
    for spine in ax.spines.values():
        spine.set_color('white')
    ax.title.set_color('white')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.tick_params(colors='white')


class ScatterCanvas(FigureCanvasQTAgg):
    """Two-axis scatter of paired intensities."""

    def __init__(self):
        # constrained_layout reflows axis labels and titles on
        # every resize event, so the plot does not get cropped
        # when the dock widget is narrow.
        self._figure = Figure(
            figsize=(4, 3), constrained_layout=True, facecolor='black'
        )
        super().__init__(self._figure)
        self._ax = self._figure.add_subplot(111)
        # Expanding policies let the canvas grow into the layout
        # cell that owns it (stretch=1 in the parent QVBoxLayout).
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumHeight(200)
        self.clear()

    def _apply_dark_style(self):
        """Apply black background / white axis styling and remove ticks."""
        self._figure.patch.set_facecolor('black')
        self._ax.set_facecolor('black')
        for spine in self._ax.spines.values():
            spine.set_color('white')
        self._ax.title.set_color('white')
        self._ax.xaxis.label.set_color('white')
        self._ax.yaxis.label.set_color('white')
        # remove ticks and tick labels
        # self._ax.set_xticks([])
        # self._ax.set_yticks([])
        self._ax.tick_params(colors='white')

    def clear(self):
        self._ax.clear()
        self._apply_dark_style()
        self._ax.set_xlabel('Channel A intensity')
        self._ax.set_ylabel('Channel B intensity')
        self._ax.text(
            0.5,
            0.5,
            'Run analysis to populate',
            ha='center',
            va='center',
            transform=self._ax.transAxes,
            color='white',
            alpha=0.7,
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
        slope=None,
        intercept=None,
        xlim=None,
        ylim=None,
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
        slope, intercept : float or None
            If both finite, draw the Costes regression line
            ``b = slope * a + intercept`` (cyan dashed) over the
            data x-range.
        xlim, ylim : tuple of float or None
            Explicit axis ranges; when given they override the
            auto-fit, so plots are comparable across rows/images.
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

        self._ax.clear()
        self._apply_dark_style()
        self._ax.set_xlabel('Channel A intensity')
        self._ax.set_ylabel('Channel B intensity')

        # Inferno trimmed at the dark end — the bottom of the
        # default ramp is near-black and disappears against the
        # black axes facecolor.
        from matplotlib import cm
        from matplotlib.colors import ListedColormap

        cmap = ListedColormap(cm.inferno(np.linspace(0.2, 1.0, 256)))

        if xs.size:
            # hexbin aggregates points into hex cells, so render cost
            # is O(gridsize²) regardless of N. bins='log' applies log
            # color mapping so a few dense background cells don't wash
            # out signal cells. mincnt=1 leaves empty cells transparent
            # against the black axes facecolor.
            self._ax.hexbin(
                xs,
                ys,
                gridsize=80,
                bins='log',
                cmap=cmap,
                mincnt=1,
                linewidths=0,
            )

        self._ax.set_title(title, color='white')

        if (
            threshold_a is not None
            and threshold_b is not None
            and np.isfinite(threshold_a)
            and np.isfinite(threshold_b)
        ):
            # draw threshold lines but avoid letting them expand autoscale
            self._ax.axvline(threshold_a, color='red', linewidth=1)
            self._ax.axhline(threshold_b, color='red', linewidth=1)

        # Costes regression line the auto-threshold was found along.
        # Drawn before the explicit set_xlim/set_ylim below so it is
        # clipped to the data range rather than expanding autoscale.
        if (
            slope is not None
            and intercept is not None
            and np.isfinite(slope)
            and np.isfinite(intercept)
            and xs.size
        ):
            x_line = np.array([xs.min(), xs.max()], dtype=float)
            self._ax.plot(
                x_line,
                slope * x_line + intercept,
                color='cyan',
                linestyle='--',
                linewidth=1,
            )

        # ensure axes limits are determined by the scatter data alone
        if xs.size:
            xmin, xmax = float(xs.min()), float(xs.max())
            ymin, ymax = float(ys.min()), float(ys.max())
            # protect against zero-range data
            if xmax <= xmin:
                xmin -= 0.5
                xmax += 0.5
            if ymax <= ymin:
                ymin -= 0.5
                ymax += 0.5
            # add small padding
            xpad = 0.02 * (xmax - xmin)
            ypad = 0.02 * (ymax - ymin)
            if xpad == 0:
                xpad = 0.5
            if ypad == 0:
                ypad = 0.5
            self._ax.set_xlim(xmin - xpad, xmax + xpad)
            self._ax.set_ylim(ymin - ypad, ymax + ypad)

        # Explicit bounds (e.g. "fixed axes") win over the auto-fit so
        # the same scale is used for every row/region/image.
        if xlim is not None:
            self._ax.set_xlim(*xlim)
        if ylim is not None:
            self._ax.set_ylim(*ylim)

        if annotation:
            self._ax.text(
                0.02,
                0.98,
                annotation,
                transform=self._ax.transAxes,
                va='top',
                ha='left',
                fontsize=8,
                color='white',
                bbox={
                    'facecolor': 'white',
                    'alpha': 0.15,
                    'edgecolor': 'none',
                },
            )

        self.draw_idle()

    def save_figure(self, path, width_in, height_in, dpi):
        """Write the current figure to ``path`` at the given size.

        forward=False resizes for the save only, without propagating to
        the on-screen canvas (avoids a flicker / dock relayout).
        """
        orig_size = self._figure.get_size_inches()
        self._figure.set_size_inches(width_in, height_in, forward=False)
        try:
            self._figure.savefig(
                path,
                dpi=dpi,
                facecolor=self._figure.get_facecolor(),
            )
        finally:
            self._figure.set_size_inches(orig_size, forward=False)


class DiagnosticCanvas(FigureCanvasQTAgg):
    """Canvas for the Diagnostics tab.

    One canvas reused for every diagnostic; each ``plot_*`` method
    clears the figure and lays out the subplot(s) that diagnostic
    needs (a histogram, a line, or two scatter panels).
    """

    # Inferno trimmed at the dark end so faint cells don't vanish
    # against the black facecolor — shared with ScatterCanvas's look.
    def _hexbin_cmap(self):
        from matplotlib import cm
        from matplotlib.colors import ListedColormap

        return ListedColormap(cm.inferno(np.linspace(0.2, 1.0, 256)))

    def __init__(self):
        self._figure = Figure(
            figsize=(4, 3), constrained_layout=True, facecolor='black'
        )
        super().__init__(self._figure)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumHeight(220)
        self.clear()

    def clear(self, message='Run a diagnostic to populate'):
        self._figure.clear()
        self._figure.patch.set_facecolor('black')
        ax = self._figure.add_subplot(111)
        _style_axes_dark(ax)
        ax.text(
            0.5,
            0.5,
            message,
            ha='center',
            va='center',
            transform=ax.transAxes,
            color='white',
            alpha=0.7,
        )
        self.draw_idle()

    def plot_ccf(self, shifts, ccf, title=''):
        """Van Steensel CCF: Pearson r as a function of pixel shift."""
        self._figure.clear()
        self._figure.patch.set_facecolor('black')
        ax = self._figure.add_subplot(111)
        _style_axes_dark(ax)
        ax.plot(shifts, ccf, color='cyan')
        ax.axvline(0, color='white', linewidth=0.8, alpha=0.5)
        if np.any(np.isfinite(ccf)):
            peak = int(np.nanargmax(ccf))
            ax.plot(shifts[peak], ccf[peak], 'o', color='red')
        ax.set_xlabel('Shift (px)')
        ax.set_ylabel('Pearson r')
        ax.set_title(title, color='white')
        self.draw_idle()

    def plot_costes(self, observed, null, p_value, z_score, title=''):
        """Costes randomization: null PCC histogram + observed line."""
        self._figure.clear()
        self._figure.patch.set_facecolor('black')
        ax = self._figure.add_subplot(111)
        _style_axes_dark(ax)
        finite = np.asarray(null)[np.isfinite(null)]
        if finite.size:
            ax.hist(finite, bins=40, color='gray', alpha=0.8)
        if np.isfinite(observed):
            ax.axvline(observed, color='red', linewidth=1.5, label='observed')
        ax.set_xlabel('Pearson r (scrambled)')
        ax.set_ylabel('count')
        ax.set_title(title, color='white')
        annotation = f'observed = {observed:.4g}'
        if np.isfinite(p_value):
            annotation += f'\np = {p_value:.4g}'
        if np.isfinite(z_score):
            annotation += f'\nz = {z_score:.3g}'
        ax.text(
            0.02,
            0.98,
            annotation,
            transform=ax.transAxes,
            va='top',
            ha='left',
            fontsize=8,
            color='white',
            bbox={'facecolor': 'white', 'alpha': 0.15, 'edgecolor': 'none'},
        )
        self.draw_idle()

    def plot_ica(self, a, b, products, names=('A', 'B'), title=''):
        """Li ICA: two panels of intensity vs covariance product."""
        self._figure.clear()
        self._figure.patch.set_facecolor('black')
        cmap = self._hexbin_cmap()
        for i, (channel, name) in enumerate(((a, names[0]), (b, names[1]))):
            ax = self._figure.add_subplot(1, 2, i + 1)
            _style_axes_dark(ax)
            if np.asarray(channel).size:
                ax.hexbin(
                    products,
                    channel,
                    gridsize=60,
                    bins='log',
                    cmap=cmap,
                    mincnt=1,
                    linewidths=0,
                )
            ax.axvline(0, color='white', linewidth=0.8, alpha=0.5)
            ax.set_xlabel('(A-Ā)(B-B̄)')
            ax.set_ylabel(f'{name} intensity')
        self._figure.suptitle(title, color='white')
        self.draw_idle()

    def save_figure(self, path, width_in, height_in, dpi):
        """Write the current figure to ``path`` at the given size."""
        orig_size = self._figure.get_size_inches()
        self._figure.set_size_inches(width_in, height_in, forward=False)
        try:
            self._figure.savefig(
                path, dpi=dpi, facecolor=self._figure.get_facecolor()
            )
        finally:
            self._figure.set_size_inches(orig_size, forward=False)
