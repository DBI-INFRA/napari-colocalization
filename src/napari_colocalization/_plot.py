"""Embedded matplotlib canvases for the results dock.

``ScatterCanvas`` plots channel-A intensity against channel-B
intensity for the currently selected (region, channel-pair) and
overlays the Manders thresholds and Costes regression line. Uses
hexbin to stay responsive on large images. ``DiagnosticCanvas``
backs the Diagnostics tab, rendering the Costes randomization null
distribution, the Van Steensel CCF curve, or the Li ICA scatters.

Both default to napari's dark look and take their background /
foreground colours from ``set_theme`` - the colours are passed in by
the dock widget, so this module stays free of napari imports.
"""

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from qtpy.QtWidgets import QSizePolicy

DEFAULT_BACKGROUND = 'black'
DEFAULT_FOREGROUND = 'white'


def _hexbin_cmap():
    """Inferno with the near-black low end trimmed (visible on black)."""
    from matplotlib import cm
    from matplotlib.colors import ListedColormap

    return ListedColormap(cm.inferno(np.linspace(0.2, 1.0, 256)))


def _save_figure(figure, path, width_in, height_in, dpi):
    """Write ``figure`` to ``path`` at the given size, then restore it.

    ``forward=False`` resizes for the save only, without propagating to
    the on-screen canvas (avoids a flicker / dock relayout).
    """
    orig_size = figure.get_size_inches()
    figure.set_size_inches(width_in, height_in, forward=False)
    try:
        figure.savefig(path, dpi=dpi, facecolor=figure.get_facecolor())
    finally:
        figure.set_size_inches(orig_size, forward=False)


class _ThemedCanvas(FigureCanvasQTAgg):
    """Canvas whose figure follows a two-colour theme.

    ``_background`` paints the figure and axes faces, ``_foreground``
    everything drawn on top of them (spines, ticks, labels, text).
    Accent colours (thresholds, regression line, CCF curve) are
    deliberately theme-independent so they read the same either way.
    """

    def __init__(self, min_height):
        self._background = DEFAULT_BACKGROUND
        self._foreground = DEFAULT_FOREGROUND
        # constrained_layout reflows axis labels and titles on
        # every resize event, so the plot does not get cropped
        # when the dock widget is narrow.
        self._figure = Figure(
            figsize=(4, 3),
            constrained_layout=True,
            facecolor=self._background,
        )
        super().__init__(self._figure)
        # Expanding policies let the canvas grow into the layout
        # cell that owns it (stretch=1 in the parent QVBoxLayout).
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumHeight(min_height)

    def _style_axes(self, ax):
        """Paint one axes in the current theme colours."""
        ax.set_facecolor(self._background)
        for spine in ax.spines.values():
            spine.set_color(self._foreground)
        ax.title.set_color(self._foreground)
        ax.xaxis.label.set_color(self._foreground)
        ax.yaxis.label.set_color(self._foreground)
        ax.tick_params(colors=self._foreground)

    def _annotation_bbox(self):
        """Translucent backing box for corner annotations."""
        return {
            'facecolor': self._foreground,
            'alpha': 0.15,
            'edgecolor': 'none',
        }

    def set_theme(self, background, foreground):
        """Restyle the figure for new background/foreground colours."""
        self._background = background
        self._foreground = foreground
        self._figure.patch.set_facecolor(background)
        for ax in self._figure.axes:
            self._style_axes(ax)
        self.draw_idle()

    def save_figure(self, path, width_in, height_in, dpi):
        """Write the current figure to ``path`` at the given size."""
        _save_figure(self._figure, path, width_in, height_in, dpi)


class ScatterCanvas(_ThemedCanvas):
    """Two-axis scatter of paired intensities."""

    def __init__(self):
        super().__init__(min_height=200)
        self._ax = self._figure.add_subplot(111)
        self.clear()

    def clear(self):
        self._ax.clear()
        self._figure.patch.set_facecolor(self._background)
        self._style_axes(self._ax)
        self._ax.set_xlabel('Channel A intensity')
        self._ax.set_ylabel('Channel B intensity')
        self._ax.text(
            0.5,
            0.5,
            'Run analysis to populate',
            ha='center',
            va='center',
            transform=self._ax.transAxes,
            color=self._foreground,
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
        xlabel='Channel A intensity',
        ylabel='Channel B intensity',
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
        xlabel, ylabel : str
            Axis labels (typically the two channel/layer names).
        title : str
            Axes title (typically the region + channel-pair label).
        annotation : str
            Multi-line text drawn in the upper-left corner - used
            to display metric values for the selected row.
        """
        if mask is None:
            xs = np.asarray(a).ravel()
            ys = np.asarray(b).ravel()
        else:
            xs = np.asarray(a)[mask]
            ys = np.asarray(b)[mask]

        self._ax.clear()
        self._figure.patch.set_facecolor(self._background)
        self._style_axes(self._ax)
        self._ax.set_xlabel(xlabel)
        self._ax.set_ylabel(ylabel)

        cmap = _hexbin_cmap()

        if xs.size:
            # hexbin aggregates points into hex cells, so render cost
            # is O(gridsize²) regardless of N. bins='log' applies log
            # color mapping so a few dense background cells don't wash
            # out signal cells. mincnt=1 leaves empty cells transparent
            # against the axes facecolor.
            self._ax.hexbin(
                xs,
                ys,
                gridsize=80,
                bins='log',
                cmap=cmap,
                mincnt=1,
                linewidths=0,
            )

        self._ax.set_title(title, color=self._foreground)

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
                color=self._foreground,
                bbox=self._annotation_bbox(),
            )

        self.draw_idle()


class DiagnosticCanvas(_ThemedCanvas):
    """Canvas for the Diagnostics tab.

    One canvas reused for every diagnostic; each ``plot_*`` method
    clears the figure and lays out the subplot(s) that diagnostic
    needs (a histogram, a line, or two scatter panels).
    """

    def __init__(self):
        super().__init__(min_height=220)
        # The placeholder message while no diagnostic has been run, so
        # a theme change can redraw it in the new colours.
        self._placeholder = None
        self.clear()

    def _reset_figure(self):
        self._figure.clear()
        self._figure.patch.set_facecolor(self._background)

    def set_theme(self, background, foreground):
        super().set_theme(background, foreground)
        if self._placeholder is not None:
            self.clear(self._placeholder)

    def clear(self, message='Run a diagnostic to populate'):
        self._placeholder = message
        self._reset_figure()
        ax = self._figure.add_subplot(111)
        self._style_axes(ax)
        ax.text(
            0.5,
            0.5,
            message,
            ha='center',
            va='center',
            transform=ax.transAxes,
            color=self._foreground,
            alpha=0.7,
        )
        self.draw_idle()

    def plot_ccf(self, shifts, ccf, title=''):
        """Van Steensel CCF: Pearson r as a function of pixel shift."""
        self._placeholder = None
        self._reset_figure()
        ax = self._figure.add_subplot(111)
        self._style_axes(ax)
        ax.plot(shifts, ccf, color='cyan')
        ax.axvline(0, color=self._foreground, linewidth=0.8, alpha=0.5)
        if np.any(np.isfinite(ccf)):
            peak = int(np.nanargmax(ccf))
            ax.plot(shifts[peak], ccf[peak], 'o', color='red')
        ax.set_xlabel('Shift (px)')
        ax.set_ylabel('Pearson r')
        ax.set_title(title, color=self._foreground)
        self.draw_idle()

    def plot_costes(self, observed, null, p_value, z_score, title=''):
        """Costes randomization: null PCC histogram + observed line."""
        self._placeholder = None
        self._reset_figure()
        ax = self._figure.add_subplot(111)
        self._style_axes(ax)
        finite = np.asarray(null)[np.isfinite(null)]
        if finite.size:
            ax.hist(finite, bins=40, color='gray', alpha=0.8)
        if np.isfinite(observed):
            ax.axvline(observed, color='red', linewidth=1.5, label='observed')
        ax.set_xlabel('Pearson r (scrambled)')
        ax.set_ylabel('count')
        ax.set_title(title, color=self._foreground)
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
            color=self._foreground,
            bbox=self._annotation_bbox(),
        )
        self.draw_idle()

    def plot_ica(self, a, b, products, names=('A', 'B'), title=''):
        """Li ICA: two panels of intensity vs covariance product."""
        self._placeholder = None
        self._reset_figure()
        cmap = _hexbin_cmap()
        for i, (channel, name) in enumerate(((a, names[0]), (b, names[1]))):
            ax = self._figure.add_subplot(1, 2, i + 1)
            self._style_axes(ax)
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
            ax.axvline(0, color=self._foreground, linewidth=0.8, alpha=0.5)
            ax.set_xlabel('(A-Ā)(B-B̄)')
            ax.set_ylabel(f'{name} intensity')
        self._figure.suptitle(title, color=self._foreground)
        self.draw_idle()
