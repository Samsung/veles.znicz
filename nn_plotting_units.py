"""
Created on May 5, 2014

Copyright (c) 2014, Samsung Electronics, Co., Ltd.
"""

from __future__ import division
import numpy
import numpy.linalg as linalg
from zope.interface import implementer

import veles.config as config
import veles.formats as formats
from veles.mutable import Bool
import veles.plotter as plotter
import veles.opencl_types as opencl_types


@implementer(plotter.IPlotter)
class Weights2D(plotter.Plotter):
    """Plotter for drawing weights as 2D.

    Must be assigned before initialize():
        input

    Updates after run():

    Creates within initialize():

    """
    def __init__(self, workflow, **kwargs):
        kwargs["name"] = kwargs.get("name", "Weights")
        super(Weights2D, self).__init__(workflow, **kwargs)
        self.demand("input")
        self.get_shape_from = None
        self.limit = kwargs.get("limit", 64)
        self.transposed = False
        self.yuv = Bool(kwargs.get("yuv", False))
        self.cm = None
        self.pp = None
        self.show_figure = self.nothing
        self._pics_to_draw = []
        self.redraw_threshold = 1.5

    def __getstate__(self):
        state = super(Weights2D, self).__getstate__()
        if self.stripped_pickle:
            inp = state["input"][:self.limit]
            state["input"] = None
            state["_pics_to_draw"] = self._prepare_pics(inp)
        return state

    def _prepare_pics(self, inp):
        pics = []

        if type(inp) != numpy.ndarray or len(inp.shape) < 2:
            raise ValueError("input should be a numpy array (2D at least)")

        inp = inp.reshape(inp.shape[0], inp.size // inp.shape[0])

        if self.transposed:
            inp = inp.transpose()

        n_channels = 1
        if self.get_shape_from is None:
            sx = int(numpy.round(numpy.sqrt(inp.shape[1])))
            sy = int(inp.shape[1]) // sx
        elif isinstance(self.get_shape_from, formats.Vector):
            sx = self.get_shape_from.mem.shape[2]
            sy = self.get_shape_from.mem.shape[1]
        else:
            if len(self.get_shape_from) == 2:
                sx = self.get_shape_from[0]
                sy = self.get_shape_from[1]
            else:
                sx = self.get_shape_from[-2]
                sy = self.get_shape_from[-3]
                n_channels = self.get_shape_from[-1]

        sz = sx * sy * n_channels

        for i in range(inp.shape[0]):
            mem = inp[i].ravel()[:sz]
            if n_channels > 1:
                w = mem.reshape(sy, sx, n_channels)
                if n_channels == 2:
                    w = w[:, :, 0].reshape(sy, sx)
                elif n_channels > 3:
                    w = w[:, :, :3].reshape(sy, sx, 3)
                pics.append(formats.norm_image(w, self.yuv))
            else:
                pics.append(formats.norm_image(mem.reshape(sy, sx), self.yuv))
        return pics

    def redraw(self):
        figure = self.pp.figure(self.name)
        figure.clf()

        pics = self._pics_to_draw

        n_cols = int(numpy.round(numpy.sqrt(len(pics))))
        n_rows = int(numpy.ceil(len(pics) / n_cols))

        i = 0
        for _row in range(n_rows):
            for _col in range(n_cols):
                ax = figure.add_subplot(n_rows, n_cols, i + 1)
                ax.cla()
                ax.axis('off')
                # ax.set_title(self.name)
                if len(pics[i].shape) == 3:
                    ax.imshow(pics[i], interpolation="nearest")
                else:
                    ax.imshow(pics[i], interpolation="nearest",
                              cmap=self.cm.gray)
                i += 1
                if i >= len(pics):
                    break
            if i >= len(pics):
                break

        self.show_figure(figure)
        figure.canvas.draw()
        return figure


@implementer(plotter.IPlotter)
class MSEHistogram(plotter.Plotter):
    """Plotter for drawing histogram.

    Must be assigned before initialize():
        mse

    Updates after run():

    Creates within initialize():

    """
    def __init__(self, workflow, **kwargs):
        name = kwargs.get("name", "Histogram")
        n_bars = kwargs.get("n_bars", 35)
        kwargs["name"] = name
        kwargs["n_bars"] = n_bars
        super(MSEHistogram, self).__init__(workflow, **kwargs)
        self.val_mse = None
        self.mse_min = None
        self.mse_max = None
        self.n_bars = n_bars
        self.demand("mse")
        self.pp = None
        self.show_figure = self.nothing

    def initialize(self, **kwargs):
        super(MSEHistogram, self).initialize(**kwargs)
        self.val_mse = numpy.zeros(
            self.n_bars, dtype=opencl_types.dtypes[config.root.common.dtype])

    def redraw(self):
        fig = self.pp.figure(self.name)
        fig.clf()
        fig.patch.set_facecolor('#E8D6BB')
        # fig.patch.set_alpha(0.45)

        ax = fig.add_subplot(1, 1, 1)
        ax.cla()
        ax.patch.set_facecolor('#ffe6ca')
        # ax.patch.set_alpha(0.45)

        ymin = self.val_min
        ymax = (self.val_max) * 1.3
        xmin = self.mse_min
        xmax = self.mse_max
        width = ((xmax - xmin) / self.n_bars) * 0.8
        t0 = 0.65 * ymax
        l1 = width * 0.5

        if self.n_bars < 11:
            l3 = 20
            koef = 0.5 * ymax
            l2 = 0.235 * ymax

        if self.n_bars < 31 and self.n_bars > 10:
            l3 = 25 - (0.5) * self.n_bars
            koef = 0.635 * ymax - 0.0135 * self.n_bars * ymax
            l2 = 0.2975 * ymax - 0.00625 * self.n_bars * ymax

        if self.n_bars < 41 and self.n_bars > 30:
            l3 = 16 - (0.2) * self.n_bars
            koef = 0.32 * ymax - 0.003 * self.n_bars * ymax
            l2 = 0.17 * ymax - 0.002 * self.n_bars * ymax

        if self.n_bars < 51 and self.n_bars > 40:
            l3 = 8
            koef = 0.32 * ymax - 0.003 * self.n_bars * ymax
            l2 = 0.17 * ymax - 0.002 * self.n_bars * ymax

        if self.n_bars > 51:
            l3 = 8
            koef = 0.17 * ymax
            l2 = 0.07 * ymax

        N = numpy.linspace(self.mse_min, self.mse_max, num=self.n_bars,
                           endpoint=True)
        ax.bar(N, self.val_mse, color='#ffa0ef', width=width,
               edgecolor='lavender')
        # , edgecolor='red')
        # D889B8
        # B96A9A
        ax.set_xlabel('Errors', fontsize=20)
        ax.set_ylabel('Input Data', fontsize=20)
        ax.set_title(self.name.replace("Histogram ", ""))
        ax.axis([xmin, xmax + ((xmax - xmin) / self.n_bars), ymin, ymax])
        ax.grid(True)
        leg = ax.legend(self.name.replace("Histogram ", ""))  # 'upper center')
        frame = leg.get_frame()
        frame.set_facecolor('#E8D6BB')
        for t in leg.get_texts():
            t.set_fontsize(18)
        for l in leg.get_lines():
            l.set_linewidth(1.5)

        for x, y in zip(N, self.val_mse):
            if y > koef - l2 * 0.75:
                self.pp.text(x + l1, y - l2 * 0.75, '%.0f' % y, ha='center',
                             va='bottom', fontsize=l3, rotation=90)
            else:
                self.pp.text(x + l1, t0, '%.0f' % y, ha='center', va='bottom',
                             fontsize=l3, rotation=90)

        self.show_figure(fig)
        fig.canvas.draw()
        return fig

    def run(self):
        mx = self.mse.mem.max()
        mi = self.mse.mem.min()
        self.mse_max = mx
        self.mse_min = mi
        d = mx - mi
        if not d:
            return
        d = (self.n_bars - 1) / d
        self.val_mse[:] = 0
        for mse in self.mse.mem:
            i_bar = int(numpy.floor((mse - mi) * d))
            self.val_mse[i_bar] += 1

        self.val_max = self.val_mse.max()
        self.val_min = self.val_mse.min()

        super(MSEHistogram, self).run()


@implementer(plotter.IPlotter)
class KohonenHits(plotter.Plotter):
    """Draws the Kohonen classification win numbers.

    Must be assigned before initialize():
        input
        shape
    """

    SIZE_TEXT_THRESHOLD = 0.33

    def __init__(self, workflow, **kwargs):
        name = kwargs.get("name", "Kohonen Hits")
        kwargs["name"] = name
        super(KohonenHits, self).__init__(workflow, **kwargs)
        self._color_bins = kwargs.get("color_bins", "#666699")
        self._color_text = kwargs.get("color_text", "white")
        self.demand("input", "shape")

    @property
    def width(self):
        return self.shape[0]

    @property
    def height(self):
        return self.shape[1]

    @property
    def color_bins(self):
        return self._color_bins

    @color_bins.setter
    def color_bins(self, value):
        self._color_bins = value

    @property
    def color_text(self):
        return self._color_text

    @color_text.setter
    def color_text(self, value):
        self._color_text = value

    def redraw(self):
        fast_redraw = self.name in self.pp.get_figlabels()
        fig = self.pp.figure(self.name)
        axes = fig.add_subplot(111)

        if not fast_redraw:
            # Draw the hexbin grid
            diag = 1.0 / numpy.sqrt(3)
            vlength = 2 * self.height + 2
            # Cloned primitive
            subline = numpy.empty((4, 2))
            subline[0, 0] = 0.0
            subline[0, 1] = -diag
            subline[1, 0] = -0.5
            subline[1, 1] = -diag / 2
            subline[2, 0] = -0.5
            subline[2, 1] = diag / 2
            subline[3, 0] = 0.0
            subline[3, 1] = diag
            # Tile sublines into line
            line = numpy.empty((vlength, 2))
            for rep in range(vlength // 4):
                line[rep * 4:rep * 4 + 4, :] = subline
                subline[:, 1] += diag * 3
            if not self.height & 1:
                line[-2:, :] = subline[:2]
            # Fill the grid vertices
            hlength = self.width * 2 + 1
            vertices = numpy.empty((hlength, vlength, 2))
            for rep in range(self.width):
                vertices[rep, :, :] = line
                # Right side
                line[1:vlength:4, 0] += 1.0
                line[2:vlength:4, 0] += 1.0
                vertices[self.width + 1 + rep, :, :] = line
                line[0:vlength:4, 0] += 1.0
                line[3:vlength:4, 0] += 1.0
            # The last right side
            vertices[self.width, :vlength - 1, :] = line[1:, :]
            # Line ending fixes
            if self.height & 1:
                vertices[self.width, -2, :] = vertices[self.width, -3, :]
            else:
                vertices[0, -1, :] = vertices[0, -2, :]
            vertices[self.width, -1, :] = vertices[self.width, -2, :]
            # Add the constructed vertices as PolyCollection
            col = self.matplotlib.collections.PolyCollection(
                vertices, closed=False, edgecolors='black', facecolors='none')
            # Resize together with the axes
            col.set_transform(axes.transData)
            axes.add_collection(col)
            axes.set_xlim(-1.0, self.width + 0.5)
            axes.set_ylim(-1.0, numpy.round(self.height * numpy.sqrt(3.) / 2.))
            axes.set_xticks([])
            axes.set_yticks([])

        if fast_redraw:
            while len(axes.texts):
                axes.texts[0].remove()

        # Draw the inner hexagons with text
        # Initialize sizes
        hits_max = numpy.max(self.input)
        if hits_max == 0:
            hits_max = 1
        patches = []
        # Add hexagons one by one
        for y in range(self.height):
            for x in range(self.width):
                number = self.input[y * self.width + x]
                # square is proportional to the square root of the linear
                # size / the hits number
                self._add_hexagon(axes, patches, x, y,
                                  numpy.sqrt(number / hits_max),
                                  number)
        col = self.matplotlib.collections.PatchCollection(
            patches, edgecolors='none', facecolors=self.color_bins)
        if fast_redraw:
            axes.collections[-1].remove()
        axes.add_collection(col)

        self.show_figure(fig)
        fig.canvas.draw()
        return fig

    def _add_hexagon(self, axes, patches, x, y, size, number):
        r = size / numpy.sqrt(3)
        cx = x if not (y & 1) else x + 0.5
        cy = y * (1.5 / numpy.sqrt(3))
        patches.append(self.patches.RegularPolygon((cx, cy), 6, radius=r))
        if size > KohonenHits.SIZE_TEXT_THRESHOLD:
            axes.annotate(number, xy=(cx, cy),
                          verticalalignment="center",
                          horizontalalignment="center",
                          color=self.color_text, size=12)


@implementer(plotter.IPlotter)
class KohonenInputMaps(plotter.Plotter):
    """Draws the Kohonen input weight maps.

    Must be assigned before initialize():
        input
        shape
    """

    def __init__(self, workflow, **kwargs):
        name = kwargs.get("name", "Kohonen Maps")
        kwargs["name"] = name
        super(KohonenInputMaps, self).__init__(workflow, **kwargs)
        self._color_scheme = kwargs.get("color_scheme", "YlOrRd")
        self._color_grid = kwargs.get("color_grid", "none")
        self.demand("input", "shape")

    @property
    def width(self):
        return self.shape[0]

    @property
    def height(self):
        return self.shape[1]

    @property
    def color_scheme(self):
        return self._color_scheme

    @color_scheme.setter
    def color_scheme(self, value):
        self._color_scheme = value

    @property
    def color_grid(self):
        return self._color_grid

    @color_grid.setter
    def color_grid(self, value):
        self._color_grid = value

    def redraw(self):
        fast_redraw = self.name in self.pp.get_figlabels()
        fig = self.pp.figure(self.name)
        if not fast_redraw:
            fig.clf()
        length = self.input.shape[1]
        if length < 3:
            grid_shape = (length, 1)
        elif length < 5:
            grid_shape = (2, length - 2)
        elif length < 7:
            grid_shape = (3, length - 3)
        else:
            grid_shape = (4, int(numpy.ceil(length / 4)))
        for index in range(length):
            axes = fig.add_subplot(grid_shape[1], grid_shape[0], index)
            if not fast_redraw:
                patches = []
                # Add hexagons to patches one by one
                for y in range(self.height):
                    for x in range(self.width):
                        self._add_hexagon(axes, patches, x, y)
                # Add the collection
                col = self.matplotlib.collections.PatchCollection(
                    patches, cmap=getattr(self.cm, self.color_scheme),
                    edgecolor=self.color_grid)
                axes.add_collection(col)
            else:
                col = axes.collections[0]
            arr = self.input[:, index]
            amax = numpy.max(arr)
            amin = numpy.min(arr)
            col.set_array((arr - amin) / (amax - amin))
            if not fast_redraw:
                axes.set_xlim(-1.0, self.width + 0.5)
                axes.set_ylim(-1.0,
                              numpy.round(self.height * numpy.sqrt(3.0) / 2))
                axes.set_xticks([])
                axes.set_yticks([])
        if not fast_redraw:
            fig.colorbar(col)
        self.show_figure(fig)
        fig.canvas.draw()
        return fig

    def _add_hexagon(self, axes, patches, x, y):
        r = 1.0 / numpy.sqrt(3)
        cx = x if not (y & 1) else x + 0.5
        cy = y * (1.5 / numpy.sqrt(3))
        patches.append(self.patches.RegularPolygon((cx, cy), 6, radius=r))


@implementer(plotter.IPlotter)
class KohonenNeighborMap(plotter.Plotter):
    """Draws the Kohonen neighbor weight distances.

    Must be assigned before initialize():
        input
        shape
    """

    NEURON_SIZE = 0.4

    def __init__(self, workflow, **kwargs):
        name = kwargs.get("name", "Kohonen Neighbor Weight Distances")
        kwargs["name"] = name
        super(KohonenNeighborMap, self).__init__(workflow, **kwargs)
        self._color_neurons = kwargs.get("color_neurons", "#666699")
        self._color_scheme = kwargs.get("color_scheme", "YlOrRd")
        self.demand("input", "shape")

    @property
    def width(self):
        return self.shape[0]

    @property
    def height(self):
        return self.shape[1]

    @property
    def color_neurons(self):
        return self._color_neurons

    @color_neurons.setter
    def color_neurons(self, value):
        self._color_neurons = value

    @property
    def color_scheme(self):
        return self._color_scheme

    @color_scheme.setter
    def color_scheme(self, value):
        self._color_scheme = value

    def redraw(self):
        self._scheme = getattr(self.cm, self.color_scheme)

        fast_redraw = self.name in self.pp.get_figlabels()
        fig = self.pp.figure(self.name)
        axes = fig.add_subplot(111)

        # Calculate the links patches
        link_values = numpy.empty((self.width - 1) * self.height +
                                  (self.width * 2 - 1) * (self.height - 1))

        links = []
        lvi = 0
        # Add horizontal links
        for y in range(self.height):
            for x in range(self.width - 1):
                n1 = (x, y)
                n2 = (x + 1, y)
                if not fast_redraw:
                    self._add_link(axes, links, n1, n2)
                link_values[lvi] = self._calc_link_value(n1, n2)
                lvi += 1
        # Add vertical links
        for y in range(self.height - 1):
            for x in range(self.width):
                n1 = (x, y)
                n2 = (x, y + 1)
                if not fast_redraw:
                    self._add_link(axes, links, n1, n2)
                link_values[lvi] = self._calc_link_value(n1, n2)
                lvi += 1
                n1 = (x, y)
                if y & 1:
                    if x == self.width - 1:
                        continue
                    n2 = (x + 1, y + 1)
                else:
                    if x == 0:
                        continue
                    n2 = (x - 1, y + 1)
                if not fast_redraw:
                    self._add_link(axes, links, n1, n2)
                link_values[lvi] = self._calc_link_value(n1, n2)
                lvi += 1

        if not fast_redraw:
            # Draw the neurons
            patches = []
            for y in range(self.height):
                for x in range(self.width):
                    self._add_hexagon(axes, patches, x, y)
            col = self.matplotlib.collections.PatchCollection(
                patches, edgecolors='black', facecolors=self.color_neurons)
            axes.add_collection(col)

            # Draw the links
            col = self.matplotlib.collections.PatchCollection(
                links, cmap=getattr(self.cm, self.color_scheme),
                edgecolor='none')
            axes.add_collection(col)
            axes.set_xlim(-1.0, self.width + 0.5)
            axes.set_ylim(-1.0, numpy.round(self.height * numpy.sqrt(3.) / 2.))
            axes.set_xticks([])
            axes.set_yticks([])
        else:
            col = axes.collections[-1]
        amax = numpy.max(link_values)
        amin = numpy.min(link_values)
        col.set_array((link_values - amin) / (amax - amin))
        if not fast_redraw:
            fig.colorbar(col)

        self.show_figure(fig)
        fig.canvas.draw()
        return fig

    def _add_hexagon(self, axes, patches, x, y):
        r = KohonenNeighborMap.NEURON_SIZE / numpy.sqrt(3)
        cx = x if not (y & 1) else x + 0.5
        cy = y * (1.5 / numpy.sqrt(3))
        patches.append(self.patches.RegularPolygon((cx, cy), 6, radius=r))

    def _calc_link_value(self, n1, n2):
        n1x, n1y = n1
        n2x, n2y = n2
        weights1 = self.input[n1y * self.width + n1x, :]
        weights2 = self.input[n2y * self.width + n2x, :]
        return linalg.norm(weights1 - weights2)

    def _add_link(self, axes, links, n1, n2):
        n1x, n1y = n1
        n2x, n2y = n2
        vertices = numpy.empty((6, 2))
        diag = 1.0 / numpy.sqrt(3)
        ratio = 1.0 - KohonenNeighborMap.NEURON_SIZE
        # LET THE GEOMETRIC PORN BEGIN!!!
        if n1y == n2y:
            # Horizontal hexagon
            cx = (n1x + n2x) / 2 + (0.5 if n1y & 1 else 0)
            cy = n1y * (1.5 / numpy.sqrt(3))
            vertices[0, :] = (cx, cy - diag / 2)
            vertices[1, :] = (cx - 0.5 * ratio, cy - (diag / 2) * (1 - ratio))
            vertices[2, :] = (cx - 0.5 * ratio, cy + (diag / 2) * (1 - ratio))
            vertices[3, :] = (cx, cy + diag / 2)
            vertices[4, :] = (cx + 0.5 * ratio, cy + (diag / 2) * (1 - ratio))
            vertices[5, :] = (cx + 0.5 * ratio, cy - (diag / 2) * (1 - ratio))
        elif (n1x == n2x and n2y & 1) or (n2x > n1x and n1y & 1):
            # Right hexagon
            sx = n1x + (0.5 if n1y & 1 else 0)
            sy = n1y * (1.5 / numpy.sqrt(3)) + diag
            vertices[0, :] = (sx, sy)
            vertices[1, :] = (sx + 0.5 * ratio, sy + (diag / 2) * ratio)
            vertices[2, :] = (sx + 0.5, sy + diag * (ratio - 0.5))
            vertices[3, :] = (sx + 0.5, sy - diag / 2)
            vertices[4, :] = (sx + 0.5 * (1 - ratio),
                              sy - (diag / 2) * (1 + ratio))
            vertices[5, :] = (sx, sy - diag * ratio)
        else:
            # Left hexagon
            sx = n2x + (0 if n1x == n2x else 0.5)
            sy = n2y * (1.5 / numpy.sqrt(3)) - diag
            vertices[0, :] = (sx, sy)
            vertices[1, :] = (sx, sy + diag * ratio)
            vertices[2, :] = (sx + 0.5 * (1 - ratio),
                              sy + (diag / 2) * (1 + ratio))
            vertices[3, :] = (sx + 0.5, sy + diag / 2)
            vertices[4, :] = (sx + 0.5, sy - diag * (ratio - 0.5))
            vertices[5, :] = (sx + 0.5 * ratio, sy - (diag / 2) * ratio)
        links.append(self.patches.Polygon(vertices))
