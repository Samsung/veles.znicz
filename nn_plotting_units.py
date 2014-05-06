"""
Created on May 5, 2014

Copyright (c) 2014, Samsung Electronics, Co., Ltd.
"""


import numpy

import veles.config as config
import veles.formats as formats
import veles.plotter as plotter
import veles.opencl_types as opencl_types


class Weights2D(plotter.Plotter):
    """Plotter for drawing weights as 2D.

    Should be assigned before initialize():
        input
        input_field

    Updates after run():

    Creates within initialize():

    """
    def __init__(self, workflow, **kwargs):
        name = kwargs.get("name", "Weights")
        limit = kwargs.get("limit", 64)
        yuv = kwargs.get("yuv", False)
        kwargs["name"] = name
        kwargs["limit"] = limit
        kwargs["yuv"] = yuv
        super(Weights2D, self).__init__(workflow, **kwargs)
        self.input = None
        self.input_field = None
        self.get_shape_from = None
        self.limit = limit
        self.transposed = False
        self.yuv = [1 if yuv else 0]
        self.cm = None
        self.pp = None
        self.show_figure = self.nothing

    def redraw(self):
        if type(self.input_field) == int:
            if self.input_field < 0 or self.input_field >= len(self.input):
                return
            value = self.input[self.input_field]
        else:
            value = self.input.__dict__[self.input_field]

        if type(value) != numpy.ndarray or len(value.shape) != 2:
            return

        if self.transposed:
            value = value.transpose()

        if value.shape[0] > self.limit:
            value = value[:self.limit]

        figure = self.pp.figure(self.name)
        figure.clf()

        color = False
        if self.get_shape_from is None:
            sx = int(numpy.round(numpy.sqrt(value.shape[1])))
            sy = int(value.shape[1]) // sx
        elif type(self.get_shape_from) == list:
            sx = self.get_shape_from[0]
            sy = self.get_shape_from[1]
        elif "v" in self.get_shape_from.__dict__:
            if len(self.get_shape_from.v.shape) == 3:
                sx = self.get_shape_from.v.shape[2]
                sy = self.get_shape_from.v.shape[1]
            else:
                if self.get_shape_from.v.shape[-1] == 3:
                    sx = self.get_shape_from.v.shape[-2]
                    sy = self.get_shape_from.v.shape[-3]
                    interleave = False
                else:
                    sx = self.get_shape_from.v.shape[-3]
                    sy = self.get_shape_from.v.shape[-2]
                    interleave = True
                color = True
        else:
            sx = self.get_shape_from.shape[1]
            sy = self.get_shape_from.shape[0]

        if color:
            sz = sx * sy * 3
        else:
            sz = sx * sy

        n_cols = int(numpy.round(numpy.sqrt(value.shape[0])))
        n_rows = int(numpy.ceil(value.shape[0] / n_cols))

        i = 0
        for _ in range(0, n_rows):
            for _ in range(0, n_cols):
                ax = figure.add_subplot(n_rows, n_cols, i + 1)
                ax.cla()
                ax.axis('off')
                v = value[i].ravel()[:sz]
                if color:
                    if interleave:
                        w = formats.interleave(v.reshape(3, sy, sx))
                    else:
                        w = v.reshape(sy, sx, 3)
                    ax.imshow(formats.norm_image(w, self.yuv[0]),
                              interpolation="nearest")
                else:
                    ax.imshow(formats.norm_image(v.reshape(sy, sx),
                                                 self.yuv[0]),
                              interpolation="nearest", cmap=self.cm.gray)
                i += 1
                if i >= value.shape[0]:
                    break
            if i >= value.shape[0]:
                break

        self.show_figure(figure)
        figure.canvas.draw()

        super(Weights2D, self).redraw()


class MSEHistogram(plotter.Plotter):
    """Plotter for drawing histogram.

    Should be assigned before initialize():
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
        self.mse = None  # formats.Vector()
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
        ax.set_title('Histogram', fontsize=25)
        ax.axis([xmin, xmax + ((xmax - xmin) / self.n_bars), ymin, ymax])
        ax.grid(True)
        leg = ax.legend((self.name.replace("Histogram ", "")))
                        # 'upper center')
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
        super(MSEHistogram, self).redraw()

    def run(self):
        mx = self.mse.v.max()
        mi = self.mse.v.min()
        self.mse_max = mx
        self.mse_min = mi
        d = mx - mi
        if not d:
            return
        d = (self.n_bars - 1) / d
        self.val_mse[:] = 0
        for mse in self.mse.v:
            i_bar = int(numpy.floor((mse - mi) * d))
            self.val_mse[i_bar] += 1

        self.val_max = self.val_mse.max()
        self.val_min = self.val_mse.min()

        super(MSEHistogram, self).run()


class KohonenHits(plotter.Plotter):
    """Draws the Kohonen classification win numbers.

    Should be assigned before initialize():
        input
    """

    SIZE_TEXT_THRESHOLD = 0.33

    def __init__(self, workflow, **kwargs):
        name = kwargs.get("name", "Kohonen Hits")
        kwargs["name"] = name
        super(KohonenHits, self).__init__(workflow, **kwargs)
        self._color_bins = kwargs.get("color_bins", "#666699")
        self._color_text = kwargs.get("color_text", "white")
        self._input = None
        self._width = 0
        self._height = 0

    @property
    def input(self):
        return self._input

    @input.setter
    def input(self, value):
        self._input = value
        self._width = self.input.shape[0]
        self._height = self.input.shape[1]

    @property
    def width(self):
        return self._width

    @property
    def height(self):
        return self._height

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
        fig = self.pp.figure(self.name)
        fig.clf()
        axes = fig.add_subplot(111)

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

        # Draw the inner hexagons with text
        # Initialize sizes
        hits_max = numpy.max(self.input)
        sizes = self.input / hits_max
        self._fill_sizes(sizes)
        vertices = numpy.empty((self.width * self.height, 6, 2))
        # Add hexagons one by one
        for y in range(self.height):
            for x in range(self.width):
                self._add_hexagon(axes, vertices, x, y, sizes[x, y],
                                  self.input[x, y])
        col = self.matplotlib.collections.PolyCollection(
            vertices, closed=True, edgecolors='none',
            facecolors=self.color_bins)
        col.set_transform(axes.transData)
        axes.add_collection(col)

        axes.set_xlim(-1.0, self.width + 0.5)
        axes.set_ylim(-1.0, numpy.round(self.height * numpy.sqrt(3.0) / 2.0))
        axes.set_xticks([])
        axes.set_yticks([])

    def _add_hexagon(self, axes, vertices, x, y, size, number):
        index = y * self.width + x
        hsz = size / 2
        diag = size / numpy.sqrt(3)
        cx = x if not (y & 1) else x + 0.5
        cy = y * (1.5 / numpy.sqrt(3))
        vertices[index, 0, :] = (cx, cy - diag)
        vertices[index, 1, :] = (cx - hsz, cy - diag / 2)
        vertices[index, 2, :] = (cx - hsz, cy + diag / 2)
        vertices[index, 3, :] = (cx, cy + diag)
        vertices[index, 4, :] = (cx + hsz, cy + diag / 2)
        vertices[index, 5, :] = (cx + hsz, cy - diag / 2)
        if size > KohonenHits.SIZE_TEXT_THRESHOLD:
            axes.annotate(number, xy=(cx, cy),
                          verticalalignment="center",
                          horizontalalignment="center",
                          color=self.color_text, size=12)


class KohonenInputMaps(plotter.Plotter):
    """Draws the Kohonen input weight maps.

    Should be assigned before initialize():
        input
        width
        height
    """

    def __init__(self, workflow, **kwargs):
        name = kwargs.get("name", "Kohonen Maps")
        kwargs["name"] = name
        super(KohonenInputMaps, self).__init__(workflow, **kwargs)
        self._color_scheme = kwargs.get("color_scheme", "YlOrRd")
        self._color_grid = kwargs.get("color_grid", "none")
        self._input = None
        self.width = 0
        self.height = 0

    @property
    def input(self):
        return self._input

    @input.setter
    def input(self, value):
        self._input = value

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
        fig = self.pp.figure(self.name)
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
            axes = fig.add_subplot(grid_shape[0], grid_shape[1], index)

            weights = self.input[:, index]
            min_weight = numpy.min(weights)
            max_weight = numpy.max(weights)
            delta = max_weight - min_weight
            weights /= max_weight

            patches = []
            # Add hexagons to patches one by one
            for y in range(self.height):
                for x in range(self.width):
                    value = self.input[y * self.width + x, index]
                    value = (value - min_weight) / delta
                    self._add_hexagon(axes, patches, x, y, value)
            # Add the collection
            col = self.matplotlib.collections.PatchCollection(
                patches, cmap=getattr(self.cm, self.color_scheme),
                edgecolor=self.color_grid)
            axes.add_collection(col)

            axes.set_xlim(-1.0, self.width + 0.5)
            axes.set_ylim(-1.0, numpy.round(self.height * numpy.sqrt(3.0) / 2))
            axes.set_xticks([])
            axes.set_yticks([])

    def _add_hexagon(self, axes, patches, x, y, value):
        r = 1.0 / numpy.sqrt(3)
        cx = x if not (y & 1) else x + 0.5
        cy = y * (1.5 / numpy.sqrt(3))
        patches.append(self.patches.RegularPolygon((cx, cy), 6, radius=r))


class KohonenNeighborMap(plotter.Plotter):
    """Draws the Kohonen neighbor weight distances.

    Should be assigned before initialize():
        input
        width
        height
    """

    NEURON_SIZE = 0.2

    def __init__(self, workflow, **kwargs):
        name = kwargs.get("name", "Kohonen Neighbor Weight Distances")
        kwargs["name"] = name
        super(KohonenHits, self).__init__(workflow, **kwargs)
        self._color_neurons = kwargs.get("color_neurons", "#666699")
        self._color_scheme = kwargs.get("color_scheme", "YlOrRd")
        self._input = None
        self._width = 0
        self._height = 0

    @property
    def input(self):
        return self._input

    @input.setter
    def input(self, value):
        self._input = value

    @property
    def width(self):
        return self._width

    @property
    def height(self):
        return self._height

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
        fig = self.pp.figure(self.name)
        fig.clf()
        axes = fig.add_subplot(111)

        # Draw the neurons
        vertices = numpy.empty((self.width * self.height, 6, 2))
        for y in range(self.height):
            for x in range(self.width):
                self._add_hexagon(axes, vertices, x, y)
        col = self.matplotlib.collections.PolyCollection(
            vertices, closed=True, edgecolors='none',
            facecolors=self.color_neurons)
        col.set_transform(axes.transData)
        axes.add_collection(col)

        # Draw the links
        links = []

        # Add the collection
        col = self.matplotlib.collections.PatchCollection(
            links, cmap=getattr(self.cm, self.color_scheme),
            edgecolor='none')
        axes.add_collection(col)

        axes.set_xlim(-1.0, self.width + 0.5)
        axes.set_ylim(-1.0, numpy.round(self.height * numpy.sqrt(3.0) / 2.0))
        axes.set_xticks([])
        axes.set_yticks([])

    def _add_hexagon(self, axes, vertices, x, y):
        size = KohonenNeighborMap.NEURON_SIZE
        index = y * self.width + x
        hsz = size / 2
        diag = size / numpy.sqrt(3)
        cx = x if not (y & 1) else x + 0.5
        cy = y * (1.5 / numpy.sqrt(3))
        vertices[index, 0, :] = (cx, cy - diag)
        vertices[index, 1, :] = (cx - hsz, cy - diag / 2)
        vertices[index, 2, :] = (cx - hsz, cy + diag / 2)
        vertices[index, 3, :] = (cx, cy + diag)
        vertices[index, 4, :] = (cx + hsz, cy + diag / 2)
        vertices[index, 5, :] = (cx + hsz, cy - diag / 2)

    def _add_link(self, axes, links, x, y, value):
        # TODO(v.markovtsev): implement this
        pass
