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
