"""
Created on May 17, 2013

@author: Kumok Akim <a.kumok@samsung.com>
"""
import matplotlib.pyplot as pp
import matplotlib.patches as patches
import matplotlib.lines as lines
import matplotlib.cm as cm
import pylab
import units
import tkinter
import queue
import threading
import numpy
import formats
import logging
import config


pp.ion()


class Graphics:
    """ Class handling all interaction with main graphics window
        NOTE: This class should be created ONLY within one thread
        (preferably main)

    Attributes:
        _instance: instance of MainGraphicsData class. Used for implementing
            Singleton pattern for this class.
        root: TKinter graphics root.
        event_queue: Queue of all pending changes created by other threads.
        run_lock: Lock to determine whether graphics window is running
        registered_plotters: List of registered plotters
        is_initialized: whether this class was already initialized.
    """

    _instance = None
    root = None
    event_queue = None
    run_lock = None
    registered_plotters = None
    is_initialized = False

    def __new__(cls):
        if not cls._instance:
            cls._instance = super(Graphics, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if not self.is_initialized:
            self.is_initialized = True
            self.event_queue = queue.Queue()
            self.initialize_lock = threading.Lock()
            self.registered_plotters = {}
            threading.Thread(target=self.run).start()

    def run(self):
        """Creates and runs main graphics window.
        Note that this function should be called only by __init__()
        """
        self.run_lock = threading.Lock()
        self.run_lock.acquire()
        self.root = tkinter.Tk()
        self.root.withdraw()
        self.root.after(100, self.update)
        tkinter.mainloop()  # Wait for user to close the window
        self.run_lock.release()

    def process_event(self, plotter):
        """Processes scheduled redraw event
        """
        plotter.redraw()

    def update(self):
        """Processes all events scheduled for plotting
        """
        try:
            while True:
                plotter = self.event_queue.get_nowait()
                self.process_event(plotter)
        except queue.Empty:
            pass
        self.root.after(100, self.update)

    def wait_finish(self):
        """Waits for user to close the window.
        """
        logging.info("Waiting for user to close the window...")
        self.root.destroy()
        self.run_lock.acquire()
        self.run_lock.release()
        logging.info("Done")


class Plotter(units.Unit):
    """Base class for all plotters

    Attributes:
        lock_: lock.
    """
    def __init__(self, device=None, unpickling=0):
        super(Plotter, self).__init__(unpickling=unpickling)
        self.lock_ = threading.Lock()
        self.lock_.acquire()
        if unpickling:
            return

    def redraw(self):
        """ Do the actual drawing here
        """
        self.lock_.release()

    def run(self):
        self.lock_.acquire()


class SimplePlotter(Plotter):
    """ Plotter for given values

    Should be assigned before initialize():
        input
        input_field

    Updates after run():

    Creates within initialize():

    Attributes:
        values: history of all parameter values given to plotter.
        input: connector to take values from.
        input_field: name of field in input we want to plot.
        figure_label: label of figure used for drawing. If two ploters share
            the same figure_label, their plots will appear together.
        plot_style: Style of lines used for plotting. See
            http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.plot
            for reference.
    """
    def __init__(self, figure_label="num errorrs",
                 plot_style="k-",
                 clear_plot=False,
                 unpickling=0):
        super(SimplePlotter, self).__init__(unpickling=unpickling)
        if unpickling:
            if "clear_plot" not in self.__dict__:
                self.clear_plot = False
            return
        self.values = list()
        self.input = None  # Connector
        self.input_field = None
        self.figure_label = figure_label
        self.plot_style = plot_style
        self.input_offs = 0
        self.clear_plot = clear_plot

    def redraw(self):
        figure_label = self.figure_label
        figure = pp.figure(figure_label)
        if self.clear_plot:
            figure.clf()
        axes = figure.add_subplot(111)  # Main axes
        if self.clear_plot:
            axes.cla()
        axes.plot(self.values, self.plot_style)
        figure.show()
        super(SimplePlotter, self).redraw()

    def run(self):
        if type(self.input_field) == int:
            if self.input_field < 0 or self.input_field >= len(self.input):
                return
            value = self.input[self.input_field]
        else:
            value = self.input.__dict__[self.input_field]
        if type(value) == numpy.ndarray:
            value = value[self.input_offs]
        self.values.append(value)
        Graphics().event_queue.put(self, block=True)
        super(SimplePlotter, self).run()


class MatrixPlotter(Plotter):
    """Plotter for drawing matrixes

    Should be assigned before initialize():
        input
        input_field

    Updates after run():

    Creates within initialize():

    """
    def __init__(self, figure_label="Matrix", unpickling=0):
        super(MatrixPlotter, self).__init__(unpickling=unpickling)
        if unpickling:
            return
        self.value = None
        self.input = None  # Connector
        self.input_field = None
        self.figure_label = figure_label

    def redraw(self):
        if type(self.input_field) == int:
            if self.input_field < 0 or self.input_field >= len(self.input):
                return
            value = self.input[self.input_field]
        else:
            value = self.input.__dict__[self.input_field]

        figure_label = self.figure_label
        figure = pp.figure(figure_label)
        figure.clf()
        num_rows = len(value) + 2
        num_columns = len(value[0]) + 2

        main_axes = figure.add_axes([0, 0, 1, 1])
        main_axes.cla()

        # First cell color
        rc = patches.Rectangle(
            (0, (num_rows - 1) / num_rows),
            1.0 / num_rows, 1.0 / num_columns, color='gray')
        main_axes.add_patch(rc)
        # First row last cell color
        rc = patches.Rectangle(
            ((num_columns - 1) / num_columns, (num_rows - 1) / num_rows),
            1.0 / num_rows, 1.0 / num_columns, color='gray')
        main_axes.add_patch(rc)
        # First column last cell color
        rc = patches.Rectangle(
            (0, 0),
            1.0 / num_rows, 1.0 / num_columns, color='gray')
        main_axes.add_patch(rc)
        # Last cell color
        rc = patches.Rectangle(
            ((num_columns - 1) / num_columns, 0),
            1.0 / num_rows, 1.0 / num_columns, color='silver')
        main_axes.add_patch(rc)
        # Data cells colors
        sum_total = value.sum()
        sum_ok = 0
        max_vle = 0
        for row in range(0, num_rows - 2):
            for column in range(0, num_columns - 2):
                if row != column:
                    max_vle = max(max_vle, value[row, column])
                else:
                    sum_ok += value[row, column]
        #sum_by_row = value.sum(axis=0)
        for row in range(1, num_rows - 1):
            for column in range(1, num_columns - 1):
                n_elem = value[row - 1, column - 1]
                color = 'white'
                if row == column:
                    if n_elem > 0:
                        color = 'cyan'
                else:
                    if n_elem > 0:
                        v = int(numpy.round((1.0 - n_elem / max_vle) * 255.0))
                        color = "#FF%02X%02X" % (v, v)
                    else:
                        color = 'green'
                rc = patches.Rectangle(
                    (column / num_columns, (num_rows - row - 1) / num_rows),
                    1.0 / num_rows, 1.0 / num_columns, color=color)
                main_axes.add_patch(rc)

        for row in range(num_rows):
            y = row / num_rows
            main_axes.add_line(lines.Line2D([0, 1], [y, y]))
        for column in range(num_columns):
            x = column / num_columns
            main_axes.add_line(lines.Line2D([x, x], [0, 1]))

        # First cell
        column = 0
        row = 0
        pp.figtext(label="0",
            s="target",
            x=(column + 0.9) / num_columns,
            y=(num_rows - row - 0.33) / num_rows,
            verticalalignment="center",
            horizontalalignment="right")
        pp.figtext(label="0",
            s="value",
            x=(column + 0.1) / num_columns,
            y=(num_rows - row - 0.66) / num_rows,
            verticalalignment="center",
            horizontalalignment="left")

        # Headers in first row
        row = 0
        for column in range(1, num_columns - 1):
            pp.figtext(label=("C%d" % (column - 1,)),
                            s=(column - 1),
                            x=(column + 0.5) / num_columns,
                            y=(num_rows - row - 0.5) / num_rows,
                            verticalalignment="center",
                            horizontalalignment="center")
        # Headers in first column
        column = 0
        for row in range(1, num_rows - 1):
            pp.figtext(label=("R%d" % (row - 1,)),
                            s=(row - 1),
                            x=(column + 0.5) / num_columns,
                            y=(num_rows - row - 0.5) / num_rows,
                            verticalalignment="center",
                            horizontalalignment="center")
        # Data
        for row in range(1, num_rows - 1):
            for column in range(1, num_columns - 1):
                n_elem = value[row - 1, column - 1]
                #n = sum_by_row[row - 1]
                #pt_elem = 100.0 * n_elem / n if n else 0
                n = sum_total
                pt_total = 100.0 * n_elem / n if n else 0
                label = "%d as %d" % (column - 1, row - 1)
                pp.figtext(
                    label=label,
                    s=n_elem,
                    x=(column + 0.5) / num_columns,
                    y=(num_rows - row - 0.33) / num_rows,
                    verticalalignment="center",
                    horizontalalignment="center")
                #pp.figtext(
                #    label=label,
                #    s=("%.2f%%" % (pt_elem, )),
                #    x=(column + 0.1) / num_columns,
                #    y=(num_rows - row - 0.75) / num_rows,
                #    verticalalignment="center",
                #    horizontalalignment="left")
                pp.figtext(
                    label=label,
                    s=("%.2f%%" % (pt_total,)),
                    x=(column + 0.5) / num_columns,
                    y=(num_rows - row - 0.66) / num_rows,
                    verticalalignment="center",
                    horizontalalignment="center")
        # Last cell
        n = sum_total
        pt_total = 100.0 * sum_ok / n if n else 0
        label = "Totals"
        row = num_rows - 1
        column = num_columns - 1
        pp.figtext(
            label=label,
            s=sum_ok,
            x=(column + 0.5) / num_columns,
            y=(num_rows - row - 0.33) / num_rows,
            verticalalignment="center",
            horizontalalignment="center")
        pp.figtext(
            label=label,
            s=("%.2f%%" % (pt_total,)),
            x=(column + 0.5) / num_columns,
            y=(num_rows - row - 0.66) / num_rows,
            verticalalignment="center",
            horizontalalignment="center")
        figure.show()
        super(MatrixPlotter, self).redraw()

    def run(self):
        if type(self.input_field) == int:
            if self.input_field < 0 or self.input_field >= len(self.input):
                return
            self.value = self.input[self.input_field]
        else:
            self.value = self.input.__dict__[self.input_field]
        Graphics().event_queue.put(self, block=True)
        super(MatrixPlotter, self).run()


class Weights2D(Plotter):
    """Plotter for drawing weights as 2D.

    Should be assigned before initialize():
        input
        input_field

    Updates after run():

    Creates within initialize():

    """
    def __init__(self, figure_label="Weights", limit=256, unpickling=0):
        super(Weights2D, self).__init__(unpickling=unpickling)
        if unpickling:
            return
        self.value = None
        self.input = None  # Connector
        self.input_field = None
        self.figure_label = figure_label
        self.get_shape_from = None
        self.limit = limit
        self.transposed = False

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

        figure_label = self.figure_label
        figure = pp.figure(figure_label)
        figure.clf()

        if self.get_shape_from == None:
            sx = int(numpy.round(numpy.sqrt(value.shape[1])))
            sy = int(value.shape[1]) // sx
        elif type(self.get_shape_from) == list:
            sx = self.get_shape_from[0][1]
            sy = self.get_shape_from[0][0]
        elif "batch" in self.get_shape_from.__dict__:
            sx = self.get_shape_from.batch.shape[2]
            sy = self.get_shape_from.batch.shape[1]
        elif "v"  in self.get_shape_from.__dict__:
            sx = self.get_shape_from.v.shape[1]
            sy = self.get_shape_from.v.shape[0]
        else:
            sx = self.get_shape_from.shape[1]
            sy = self.get_shape_from.shape[0]

        sz = sx * sy

        n_cols = int(numpy.round(numpy.sqrt(value.shape[0])))
        n_rows = int(numpy.ceil(value.shape[0] / n_cols))

        i = 0
        for row in range(0, n_rows):
            for col in range(0, n_cols):
                ax = figure.add_subplot(n_rows, n_cols, i)
                ax.cla()
                v = value[i].ravel()[:sz]
                ax.imshow(v.reshape(sy, sx),
                    interpolation="nearest", cmap=cm.gray)
                i += 1
                if i >= value.shape[0]:
                    break
            if i >= value.shape[0]:
                break

        figure.show()

        super(Weights2D, self).redraw()

    def run(self):
        if type(self.input_field) == int:
            if self.input_field < 0 or self.input_field >= len(self.input):
                return
            self.value = self.input[self.input_field]
        else:
            self.value = self.input.__dict__[self.input_field]
        Graphics().event_queue.put(self, block=True)
        super(Weights2D, self).run()


class Image2(Plotter):
    """Plotter for drawing 2 images.

    Should be assigned before initialize():
        input
        input_field
        input_field2

    Updates after run():

    Creates within initialize():

    """
    def __init__(self, figure_label="Image", unpickling=0):
        super(Image2, self).__init__(unpickling=unpickling)
        if unpickling:
            return
        self.value = None
        self.input = None  # Connector
        self.input_field = None
        self.figure_label = figure_label

    def redraw(self):
        if type(self.input_field) == int:
            if self.input_field < 0 or self.input_field >= len(self.input):
                return
            value = self.input[self.input_field]
            value2 = self.input[self.input_field2]
        else:
            value = self.input.__dict__[self.input_field]
            value2 = self.input.__dict__[self.input_field2]

        if type(value) != numpy.ndarray:
            return

        figure_label = self.figure_label
        figure = pp.figure(figure_label)
        figure.clf()

        if len(value.shape) == 2:
            sy1 = value.shape[0]
            sx1 = value.shape[1]
        elif len(value2.shape) == 2:
            sy2 = value2.shape[0]
            sx2 = value2.shape[1]
            sy1 = sy2
            sx1 = sx2
        if len(value2.shape) != 2:
            sy2 = sy1
            sx2 = sx1

        ax = figure.add_subplot(2, 1, 0)
        ax.cla()
        ax.imshow(value.reshape(sy1, sx1), interpolation="nearest", cmap=cm.gray)
        ax = figure.add_subplot(2, 1, 1)
        ax.cla()
        ax.imshow(value2.reshape(sy2, sx2), interpolation="nearest", cmap=cm.gray)

        figure.show()
        super(Image2, self).redraw()

    def run(self):
        Graphics().event_queue.put(self, block=True)
        super(Image2, self).run()


class Image3(Plotter):
    """Plotter for drawing 3 images.

    Should be assigned before initialize():
        input
        input_field
        input_field2
        input_field3

    Updates after run():

    Creates within initialize():

    """
    def __init__(self, figure_label="Image", unpickling=0):
        super(Image3, self).__init__(unpickling=unpickling)
        if unpickling:
            return
        self.value = None
        self.input = None  # Connector
        self.input_field = None
        self.figure_label = figure_label

    def redraw(self):
        if type(self.input_field) == int:
            if self.input_field < 0 or self.input_field >= len(self.input):
                return
            value = self.input[self.input_field]
            value2 = self.input[self.input_field2]
            value3 = self.input[self.input_field3]
        else:
            value = self.input.__dict__[self.input_field]
            value2 = self.input.__dict__[self.input_field2]
            value3 = self.input.__dict__[self.input_field3]

        if type(value) != numpy.ndarray:
            return

        if len(value.shape) != 2:
            sx = int(numpy.round(numpy.sqrt(value.size)))
            sy = int(numpy.round(value.size / sx))
            value = value.reshape(sy, sx)
        if len(value2.shape) != 2:
            sx = int(numpy.round(numpy.sqrt(value2.size)))
            sy = int(numpy.round(value2.size / sx))
            value2 = value2.reshape(sy, sx)
        if len(value3.shape) != 2:
            sx = int(numpy.round(numpy.sqrt(value3.size)))
            sy = int(numpy.round(value3.size / sx))
            value3 = value3.reshape(sy, sx)

        figure_label = self.figure_label
        figure = pp.figure(figure_label)
        figure.clf()

        ax = figure.add_subplot(3, 1, 1)
        ax.cla()
        ax.imshow(value, interpolation="nearest", cmap=cm.gray)

        ax = figure.add_subplot(3, 1, 2)
        ax.cla()
        ax.imshow(value2, interpolation="nearest", cmap=cm.gray)

        ax = figure.add_subplot(3, 1, 3)
        ax.cla()
        ax.imshow(value3, interpolation="nearest", cmap=cm.gray)

        figure.show()
        super(Image3, self).redraw()

    def run(self):
        Graphics().event_queue.put(self, block=True)
        super(Image3, self).run()


class Image1(Plotter):
    """Plotter for drawing 1 image.

    Should be assigned before initialize():
        input

    Updates after run():

    Creates within initialize():

    """
    def __init__(self, figure_label="Image", unpickling=0):
        super(Image1, self).__init__(unpickling=unpickling)
        if unpickling:
            return
        self.value = None
        self.input = None  # formats.Batch
        self.figure_label = figure_label

    def initialize(self):
        if type(self.input) != formats.Batch:
            return
        self.value = numpy.zeros_like(self.input.batch[0])

    def redraw(self):
        figure_label = self.figure_label
        figure = pp.figure(figure_label)
        figure.clf()

        ax = figure.add_subplot(111)
        ax.cla()
        ax.imshow(self.value, interpolation="nearest", cmap=cm.gray)
        figure.show()
        super(Image1, self).redraw()

    def run(self):
        if type(self.input) != formats.Batch:
            return
        self.input.sync(read_only=True)
        numpy.copyto(self.value, self.input.batch[0])
        Graphics().event_queue.put(self, block=True)
        super(Image1, self).run()


class ResultPlotter(Plotter):
    """Plotter for drawing result.

    Should be assigned before initialize():
        input

    Updates after run():

    Creates within initialize():

    """
    def __init__(self, figure_label="Result", unpickling=0):
        super(ResultPlotter, self).__init__(unpickling=unpickling)
        if unpickling:
            return
        self.values = []
        self.img = None
        self.input = None  # formats.Batch
        self.image = None  # formats.Batch
        self.figure_label = figure_label
        self.names = {0: "No channel",
                      1: "CCTV 4",
                      2: "CCTV Arabic",
                      3: "CCTV Espanol",
                      4: "CCTV Russian",
                      5: "CCTV Documentary",
                      6: "Duna",
                      7: "Duna World",
                      8: "FashionTV",
                      9: "M1",
                      10: "M2",
                      11: "Newros",
                      12: "Sterk"}
        self.lock = threading.Lock()

    def initialize(self):
        if type(self.image) != formats.Batch:
            return
        self.img = numpy.zeros_like(self.image.batch[0])
        self.lock.acquire()

    def redraw(self):
        figure_label = self.figure_label
        figure = pp.figure(figure_label)
        figure.clf()

        ax = figure.add_subplot(3, 1, 1)
        ax.cla()
        ax.plot(self.values, "b-")

        ax = figure.add_subplot(3, 1, 2)
        ax.cla()
        ax.axis('off')
        ax.imshow(self.img, interpolation="nearest", cmap=cm.gray)

        ax = figure.add_subplot(3, 1, 3)
        ax.cla()
        ax.axis('off')
        ax.text(0.5, 0.5, self.names.get(self.values[-1], ""),
                ha='center', va='center', fontsize=40)

        figure.show()
        super(ResultPlotter, self).redraw()

    def run(self):
        if type(self.input) != formats.Batch:
            return
        self.input.sync(read_only=True)
        self.values.append(self.input.batch[0])
        if type(self.image) == formats.Batch:
            self.image.sync(read_only=True)
            numpy.copyto(self.img, self.image.batch[0])
        Graphics().event_queue.put(self, block=True)
        super(ResultPlotter, self).run()


class MSEHistogram(Plotter):
    """Plotter for drawing histogram.

    Should be assigned before initialize():
        mse

    Updates after run():

    Creates within initialize():

    """
    def __init__(self, figure_label="Histogram", n_bars=35, unpickling=0):
        super(MSEHistogram, self).__init__(unpickling=unpickling)
        if unpickling:
            return
        self.figure_label = figure_label
        self.val_mse = None
        self.mse_min = None
        self.mse_max = None
        self.n_bars = n_bars
        self.mse = None  # formats.Vector()

    def initialize(self):
        self.val_mse = numpy.zeros(self.n_bars,
                                   dtype=config.dtypes[config.dtype])

    def redraw(self):
        fig = pp.figure(self.figure_label)
        fig.clf()
        fig.patch.set_facecolor('#E8D6BB')
        #fig.patch.set_alpha(0.45)

        ax = fig.add_subplot(1, 1, 1)
        ax.cla()
        ax.patch.set_facecolor('#ffe6ca')
        #ax.patch.set_alpha(0.45)

        ymin = self.val_min
        ymax = (self.val_max) * 1.3
        xmin = self.mse_min
        xmax = self.mse_max
        width = ((xmax - xmin) / self.n_bars) * 0.8
        t0 = 0.65 * ymax
        l1 = width * 0.5
        #l3 = 20
        #koef = 0.5 * ymax
        #l2 = 0.235 * ymax

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
        #, edgecolor='red')
        #D889B8
        #B96A9A
        ax.set_xlabel('Errors', fontsize=20)
        ax.set_ylabel('Input Data', fontsize=20)
        ax.set_title('Histogram', fontsize=25)
        ax.axis([xmin, xmax + ((xmax - xmin) / self.n_bars), ymin, ymax])
        ax.grid(True)
        leg = ax.legend((self.figure_label.replace("Histogram ", ""),),)
                        #'upper center')
        frame = leg.get_frame()
        frame.set_facecolor('#E8D6BB')
        for t in leg.get_texts():
            t.set_fontsize(18)
        for l in leg.get_lines():
            l.set_linewidth(1.5)

        for x, y in zip(N, self.val_mse):
            if y > koef - l2:
                pylab.text(x + l1, y - l2, '%.2f' % y,
                           ha='center', va='bottom',
                           fontsize=l3, rotation=90)
            else:
                pylab.text(x + l1, t0, '%.2f' % y,
                           ha='center', va='bottom',
                           fontsize=l3, rotation=90)

        fig.show()
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

        Graphics().event_queue.put(self, block=True)
        super(MSEHistogram, self).run()
