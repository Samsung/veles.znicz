"""
Created on May 17, 2013

@author: Kumok Akim <a.kumok@samsung.com>
"""
import matplotlib.pyplot
import matplotlib.patches
import matplotlib.cm as cm
import units
import tkinter
import queue
import threading
import numpy
import formats


matplotlib.pyplot.ion()


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
        print("Waiting for user to close the window...")
        self.root.destroy()
        self.run_lock.acquire()
        self.run_lock.release()
        print("Done")


class Plotter(units.OpenCLUnit):
    """Base class for all plotters
    """

    def __init__(self, device=None, unpickling=0):
        super(Plotter, self).__init__(unpickling=unpickling,
                                      device=device)
        if unpickling:
            return

    def redraw(self):
        """ Do the actual drawing here
        """
        pass


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
                 device=None,
                 unpickling=0):
        super(SimplePlotter, self).__init__(unpickling=unpickling,
                                            device=device)
        if unpickling:
            if "clear_plot" not in self.__dict__:
                self.clear_plot = False
            return
        self.values = list()
        self.input = None   # Connector
        self.input_field = None
        self.figure_label = figure_label
        self.plot_style = plot_style
        self.input_offs = 0
        self.clear_plot = clear_plot

    def redraw(self):
        figure_label = self.figure_label
        figure = matplotlib.pyplot.figure(figure_label)
        if self.clear_plot:
            figure.clf()
        axes = figure.add_subplot(111)  # Main axes
        if self.clear_plot:
            axes.cla()
        axes.plot(self.values, self.plot_style)
        figure.show()

    def cpu_run(self):
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


class MatrixPlotter(Plotter):
    """Plotter for drawing matrixes

    Should be assigned before initialize():
        input
        input_field

    Updates after run():

    Creates within initialize():

    """
    def __init__(self, figure_label="Matrix",
                 device=None,
                 unpickling=0):
        super(MatrixPlotter, self).__init__(unpickling=unpickling,
                                            device=device)
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
        figure = matplotlib.pyplot.figure(figure_label)
        figure.clf()
        num_rows = len(value) + 2
        num_columns = len(value[0]) + 2

        main_axes = figure.add_axes([0, 0, 1, 1])
        main_axes.cla()

        # First cell color
        rc = matplotlib.patches.Rectangle(
            (0, (num_rows - 1) / num_rows),
            1.0 / num_rows, 1.0 / num_columns, color='gray')
        main_axes.add_patch(rc)
        # First row last cell color
        rc = matplotlib.patches.Rectangle(
            ((num_columns - 1) / num_columns, (num_rows - 1) / num_rows),
            1.0 / num_rows, 1.0 / num_columns, color='gray')
        main_axes.add_patch(rc)
        # First column last cell color
        rc = matplotlib.patches.Rectangle(
            (0, 0),
            1.0 / num_rows, 1.0 / num_columns, color='gray')
        main_axes.add_patch(rc)
        # Last cell color
        rc = matplotlib.patches.Rectangle(
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
                rc = matplotlib.patches.Rectangle(
                    (column / num_columns, (num_rows - row - 1) / num_rows),
                    1.0 / num_rows, 1.0 / num_columns, color=color)
                main_axes.add_patch(rc)

        for row in range(num_rows):
            y = row / num_rows
            main_axes.add_line(matplotlib.lines.Line2D([0, 1], [y, y]))
        for column in range(num_columns):
            x = column / num_columns
            main_axes.add_line(matplotlib.lines.Line2D([x, x], [0, 1]))

        # First cell
        column = 0
        row = 0
        matplotlib.pyplot.figtext(label="0",
            s="target",
            x=(column + 0.9) / num_columns,
            y=(num_rows - row - 0.33) / num_rows,
            verticalalignment="center",
            horizontalalignment="right")
        matplotlib.pyplot.figtext(label="0",
            s="value",
            x=(column + 0.1) / num_columns,
            y=(num_rows - row - 0.66) / num_rows,
            verticalalignment="center",
            horizontalalignment="left")

        # Headers in first row
        row = 0
        for column in range(1, num_columns - 1):
            matplotlib.pyplot.figtext(label=("C%d" % (column - 1, )),
                            s=(column - 1),
                            x=(column + 0.5) / num_columns,
                            y=(num_rows - row - 0.5) / num_rows,
                            verticalalignment="center",
                            horizontalalignment="center")
        # Headers in first column
        column = 0
        for row in range(1, num_rows - 1):
            matplotlib.pyplot.figtext(label=("R%d" % (row - 1, )),
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
                matplotlib.pyplot.figtext(
                    label=label,
                    s=n_elem,
                    x=(column + 0.5) / num_columns,
                    y=(num_rows - row - 0.33) / num_rows,
                    verticalalignment="center",
                    horizontalalignment="center")
                #matplotlib.pyplot.figtext(
                #    label=label,
                #    s=("%.2f%%" % (pt_elem, )),
                #    x=(column + 0.1) / num_columns,
                #    y=(num_rows - row - 0.75) / num_rows,
                #    verticalalignment="center",
                #    horizontalalignment="left")
                matplotlib.pyplot.figtext(
                    label=label,
                    s=("%.2f%%" % (pt_total, )),
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
        matplotlib.pyplot.figtext(
            label=label,
            s=sum_ok,
            x=(column + 0.5) / num_columns,
            y=(num_rows - row - 0.33) / num_rows,
            verticalalignment="center",
            horizontalalignment="center")
        matplotlib.pyplot.figtext(
            label=label,
            s=("%.2f%%" % (pt_total, )),
            x=(column + 0.5) / num_columns,
            y=(num_rows - row - 0.66) / num_rows,
            verticalalignment="center",
            horizontalalignment="center")
        figure.show()

    def cpu_run(self):
        if type(self.input_field) == int:
            if self.input_field < 0 or self.input_field >= len(self.input):
                return
            self.value = self.input[self.input_field]
        else:
            self.value = self.input.__dict__[self.input_field]
        Graphics().event_queue.put(self, block=True)


class Weights2D(Plotter):
    """Plotter for drawing weights as 2D.

    Should be assigned before initialize():
        input
        input_field

    Updates after run():

    Creates within initialize():

    """
    def __init__(self, figure_label="Weights",
                 device=None,
                 unpickling=0):
        super(Weights2D, self).__init__(unpickling=unpickling,
                                            device=device)
        if unpickling:
            return
        self.value = None
        self.input = None  # Connector
        self.input_field = None
        self.figure_label = figure_label
        self.get_shape_from = None

    def redraw(self):
        if type(self.input_field) == int:
            if self.input_field < 0 or self.input_field >= len(self.input):
                return
            value = self.input[self.input_field]
        else:
            value = self.input.__dict__[self.input_field]

        if type(value) != numpy.ndarray or len(value.shape) != 2:
            return

        figure_label = self.figure_label
        figure = matplotlib.pyplot.figure(figure_label)
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
                    interpolation="none", cmap=cm.gray)
                i += 1
                if i >= value.shape[0]:
                    break
            if i >= value.shape[0]:
                break

        figure.show()

    def cpu_run(self):
        if type(self.input_field) == int:
            if self.input_field < 0 or self.input_field >= len(self.input):
                return
            self.value = self.input[self.input_field]
        else:
            self.value = self.input.__dict__[self.input_field]
        Graphics().event_queue.put(self, block=True)


class Image2(Plotter):
    """Plotter for drawing 2 images.

    Should be assigned before initialize():
        input
        input_field
        input_field2

    Updates after run():

    Creates within initialize():

    """
    def __init__(self, figure_label="Image",
                 device=None,
                 unpickling=0):
        super(Image2, self).__init__(unpickling=unpickling,
                                     device=device)
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
        figure = matplotlib.pyplot.figure(figure_label)
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
        ax.imshow(value.reshape(sy1, sx1), interpolation="none", cmap=cm.gray)
        ax = figure.add_subplot(2, 1, 1)
        ax.cla()
        ax.imshow(value2.reshape(sy2, sx2), interpolation="none", cmap=cm.gray)

        figure.show()

    def cpu_run(self):
        Graphics().event_queue.put(self, block=True)


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
    def __init__(self, figure_label="Image",
                 device=None,
                 unpickling=0):
        super(Image3, self).__init__(unpickling=unpickling,
                                     device=device)
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

        figure_label = self.figure_label
        figure = matplotlib.pyplot.figure(figure_label)
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

        ax = figure.add_subplot(111)
        ax.cla()
        img = numpy.empty([sy1 * 3, sx1], dtype=value.dtype)
        img[:sy1, :] = value.reshape(sy1, sx1)[:]
        img[sy1:sy1 * 2, :] = value2.reshape(sy1, sx1)[:]
        img[sy1 * 2:, :] = value3.reshape(sy1, sx1)[:]
        ax.imshow(img, interpolation="none", cmap=cm.gray)
        figure.show()

    def cpu_run(self):
        Graphics().event_queue.put(self, block=True)


class Image1(Plotter):
    """Plotter for drawing 1 image.

    Should be assigned before initialize():
        input

    Updates after run():

    Creates within initialize():

    """
    def __init__(self, figure_label="Image",
                 device=None,
                 unpickling=0):
        super(Image1, self).__init__(unpickling=unpickling,
                                     device=device)
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
        figure = matplotlib.pyplot.figure(figure_label)
        figure.clf()

        ax = figure.add_subplot(111)
        ax.cla()
        ax.imshow(self.value, interpolation="none", cmap=cm.gray)
        figure.show()

    def cpu_run(self):
        if type(self.input) != formats.Batch:
            return
        self.input.sync(read_only=True)
        numpy.copyto(self.value, self.input.batch[0])
        Graphics().event_queue.put(self, block=True)


class ResultPlotter(Plotter):
    """Plotter for drawing result.

    Should be assigned before initialize():
        input

    Updates after run():

    Creates within initialize():

    """
    def __init__(self, figure_label="Result",
                 device=None,
                 unpickling=0):
        super(ResultPlotter, self).__init__(unpickling=unpickling,
            device=device)
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
        figure = matplotlib.pyplot.figure(figure_label)
        figure.clf()

        ax = figure.add_subplot(3, 1, 1)
        ax.cla()
        ax.plot(self.values, "b-")

        ax = figure.add_subplot(3, 1, 2)
        ax.cla()
        ax.axis('off')
        ax.imshow(self.img, interpolation="none", cmap=cm.gray)

        ax = figure.add_subplot(3, 1, 3)
        ax.cla()
        ax.axis('off')
        ax.text(0.5, 0.5, self.names.get(self.values[-1], ""),
                ha='center', va='center', fontsize=40)

        figure.show()

        self.lock.release()

    def cpu_run(self):
        if type(self.input) != formats.Batch:
            return
        self.input.sync(read_only=True)
        self.values.append(self.input.batch[0])
        if type(self.image) == formats.Batch:
            self.image.sync(read_only=True)
            numpy.copyto(self.img, self.image.batch[0])
        Graphics().event_queue.put(self, block=True)
        self.lock.acquire()
