import numpy
import threading
import plotters
import formats


class ResultPlotter(plotters.Plotter):
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
        plotters.Graphics().event_queue.put(self, block=True)
        super(ResultPlotter, self).run()