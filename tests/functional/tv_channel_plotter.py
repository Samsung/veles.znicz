"""
Plotter for TV-logo recognition demo.

@author: Kazantsev Alexey <a.kazantsev@samsung.com>
"""
import matplotlib.pyplot as pp
import matplotlib.cm as cm
import numpy
import plotters
import formats


class ResultPlotter(plotters.Plotter):
    """Plotter for drawing result.

    Should be assigned before initialize():
        input

    Updates after run():

    Creates within initialize():

    """
    def __init__(self, figure_label="Result"):
        super(ResultPlotter, self).__init__()
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
        if self.img != None:
            ax.imshow(self.img, interpolation="nearest", cmap=cm.gray)

        ax = figure.add_subplot(3, 1, 3)
        ax.cla()
        ax.axis('off')
        nme = self.names.get(self.values[-1], "")
        self.log().info(nme)
        ax.text(0.5, 0.5, nme,
                ha='center', va='center', fontsize=40)

        figure.show()
        super(ResultPlotter, self).redraw()

    def run(self):
        if type(self.input) != formats.Vector:
            return
        self.input.sync()
        self.values.append(float(self.input.v[0]))
        if type(self.image) == formats.Vector:
            self.image.sync()
            self.img = plotters.norm_image(self.image.v[0])
        super(ResultPlotter, self).run()
