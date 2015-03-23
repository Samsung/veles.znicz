"""
Created on May 7, 2014

Copyright (c) 2014, Samsung Electronics, Co., Ltd.
"""


import logging
import matplotlib
matplotlib.use("cairo")
import matplotlib.cm
import matplotlib.pyplot
import matplotlib.patches
import numpy
import os
import unittest

import veles.znicz.nn_plotting_units as nnpu
from veles.prng import get as get_prng
prng = get_prng()


STORE_IMAGES = True


class Test(unittest.TestCase):

    def init_plotter(self, name):
        plotter = getattr(nnpu, name)(self.parent)
        plotter.matplotlib = matplotlib
        plotter.cm = matplotlib.cm
        plotter.pp = matplotlib.pyplot
        plotter.patches = matplotlib.patches
        plotter.show_figure = plotter.nothing
        return plotter

    def plot(self, plotter):
        plotter.redraw()
        tmp_file_name = "/tmp/%s.png" % plotter.name
        plotter.pp.savefig(tmp_file_name)
        if not STORE_IMAGES:
            os.remove(tmp_file_name)

    def testKohonenHits(self):
        kh = self.init_plotter("KohonenHits")
        kh.input = numpy.empty((10, 9))
        kh.input = numpy.digitize(prng.uniform(
            size=kh.input.size), numpy.arange(0.05, 1.05, 0.05))
        kh.shape = (10, 9)
        self.plot(kh)

    def testKohonenInputMaps(self):
        kim = self.init_plotter("KohonenInputMaps")
        kim.input = numpy.empty((100, 4))
        kim.input = prng.uniform(size=kim.input.size).reshape(
            kim.input.shape)
        kim.shape = (10, 10)
        self.plot(kim)

    def testKohonenNeighborMap(self):
        knm = self.init_plotter("KohonenNeighborMap")
        knm.input = numpy.empty((100, 4))
        knm.input = prng.uniform(size=knm.input.size).reshape(
            knm.input.shape)
        knm.shape = (10, 10)
        self.plot(knm)

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()
