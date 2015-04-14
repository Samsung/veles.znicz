# -*- coding: utf-8 -*-
"""
.. invisible:
     _   _ _____ _     _____ _____
    | | | |  ___| |   |  ___/  ___|
    | | | | |__ | |   | |__ \ `--.
    | | | |  __|| |   |  __| `--. \
    \ \_/ / |___| |___| |___/\__/ /
     \___/\____/\_____|____/\____/

Created on May 7, 2014

███████████████████████████████████████████████████████████████████████████████

Licensed to the Apache Software Foundation (ASF) under one
or more contributor license agreements.  See the NOTICE file
distributed with this work for additional information
regarding copyright ownership.  The ASF licenses this file
to you under the Apache License, Version 2.0 (the
"License"); you may not use this file except in compliance
with the License.  You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing,
software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
KIND, either express or implied.  See the License for the
specific language governing permissions and limitations
under the License.

███████████████████████████████████████████████████████████████████████████████
"""


import logging
import matplotlib
from veles.dummy import DummyWorkflow
from veles.units import nothing

matplotlib.use("cairo")
import matplotlib.cm
import matplotlib.pyplot
import matplotlib.patches
import numpy
from tempfile import NamedTemporaryFile
import unittest

import veles.znicz.nn_plotting_units as nnpu
from veles.prng import get as get_prng
prng = get_prng()


STORE_IMAGES = True


class Test(unittest.TestCase):
    def setUp(self):
        self.parent = DummyWorkflow()

    def tearDown(self):
        del self.parent

    def init_plotter(self, name):
        plotter = getattr(nnpu, name)(self.parent)
        plotter.matplotlib = matplotlib
        plotter.cm = matplotlib.cm
        plotter.pp = matplotlib.pyplot
        plotter.patches = matplotlib.patches
        plotter.show_figure = nothing
        return plotter

    def plot(self, plotter):
        plotter.redraw()
        with NamedTemporaryFile(suffix="-%s.png" % plotter.name,
                                delete=not STORE_IMAGES) as fout:
            logging.debug("Created %s", fout.name)
            plotter.pp.savefig(fout)

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
