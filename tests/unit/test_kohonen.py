#!/usr/bin/python3.3 -O
"""
Created on November 18, 2013

Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""


import logging
import numpy
import unittest

from veles.config import root
import veles.formats as formats
import veles.opencl as opencl
import veles.opencl_types as opencl_types
from veles.tests.dummy_workflow import DummyWorkflow
import veles.znicz.kohonen as kohonen


class TestKohonen(unittest.TestCase):
    def setUp(self):
        root.common.unit_test = True
        root.common.plotters_disabled = True
        self.device = opencl.Device()
        self.dtype = opencl_types.dtypes[root.common.dtype]
        self.input = numpy.array([[1, 2, 3, 2, 1],
                                  [0, 1, 2, 1, 0],
                                  [0, 1, 0, 1, 0],
                                  [2, 0, 1, 0, 2],
                                  [1, 0, 1, 0, 1]],
                                 dtype=self.dtype)
        self.weights = numpy.array([[1, 0, 2, 1, -1],
                                    [3, 1, 0, 2, 3],
                                    [-1, 2, 0, 1, 3],
                                    [0, 1, -1, 0, 1],
                                    [-1, -1, 1, 1, 1],
                                    [1, -2, -1, -1, 3],
                                    [-1, -2, 1, 3, 1],
                                    [-1, -1, 3, 0, 2],
                                    [1, 0, 3, 2, -1]],
                                   dtype=self.dtype)
        self.output = numpy.array([[8, 12, 8, 0, 3, -5, 5, 8, 13],
                                   [5, 3, 3, -1, 2, -5, 3, 5, 8],
                                   [1, 3, 3, 1, 0, -3, 1, -1, 2],
                                   [2, 12, 4, 1, 1, 7, 1, 5, 3],
                                   [2, 6, 2, 0, 1, 3, 1, 4, 3]],
                                  dtype=self.dtype)

        # TODO(a.kazantsev): put right values here
        self.gradient_weights = numpy.array([[1, 0, 2, 1, -1],
                                             [3, 1, 0, 2, 3],
                                             [-1, 2, 0, 1, 3],
                                             [0, 1, -1, 0, 1],
                                             [-1, -1, 1, 1, 1],
                                             [1, -2, -1, -1, 3],
                                             [-1, -2, 1, 3, 1],
                                             [-1, -1, 3, 0, 2],
                                             [1, 0, 3, 2, -1]],
                                            dtype=self.dtype)

    def tearDown(self):
        del self.device

    def test_forward(self):
        logging.info("Will test Kohonen unit forward pass")
        inp = formats.Vector()
        inp.v = self.input.copy()
        c = kohonen.Kohonen(DummyWorkflow(), output_shape=[3, 3])
        c.input = inp

        c.initialize(device=self.device)

        c.weights.map_invalidate()  # rewrite weights
        c.weights.v[:] = self.weights.reshape(c.weights.v.shape)[:]

        c.run()
        c.output.map_read()  # get results back

        y = c.output.v.ravel()

        max_diff = numpy.fabs(self.output.ravel() - y.ravel()).max()
        self.assertLess(max_diff, 0.0001,
                        "Result differs by %.6f" % (max_diff))
        logging.info("All Ok")

    def test_train(self):
        logging.info("Will test Kohonen unit train pass")
        inp = formats.Vector()
        inp.v = self.input.copy()
        out = formats.Vector()
        out.v = self.output.copy()
        c = kohonen.KohonenTrain(DummyWorkflow())
        c.h = inp
        c.y = out
        c.weights = formats.Vector()
        c.weights.v = self.weights.copy()

        c.initialize(device=self.device)

        c.run()

        c.gradient_weights.map_read()  # get results back

        gradient_weights = c.gradient_weights.v.ravel()

        max_diff = numpy.fabs(self.gradient_weights.ravel() -
                              gradient_weights.ravel()).max()
        self.assertLess(max_diff, 0.0001,
                        "Result differs by %.6f" % (max_diff))
        logging.info("All Ok")


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
