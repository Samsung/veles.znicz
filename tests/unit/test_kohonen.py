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
        self.new_weights = numpy.array(
            [[0.0095, 3.4077, -1.1363, -0.2331, 7.291],
             [-7.3005, -0.141, 6.3435, -3.8349, -7.3005],
             [7.0286, -3.361, 6.0806, 0.0389, -6.5709],
             [3.7339, -0.0242, 10.1774, 3.7097, 0.],
             [7.3563, 7.3657, 2.8212, 0.121, 0.1115],
             [0.2757, 9.8744, 9.1679, 6.6905, -6.0922],
             [6.6781, 10.1743, 2.6081, -6.4829, 0.0152],
             [6.8938, 7.1, -4.0215, 3.6939, -3.3245],
             [0.213, 3.6071, -3.4613, -2.7033, 6.5233]],
            dtype=self.dtype)
        self.winners = numpy.array([2, 1, 0, 1, 0, 0, 0, 0, 1],
                                   dtype=numpy.int)

    def tearDown(self):
        del self.device

    def test_forward(self):
        logging.info("Will test KohonenForward unit forward pass")
        c = kohonen.KohonenForward(DummyWorkflow(), output_shape=[3, 3])
        c.input = formats.Vector()
        c.input.v = self.input[:]
        c.weights = formats.Vector()
        c.weights.v = self.weights[:]
        c.shape = (3, 3)
        c.epoch_ended = False
        c.initialize(device=self.device)

        c.run()
        c.output.map_read()  # get results back

        y = c.output.v.ravel()

        max_diff = numpy.fabs(self.output.ravel() - y.ravel()).max()
        self.assertLess(max_diff, 0.0001,
                        "Result differs by %.6f" % (max_diff))

    def test_train(self):
        logging.info("Will test KohonenForward unit train pass")

        c = kohonen.KohonenTrainer(DummyWorkflow(), shape=(3, 3))
        c.input = formats.Vector()
        c.input.v = self.input[:]
        c.gradient_decay = lambda t: 1.0 / (1.0 + t)
        c.weights.v = self.weights[:]

        c.initialize(device=self.device)

        c.cpu_run()

        weights = c.weights.v.ravel()
        winners = c.winners.v
        max_diff = numpy.fabs(self.new_weights.ravel() - weights.ravel()).max()
        self.assertLess(max_diff, 0.0001,
                        "Result differs by %.6f" % (max_diff))
        self.assertTrue(all(winners == self.winners),
                        "Wrong winners %s" % str(winners))

        c.weights.map_invalidate()
        c.weights.v[:] = self.weights
        c.winners.map_invalidate()
        c.winners.v[:] = 0
        c.weights.unmap()
        c.winners.unmap()
        c.time = 0

        c.ocl_run()

        c.weights.map_read()
        c.winners.map_read()
        weights = c.weights.v.ravel()
        winners = c.winners.v

        max_diff = numpy.fabs(self.new_weights.ravel() - weights.ravel()).max()
        self.assertLess(max_diff, 0.0001,
                        "Result differs by %.6f" % (max_diff))
        self.assertTrue(all(winners == self.winners),
                        "Wrong winners %s" % str(winners))


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
