"""
Created on Mart 31, 2014

Unit test for RELU convolutional layer forward propagation.

Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""


import logging
import numpy
import unittest

from veles.config import root
import veles.formats as formats
import veles.opencl as opencl
import veles.opencl_types as opencl_types
import veles.znicz.conv as conv
from veles.tests.dummy_workflow import DummyWorkflow


class TestConvRelu(unittest.TestCase):
    def setUp(self):
        root.common.unit_test = True
        root.common.plotters_disabled = True
        self.device = opencl.Device()

    def tearDown(self):
        del self.device

    def test_fixed(self):
        logging.info("Will test RELU convolutional layer forward propagation")

        inp = formats.Vector()
        dtype = opencl_types.dtypes[root.common.dtype]
        inp.v = numpy.array([[[1, 2, 3, 2, 1],
                              [0, 1, 2, 1, 0],
                              [0, 1, 0, 1, 0],
                              [2, 0, 1, 0, 2],
                              [1, 0, 1, 0, 1]]], dtype=dtype)

        weights = numpy.array([[[-1, -1, -1],
                                [-1, 8, -1],
                                [-1, -1, -1]],
                               [[1.1, 2.1, 3.1],
                                [-1.1, -0.5, 1.3],
                                [1.7, -1.4, 0.05]]], dtype=dtype)
        bias = numpy.array([10, -10], dtype=dtype)

        c = conv.ConvRELU(DummyWorkflow(), n_kernels=2, kx=3, ky=3)
        c.input = inp

        c.initialize(device=self.device)

        c.weights.map_invalidate()  # rewrite weights
        c.weights.v[:] = weights.reshape(c.weights.v.shape)[:]
        c.bias.map_invalidate()  # rewrite bias
        c.bias.v[:] = bias[:]

        c.run()
        c.output.map_read()  # get results back
        nz = numpy.count_nonzero(c.output.vv[c.output.v.shape[0]:].ravel())
        self.assertEqual(nz, 0, "Overflow occured")

        y = c.output.v.ravel()
        t = numpy.array([9, 5.3, 15, 5.65, 9, -3.5,
                         12, 1.25, 3, -2.8, 12, -4.4,
                         4, -7.05, 15, -7.7, 4, -4.65], dtype=dtype)
        t = numpy.where(t > 15, t, numpy.log(numpy.exp(t) + 1.0))
        max_diff = numpy.fabs(t - y).max()
        self.assertLess(max_diff, 0.0001,
                        "Result differs by %.6f" % (max_diff))

        logging.info("All Ok")


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
