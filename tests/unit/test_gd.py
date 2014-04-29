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
import veles.rnd as rnd
import veles.znicz.gd as gd
from veles.tests.dummy_workflow import DummyWorkflow


class TestGD(unittest.TestCase):
    def setUp(self):
        root.common.unit_test = True
        root.common.plotters_disabled = True
        self.device = opencl.Device()

    def tearDown(self):
        del self.device

    def _do_tst(self, device, Unit):
        inp = formats.Vector()
        dtype = opencl_types.dtypes[root.common.dtype]
        inp.v = numpy.empty([5, 5], dtype=dtype)
        rnd.default.fill(inp.v)

        if device is not None:
            self.x = inp.v.copy()
        else:
            inp.v[:] = self.x[:]

        c = Unit(DummyWorkflow())
        c.h = inp

        weights = numpy.array([[1, 0, 2, 1, -1],
                              [3, 1, 0, 2, 3],
                              [-1, 2, 0, 1, 3],
                              [1, -1, 3, 2, 4]], dtype=dtype)
        bias = numpy.array([10, -10, 5, 2], dtype=dtype)
        c.err_y = formats.Vector()
        c.err_y.v = numpy.array([[-1, 3, 0, 2],
                                 [8, 2, 1, 3],
                                 [0, 1, -2, 1],
                                 [2, 3, -1, 0],
                                 [1, 0, 1, 1]], dtype=dtype)

        c.weights = formats.Vector()
        c.weights.v = weights
        c.bias = formats.Vector()
        c.bias.v = bias
        c.y = formats.Vector()
        c.y.v = c.err_y.v.copy()
        c.initialize(device=device)

        if device is None:
            c.weights.map_invalidate()
            c.bias.map_invalidate()
            c.weights.v[:] = self.W[:]
            c.bias.v[:] = self.b[:]
            c.cpu_run()
            c.weights.map_read()
            self.W_cpu = c.weights.v.copy()
            c.bias.map_read()
            self.b_cpu = c.bias.v.copy()
        else:
            self.W = c.weights.v.copy()
            self.b = c.bias.v.copy()
            c.ocl_run()
            c.weights.map_read()
            self.W_gpu = c.weights.v.copy()
            c.bias.map_read()
            self.b_gpu = c.bias.v.copy()
        c.err_h.map_read()  # get results back

        return c.err_h.v

    def _do_test_gpu_cpu(self, Unit):
        y_gpu = self._do_tst(self.device, Unit)
        y_cpu = self._do_tst(None, Unit)
        max_diff = numpy.fabs(y_gpu.ravel() - y_cpu.ravel()).max()
        self.assertLess(max_diff, 0.0001,
                        "Result differs by %.6f" % (max_diff))
        max_diff = numpy.fabs(self.W_gpu.ravel() - self.W_cpu.ravel()).max()
        self.assertLess(max_diff, 0.0001,
                        "Weights differs by %.6f" % (max_diff))
        logging.info("All Ok")

    def test_gpu_cpu_linear(self):
        logging.info("Will test linear gd unit for gpu/cpu correctness")
        self._do_test_gpu_cpu(gd.GradientDescent)

    def test_gpu_cpu_relu(self):
        logging.info("Will test RELU gd unit for gpu/cpu correctness")
        self._do_test_gpu_cpu(gd.GDRELU)

    def test_gpu_cpu_softmax(self):
        logging.info("Will test SoftMax gd unit for gpu/cpu correctness")
        self._do_test_gpu_cpu(gd.GDSM)

    def test_gpu_cpu_tanh(self):
        logging.info("Will test Tanh gd unit for gpu/cpu correctness")
        self._do_test_gpu_cpu(gd.GDTanh)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
