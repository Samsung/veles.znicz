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
import veles.random_generator as rnd
import veles.znicz.gd as gd
from veles.tests.dummy_workflow import DummyWorkflow


class TestGD(unittest.TestCase):
    def setUp(self):
        root.common.unit_test = True
        root.common.plotters_disabled = True
        self.device = opencl.Device()
        self.W = None
        self.b = None
        self.state = rnd.get().state

    def tearDown(self):
        del self.device

    def _do_test(self, device, Unit):
        rnd.get().state = self.state

        inp = formats.Vector()
        dtype = opencl_types.dtypes[root.common.dtype]
        inp.mem = numpy.empty([1, 1], dtype=dtype)
        rnd.get().fill(inp.mem)

        if device is not None:
            self.x = inp.mem.copy()
        else:
            inp.mem[:] = self.x[:]

        c = Unit(DummyWorkflow(), gradient_moment=0.0)
        c.input = inp

        weights = numpy.empty([1, 1], dtype=dtype)
        rnd.get().fill(weights)
        bias = numpy.empty(1, dtype=dtype)
        rnd.get().fill(bias)
        c.err_output = formats.Vector()
        c.err_output.mem = numpy.empty([1, 1], dtype=dtype)
        rnd.get().fill(c.err_output.mem)

        c.weights = formats.Vector()
        c.weights.mem = weights
        c.bias = formats.Vector()
        c.bias.mem = bias
        c.output = formats.Vector()
        c.output.mem = c.err_output.mem.copy()
        c.initialize(device=device)

        if device is None:
            c.weights.map_invalidate()
            c.bias.map_invalidate()
            c.weights.mem[:] = self.W[:]
            c.bias.mem[:] = self.b[:]
            c.cpu_run()
            c.weights.map_read()
            self.W_cpu = c.weights.mem.copy()
            c.bias.map_read()
            self.b_cpu = c.bias.mem.copy()
        else:
            self.W = c.weights.mem.copy()
            self.b = c.bias.mem.copy()
            c.ocl_run()
            c.weights.map_read()
            self.W_gpu = c.weights.mem.copy()
            c.bias.map_read()
            self.b_gpu = c.bias.mem.copy()
        c.err_input.map_read()  # get results back

        return c.err_input.mem.copy()

    def _do_test_gpu_cpu(self, Unit):
        output_gpu = self._do_test(self.device, Unit)
        output_cpu = self._do_test(None, Unit)
        max_diff = numpy.fabs(output_gpu.ravel() - output_cpu.ravel()).max()
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
    logging.basicConfig(level=logging.INFO)
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
