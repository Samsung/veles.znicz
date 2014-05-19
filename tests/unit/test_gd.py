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
import veles.znicz.all2all as all2all


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

    def _do_test(self, device, Unit, Forward):
        batch_size = 1
        input_size = 25
        n_neurons = 7

        rnd.get().state = self.state

        dtype = opencl_types.dtypes[root.common.dtype]
        inp = numpy.empty([batch_size, input_size], dtype=dtype)
        rnd.get().fill(inp)

        c = Unit(DummyWorkflow(), gradient_moment=0.0)

        weights = numpy.empty([n_neurons, input_size], dtype=dtype)
        rnd.get().fill(weights)

        bias = numpy.empty(n_neurons, dtype=dtype)
        rnd.get().fill(bias)

        err_output = numpy.empty([batch_size, n_neurons], dtype=dtype)
        rnd.get().fill(err_output)

        if device is not None:
            self.x = inp.copy()
            self.weights = weights.copy()
            self.bias = bias.copy()
            self.err_output = err_output.copy()
        else:
            inp[:] = self.x[:]
            weights[:] = self.weights[:]
            bias[:] = self.bias[:]
            err_output[:] = self.err_output[:]

        c.input = formats.Vector()
        c.input.mem = inp.copy()
        c.err_output = formats.Vector()
        c.err_output.mem = err_output.copy()
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

        logging.info("Will check with numeric differentiation")
        forward = Forward(DummyWorkflow(), output_shape=err_output.shape[1:])
        forward.input = formats.Vector()
        forward.input.mem = inp.copy()
        forward.initialize(device=self.device)
        forward.weights.map_invalidate()
        forward.weights.mem[:] = weights[:]
        forward.bias.map_invalidate()
        forward.bias.mem[:] = bias[:]
        forward.run()
        forward.output.map_read()
        target = forward.output.mem.ravel() - err_output.ravel()
        h = 1.0e-8
        points = (2.0 * h, h, -h, -2.0 * h)
        coeffs = numpy.array([-1.0, 8.0, -8.0, 1.0], dtype=numpy.float64)
        divizor = 12.0 * h
        errs = numpy.zeros_like(points)
        err_input = c.err_input.mem.ravel()
        offs = 0
        for i_sample in range(inp.shape[0]):
            for x in range(inp.shape[1]):
                for i, p in enumerate(points):
                    forward.input.map_invalidate()
                    forward.input.mem[:] = inp[:]
                    forward.input.mem[i_sample, x] = inp[i_sample, x] + p
                    forward.run()
                    forward.output.map_read()
                    out = forward.output.mem.ravel()
                    errs[i] = numpy.square(out - target).sum() * 0.5

                derivative = (errs * coeffs).sum() / divizor
                d = numpy.fabs(derivative - err_input[offs])
                logging.info("%.2f %.2f %.2f" %
                             (derivative, err_input[offs], d))
                self.assertLess(d, 0.5, "Numeric diff test failed")
                offs += 1

        return c.err_input.mem.copy()

    def _do_test_gpu_cpu(self, Unit, Forward):
        output_gpu = self._do_test(self.device, Unit, Forward)
        output_cpu = self._do_test(None, Unit, Forward)
        max_diff = numpy.fabs(output_gpu.ravel() - output_cpu.ravel()).max()
        self.assertLess(max_diff, 0.0001,
                        "Result differs by %.6f" % (max_diff))
        max_diff = numpy.fabs(self.W_gpu.ravel() - self.W_cpu.ravel()).max()
        self.assertLess(max_diff, 0.0001,
                        "Weights differs by %.6f" % (max_diff))
        logging.info("All Ok")

    def test_gpu_cpu_linear(self):
        logging.info("Will test linear gd unit for gpu/cpu correctness")
        self._do_test_gpu_cpu(gd.GradientDescent, all2all.All2All)

    def _test_gpu_cpu_relu(self):
        logging.info("Will test RELU gd unit for gpu/cpu correctness")
        self._do_test_gpu_cpu(gd.GDRELU, all2all.All2AllRELU)

    def _test_gpu_cpu_softmax(self):
        logging.info("Will test SoftMax gd unit for gpu/cpu correctness")
        self._do_test_gpu_cpu(gd.GDSM, all2all.All2AllSoftmax)

    def _test_gpu_cpu_tanh(self):
        logging.info("Will test Tanh gd unit for gpu/cpu correctness")
        self._do_test_gpu_cpu(gd.GDTanh, all2all.All2AllTanh)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
