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
from veles.znicz.tests.unit.gd_numdiff import GDNumDiff


class TestGD(unittest.TestCase, GDNumDiff):
    def setUp(self):
        root.common.unit_test = True
        root.common.plotters_disabled = True
        self.state = rnd.get().state
        if not hasattr(self, "device"):
            self.device = opencl.Device()

    def _do_test(self, device, Forward, GD):
        batch_size = 2
        input_size = 25
        n_neurons = 7

        rnd.get().state = self.state

        dtype = opencl_types.dtypes[root.common.dtype]
        inp = numpy.zeros([batch_size, input_size], dtype=dtype)
        rnd.get().fill(inp)
        forward = Forward(DummyWorkflow(), output_shape=[n_neurons])
        forward.input = formats.Vector()
        forward.input.mem = inp.copy()
        forward.initialize(device=self.device)
        forward.run()

        forward.output.map_read()
        target = numpy.zeros_like(forward.output.mem)
        rnd.get().fill(target)
        if isinstance(forward, all2all.All2AllSoftmax):
            for sample in target:
                im = sample.argmax()
                sample[:] = 0.0
                sample[im] = 1.0
        out = forward.output.mem.copy()
        err_output = out - target
        forward.weights.map_read()
        weights = forward.weights.mem.copy()
        forward.bias.map_read()
        bias = forward.bias.mem.copy()

        c = GD(DummyWorkflow(),
               gradient_moment=0, gradient_moment_bias=0,
               learning_rate=-1, weights_decay=0,
               learning_rate_bias=-1, weights_decay_bias=0)

        c.err_output = formats.Vector()
        c.err_output.mem = err_output.copy()
        c.input = formats.Vector()
        c.input.mem = inp.copy()
        c.weights = formats.Vector()
        c.weights.mem = weights.copy()
        c.bias = formats.Vector()
        c.bias.mem = bias.copy()
        c.output = formats.Vector()
        c.output.mem = out.copy()
        c.initialize(device=device)
        c.run()
        c.err_input.map_read()
        c.weights.map_read()
        c.bias.map_read()

        err_input = c.err_input.mem.ravel()
        weights_derivative = c.weights.mem - weights
        bias_derivative = c.bias.mem - bias

        self.numdiff_check_gd(forward, inp, weights, bias, target,
                              err_input, weights_derivative, bias_derivative,
                              logging.info, self.assertLess)

        return c.err_input.mem.copy(), c.weights.mem.copy(), c.bias.mem.copy()

    def _do_test_gpu_cpu(self, Forward, GD):
        err_gpu, weights_gpu, bias_gpu = self._do_test(self.device,
                                                       Forward, GD)
        err_cpu, weights_cpu, bias_cpu = self._do_test(None,
                                                       Forward, GD)
        max_diff = numpy.fabs(err_gpu.ravel() - err_cpu.ravel()).max()
        self.assertLess(max_diff, 0.0001,
                        "GPU-CPU err_input differs by %.6f" % (max_diff))
        max_diff = numpy.fabs(weights_gpu.ravel() - weights_cpu.ravel()).max()
        self.assertLess(max_diff, 0.0001,
                        "GPU-CPU weights differs by %.6f" % (max_diff))
        max_diff = numpy.fabs(bias_gpu.ravel() - bias_cpu.ravel()).max()
        self.assertLess(max_diff, 0.0001,
                        "GPU-CPU bias differs by %.6f" % (max_diff))
        logging.info("All Ok")

    def test_gpu_cpu_linear(self):
        logging.info("Will test linear gd unit for gpu/cpu correctness")
        self._do_test_gpu_cpu(all2all.All2All, gd.GradientDescent)

    def test_gpu_cpu_relu(self):
        logging.info("Will test RELU gd unit for gpu/cpu correctness")
        self._do_test_gpu_cpu(all2all.All2AllRELU, gd.GDRELU)

    def test_gpu_cpu_softmax(self):
        logging.info("Will test SoftMax gd unit for gpu/cpu correctness")
        self._do_test_gpu_cpu(all2all.All2AllSoftmax, gd.GDSM)

    def test_gpu_cpu_tanh(self):
        logging.info("Will test Tanh gd unit for gpu/cpu correctness")
        self._do_test_gpu_cpu(all2all.All2AllTanh, gd.GDTanh)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
