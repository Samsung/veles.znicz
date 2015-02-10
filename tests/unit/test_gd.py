#!/usr/bin/python3 -O
"""
Created on November 18, 2013

Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""

import gc
import logging
import unittest

import numpy

from veles.config import root
from veles.memory import Vector
import veles.backends as opencl
import veles.opencl_types as opencl_types
import veles.prng as prng
from veles.znicz.gd import GradientDescent, GDRELU, GDSoftmax, GDTanh
from veles.dummy import DummyWorkflow
import veles.znicz.all2all as all2all
from veles.znicz.nn_units import GradientDescentBase
from veles.znicz.tests.unit.gd_numdiff import GDNumDiff
from veles.tests import timeout
from veles.tests.doubling_reset import patch


class PatchedGradientDescentBase(GradientDescentBase):
    def __init__(self, workflow, **kwargs):
        super(PatchedGradientDescentBase, self).__init__(workflow, **kwargs)
        patch(self, self.err_input, lambda: self.input.shape,
              lambda: self.err_output.dtype)


class PatchedGradientDescent(GradientDescent, PatchedGradientDescentBase):
    pass


class PatchedGDRELU(GDRELU, PatchedGradientDescentBase):
    pass


class PatchedGDSoftmax(GDSoftmax, PatchedGradientDescentBase):
    pass


class PatchedGDTanh(GDTanh, PatchedGradientDescentBase):
    pass


class TestGD(unittest.TestCase, GDNumDiff):
    def setUp(self):
        self.device = opencl.Device()

    def tearDown(self):
        gc.collect()
        del self.device

    def _do_test(self, device, Forward, GD):
        batch_size = 2
        input_size = 25
        n_neurons = 7

        prng.get().seed(123)

        dtype = opencl_types.dtypes[root.common.precision_type]
        inp = numpy.zeros([batch_size, input_size], dtype=dtype)
        prng.get().fill(inp)
        forward = Forward(DummyWorkflow(), output_sample_shape=[n_neurons])
        forward.input = Vector()
        forward.input.mem = inp.copy()
        forward.initialize(device=self.device)
        forward.run()

        forward.output.map_read()
        target = numpy.zeros_like(forward.output.mem)
        prng.get().fill(target)
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

        c.err_output = Vector()
        c.err_output.mem = err_output.copy()
        c.input = Vector()
        c.input.mem = inp.copy()
        c.weights = Vector()
        c.weights.mem = weights.copy()
        c.bias = Vector()
        c.bias.mem = bias.copy()
        c.output = Vector()
        c.output.mem = out.copy()
        c.initialize(device=device)
        c.run()
        c.err_input.map_read()
        c.weights.map_read()
        c.bias.map_read()

        upper = c.err_input.unit_test_mem[c.err_input.shape[0]:].ravel()
        nz = numpy.count_nonzero(numpy.isnan(upper))
        self.assertEqual(
            nz, upper.size,
            "Written some values outside of the target array bounds")

        err_input = c.err_input.mem.ravel()
        weights_derivative = c.weights.mem - weights
        bias_derivative = c.bias.mem - bias

        self.numdiff_check_gd(forward, inp, weights, bias, target,
                              err_input, weights_derivative, bias_derivative,
                              logging.info, self.assertLess,
                              error_function_averaged=False)

        return c.err_input.mem.copy(), c.weights.mem.copy(), c.bias.mem.copy()

    def _do_test_gpu_cpu(self, Forward, GD):
        err_gpu, weights_gpu, bias_gpu = self._do_test(self.device,
                                                       Forward, GD)
        err_cpu, weights_cpu, bias_cpu = self._do_test(None,
                                                       Forward, GD)
        max_diff = numpy.fabs(err_gpu.ravel() - err_cpu.ravel()).max()
        logging.info("err_input difference is %.12f", max_diff)
        self.assertLess(max_diff, 0.0001,
                        "GPU-CPU err_input differs by %.6f" % (max_diff))
        max_diff = numpy.fabs(weights_gpu.ravel() - weights_cpu.ravel()).max()
        logging.info("weights difference is %.12f", max_diff)
        self.assertLess(max_diff, 0.0001,
                        "GPU-CPU weights differs by %.6f" % (max_diff))
        max_diff = numpy.fabs(bias_gpu.ravel() - bias_cpu.ravel()).max()
        logging.info("bias difference is %.12f", max_diff)
        self.assertLess(max_diff, 0.0001,
                        "GPU-CPU bias differs by %.6f" % (max_diff))
        logging.info("All Ok")

    @timeout()
    def test_gpu_cpu_linear(self):
        logging.info("Will test linear gd unit for gpu/cpu correctness")
        self._do_test_gpu_cpu(all2all.All2All, PatchedGradientDescent)

    @timeout()
    def test_gpu_cpu_relu(self):
        logging.info("Will test RELU gd unit for gpu/cpu correctness")
        self._do_test_gpu_cpu(all2all.All2AllRELU, PatchedGDRELU)

    @timeout()
    def test_gpu_cpu_softmax(self):
        logging.info("Will test SoftMax gd unit for gpu/cpu correctness")
        self._do_test_gpu_cpu(all2all.All2AllSoftmax, PatchedGDSoftmax)

    @timeout()
    def test_gpu_cpu_tanh(self):
        logging.info("Will test Tanh gd unit for gpu/cpu correctness")
        self._do_test_gpu_cpu(all2all.All2AllTanh, PatchedGDTanh)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
