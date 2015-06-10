#!/usr/bin/env python3
# -*-coding: utf-8 -*-
"""
.. invisible:
     _   _ _____ _     _____ _____
    | | | |  ___| |   |  ___/  ___|
    | | | | |__ | |   | |__ \ `--.
    | | | |  __|| |   |  __| `--. \
    \ \_/ / |___| |___| |___/\__/ /
     \___/\____/\_____|____/\____/

Created on November 18, 2013

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


import numpy
from veles.backends import NumpyDevice

from veles.config import root
from veles.memory import Array
import veles.opencl_types as opencl_types
import veles.prng as prng
from veles.znicz.gd import (GradientDescent, GDRELU, GDSoftmax, GDTanh,
                            GDSigmoid)
import veles.znicz.all2all as all2all
from veles.znicz.nn_units import GradientDescentBase
from veles.znicz.tests.unit.gd_numdiff import GDNumDiff
from veles.tests import timeout, AcceleratedTest, assign_backend
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


class PatchedGDSigmoid(GDSigmoid, PatchedGradientDescentBase):
    pass


class TestGD(AcceleratedTest, GDNumDiff):
    ABSTRACT = True

    def _do_test(self, device, Forward, GD, weights_transposed):
        batch_size = 2
        input_size = 25
        n_neurons = 7

        prng.get().seed(123)

        dtype = opencl_types.dtypes[root.common.precision_type]
        inp = numpy.zeros([batch_size, input_size], dtype=dtype)
        prng.get().fill(inp)
        forward = Forward(self.parent, output_sample_shape=[n_neurons],
                          weights_transposed=weights_transposed)
        forward.input = Array()
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

        c = GD(self.parent,
               gradient_moment=0, gradient_moment_bias=0,
               learning_rate=-1, weights_decay=0,
               learning_rate_bias=-1, weights_decay_bias=0,
               weights_transposed=weights_transposed)

        c.err_output = Array()
        c.err_output.mem = err_output.copy()
        c.input = Array()
        c.input.mem = inp.copy()
        c.weights = Array()
        c.weights.mem = weights.copy()
        c.bias = Array()
        c.bias.mem = bias.copy()
        c.output = Array()
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
                              self.info, self.assertLess,
                              mean=False)

        return c.err_input.mem.copy(), c.weights.mem.copy(), c.bias.mem.copy()

    def _do_test_gpu_cpu(self, Forward, GD):
        self._do_test_gpu_cpu_transposed(Forward, GD, False)
        self._do_test_gpu_cpu_transposed(Forward, GD, True)

    def _do_test_gpu_cpu_transposed(self, Forward, GD, weights_transposed):
        err_gpu, weights_gpu, bias_gpu = self._do_test(
            self.device, Forward, GD, weights_transposed)
        err_cpu, weights_cpu, bias_cpu = self._do_test(
            NumpyDevice(), Forward, GD, weights_transposed)
        max_diff = numpy.fabs(err_gpu.ravel() - err_cpu.ravel()).max()
        self.info("err_input difference is %.12f", max_diff)
        self.assertLess(max_diff, 0.0001,
                        "GPU-CPU err_input differs by %.6f" % (max_diff))
        max_diff = numpy.fabs(weights_gpu.ravel() - weights_cpu.ravel()).max()
        self.info("weights difference is %.12f", max_diff)
        self.assertLess(max_diff, 0.0001,
                        "GPU-CPU weights differs by %.6f" % (max_diff))
        max_diff = numpy.fabs(bias_gpu.ravel() - bias_cpu.ravel()).max()
        self.info("bias difference is %.12f", max_diff)
        self.assertLess(max_diff, 0.0001,
                        "GPU-CPU bias differs by %.6f" % (max_diff))
        self.info("All Ok")

    @timeout()
    def test_gpu_cpu_linear(self):
        self.info("Will test linear gd unit for gpu/cpu correctness")
        self._do_test_gpu_cpu(all2all.All2All, PatchedGradientDescent)

    @timeout()
    def test_gpu_cpu_relu(self):
        self.info("Will test RELU gd unit for gpu/cpu correctness")
        self._do_test_gpu_cpu(all2all.All2AllRELU, PatchedGDRELU)

    @timeout()
    def test_gpu_cpu_softmax(self):
        self.info("Will test SoftMax gd unit for gpu/cpu correctness")
        self._do_test_gpu_cpu(all2all.All2AllSoftmax, PatchedGDSoftmax)

    @timeout()
    def test_gpu_cpu_tanh(self):
        self.info("Will test Tanh gd unit for gpu/cpu correctness")
        self._do_test_gpu_cpu(all2all.All2AllTanh, PatchedGDTanh)

    @timeout()
    def test_gpu_cpu_sigmoid(self):
        self.info("Will test Sigmoid gd unit for gpu/cpu correctness")
        self._do_test_gpu_cpu(all2all.All2AllSigmoid, PatchedGDSigmoid)


@assign_backend("ocl")
class OpenCLTestGD(TestGD):
    pass


@assign_backend("cuda")
class CUDATestGD(TestGD):
    pass


if __name__ == "__main__":
    AcceleratedTest.main()
