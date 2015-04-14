# -*- coding: utf-8 -*-
"""
.. invisible:
     _   _ _____ _     _____ _____
    | | | |  ___| |   |  ___/  ___|
    | | | | |__ | |   | |__ \ `--.
    | | | |  __|| |   |  __| `--. \
    \ \_/ / |___| |___| |___/\__/ /
     \___/\____/\_____|____/\____/

Created on Nov 7, 2013

Unit test for convolutional layer back propagation.

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
from veles.dummy import DummyUnit
from veles.memory import assert_addr, Vector
import veles.opencl_types as opencl_types
from veles.tests import AcceleratedTest, assign_backend
from veles.znicz.gd_conv import GradientDescentConv, GDRELUConv, \
    GDStrictRELUConv, GDTanhConv, GDSigmoidConv
import veles.znicz.conv as conv
import veles.prng as prng
from veles.znicz.tests.unit.gd_numdiff import GDNumDiff
from veles.znicz.tests.unit.test_gd import PatchedGradientDescentBase


class PatchedGradientDescentConv(GradientDescentConv,
                                 PatchedGradientDescentBase):
    pass


class PatchedGDRELUConv(GDRELUConv, PatchedGradientDescentBase):
    pass


class PatchedGDStrictRELUConv(GDStrictRELUConv, PatchedGradientDescentBase):
    pass


class PatchedGDTanhConv(GDTanhConv, PatchedGradientDescentBase):
    pass


class PatchedGDSigmoidConv(GDSigmoidConv, PatchedGradientDescentBase):
    pass


class TestGDConv(AcceleratedTest, GDNumDiff):
    ABSTRACT = True

    def test_err_h_gpu(self):
        self._test_err_h(self.device)

    def test_err_h_cpu(self):
        self._test_err_h(NumpyDevice())

    def _test_err_h(self, device):
        self.info("Will test convolutional layer back propagation")

        dtype = opencl_types.dtypes[root.common.precision_type]
        inp = numpy.array([[[-1, 0, 2, 0, 3],
                            [0, 1, -2, 1, 2],
                            [2, 0, 1, 1, 0],
                            [-1, 1, 1, 0, 2],
                            [1, 0, 1, 0, 1]],

                           [[-1, 0, 2, 0, 3],
                            [0, 1, -2, 1, 2],
                            [2, 0, 1, 1, 0],
                            [-1, 1, 1, 0, 2],
                            [1, 0, 1, 0, 1]]], dtype=dtype)
        inp.shape = inp.shape + (1,)

        a = numpy.array([[-1, 0, 2, 0, 1, -2, 2, 0, 1],
                         [0, 2, 0, 1, -2, 1, 0, 1, 1],
                         [2, 0, 3, -2, 1, 2, 1, 1, 0],
                         [0, 1, -2, 2, 0, 1, -1, 1, 1],
                         [1, -2, 1, 0, 1, 1, 1, 1, 0],
                         [-2, 1, 2, 1, 1, 0, 1, 0, 2],
                         [2, 0, 1, -1, 1, 1, 1, 0, 1],
                         [0, 1, 1, 1, 1, 0, 0, 1, 0],
                         [1, 1, 0, 1, 0, 2, 1, 0, 1],

                         [-1, 0, 2, 0, 1, -2, 2, 0, 1],
                         [0, 2, 0, 1, -2, 1, 0, 1, 1],
                         [2, 0, 3, -2, 1, 2, 1, 1, 0],
                         [0, 1, -2, 2, 0, 1, -1, 1, 1],
                         [1, -2, 1, 0, 1, 1, 1, 1, 0],
                         [-2, 1, 2, 1, 1, 0, 1, 0, 2],
                         [2, 0, 1, -1, 1, 1, 1, 0, 1],
                         [0, 1, 1, 1, 1, 0, 0, 1, 0],
                         [1, 1, 0, 1, 0, 2, 1, 0, 1]], dtype=dtype)

        weights = numpy.array([[-1, -1, 4, 1, 8, -1, -1, 3, 2],
                               [2, 1, -1, 3, 0, 1, 4, 1, 3]], dtype=dtype)

        bias = numpy.array([10, -10], dtype=dtype)

        err_output = numpy.array([[[[-1, 3], [8, 2], [0, 1]],
                                   [[4, -1], [-1, 2], [0, 1]],
                                   [[-2, 3], [1, 2], [1, 1]]],

                                  [[[-1, 3], [8, 2], [0, 1]],
                                   [[4, -1], [-1, 2], [0, 1]],
                                   [[-2, 3], [1, 2], [1, 1]]]], dtype=dtype)

        c = PatchedGradientDescentConv(self.parent, gradient_moment=0.9)
        u = DummyUnit(kx=3, ky=3, n_kernels=2,
                      padding=(0, 0, 0, 0), sliding=(1, 1),
                      err_output=Vector(err_output.copy()),
                      input=Vector(inp.copy()),
                      weights=Vector(weights.copy()),
                      bias=Vector(bias.copy()),
                      output=Vector(err_output.copy()),
                      unpack_size=1, unpack_data=Vector())
        c.link_conv_attrs(u)
        c.link_attrs(u, "err_output", "input", "output", "weights", "bias")

        batch_size = c.err_output.shape[0]
        b = c.err_output.mem.reshape(9 * batch_size, 2)
        gradient_weights = numpy.dot(b.transpose(), a)
        weights_derivative = gradient_weights.copy()
        gradient_weights *= -c.learning_rate
        gradient_weights += weights * (-1) * (c.learning_rate *
                                              c.weights_decay)
        weights_new = weights + gradient_weights
        bias_derivative = b.sum(axis=0)
        gradient_bias = bias_derivative * (-c.learning_rate_bias)
        bias_new = bias + gradient_bias

        c.initialize(device=device)
        c.run()
        c.err_input.map_read()
        c.weights.map_read()
        c.bias.map_read()

        t = numpy.array([[[7, 0, -11, 31, -1],
                          [2, 6, 93, -11, 0],
                          [22, 45, 18, 28, 7],
                          [-1, 11, 25, 14, 3],
                          [14, 4, 13, 12, 5]],

                         [[7, 0, -11, 31, -1],
                          [2, 6, 93, -11, 0],
                          [22, 45, 18, 28, 7],
                          [-1, 11, 25, 14, 3],
                          [14, 4, 13, 12, 5]]], dtype=dtype)
        max_diff = numpy.fabs(t.ravel() - c.err_input.mem.ravel()).max()
        self.assertLess(max_diff, 0.0001,
                        "Err_input differs by %.6f" % (max_diff))
        self.info("Err_input Ok")

        max_diff = numpy.fabs(weights_new.ravel() -
                              c.weights.mem.ravel()).max()
        self.assertLess(max_diff, 0.0001,
                        "Weights differ by %.6f" % (max_diff))
        self.info("Weights Ok")

        max_diff = numpy.fabs(bias_new.ravel() - c.bias.mem.ravel()).max()
        self.assertLess(max_diff, 0.0001,
                        "Bias differs by %.6f" % (max_diff))
        self.info("Bias Ok")

        err_input = c.err_input.mem.ravel()
        forward = conv.Conv(self.parent, n_kernels=2, kx=3, ky=3)
        target = self._numdiff_init_forward(forward, inp, weights, bias,
                                            err_output)

        self.numdiff_check_gd(forward, inp, weights, bias, target,
                              err_input, weights_derivative, bias_derivative,
                              self.info, self.assertLess,
                              error_function_averaged=False)

    def _numdiff_init_forward(self, forward, inp, weights, bias, err_output):
        forward.input = Vector()
        forward.input.mem = inp.copy()
        forward.initialize(device=self.device)
        forward.weights.map_invalidate()
        forward.weights.mem[:] = weights[:]
        forward.bias.map_invalidate()
        forward.bias.mem[:] = bias[:]
        forward.run()
        forward.output.map_read()
        target = forward.output.mem.ravel() - err_output.ravel()
        return target

    def test_padding_sliding_gpu(self):
        self._test_padding_sliding(self.device)

    def test_padding_sliding_cpu(self):
        self._test_padding_sliding(NumpyDevice())

    def _test_padding_sliding(self, device):
        self.info("Will test convolutional layer back propagation")

        dtype = opencl_types.dtypes[root.common.precision_type]
        inp = numpy.array([[[1, 2, 3, 2, 1],
                            [0, 1, 2, 1, 0],
                            [0, 1, 0, 1, 0],
                            [2, 0, 1, 0, 2],
                            [1, 0, 1, 0, 1]]], dtype=dtype)
        inp.shape = inp.shape + (1,)

        a = numpy.array([[0, 0, 0, 0, 0, 0, 0, 1, 2],
                         [0, 0, 0, 0, 0, 0, 2, 3, 2],
                         [0, 0, 0, 0, 0, 0, 2, 1, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 1, 0, 0, 1, 0, 2, 0],
                         [1, 2, 1, 1, 0, 1, 0, 1, 0],
                         [1, 0, 0, 1, 0, 0, 0, 2, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 1, 0, 0, 0, 0, 0, 0, 0],
                         [0, 1, 0, 0, 0, 0, 0, 0, 0],
                         [0, 1, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=dtype)

        weights = numpy.array([[-1, -1, -1, -1, 8, -1, -1, -1, -1],
                               [1.1, 2.1, 3.1, -1.1, -0.5, 1.3, 1.7, -1.4,
                                0.05]], dtype=dtype)

        bias = numpy.array([10, -10], dtype=dtype)

        err_output = numpy.array([[[[-1, 3], [8, 2], [0, 1], [4, -1]],
                                   [[-1, 2], [0, 1], [-2, 3], [1, 2]],
                                   [[1, 1], [1, -2], [0, 5], [2, 3]]]],
                                 dtype=dtype)

        c = PatchedGradientDescentConv(self.parent, gradient_moment=0.9)
        u = DummyUnit(n_kernels=2, kx=3, ky=3,
                      padding=(1, 2, 3, 4), sliding=(2, 3),
                      err_output=Vector(err_output.copy()),
                      input=Vector(inp.copy()),
                      weights=Vector(weights.copy()),
                      bias=Vector(bias.copy()),
                      output=Vector(err_output.copy()),
                      unpack_size=1, unpack_data=Vector())
        c.link_conv_attrs(u)
        c.link_attrs(u, "err_output", "input", "output", "weights", "bias")

        batch_size = c.err_output.shape[0]
        b = c.err_output.mem.reshape(12 * batch_size, 2)
        gradient_weights = numpy.dot(b.transpose(), a)
        weights_derivative = gradient_weights.copy()
        gradient_weights *= -c.learning_rate
        gradient_weights += weights * (-1) * (c.learning_rate *
                                              c.weights_decay)
        weights_new = weights + gradient_weights
        bias_derivative = b.sum(axis=0)
        gradient_bias = bias_derivative * (-c.learning_rate_bias)
        bias_new = bias + gradient_bias

        c.initialize(device=device)
        c.run()
        c.err_input.map_read()
        c.weights.map_read()
        c.bias.map_read()

        t = numpy.array([[[-3.2, -3.45, -10.8, -6.2, -1.4],
                          [5.2, 8.3, 2.1, 8.4, 8.3],
                          [-9, 2.5, -0.5, 0, -17.5],
                          [-1.8, 2.8, -1.4, 7.15, -2.2],
                          [1.1, -1.1, -5.2, -1.7, 10.5]]], dtype=dtype)
        max_diff = numpy.fabs(t.ravel() - c.err_input.mem.ravel()).max()
        self.assertLess(max_diff, 0.0001,
                        "Err_input differs by %.6f\nTarget is:\n%s\nGot:\n%s" %
                        (max_diff, " ".join("%.2f" % x for x in t.ravel()),
                         " ".join("%.2f" % x for x in
                                  c.err_input.mem.ravel())))
        self.info("Err_input Ok")

        max_diff = numpy.fabs(weights_new.ravel() -
                              c.weights.mem.ravel()).max()
        self.assertLess(max_diff, 0.0001,
                        "Weights differ by %.6f" % (max_diff))
        self.info("Weights Ok")

        max_diff = numpy.fabs(bias_new.ravel() - c.bias.mem.ravel()).max()
        self.assertLess(max_diff, 0.0001,
                        "Bias differs by %.6f" % (max_diff))
        self.info("Bias Ok")

        err_input = c.err_input.mem.ravel()
        forward = conv.Conv(self.parent, n_kernels=2, kx=3, ky=3,
                            padding=(1, 2, 3, 4), sliding=(2, 3))
        target = self._numdiff_init_forward(forward, inp, weights, bias,
                                            err_output)

        self.numdiff_check_gd(forward, inp, weights, bias, target,
                              err_input, weights_derivative, bias_derivative,
                              self.info, self.assertLess,
                              error_function_averaged=False)

    def test_random_numeric_gpu(self):
        self._test_random_numeric(self.device, conv.Conv,
                                  PatchedGradientDescentConv)

    def test_random_numeric_gpu_tanh(self):
        self._test_random_numeric(self.device, conv.ConvTanh,
                                  PatchedGDTanhConv)

    def test_random_numeric_gpu_sigmoid(self):
        self._test_random_numeric(self.device, conv.ConvSigmoid,
                                  PatchedGDSigmoidConv)

    def test_random_numeric_gpu_relu(self):
        self._test_random_numeric(self.device, conv.ConvRELU,
                                  PatchedGDRELUConv)

    def test_random_numeric_cpu(self):
        self._test_random_numeric(NumpyDevice(), conv.Conv,
                                  PatchedGradientDescentConv)

    def test_random_numeric_cpu_tanh(self):
        self._test_random_numeric(NumpyDevice(), conv.ConvTanh,
                                  PatchedGDTanhConv)

    def test_random_numeric_cpu_sigmoid(self):
        self._test_random_numeric(NumpyDevice(), conv.ConvSigmoid,
                                  PatchedGDSigmoidConv)

    def test_random_numeric_cpu_relu(self):
        self._test_random_numeric(NumpyDevice(), conv.ConvRELU,
                                  PatchedGDRELUConv)

    def _test_random_numeric(self, device, Forward, GD):
        self._test_random_numeric_transposed(device, Forward, GD, False)
        self._test_random_numeric_transposed(device, Forward, GD, True)

    def _test_random_numeric_transposed(self, device, Forward, GD,
                                        weights_transposed):
        self.info("Will test convolutional layer forward-backward "
                  "via numeric differentiation (weights_transposed = %s)",
                  str(weights_transposed))

        dtype = opencl_types.dtypes[root.common.precision_type]
        inp = numpy.zeros([2, 5, 5, 3], dtype=dtype)
        prng.get().fill(inp)
        forward = Forward(self.parent, n_kernels=2, kx=3, ky=3,
                          padding=(1, 2, 3, 4), sliding=(2, 3),
                          weights_transposed=weights_transposed)
        sh = list(inp.shape)
        sh[0] <<= 1
        forward.input = Vector(numpy.zeros(sh, dtype=dtype))
        forward.input.initialize(device)
        forward.input.map_write()
        forward.input.unit_test_mem = forward.input.mem
        sh[0] >>= 1
        forward.input.mem = forward.input.unit_test_mem[:sh[0]]
        assert_addr(forward.input.mem, forward.input.unit_test_mem)
        forward.input.mem[:] = inp[:]
        forward.input.unit_test_mem[sh[0]:] = numpy.nan
        forward.initialize(device=device)
        forward.run()

        forward.output.map_read()
        target = numpy.zeros_like(forward.output.mem)
        prng.get().fill(target)
        out = forward.output.mem.copy()
        err_output = out - target
        forward.weights.map_read()
        weights = forward.weights.mem.copy()
        forward.bias.map_read()
        bias = forward.bias.mem.copy()

        c = GD(
            self.parent,
            gradient_moment=0, gradient_moment_bias=0,
            learning_rate=-1, weights_decay=0,
            learning_rate_bias=-1, weights_decay_bias=0,
            weights_transposed=weights_transposed)
        c.link_conv_attrs(forward)
        c.link_attrs(forward, "input", "output", "weights", "bias")
        sh = list(err_output.shape)
        sh[0] <<= 1
        c.err_output = Vector(numpy.zeros(sh, dtype=dtype))
        c.err_output.initialize(device)
        c.err_output.map_write()
        c.err_output.unit_test_mem = c.err_output.mem
        sh[0] >>= 1
        c.err_output.mem = c.err_output.unit_test_mem[:sh[0]]
        assert_addr(c.err_output.mem, c.err_output.unit_test_mem)
        c.err_output.mem[:] = err_output[:]
        c.err_output.unit_test_mem[sh[0]:] = numpy.nan
        c.initialize(device)
        c.run()
        c.err_input.map_read()
        c.weights.map_read()
        c.bias.map_read()

        nz = numpy.count_nonzero(numpy.isnan(c.err_input.mem))
        self.assertEqual(nz, 0, "NaNs encountered in err_input")

        nz = numpy.count_nonzero(numpy.isnan(c.weights.mem))
        self.assertEqual(nz, 0, "NaNs encountered in weights")

        nz = numpy.count_nonzero(numpy.isnan(c.bias.mem))
        self.assertEqual(nz, 0, "NaNs encountered in bias")

        err_input = c.err_input.mem.ravel()
        weights_derivative = c.weights.mem - weights
        bias_derivative = c.bias.mem - bias

        self.numdiff_check_gd(forward, inp, weights, bias, target,
                              err_input, weights_derivative, bias_derivative,
                              self.info, self.assertLess,
                              error_function_averaged=False)


@assign_backend("ocl")
class OpenCLTestGDConv(TestGDConv):
    pass


@assign_backend("cuda")
class CUDATestGDConv(TestGDConv):
    pass


if __name__ == "__main__":
    AcceleratedTest.main()
