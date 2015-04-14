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

Unit test for convolutional layer forward propagation.

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
import time
from veles.backends import NumpyDevice

from veles.memory import Vector
from veles.tests import AcceleratedTest, assign_backend
from veles.znicz.conv import Conv
import veles.prng as prng
from veles.tests.doubling_reset import patch


class PatchedConv(Conv):
    def __init__(self, workflow, **kwargs):
        super(PatchedConv, self).__init__(workflow, **kwargs)
        patch(self, self.output, lambda: (
            self._batch_size, self._ky_app, self._kx_app,
            self.n_kernels), lambda: self.input.dtype)


class TestConvBase(AcceleratedTest):
    ABSTRACT = True

    def _run_test(self, unit, device, input_data, weights, bias):
        """Run test for specified unit with specified device.

        Tested unit should be an instance of conv.Conv class.
        input_data.shape = [batch_size, y_size, x_size, channels_num]
        weights.shape = [kernels_num, ky, kx]
        bias.shape = [kernels_num]

        Args:
            unit: Veles unit which is tested.
            device: Acceleration device instance.
            input_data: numpy array which is passed to unit as its input.
            weights: numpy array which is passed to unit as NN weights.
            bias: numpy array which is passed to unit as NN bias.

        Returns:
            output: output data of unit.run()
        """
        assert unit.__class__ == PatchedConv
        # set unit input and start initialization
        input_vector = Vector()
        input_vector.mem = input_data
        unit.input = input_vector
        unit.initialize(device=device)

        # set weights and bias using allocated memory during the initialization
        unit.weights.map_invalidate()
        unit.weights.mem[:] = weights.reshape(unit.weights.shape)
        unit.bias.map_invalidate()
        unit.bias.mem[:] = bias.reshape(unit.bias.shape)

        unit.run()
        if not isinstance(device, NumpyDevice):
            unit.output.map_read()
            nz = numpy.count_nonzero(numpy.isnan(
                unit.output.unit_test_mem[unit.output.shape[0]:]))
            self.assertEqual(nz, unit.output.size, "Overflow occured")
        nz = numpy.count_nonzero(numpy.isnan(unit.output.mem))
        self.assertEqual(nz, 0, "NaNs encountered")
        return unit.output.mem

    def _run_check(self, unit, device, input_data, weights, bias, gold_output):
        """Run test for specified unit with specified device and compare result
        with gold_output.

        Tested unit should be an instance of conv.Conv class.
        input_data.shape = [batch_size, y_size, x_size, channels_num]
        weights.shape = [kernels_num, ky, kx]
        bias.shape = [kernels_num]
        gold_output = [batch_size, out_y, out_x, kernels_num]

        Args:
            unit: Veles unit which is tested.
            device: Acceleration device instance.
            input_data: numpy array which is passed to unit as its input.
            weights: numpy array which is passed to unit as NN weights.
            bias: numpy array which is passed to unit as NN bias.
            gold_output: gold result (numpy array) of unit execution

        Raises:
            AssertLess: if unit output is wrong.
        """
        output = self._run_test(unit, device, input_data, weights, bias)
        max_diff = numpy.fabs(output.ravel() -
                              gold_output.ravel()).max()
        self.assertLess(max_diff, 1E-06, "Result differs by %.2e" % (max_diff))

    def _do_trivial_test(self, device, input_shape, weights_shape,
                         sliding=(1, 1), padding=(0, 0, 0, 0)):
        """ Run test, which checks trivial cases with specified padding and
        sliding.

        Args:
            device: acceleration device instance.
            input_shape: Shape of input data. Its format is
                (batch_size, y_size, x_size, channels_num).
            weights_shape: Shape of weights. Its format is
                (kernels_num, ky, kx, channels_num).
            sliding: Kernel sliding for selecting input data. Its format is
                (x_sliding, y_sliging)
            padding: Expands size of input data with zero fields on the each
                side with corresponding value (left, top, right, bottom).
        """
        # calculate x and y size of unit output
        out_y = (input_shape[1] + padding[1] + padding[3] -
                 weights_shape[1]) // sliding[1] + 1
        out_x = (input_shape[2] + padding[0] + padding[2] -
                 weights_shape[2]) // sliding[0] + 1

        unit = PatchedConv(self.parent, n_kernels=weights_shape[0],
                           ky=weights_shape[1], kx=weights_shape[2],
                           sliding=sliding, padding=padding)

        self.info("run conv with input = 0, random weights, random bias...")
        input_data = numpy.zeros(input_shape)
        weights = prng.get().rand(*weights_shape)
        bias = prng.get().rand(weights_shape[0])
        gold_output = numpy.empty((input_shape[0], out_y, out_x,
                                   weights_shape[0]))
        for batch, i, j in ((batch, i, j) for batch in range(input_shape[0])
                            for i in range(out_y) for j in range(out_x)):
            gold_output[batch, i, j, :] = bias[:]
        self._run_check(unit, device, input_data, weights, bias, gold_output)

        self.info("run conv with random input, weights = 0, random bias...")
        input_data = prng.get().rand(*input_shape)
        weights = numpy.zeros(weights_shape)
        bias = prng.get().rand(weights_shape[0])
        gold_output = numpy.empty((input_shape[0], out_y, out_x,
                                   weights_shape[0]))
        for batch, i, j in ((batch, i, j) for batch in range(input_shape[0])
                            for i in range(out_y) for j in range(out_x)):
            gold_output[batch, i, j, :] = bias[:]
        self._run_check(unit, device, input_data, weights, bias, gold_output)


class TestConvNoPadding(TestConvBase):
    """Tests convolutional layer forward propagation without padding and with
    sliding = (1, 1) and linear activation function.
    """

    def test_trivial_cases_ocl(self):
        self.info("start trivial OpenCL test [no padding, "
                  "sliding = (1, 1)]...")
        input_shape = (3, 7, 9, 3)
        weights_shape = (2, 4, 3, 3)
        self._do_trivial_test(self.device, input_shape, weights_shape)
        self.info("TEST PASSED")

    def test_trivial_cases_cpu(self):
        self.info("start trivial CPU test [no padding, "
                  "sliding = (1, 1)]...")
        input_shape = (3, 7, 9, 3)
        weights_shape = (2, 4, 3, 3)
        self._do_trivial_test(NumpyDevice(), input_shape, weights_shape)
        self.info("TEST PASSED")

    def _do_test_all_1(self, device, input_shape, weights_shape):
        """ Run test, which checks result of conv without padding and
        tricky sliding when input data, weights and bias fills with 1.

        Args:
            device: Acceleration device instance.
            input_shape: Shape of input data. Its format is
                (batch_size, y_size, x_size, channels_num).
            weights_shape: Shape of weights. Its format is
                (kernels_num, ky, kx, channels_num).
        """
        # set data size
        sliding = (1, 1)  # (sliding_x, sliding_y)
        padding = (0, 0, 0, 0)  # (left, top, right, bottom)

        # calculate x and y size of unit output
        out_y = (input_shape[1] + padding[1] + padding[3] -
                 weights_shape[1]) // sliding[1] + 1
        out_x = (input_shape[2] + padding[0] + padding[2] -
                 weights_shape[2]) // sliding[0] + 1

        unit = PatchedConv(self.parent, n_kernels=weights_shape[0],
                           ky=weights_shape[1], kx=weights_shape[2],
                           sliding=sliding, padding=padding)

        self.info("run conv with input = 1, weights = 1, bias = 1...")
        input_data = numpy.empty(input_shape)
        input_data.fill(1)
        weights = numpy.empty(weights_shape)
        weights.fill(1)
        bias = numpy.empty(weights_shape[0])
        bias_val = 1
        bias.fill(bias_val)
        gold_output = numpy.empty((input_shape[0], out_y, out_x,
                                   weights_shape[0]))
        gold_output.fill(weights_shape[1] * weights_shape[2] * input_shape[3] +
                         bias_val)
        self._run_check(unit, device, input_data, weights, bias, gold_output)

    def test_all_1_ocl(self):
        self.info("start 'all 1' OpenCL test [no padding, "
                  "sliding = (1, 1)]...")
        input_shape = (3, 7, 9, 3)
        weights_shape = (2, 4, 3, 3)
        self._do_test_all_1(self.device, input_shape, weights_shape)
        self.info("TEST PASSED")

    def test_all_1_cpu(self):
        self.info("start 'all 1' CPU test [no padding, "
                  "sliding = (1, 1)]...")
        input_shape = (3, 7, 9, 3)
        weights_shape = (2, 4, 3, 3)
        self._do_test_all_1(NumpyDevice(), input_shape, weights_shape)
        self.info("TEST PASSED")

    def _do_1_channel_input_test(self, device):
        """ Run test with 1 channel input without padding and with
        sliding = (1, 1).

        Args:
            device: Acceleration device instance.
        """
        input_data = numpy.array([[[[1], [2], [3], [2], [1]],
                                   [[0], [1], [2], [1], [0]],
                                   [[0], [1], [0], [1], [0]],
                                   [[2], [0], [1], [0], [2]],
                                   [[1], [0], [1], [0], [1]]]],
                                 dtype=self._dtype)
        weights = numpy.array([[[-1, -1, -1],
                                [-1, 8, -1],
                                [-1, -1, -1]],
                               [[1.1, 2.1, 3.1],
                                [-1.1, -0.5, 1.3],
                                [1.7, -1.4, 0.05]]], dtype=self._dtype)
        weights = weights.reshape(2, 3, 3, 1)
        bias = numpy.array([10, -10], dtype=self._dtype)
        gold_output = numpy.array([[[[9, 5.3], [15, 5.65], [9, -3.5]],
                                    [[12, 1.25], [3, -2.8], [12, -4.4]],
                                    [[4, -7.05], [15, -7.7], [4, -4.65]]]],
                                  dtype=self._dtype)

        unit = PatchedConv(self.parent, n_kernels=weights.shape[0],
                           kx=3, ky=3)
        self._run_check(unit, device, input_data, weights, bias, gold_output)

    def test_1_channel_input_ocl(self):
        self.info("start OpenCL conv. 1 channel layer forward"
                  "propagation...")
        self._do_1_channel_input_test(self.device)
        self.info("TEST PASSED")

    def test_1_channel_input_cpu(self):
        self.info("start CPU conv. 1 channel layer forward propagation...")
        self._do_1_channel_input_test(NumpyDevice())
        self.info("TEST PASSED")


class TestConvWithPadding(TestConvBase):
    """Tests convolutional layer forward propagation with padding and tricky
    sliding and linear activation function.
    """

    def test_trivial_cases_ocl(self):
        self.info("start trivial OpenCL test (with padding and tricky "
                  "sliding)...")
        input_shape = (3, 7, 9, 3)
        weights_shape = (2, 4, 3, 3)
        sliding = (2, 3)
        padding = (2, 3, 1, 2)
        self._do_trivial_test(self.device, input_shape, weights_shape,
                              sliding, padding)
        self.info("TEST PASSED")

    def test_trivial_cases_cpu(self):
        self.info("start trivial CPU test [no padding, "
                  "sliding = (1, 1)]...")
        input_shape = (3, 7, 9, 3)
        weights_shape = (2, 4, 3, 3)
        sliding = (2, 3)
        padding = (2, 3, 1, 2)
        self._do_trivial_test(NumpyDevice(), input_shape, weights_shape,
                              sliding, padding)
        self.info("TEST PASSED")

    def _do_test_all_1(self, device, input_shape, kernels_num):
        """Run test, which checks result of conv with specified weights shape,
        padding and sliding when input data, weights and bias fills with 1.

        Args:
            device: Acceleration device instance.
            input_shape: Shape of input data. Its format is
                (batch_size, size, size, channels_num). Its important that
                x_size = y_size = size.
                weights_shape = (kernels_num, size, size, channels_num).
                sliding = (size / 2, size / 2)
                padding = (size / 2, size / 2, size / 2, size / 2).
            kernels_num: Number of conv.kernels (weights.shape[0])
        """
        size = input_shape[1]
        channels_num = input_shape[3]
        weights_shape = (kernels_num, size, size, channels_num)
        sliding = (size // 2, size // 2)
        padding = (size // 2, size // 2, size // 2, size // 2)

        # calculate x and y size of unit output
        # out_y = (input_shape[1] + padding[1] + padding[3] -
        #         weights_shape[1]) // sliding[1] + 1
        # out_x = (input_shape[2] + padding[0] + padding[2] -
        #         weights_shape[2]) // sliding[0] + 1

        unit = PatchedConv(self.parent, n_kernels=weights_shape[0],
                           ky=weights_shape[1], kx=weights_shape[2],
                           sliding=sliding, padding=padding)

        input_data = numpy.empty(input_shape)
        input_data.fill(1)
        weights = numpy.empty(weights_shape)
        weights.fill(1)
        bias = numpy.empty(weights_shape[0])
        bias_val = 1
        bias.fill(bias_val)

        gold_output = numpy.empty((input_shape[0], 3, 3, weights_shape[0]))
        quater_sum = size * size / 4 * channels_num
        for batch, k in ((batch, k)
                         for batch in range(input_shape[0])
                         for k in range(kernels_num)):
            gold_output[batch, 1, 1, k] = 4 * quater_sum + bias_val

            gold_output[batch, 0, 0, k] = quater_sum + bias_val
            gold_output[batch, 2, 0, k] = quater_sum + bias_val
            gold_output[batch, 0, 2, k] = quater_sum + bias_val
            gold_output[batch, 2, 2, k] = quater_sum + bias_val

            gold_output[batch, 1, 0, k] = 2 * quater_sum + bias_val
            gold_output[batch, 1, 2, k] = 2 * quater_sum + bias_val
            gold_output[batch, 0, 1, k] = 2 * quater_sum + bias_val
            gold_output[batch, 2, 1, k] = 2 * quater_sum + bias_val

        self._run_check(unit, device, input_data, weights, bias, gold_output)

    def test_all_1_ocl(self):
        self.info("start 'all 1' OpenCL test [with padding and sliding...")
        input_shape = (3, 8, 8, 3)
        kernels_num = 2
        self._do_test_all_1(self.device, input_shape, kernels_num)
        self.info("TEST PASSED")

    def test_all_1_cpu(self):
        self.info("start 'all 1' CPU test [with padding and sliding...")
        input_shape = (3, 8, 8, 3)
        kernels_num = 2
        self._do_test_all_1(NumpyDevice(), input_shape, kernels_num)
        self.info("TEST PASSED")

    def _do_test_fixed_arrays(self, device):
        """ Run test with fixed input data, weights and bias with tricky
        padding and sliding.

        Args:
            device: Acceleration device instance.
        """
        input_data = numpy.array([[[[1], [2], [3], [2], [1]],
                                   [[0], [1], [2], [1], [0]],
                                   [[0], [1], [0], [1], [0]],
                                   [[2], [0], [1], [0], [2]],
                                   [[1], [0], [1], [0], [1]]]],
                                 dtype=self._dtype)

        weights = numpy.array([[[-1, -1, -1],
                                [-1, 8, -1],
                                [-1, -1, -1]],
                               [[1.1, 2.1, 3.1],
                                [-1.1, -0.5, 1.3],
                                [1.7, -1.4, 0.05]]], dtype=self._dtype)
        weights = weights.reshape(2, 3, 3, 1)
        bias = numpy.array([10, -10], dtype=self._dtype)
        gold_output = numpy.array(
            [[[[7, -11.3], [3, -10.7], [7, -8], [10, -10]],
              [[6, -8.4], [3, -2.8], [6, -12.8], [10, -10]],
              [[9, -7.9], [9, -7.9], [9, -7.9], [10, -10]]]],
            dtype=self._dtype)

        unit = PatchedConv(self.parent, n_kernels=2, kx=3, ky=3,
                           padding=(1, 2, 3, 4), sliding=(2, 3))
        self._run_check(unit, device, input_data, weights, bias, gold_output)

    def test_fixed_arrays_ocl(self):
        self.info("start testing OpenCL conv. layer forward propagation "
                  "with fixed input data, weights and bias...")
        self._do_test_fixed_arrays(self.device)
        self.info("TEST PASSED")

    def test_fixed_arrays_cpu(self):
        self.info("start testing CPU conv. layer forward propagation "
                  "with fixed input data, weights and bias...")
        self._do_test_fixed_arrays(NumpyDevice())
        self.info("TEST PASSED")

    def test_compare_ocl_vs_cpu(self):
        self._ocl_vs_cpu_transposed(False)
        self._ocl_vs_cpu_transposed(True)

    def _ocl_vs_cpu_transposed(self, weights_transposed):
        """Run test with random input data, weights, bias to compare results of
        CPU and OpenCL versions of algorithm and execution time.
        """
        input_shape = (2, 256, 256, 3)
        weights_shape = (2, 4, 3, 3)
        sliding = (2, 3)
        padding = (1, 3, 2, 4)
        input_data = prng.get().rand(*input_shape)
        weights = prng.get().rand(*weights_shape)
        bias = prng.get().rand(weights_shape[0])

        unit = PatchedConv(self.parent, n_kernels=weights_shape[0],
                           ky=weights_shape[1], kx=weights_shape[2],
                           sliding=sliding, padding=padding,
                           weights_transposed=weights_transposed)
        time0 = time.time()
        ocl_output = self._run_test(unit, self.device, input_data,
                                    weights, bias)
        time1 = time.time()

        numpy_output = self._run_test(unit, NumpyDevice(), input_data, weights,
                                      bias)
        time2 = time.time()
        self.info("OpenCL is faster than CPU in %.4f times",
                  (time2 - time1) / (time1 - time0))
        max_diff = numpy.fabs(ocl_output.ravel() - numpy_output.ravel()).max()
        self.assertLess(max_diff, 1E-06, "Result differs by %.2e" % max_diff)


@assign_backend("ocl")
class OpenCLTestConvNoPadding(TestConvNoPadding):
    pass


@assign_backend("ocl")
class OpenCLTestConvWithPadding(TestConvWithPadding):
    pass


@assign_backend("cuda")
class CUDATestConvNoPadding(TestConvNoPadding):
    pass


@assign_backend("cuda")
class CUDATestConvWithPadding(TestConvWithPadding):
    pass


if __name__ == "__main__":
    AcceleratedTest.main()
