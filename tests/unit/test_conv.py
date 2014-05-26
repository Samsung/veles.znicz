"""
Created on Nov 7, 2013

Unit test for convolutional layer forward propagation.

Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""


import logging
import numpy
import scipy.signal
import time
import unittest
import operator

from veles.config import root
import veles.formats as formats
import veles.opencl as opencl
import veles.opencl_types as opencl_types
import veles.random_generator as rnd
import veles.znicz.conv as conv
from veles.tests.dummy_workflow import DummyWorkflow


class TestConvBase(unittest.TestCase):
    def setUp(self):
        root.common.unit_test = True
        root.common.plotters_disabled = True
        self._dtype = opencl_types.dtypes[root.common.dtype]

    def tearDown(self):
        pass

    def _run_test(self, unit, device, input_data, weights, bias,
                 gold_output):
        """Run test for specified unit with specified device.

        Tested unit should be an instance of conv.Conv class.
        input_data.shape = [batch_size, y_size, x_size, channels_num]
        weights.shape = [kernels_num, ky, kx]
        bias.shape = [kernels_num]
        gold_output = [batch_size, out_y, out_x, kernels_num]

        Args:
            unit: Veles unit which is tested.
            device: OpenCL device instance (if equals to None - CPU version of
                algorithm should be run).
            input_data: numpy array which is passed to unit as its input.
            weights: numpy array which is passed to unit as NN weights.
            bias: numpy array which is passed to unit as NN bias.

        Raises:
            LessError: if unit output is wrong.
        """
        assert(unit.__class__ == conv.Conv)
        # set unit input and start initialization
        input_vector = formats.Vector()
        input_vector.mem = input_data
        unit.input = input_vector
        unit.initialize(device=device)

        # set weights and bias using allocated memory during the initialization
        unit.weights.map_invalidate()
        unit.weights.mem[:] = weights.reshape(unit.weights.mem.shape)
        unit.bias.map_invalidate()
        unit.bias.mem[:] = bias.reshape(unit.bias.mem.shape)

        unit.run()
        unit.output.map_read()
        max_diff = numpy.fabs(unit.output.mem.ravel() -
                              gold_output.ravel()).max()
        self.assertLess(max_diff, 1E-06, "Result differs by %.2e" % (max_diff))

    def _do_trivial_test(self, device, input_shape, weights_shape,
                         sliding=(1, 1), padding=(0, 0, 0, 0)):
        """ Run test, which checks trivial cases with specified padding and
        sliding.

        Args:
            device: OpenCL device instance (if value is equal to None, CPU
                version of algorithm should be run. In other case -
                OpenCL version).
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

        unit = conv.Conv(DummyWorkflow(), n_kernels=weights_shape[0],
                         ky=weights_shape[1], kx=weights_shape[2],
                         sliding=sliding, padding=padding)

        logging.info("run conv with input = 0, random weights, random bias...")
        input_data = numpy.zeros(input_shape)
        weights = numpy.random.uniform(
                size=numpy.prod(weights_shape)).reshape(weights_shape)
        bias = numpy.random.uniform(size=weights_shape[0])
        gold_output = numpy.empty((input_shape[0], out_y, out_x,
                                   weights_shape[0]))
        for batch, i, j in ((batch, i, j) for batch in range(input_shape[0])
                            for i in range(out_y) for j in range(out_x)):
            gold_output[batch, i, j, :] = bias[:]
        self._run_test(unit, device, input_data, weights, bias, gold_output)

        logging.info("run conv with random input, weights = 0, random bias...")
        input_data = numpy.random.uniform(
                size=numpy.prod(input_shape)).reshape(input_shape)
        weights = numpy.zeros(weights_shape)
        bias = numpy.random.uniform(size=weights_shape[0])
        gold_output = numpy.empty((input_shape[0], out_y, out_x,
                                   weights_shape[0]))
        for batch, i, j in ((batch, i, j) for batch in range(input_shape[0])
                            for i in range(out_y) for j in range(out_x)):
            gold_output[batch, i, j, :] = bias[:]
        self._run_test(unit, device, input_data, weights, bias, gold_output)


class TestConvNoPadding(TestConvBase):
    """Tests convolutional layer forward propagation without padding and with
    sliding = (1, 1) and linear activation function.
    """

    def test_trivial_cases_ocl(self):
        logging.info("start trivial OpenCL test [no padding, "
                     "sliding = (1, 1)]...")
        input_shape = (3, 7, 9, 3)
        weights_shape = (2, 4, 3, 3)
        self._do_trivial_test(opencl.Device(), input_shape, weights_shape)
        logging.info("TEST PASSED")

    def test_trivial_cases_cpu(self):
        logging.info("start trivial CPU test [no padding, "
                     "sliding = (1, 1)]...")
        input_shape = (3, 7, 9, 3)
        weights_shape = (2, 4, 3, 3)
        self._do_trivial_test(None, input_shape, weights_shape)
        logging.info("TEST PASSED")

    def _do_test_all_1(self, device, input_shape, weights_shape):
        """ Run test, which checks result of conv without padding and
        tricky sliding when input data, weights and bias fills with 1.

        Args:
            device: OpenCL device instance (if value is equal to None, CPU
                version of algorithm should be run. In other case -
                OpenCL version).
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

        unit = conv.Conv(DummyWorkflow(), n_kernels=weights_shape[0],
                         ky=weights_shape[1], kx=weights_shape[2],
                         sliding=sliding, padding=padding)

        logging.info("run conv with input = 1, weights = 1, bias = 1...")
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
        self._run_test(unit, device, input_data, weights, bias, gold_output)

    def test_all_1_ocl(self):
        logging.info("start 'all 1' OpenCL test [no padding, "
                     "sliding = (1, 1)]...")
        input_shape = (3, 7, 9, 3)
        weights_shape = (2, 4, 3, 3)
        self._do_test_all_1(opencl.Device(), input_shape, weights_shape)
        logging.info("TEST PASSED")

    def test_all_1_cpu(self):
        logging.info("start 'all 1' CPU test [no padding, "
                     "sliding = (1, 1)]...")
        input_shape = (3, 7, 9, 3)
        weights_shape = (2, 4, 3, 3)
        self._do_test_all_1(None, input_shape, weights_shape)
        logging.info("TEST PASSED")

    def _do_1_channel_input_test(self, device):
        """ Run test with 1 channel input without padding and with
        sliding = (1, 1).

        Args:
            device: OpenCL device instance (if value is equal to None, CPU
                version of algorithm should be run. In other case -
                OpenCL version).
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

        unit = conv.Conv(DummyWorkflow(), n_kernels=weights.shape[0], kx=3,
                         ky=3)
        self._run_test(unit, device, input_data, weights, bias, gold_output)

    def test_1_channel_input_ocl(self):
        logging.info("start OpenCL conv. 1 channel layer forward"
                     "propagation...")
        self._do_1_channel_input_test(opencl.Device())
        logging.info("TEST PASSED")

    def test_1_channel_input_cpu(self):
        logging.info("start CPU conv. 1 channel layer forward propagation...")
        self._do_1_channel_input_test(None)
        logging.info("TEST PASSED")


class TestConvWithPadding(TestConvBase):
    """Tests convolutional layer forward propagation with padding and tricky
    sliding and linear activation function.
    """

    def test_trivial_cases_ocl(self):
        logging.info("start trivial OpenCL test (with padding and tricky "
                     "sliding)...")
        input_shape = (3, 7, 9, 3)
        weights_shape = (2, 4, 3, 3)
        sliding = (2, 3)
        padding = (2, 3, 1, 2)
        self._do_trivial_test(opencl.Device(), input_shape, weights_shape,
                              sliding, padding)
        logging.info("TEST PASSED")

    def test_trivial_cases_cpu(self):
        logging.info("start trivial CPU test [no padding, "
                     "sliding = (1, 1)]...")
        input_shape = (3, 7, 9, 3)
        weights_shape = (2, 4, 3, 3)
        sliding = (2, 3)
        padding = (2, 3, 1, 2)
        self._do_trivial_test(None, input_shape, weights_shape, sliding,
                              padding)
        logging.info("TEST PASSED")

    def _do_test_all_1(self, device, input_shape, kernels_num):
        """Run test, which checks result of conv with specified weights shape,
        padding and sliding when input data, weights and bias fills with 1.

        Args:
            device: OpenCL device instance (if value is equal to None, CPU
                version of algorithm should be run. In other case -
                OpenCL version).
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
        out_y = (input_shape[1] + padding[1] + padding[3] -
                 weights_shape[1]) // sliding[1] + 1
        out_x = (input_shape[2] + padding[0] + padding[2] -
                 weights_shape[2]) // sliding[0] + 1

        unit = conv.Conv(DummyWorkflow(), n_kernels=weights_shape[0],
                         ky=weights_shape[1], kx=weights_shape[2],
                         sliding=sliding, padding=padding)

        logging.info("run conv with input = 1, weights = 1, bias = 1...")
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

        self._run_test(unit, device, input_data, weights, bias, gold_output)

    def test_all_1_ocl(self):
        logging.info("start 'all 1' OpenCL test [with padding and sliding...")
        input_shape = (3, 8, 8, 3)
        kernels_num = 2
        self._do_test_all_1(opencl.Device(), input_shape, kernels_num)
        logging.info("TEST PASSED")

    def test_all_1_cpu(self):
        logging.info("start 'all 1' CPU test [with padding and sliding...")
        input_shape = (3, 8, 8, 3)
        kernels_num = 2
        self._do_test_all_1(None, input_shape, kernels_num)
        logging.info("TEST PASSED")

    def _do_test_padding_sliding(self):
        # TODO: refactor this old impl.
        logging.info("Will test convolutional layer forward propagation")

        inp = formats.Vector()
        dtype = opencl_types.dtypes[root.common.dtype]
        inp.mem = numpy.array([[[1, 2, 3, 2, 1],
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

        c = conv.Conv(DummyWorkflow(), n_kernels=2, kx=3, ky=3,
                      padding=(1, 2, 3, 4), sliding=(2, 3))
        c.input = inp

        c.initialize(device=self.device)

        c.weights.map_invalidate()  # rewrite weights
        c.weights.mem[:] = weights.reshape(c.weights.mem.shape)[:]
        c.bias.map_invalidate()  # rewrite bias
        c.bias.mem[:] = bias[:]

        c.run()
        c.output.map_read()  # get results back
        nz = numpy.count_nonzero(c.output.vv[c.output.mem.shape[0]:].ravel())
        self.assertEqual(nz, 0, "Overflow occured")

        y = c.output.mem.ravel()
        t = numpy.array([[[7, -11.3], [3, -10.7], [7, -8], [10, -10]],
                         [[6, -8.4], [3, -2.8], [6, -12.8], [10, -10]],
                         [[9, -7.9], [9, -7.9], [9, -7.9], [10, -10]]],
                        dtype=dtype).ravel()
        max_diff = numpy.fabs(t - y).max()
        self.assertLess(max_diff, 0.0001,
                        "Result differs by %.6f" % (max_diff))

        logging.info("All Ok")

    def test_compare_ocl_vs_cpu(self):
        # TODO: implement
        pass


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    unittest.main()
