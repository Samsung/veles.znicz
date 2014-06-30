"""
Created on Nov 7, 2013

Unit test for convolutional layer back propagation.

Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""


import logging
import numpy
import unittest

from veles.config import root
import veles.formats as formats
import veles.opencl_types as opencl_types
import veles.znicz.gd_conv as gd_conv
import veles.znicz.conv as conv
from veles.tests.dummy_workflow import DummyWorkflow
import veles.random_generator as prng
import veles.opencl as opencl
from veles.znicz.tests.unit.gd_numdiff import GDNumDiff


class TestGDConv(unittest.TestCase, GDNumDiff):
    def setUp(self):
        root.common.unit_test = True
        root.common.plotters_disabled = True
        if not hasattr(self, "device"):
            self.device = opencl.Device()

    def test_err_h_gpu(self):
        self._test_err_h(self.device)

    def test_err_h_cpu(self):
        self._test_err_h(None)

    def _test_err_h(self, device):
        logging.info("Will test convolutional layer back propagation")

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

        c = gd_conv.GradientDescentConv(DummyWorkflow(), n_kernels=2,
                                        kx=3, ky=3, gradient_moment=0.9,
                                        error_function_averaged=True)
        c.err_output = formats.Vector()
        c.err_output.mem = err_output.copy()
        c.input = formats.Vector()
        c.input.mem = inp.copy()
        c.weights = formats.Vector()
        c.weights.mem = weights.copy()
        c.bias = formats.Vector()
        c.bias.mem = bias.copy()
        c.output = formats.Vector()
        c.output.mem = c.err_output.mem.copy()

        batch_size = c.err_output.mem.shape[0]
        b = c.err_output.mem.reshape(9 * batch_size, 2)
        gradient_weights = numpy.dot(b.transpose(), a) / err_output.shape[0]
        weights_derivative = gradient_weights.copy()
        gradient_weights *= -c.learning_rate
        gradient_weights += weights * (-1) * (c.learning_rate *
                                              c.weights_decay)
        weights_new = weights + gradient_weights
        bias_derivative = b.sum(axis=0) / err_output.shape[0]
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
        t /= t.shape[0]
        max_diff = numpy.fabs(t.ravel() - c.err_input.mem.ravel()).max()
        self.assertLess(max_diff, 0.0001,
                        "Err_input differs by %.6f" % (max_diff))
        logging.info("Err_input Ok")

        max_diff = numpy.fabs(weights_new.ravel() -
                              c.weights.mem.ravel()).max()
        self.assertLess(max_diff, 0.0001,
                        "Weights differ by %.6f" % (max_diff))
        logging.info("Weights Ok")

        max_diff = numpy.fabs(bias_new.ravel() - c.bias.mem.ravel()).max()
        self.assertLess(max_diff, 0.0001,
                        "Bias differs by %.6f" % (max_diff))
        logging.info("Bias Ok")

        err_input = c.err_input.mem.ravel()
        forward = conv.Conv(DummyWorkflow(), n_kernels=2, kx=3, ky=3)
        target = self._numdiff_init_forward(forward, inp, weights, bias,
                                            err_output)

        self.numdiff_check_gd(forward, inp, weights, bias, target,
                              err_input, weights_derivative, bias_derivative,
                              logging.info, self.assertLess)

    def _numdiff_init_forward(self, forward, inp, weights, bias, err_output):
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
        return target

    def test_padding_sliding_gpu(self):
        self._test_padding_sliding(self.device)

    def test_padding_sliding_cpu(self):
        self._test_padding_sliding(None)

    def _test_padding_sliding(self, device):
        logging.info("Will test convolutional layer back propagation")

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

        c = gd_conv.GradientDescentConv(DummyWorkflow(), n_kernels=2,
                                        kx=3, ky=3, padding=(1, 2, 3, 4),
                                        sliding=(2, 3), gradient_moment=0.9,
                                        error_function_averaged=True)
        c.err_output = formats.Vector()
        c.err_output.mem = err_output.copy()
        c.input = formats.Vector()
        c.input.mem = inp.copy()
        c.weights = formats.Vector()
        c.weights.mem = weights.copy()
        c.bias = formats.Vector()
        c.bias.mem = bias.copy()
        c.output = formats.Vector()
        c.output.mem = err_output.copy()

        batch_size = c.err_output.mem.shape[0]
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
        logging.info("Err_input Ok")

        max_diff = numpy.fabs(weights_new.ravel() -
                              c.weights.mem.ravel()).max()
        self.assertLess(max_diff, 0.0001,
                        "Weights differ by %.6f" % (max_diff))
        logging.info("Weights Ok")

        max_diff = numpy.fabs(bias_new.ravel() - c.bias.mem.ravel()).max()
        self.assertLess(max_diff, 0.0001,
                        "Bias differs by %.6f" % (max_diff))
        logging.info("Bias Ok")

        err_input = c.err_input.mem.ravel()
        forward = conv.Conv(DummyWorkflow(), n_kernels=2, kx=3, ky=3,
                            padding=(1, 2, 3, 4), sliding=(2, 3))
        target = self._numdiff_init_forward(forward, inp, weights, bias,
                                            err_output)

        self.numdiff_check_gd(forward, inp, weights, bias, target,
                              err_input, weights_derivative, bias_derivative,
                              logging.info, self.assertLess)

    def test_random_numeric_gpu(self):
        self._test_random_numeric(self.device, conv.Conv,
                                  gd_conv.GradientDescentConv)

    def test_random_numeric_gpu_tanh(self):
        self._test_random_numeric(self.device, conv.ConvTanh,
                                  gd_conv.GDTanhConv)

    def test_random_numeric_gpu_relu(self):
        self._test_random_numeric(self.device, conv.ConvRELU,
                                  gd_conv.GDRELUConv)

    def test_random_numeric_cpu(self):
        self._test_random_numeric(None, conv.Conv,
                                  gd_conv.GradientDescentConv)

    def test_random_numeric_cpu_tanh(self):
        self._test_random_numeric(None, conv.ConvTanh,
                                  gd_conv.GDTanhConv)

    def test_random_numeric_cpu_relu(self):
        self._test_random_numeric(None, conv.ConvRELU,
                                  gd_conv.GDRELUConv)

    def _test_random_numeric(self, device, Forward, GD):
        logging.info("Will test convolutional layer forward-backward "
                     "via numeric differentiation")

        dtype = opencl_types.dtypes[root.common.precision_type]
        inp = numpy.zeros([2, 5, 5, 3], dtype=dtype)
        prng.get().fill(inp)
        forward = Forward(DummyWorkflow(), n_kernels=2, kx=3, ky=3,
                          padding=(1, 2, 3, 4), sliding=(2, 3))
        forward.input = formats.Vector()
        forward.input.mem = inp.copy()
        forward.initialize(device=self.device)
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
            DummyWorkflow(), n_kernels=forward.n_kernels,
            kx=forward.kx, ky=forward.ky,
            gradient_moment=0, gradient_moment_bias=0,
            learning_rate=-1, weights_decay=0,
            learning_rate_bias=-1, weights_decay_bias=0,
            padding=forward.padding, sliding=forward.sliding,
            error_function_averaged=True)
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


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
