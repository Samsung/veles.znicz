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
import veles.opencl as opencl
import veles.opencl_types as opencl_types
import veles.znicz.gd_conv as gd_conv
import veles.znicz.conv as conv
from veles.tests.dummy_workflow import DummyWorkflow


class TestGDConv(unittest.TestCase):
    def setUp(self):
        root.common.unit_test = True
        root.common.plotters_disabled = True
        self.device = opencl.Device()

    def tearDown(self):
        del self.device

    def test_err_h(self):
        logging.info("Will test convolutional layer back propagation")

        dtype = opencl_types.dtypes[root.common.dtype]
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

        err_output = numpy.array([[[-1, 3],
                                   [8, 2],
                                   [0, 1],
                                   [4, -1],
                                   [-1, 2],
                                   [0, 1],
                                   [-2, 3],
                                   [1, 2],
                                   [1, 1]],

                                  [[-1, 3],
                                   [8, 2],
                                   [0, 1],
                                   [4, -1],
                                   [-1, 2],
                                   [0, 1],
                                   [-2, 3],
                                   [1, 2],
                                   [1, 1]]], dtype=dtype)

        c = gd_conv.GradientDescentConv(DummyWorkflow(), n_kernels=2,
                                        kx=3, ky=3, gradient_moment=0.9)
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
        gradient_weights = numpy.dot(b.transpose(), a)
        weights_derivative = gradient_weights.copy()
        gradient_weights *= (-1) * (c.learning_rate / batch_size)
        gradient_weights += weights * (-1) * (c.learning_rate *
                                              c.weights_decay)
        weights_new = weights + gradient_weights
        bias_derivative = b.sum(axis=0)
        gradient_bias = bias_derivative * (-1) * (c.learning_rate / batch_size)
        bias_new = bias + gradient_bias

        c.initialize(device=self.device)
        c.gpu_err_output_update()
        c.gpu_err_input_update()
        c.gpu_weights_update()
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
                        "Result differs by %.6f" % (max_diff))
        logging.info("Err_input is right")

        max_diff = numpy.fabs(weights_new.ravel() -
                              c.weights.mem.ravel()).max()
        self.assertLess(max_diff, 0.0001,
                        "Result differs by %.6f" % (max_diff))
        logging.info("Weights is right")

        max_diff = numpy.fabs(bias_new.ravel() - c.bias.mem.ravel()).max()
        self.assertLess(max_diff, 0.0001,
                        "Result differs by %.6f" % (max_diff))
        logging.info("Bias is right")

        err_input = c.err_input.mem.ravel()
        forward = conv.Conv(DummyWorkflow(), n_kernels=2, kx=3, ky=3)
        target = self._numdiff_init_forward(forward, inp, weights, bias,
                                            err_output)

        self._numdiff_check_err_input(inp, forward, target, err_input)
        self._numdiff_check_weights(weights, forward, bias, target,
                                    weights_derivative)
        self._numdiff_check_bias(bias, forward, weights, target,
                                 bias_derivative)

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

    def _numdiff_check_err_input(self, inp, forward, target, err_input):
        numdiff = formats.NumDiff()

        logging.info("Checking err_input with numeric differentiation")
        offs = 0
        for i_sample in range(inp.shape[0]):
            for y in range(inp.shape[1]):
                for x in range(inp.shape[2]):
                    for i, p in enumerate(numdiff.points):
                        forward.input.map_invalidate()
                        forward.input.mem[:] = inp[:]
                        forward.input.mem[i_sample, y, x] = (
                            inp[i_sample, y, x] + p)
                        forward.run()
                        forward.output.map_read()
                        out = forward.output.mem.ravel()
                        numdiff.errs[i] = (numpy.square(out - target).sum() *
                                           0.5)

                    derivative = numdiff.derivative
                    d = numpy.fabs(derivative - err_input[offs])
                    logging.info("%.2f %.2f %.2f" %
                                 (derivative, err_input[offs], d))
                    self.assertLess(d, 0.5, "Numeric diff test failed")
                    offs += 1

    def _numdiff_check_weights(self, weights, forward, bias, target,
                               weights_derivative):
        numdiff = formats.NumDiff()

        logging.info("Checking weights with numeric differentiation")
        for y in range(weights.shape[0]):
            for x in range(weights.shape[1]):
                for i, p in enumerate(numdiff.points):
                    forward.weights.map_invalidate()
                    forward.weights.mem[:] = weights[:]
                    forward.weights.mem[y, x] = weights[y, x] + p
                    forward.bias.map_invalidate()
                    forward.bias.mem[:] = bias[:]
                    forward.run()
                    forward.output.map_read()
                    out = forward.output.mem.ravel()
                    numdiff.errs[i] = numpy.square(out - target).sum() * 0.5

                derivative = numdiff.derivative
                d = numpy.fabs(derivative - weights_derivative[y, x])
                logging.info("%.2f %.2f %.2f" %
                             (derivative, weights_derivative[y, x], d))
                self.assertLess(d, 0.5, "Numeric diff test failed")

    def _numdiff_check_bias(self, bias, forward, weights, target,
                            bias_derivative):
        numdiff = formats.NumDiff()

        logging.info("Checking bias with numeric differentiation")
        for y in range(bias.shape[0]):
            for i, p in enumerate(numdiff.points):
                forward.weights.map_invalidate()
                forward.weights.mem[:] = weights[:]
                forward.bias.map_invalidate()
                forward.bias.mem[:] = bias[:]
                forward.bias.mem[y] = bias[y] + p
                forward.run()
                forward.output.map_read()
                out = forward.output.mem.ravel()
                numdiff.errs[i] = numpy.square(out - target).sum() * 0.5

            derivative = numdiff.derivative
            d = numpy.fabs(derivative - bias_derivative[y])
            logging.info("%.2f %.2f %.2f" %
                         (derivative, bias_derivative[y], d))
            self.assertLess(d, 0.5, "Numeric diff test failed")

    def test_err_h_cpu(self):
        logging.info("Will test CPU convolutional layer back propagation")

        inp = formats.Vector()
        dtype = opencl_types.dtypes[root.common.dtype]
        inp.mem = numpy.array([[[-1, 0, 2, 0, 3],
                              [0, 1, -2, 1, 2],
                              [2, 0, 1, 1, 0],
                              [-1, 1, 1, 0, 2],
                              [1, 0, 1, 0, 1]]], dtype=dtype)

        a = numpy.array([[-1, 0, 2, 0, 1, -2, 2, 0, 1],
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

        c = gd_conv.GradientDescentConv(DummyWorkflow(), n_kernels=2,
                                        kx=3, ky=3, gradient_moment=0.9)
        c.err_output = formats.Vector()
        c.err_output.mem = numpy.array([[[-1, 3],
                                       [8, 2],
                                       [0, 1],
                                       [4, -1],
                                       [-1, 2],
                                       [0, 1],
                                       [-2, 3],
                                       [1, 2],
                                       [1, 1]]], dtype=dtype)
        c.input = inp
        c.weights = formats.Vector()
        c.weights.mem = weights
        c.bias = formats.Vector()
        c.bias.mem = bias
        c.output = formats.Vector()
        c.output.mem = c.err_output.mem.copy()

        batch_size = c.err_output.mem.shape[0]
        b = c.err_output.mem.reshape(9 * batch_size, 2)
        gradient_weights = numpy.dot(b.transpose(), a)
        gradient_weights *= (-1) * (c.learning_rate / batch_size)
        gradient_weights += weights * (-1) * (c.learning_rate *
                                              c.weights_decay)
        weights_new = weights + gradient_weights
        gradient_bias = b.sum(axis=0) * (-1) * (c.learning_rate / batch_size)
        bias_new = bias + gradient_bias

        c.initialize(device=self.device)
        c.cpu_err_input_update()
        c.cpu_weights_update()
        c.err_input.map_read()
        c.weights.map_read()
        c.bias.map_read()

        t = numpy.array([[[7, 0, -11, 31, -1],
                          [2, 6, 93, -11, 0],
                          [22, 45, 18, 28, 7],
                          [-1, 11, 25, 14, 3],
                          [14, 4, 13, 12, 5]]], dtype=dtype)
        max_diff = numpy.fabs(t.ravel() - c.err_input.mem.ravel()).max()
        self.assertLess(max_diff, 0.0001,
                        "Result differs by %.6f" % (max_diff))
        logging.info("Err_input is right")

        max_diff = numpy.fabs(weights_new.ravel() -
                              c.weights.mem.ravel()).max()
        self.assertLess(max_diff, 0.0001,
                        "Result differs by %.6f" % (max_diff))
        logging.info("Weights is right")

        max_diff = numpy.fabs(bias_new.ravel() - c.bias.mem.ravel()).max()
        self.assertLess(max_diff, 0.0001,
                        "Result differs by %.6f" % (max_diff))
        logging.info("Bias is right")

    def test_padding_sliding(self):
        logging.info("Will test convolutional layer back propagation")

        dtype = opencl_types.dtypes[root.common.dtype]
        inp = numpy.array([[[1, 2, 3, 2, 1],
                            [0, 1, 2, 1, 0],
                            [0, 1, 0, 1, 0],
                            [2, 0, 1, 0, 2],
                            [1, 0, 1, 0, 1]]], dtype=dtype)

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

        err_output = numpy.array([[[-1, 3],
                                   [8, 2],
                                   [0, 1],
                                   [4, -1],
                                   [-1, 2],
                                   [0, 1],
                                   [-2, 3],
                                   [1, 2],
                                   [1, 1],
                                   [1, -2],
                                   [0, 5],
                                   [2, 3]]], dtype=dtype)

        c = gd_conv.GradientDescentConv(DummyWorkflow(), n_kernels=2,
                                        kx=3, ky=3, padding=(1, 2, 3, 4),
                                        sliding=(2, 3), gradient_moment=0.9)
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
        gradient_weights *= (-1) * (c.learning_rate / batch_size)
        gradient_weights += weights * (-1) * (c.learning_rate *
                                              c.weights_decay)
        weights_new = weights + gradient_weights
        bias_derivative = b.sum(axis=0)
        gradient_bias = bias_derivative * (-1) * (c.learning_rate / batch_size)
        bias_new = bias + gradient_bias

        c.initialize(device=self.device)
        c.gpu_err_input_update()
        c.gpu_weights_update()
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
                        "Result differs by %.6f\nTarget is:\n%s\nGot:\n%s" %
                        (max_diff, " ".join("%.2f" % x for x in t.ravel()),
                        " ".join("%.2f" % x for x in c.err_input.mem.ravel())))
        logging.info("Err_input is right")

        max_diff = numpy.fabs(bias_new.ravel() - c.bias.mem.ravel()).max()
        self.assertLess(max_diff, 0.0001,
                        "Result differs by %.6f" % (max_diff))
        logging.info("Bias is right")

        max_diff = numpy.fabs(weights_new.ravel() -
                              c.weights.mem.ravel()).max()
        self.assertLess(max_diff, 0.0001,
                        "Result differs by %.6f" % (max_diff))
        logging.info("Weights is right")

        err_input = c.err_input.mem.ravel()
        forward = conv.Conv(DummyWorkflow(), n_kernels=2, kx=3, ky=3,
                            padding=(1, 2, 3, 4), sliding=(2, 3))
        target = self._numdiff_init_forward(forward, inp, weights, bias,
                                            err_output)

        self._numdiff_check_err_input(inp, forward, target, err_input)
        self._numdiff_check_weights(weights, forward, bias, target,
                                    weights_derivative)
        self._numdiff_check_bias(bias, forward, weights, target,
                                 bias_derivative)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
