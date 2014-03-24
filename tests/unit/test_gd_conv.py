"""
Created on Nov 7, 2013

Unit test for convolutional layer back propagation.

@author: Kazantsev Alexey <a.kazantsev@samsung.com>
"""


import numpy
import unittest

from veles.config import root
import veles.formats as formats
import veles.opencl as opencl
import veles.opencl_types as opencl_types
import veles.znicz.gd_conv as gd_conv
from veles.znicz.tests.unit.dummy_workflow import DummyWorkflow


class TestGDConv(unittest.TestCase):
    def setUp(self):
        root.common.unit_test = True
        root.common.plotters_disabled = True
        self.device = opencl.Device()

    def tearDown(self):
        del self.device

    def test_err_h(self):
        print("Will test convolutional layer back propagation")

        inp = formats.Vector()
        dtype = opencl_types.dtypes[root.common.dtype]
        inp.v = numpy.array([[[-1, 0, 2, 0, 3],
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

        c = gd_conv.GD(DummyWorkflow(), n_kernels=2, kx=3, ky=3,
                       device=self.device)
        c.err_y = formats.Vector()
        c.err_y.v = numpy.array([[[-1, 3],
                               [8, 2],
                               [0, 1],
                               [4, -1],
                               [-1, 2],
                               [0, 1],
                               [-2, 3],
                               [1, 2],
                               [1, 1]]], dtype=dtype)
        c.h = inp
        c.weights = formats.Vector()
        c.weights.v = weights
        c.bias = formats.Vector()
        c.bias.v = bias
        c.y = formats.Vector()
        c.y.v = c.err_y.v.copy()

        batch_size = c.err_y.v.shape[0]
        b = c.err_y.v.reshape(9 * batch_size, 2)
        gradient_weights = numpy.dot(b.transpose(), a)
        gradient_weights *= (-1) * (c.global_alpha / batch_size)
        gradient_weights += weights * (-1) * (c.global_alpha * c.global_lambda)
        weights_new = weights + gradient_weights
        gradient_bias = b.sum(axis=0) * (-1) * (c.global_alpha / batch_size)
        bias_new = bias + gradient_bias

        c.initialize()
        c.gpu_err_h_update()
        c.gpu_weights_update()
        c.err_h.map_read()
        c.weights.map_read()
        c.bias.map_read()

        t = numpy.array([[[7, 0, -11, 31, -1],
                          [2, 6, 93, -11, 0],
                          [22, 45, 18, 28, 7],
                          [-1, 11, 25, 14, 3],
                          [14, 4, 13, 12, 5]]], dtype=dtype)
        max_diff = numpy.fabs(t.ravel() - c.err_h.v.ravel()).max()
        self.assertLess(max_diff, 0.0001,
                        "Result differs by %.6f" % (max_diff))
        print("Err_h is right")

        max_diff = numpy.fabs(weights_new.ravel() - c.weights.v.ravel()).max()
        self.assertLess(max_diff, 0.0001,
                        "Result differs by %.6f" % (max_diff))
        print("Weights is right")

        max_diff = numpy.fabs(bias_new.ravel() - c.bias.v.ravel()).max()
        self.assertLess(max_diff, 0.0001,
                        "Result differs by %.6f" % (max_diff))
        print("Bias is right")


if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
