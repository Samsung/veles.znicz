"""
Created on Nov 7, 2013

Unit test for convolutional layer back propagation.

@author: Kazantsev Alexey <a.kazantsev@samsung.com>
"""
import unittest
import gd_conv
import opencl
import formats
import numpy
import config


class TestGDConv(unittest.TestCase):
    def test_err_h(self):
        print("Will test convolutional layer back propagation")
        device = opencl.Device()

        inp = formats.Vector()
        dtype = config.dtypes[config.dtype]
        inp.v = numpy.array([[[-1, 0, 2, 0, 3],
                              [0, 1, -2, 1, 2],
                              [2, 0, 1, 1, 0],
                              [-1, 1, 1, 0, 2],
                              [1, 0, 1, 0, 1]]], dtype=dtype)

        weights = numpy.array([[-1, -1, 4, 1, 8, -1, -1, 3, 2],
                               [2, 1, -1, 3, 0, 1, 4, 1, 3]], dtype=dtype)
        bias = numpy.array([10, -10], dtype=dtype)

        c = gd_conv.GD(n_kernels=2, kx=3, ky=3, device=device)
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
        c.initialize()
        c.gpu_err_h_update()
        c.err_h.map_read()
        t = numpy.array([[[7, 0, -11, 31, -1],
                          [2, 6, 93, -11, 0],
                          [22, 45, 18, 28, 7],
                          [-1, 11, 25, 14, 3],
                          [14, 4, 13, 12, 5]]], dtype=dtype)
        max_diff = numpy.fabs(t.ravel() - c.err_h.v.ravel()).max()
        self.assertLess(max_diff, 0.0001,
                        "Result differs by %.6f" % (max_diff))
        print("Ok")


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
