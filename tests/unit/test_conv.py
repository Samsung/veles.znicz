"""
Created on Nov 7, 2013

Unit test for convolutional layer forward propagation.

@author: ajk
"""
import unittest
import conv
import opencl
import formats
import numpy
import config


class TestConv(unittest.TestCase):
    def test(self):
        print("Will test convolutional layer forward propagation")

        cl = opencl.DeviceList()
        device = cl.get_device()

        inp = formats.Vector()
        dtype = config.dtypes[config.dtype]
        inp.v = numpy.array([[[1, 2, 3, 2, 1],
                              [0, 1, 2, 1, 0],
                              [0, 1, 0, 1, 0],
                              [2, 0, 1, 0, 2],
                              [1, 0, 1, 0, 1]]], dtype=dtype)

        weights = numpy.array([[[-1, -1, -1],
                                [-1, 8, -1],
                                [-1, -1, -1]],
                               [[1, 1, 1],
                                [1, -8, 1],
                                [1, 1, 1]]], dtype=dtype)
        bias = numpy.array([10, -10], dtype=dtype)

        c = conv.Conv(n_kernels=2, kx=3, ky=3, device=device)
        c.input = inp

        c.initialize()

        c.weights.v[:] = weights.reshape(c.weights.v.shape)[:]
        c.weights.update()
        c.bias.v[:] = bias[:]
        c.bias.update()

        c.run()

        c.output.sync()
        y = c.output.v.ravel()
        t = numpy.array([9, -9, 15, -15, 9, -9,
                         12, -12, 3, -3, 12, -12,
                         4, -4, 15, -15, 4, -4], dtype=dtype)
        for i in range(len(y)):
            self.assertAlmostEqual(y[i], t[i])


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
