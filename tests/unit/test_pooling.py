"""
Created on Dec 4, 2013

Unit test for pooling layer forward propagation.

@author: Kazantsev Alexey <a.kazantsev@samsung.com>
"""
import unittest
import pooling
import gd_pooling
import opencl
import formats
import numpy
import config
import units


class TestPooling(unittest.TestCase):
    def setUp(self):
        self.device = opencl.Device()

    def tearDown(self):
        del self.device
        units.pool.shutdown()

    def test_forward(self):
        print("Will test pooling layer forward propagation")

        inp = formats.Vector()
        dtype = config.dtypes[config.dtype]
        inp.v = numpy.array([[[3, 3, -1, 1, 2, 3, 4],
                              [-2, 4, -2, 3, -3, 1, -2],
                              [-3, -1, 2, -3, 1, -4, 0],
                              [-1, -2, 2, -1, 0, 1, 1],
                              [2, 4, -1, 1, 0, -3, -1]],
                             [[4, -1, 3, -4, -4, -2, -1],
                              [-1, 3, -2, 3, -1, 4, 2],
                              [-2, 1, -2, 0, 0, -1, -2],
                              [3, -2, 1, -2, 4, 0, -1],
                              [2, -1, -3, 2, 0, 1, 0]],
                             [[-4, -1, -1, 3, 4, 4, 1],
                              [0, 1, 0, 3, -3, 4, -1],
                              [-1, 1, -4, -4, 2, 2, 1],
                              [1, -4, 1, -3, -4, -1, -1],
                              [-1, 4, 2, -2, 3, 3, 4]]], dtype=dtype)

        c = pooling.MaxPooling(kx=2, ky=2, device=self.device)
        c.input = inp

        c.initialize()
        c.run()

        c.output.map_read()  # get results back
        y = c.output.v.ravel()
        t = numpy.array([4, 3, 3, 4, -3, -3, -4, 1, 4, -1, -3, -1,
                         4, -4, -4, 2, 3, -2, 4, -2, 2, -3, 1, 0,
                         (-4), 3, 4, 1, -4, -4, -4, 1, 4, 2, 3, 4],
                        dtype=dtype)
        max_diff = numpy.fabs(t - y).max()
        self.assertLess(max_diff, 0.0001,
                        "Result differs by %.6f" % (max_diff))

        c.input_offs.map_read()  # get results back
        y = c.input_offs.v.ravel()
        t = numpy.array([8, 10, 5, 6, 14, 17, 19, 27, 29, 30, 33, 34,
                         35, 38, 39, 48, 56, 51, 60, 55, 63, 65, 68, 69,
                         70, 73, 74, 76, 92, 86, 95, 90, 99, 100, 102, 104],
                        dtype=numpy.int32)
        max_diff = numpy.fabs(t - y).max()
        self.assertLess(max_diff, 0.0001,
                        "Result differs by %.6f" % (max_diff))

        print("All Ok")

    def test_gd(self):
        print("Will test pooling layer gradient descent")

        inp = formats.Vector()
        dtype = config.dtypes[config.dtype]
        inp.v = numpy.array([[[3, 3, -1, 1, 2, 3, 4],
                              [-2, 4, -2, 3, -3, 1, -2],
                              [-3, -1, 2, -3, 1, -4, 0],
                              [-1, -2, 2, -1, 0, 1, 1],
                              [2, 4, -1, 1, 0, -3, -1]],
                             [[4, -1, 3, -4, -4, -2, -1],
                              [-1, 3, -2, 3, -1, 4, 2],
                              [-2, 1, -2, 0, 0, -1, -2],
                              [3, -2, 1, -2, 4, 0, -1],
                              [2, -1, -3, 2, 0, 1, 0]],
                             [[-4, -1, -1, 3, 4, 4, 1],
                              [0, 1, 0, 3, -3, 4, -1],
                              [-1, 1, -4, -4, 2, 2, 1],
                              [1, -4, 1, -3, -4, -1, -1],
                              [-1, 4, 2, -2, 3, 3, 4]]], dtype=dtype)

        c = gd_pooling.GDMaxPooling(device=self.device)
        c.h = inp
        c.h_offs = formats.Vector()
        c.h_offs.v = numpy.array([8, 10, 5, 6, 14, 17, 19, 27, 29, 30, 33, 34,
            35, 38, 39, 48, 56, 51, 60, 55, 63, 65, 68, 69,
            70, 73, 74, 76, 92, 86, 95, 90, 99, 100, 102, 104],
            dtype=numpy.int32)
        c.err_y = formats.Vector()
        c.err_y.v = numpy.array([1, 3, 0.5, -4, 1, -2, -3, -1, -1, 3, -3, -0.5,
                                 4, -4, -0.3, -3, -1, -3, 2, -2, -4, 2, -1, -3,
                                 (-4), 2, 3, 2, -1, -1, -3, 4, -2, 2, 0.3, -4],
                                dtype=dtype)
        c.initialize()
        c.err_h.map_write()
        c.err_h.v[:] = 1.0e30
        c.run()

        c.err_h.map_read()  # get results back
        t = numpy.array([[[0, 0, 0, 0, 0, 0.5, -4],
                          [0, 1, 0, 3, 0, 0, 0],
                          [1, 0, 0, -2, 0, -3, 0],
                          [0, 0, 0, 0, 0, 0, -1],
                          [0, -1, 3, 0, 0, -3, -0.5]],
                         [[4, 0, 0, -4, -0.3, 0, 0],
                          [0, 0, 0, 0, 0, 0, -3],
                          [0, 0, -3, 0, 0, 0, -2],
                          [-1, 0, 0, 0, 2, 0, 0],
                          [-4, 0, 2, 0, 0, -1, -3]],
                         [[-4, 0, 0, 2, 3, 0, 2],
                          [0, 0, 0, 0, 0, 0, 0],
                          [0, 0, -1, 0, 0, 0, 4],
                          [0, -1, 0, 0, -3, 0, 0],
                          [0, -2, 2, 0, 0.3, 0, -4]]], dtype=dtype)
        max_diff = numpy.fabs(t.ravel() - c.err_h.v.ravel()).max()
        self.assertLess(max_diff, 0.0001,
                        "Result differs by %.6f" % (max_diff))

        print("All Ok")
        units.pool.shutdown()


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
