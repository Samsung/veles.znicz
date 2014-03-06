#!/usr/bin/python3.3 -O
"""
Created on November 18, 2013

@author: Lyubov Podoynitsina <lyubov.p@samsung.com>
"""
import numpy
import unittest

import config
import znicz_config
import formats
import gd
import opencl
import opencl_types
import rnd


class TestGD(unittest.TestCase):
    def setUp(self):
        config.unit_test = True
        config.plotters_disabled = True
        self.device = opencl.Device()

    def tearDown(self):
        del self.device

    def _do_tst(self, device):
        inp = formats.Vector()
        dtype = opencl_types.dtypes[config.dtype]
        inp.v = numpy.empty([5, 5], dtype=dtype)
        rnd.default.fill(inp.v)

        if device is not None:
            self.x = inp.v.copy()
        else:
            inp.v[:] = self.x[:]

        c = gd.GD(None, device=device)
        c.h = inp

        weights = numpy.array([[1, 0, 2, 1, -1],
                              [3, 1, 0, 2, 3],
                              [-1, 2, 0, 1, 3],
                              [1, -1, 3, 2, 4]], dtype=dtype)
        bias = numpy.array([10, -10, 5, 2], dtype=dtype)
        c.err_y = formats.Vector()
        c.err_y.v = numpy.array([[-1, 3, 0, 2],
                               [8, 2, 1, 3],
                               [0, 1, -2, 1],
                               [2, 3, -1, 0],
                               [1, 0, 1, 1]], dtype=dtype)

        c.weights = formats.Vector()
        c.weights.v = weights
        c.bias = formats.Vector()
        c.bias.v = bias
        c.y = formats.Vector()
        c.y.v = c.err_y.v.copy()
        c.initialize()

        if device is None:
            c.weights.map_invalidate()
            c.bias.map_invalidate()
            c.weights.v[:] = self.W[:]
            c.bias.v[:] = self.b[:]
            c.cpu_run()
        else:
            self.W = c.weights.v.copy()
            self.b = c.bias.v.copy()
            c.gpu_run()
        c.err_h.map_read()  # get results back

        return c.err_h.v

    def test_gpu_cpu(self):
        print("Will test all2all unit for gpu/cpu correctness")
        y_gpu = self._do_tst(self.device)
        y_cpu = self._do_tst(None)
        max_diff = numpy.fabs(y_gpu.ravel() - y_cpu.ravel()).max()
        self.assertLess(max_diff, 0.0001,
                        "Result differs by %.6f" % (max_diff))
        print("All Ok")


if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
