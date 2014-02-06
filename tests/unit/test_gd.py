#!/usr/bin/python3.3 -O
"""
Created on November 18, 2013

@author: Lyubov Podoynitsina <lyubov.p@samsung.com>
"""
import unittest
import gd
import opencl
import formats
import numpy
import config
import znicz_config
import units
import rnd


class TestGD(unittest.TestCase):
    def _do_tst(self, device):
        inp = formats.Vector()
        dtype = config.dtypes[config.dtype]
        inp.v = numpy.empty([5, 5], dtype=dtype)
        rnd.default.fill(inp.v)

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
        if device == None:
            c.cpu_run()
        else:
            c.gpu_run()
        c.err_h.map_read()  # get results back

        return c.err_h.v

    def test_gpu_cpu(self):
        print("Will test all2all unit for gpu/cpu correctness")
        s = rnd.default.state
        device = opencl.Device()
        y_gpu = self._do_tst(device)
        rnd.default.state = s
        y_cpu = self._do_tst(None)
        max_diff = numpy.fabs(y_gpu.ravel() - y_cpu.ravel()).max()
        self.assertLess(max_diff, 0.0001,
                        "Result differs by %.6f" % (max_diff))
        print("All Ok")


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
