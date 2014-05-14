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

from veles.config import root
import veles.formats as formats
import veles.opencl as opencl
import veles.opencl_types as opencl_types
import veles.random_generator as rnd
import veles.znicz.conv as conv
from veles.tests.dummy_workflow import DummyWorkflow


class TestConv(unittest.TestCase):
    def setUp(self):
        root.common.unit_test = True
        root.common.plotters_disabled = True
        self.device = opencl.Device()

    def tearDown(self):
        del self.device

    def test_fixed(self):
        logging.info("Will test convolutional layer forward propagation")

        inp = formats.Vector()
        dtype = opencl_types.dtypes[root.common.dtype]
        inp.v = numpy.array([[[1, 2, 3, 2, 1],
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

        c = conv.Conv(DummyWorkflow(), n_kernels=2, kx=3, ky=3)
        c.input = inp

        c.initialize(device=self.device)

        c.weights.map_invalidate()  # rewrite weights
        c.weights.v[:] = weights.reshape(c.weights.v.shape)[:]
        c.bias.map_invalidate()  # rewrite bias
        c.bias.v[:] = bias[:]

        c.run()
        c.output.map_read()  # get results back
        nz = numpy.count_nonzero(c.output.vv[c.output.v.shape[0]:].ravel())
        self.assertEqual(nz, 0, "Overflow occured")

        y = c.output.v.ravel()
        t = numpy.array([9, 5.3, 15, 5.65, 9, -3.5,
                         12, 1.25, 3, -2.8, 12, -4.4,
                         4, -7.05, 15, -7.7, 4, -4.65], dtype=dtype)
        max_diff = numpy.fabs(t - y).max()
        self.assertLess(max_diff, 0.0001,
                        "Result differs by %.6f" % (max_diff))

        logging.info("All Ok")

    def test_fixed_cpu(self):
        logging.info("Will test CPU convolutional layer forward propagation")

        inp = formats.Vector()
        dtype = opencl_types.dtypes[root.common.dtype]
        inp.v = numpy.array([[[1, 2, 3, 2, 1],
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

        c = conv.Conv(DummyWorkflow(), n_kernels=2, kx=3, ky=3)
        c.input = inp

        c.initialize(device=None)

        c.weights.map_invalidate()  # rewrite weights
        c.weights.v[:] = weights.reshape(c.weights.v.shape)[:]
        c.bias.map_invalidate()  # rewrite bias
        c.bias.v[:] = bias[:]

        c.run()
        nz = numpy.count_nonzero(c.output.vv[c.output.v.shape[0]:].ravel())
        self.assertEqual(nz, 0, "Overflow occured")

        y = c.output.v.ravel()
        t = numpy.array([9, 5.3, 15, 5.65, 9, -3.5,
                         12, 1.25, 3, -2.8, 12, -4.4,
                         4, -7.05, 15, -7.7, 4, -4.65], dtype=dtype)
        max_diff = numpy.fabs(t - y).max()
        self.assertLess(max_diff, 0.0001,
                        "Result differs by %.6f" % (max_diff))

        logging.info("All Ok")

    def test_padding_sliding(self):
        logging.info("Will test convolutional layer forward propagation")

        inp = formats.Vector()
        dtype = opencl_types.dtypes[root.common.dtype]
        inp.v = numpy.array([[[1, 2, 3, 2, 1],
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
        c.weights.v[:] = weights.reshape(c.weights.v.shape)[:]
        c.bias.map_invalidate()  # rewrite bias
        c.bias.v[:] = bias[:]

        c.run()
        c.output.map_read()  # get results back
        nz = numpy.count_nonzero(c.output.vv[c.output.v.shape[0]:].ravel())
        self.assertEqual(nz, 0, "Overflow occured")

        y = c.output.v.ravel()
        t = numpy.array([[[7, -11.3], [3, -10.7], [7, -8], [10, -10]],
                         [[6, -8.4], [3, -2.8], [6, -12.8], [10, -10]],
                         [[9, -7.9], [9, -7.9], [9, -7.9], [10, -10]]],
                        dtype=dtype).ravel()
        max_diff = numpy.fabs(t - y).max()
        self.assertLess(max_diff, 0.0001,
                        "Result differs by %.6f" % (max_diff))

        logging.info("All Ok")

    def _do_test_vs_python(self, Unit):

        logging.info("OpenCL")

        inp = formats.Vector()
        dtype = opencl_types.dtypes[root.common.dtype]
        inp.v = numpy.zeros([27, 28, 28], dtype=dtype)
        rnd.get().fill(inp.v)

        c = Unit(DummyWorkflow(), n_kernels=25, kx=9, ky=9)
        c.input = inp

        c.initialize(device=self.device)

        weights = c.weights.v.reshape(c.n_kernels, c.ky, c.kx)
        bias = c.bias.v

        t0 = time.time()
        c.run()
        dt0 = time.time() - t0
        logging.info("OpenCL convolved in %.2f seconds" % (dt0))

        c.output.map_read()  # get results back
        nz = numpy.count_nonzero(c.output.vv[c.output.v.shape[0]:].ravel())
        self.assertEqual(nz, 0, "Overflow occured")

        logging.info("Numpy")
        t0 = time.time()
        pp = []
        for v in inp.v:
            for j, w in enumerate(weights):
                ww = w.copy()
                for i in range(w.shape[0]):
                    ww[-(i + 1)] = w[i]
                www = ww.copy()
                for i in range(w.shape[1]):
                    www[:, -(i + 1)] = ww[:, i]
                out = scipy.signal.convolve2d(v, www, "valid")
                out += bias[j]
                out *= 0.6666
                numpy.tanh(out, out)
                out *= 1.7159
                pp.append(out)
        dt1 = time.time() - t0
        logging.info("Numpy convolved in %.2f seconds" % (dt1))
        logging.info("OpenCL was %.2f times faster than Numpy" % (dt1 / dt0))
        logging.info("Will compare results")
        offs = 0
        for vv in c.output.v:
            for i_kernel in range(len(weights)):
                p = pp[offs]
                v = vv[:, :, i_kernel].reshape(vv.shape[0], vv.shape[1])
                max_diff = numpy.fabs(v.ravel() - p.ravel()).max()
                self.assertLess(max_diff, 0.0001,
                                "Result differs by %.6f" % (max_diff))
                offs += 1

        logging.info("All Ok")

    def _do_test_vs_python_rgb(self, Unit):

        logging.info("OpenCL")

        inp = formats.Vector()
        dtype = opencl_types.dtypes[root.common.dtype]
        inp.v = numpy.zeros([3, 128, 128, 3], dtype=dtype)
        rnd.get().fill(inp.v)

        c = Unit(DummyWorkflow(), n_kernels=4, kx=3, ky=3)
        c.input = inp

        c.initialize(device=self.device)

        c.bias.map_invalidate()  # rewrite bias
        c.bias.v[:] = 0

        t0 = time.time()
        c.run()
        dt0 = time.time() - t0
        logging.info("OpenCL convolved in %.2f seconds" % (dt0))

        c.output.map_read()  # get results back
        nz = numpy.count_nonzero(c.output.vv[c.output.v.shape[0]:].ravel())
        self.assertEqual(nz, 0, "Overflow occured")

        logging.info("Numpy with FFT")
        t0 = time.time()
        pp = []
        for v in inp.v:
            for w_ in c.weights.v:
                w = w_.reshape(c.ky, c.kx, 3)
                ww = w.copy()
                for i in range(w.shape[0]):
                    ww[-(i + 1)] = w[i]
                www = ww.copy()
                for i in range(w.shape[1]):
                    www[:, -(i + 1)] = ww[:, i]
                wwww = www.copy()
                for i in range(w.shape[2]):
                    wwww[:, :, -(i + 1)] = www[:, :, i]
                pp.append(scipy.signal.fftconvolve(v, wwww, "valid"))
        dt1 = time.time() - t0
        logging.info("Numpy convolved in %.2f seconds" % (dt1))

        logging.info("OpenCL was %.2f times faster than Numpy" % (dt1 / dt0))

        logging.info("Will compare results")

        offs = 0
        for vv in c.output.v:
            for i_kernel in range(len(c.weights.v)):
                p = pp[offs]
                v = vv[:, :, i_kernel].reshape(vv.shape[0], vv.shape[1])
                max_diff = numpy.fabs(v.ravel() - p.ravel()).max()
                self.assertLess(max_diff, 0.0001,
                                "Result differs by %.6f" % (max_diff))
                offs += 1

        logging.info("All Ok")

    def test_vs_python_rgb(self):
        logging.info("Will test linear convolutional"
                     " layer vs python on color image")
        self._do_test_vs_python_rgb(conv.Conv)

    def test_vs_python(self):
        logging.info("Will test linear convolutional layer vs python on image")
        self._do_test_vs_python(conv.ConvTanh)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
