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
import scipy.misc
import scipy.signal
import time
import units


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
                               [[1.1, 2.1, 3.1],
                                [-1.1, -0.5, 1.3],
                                [1.7, -1.4, 0.05]]], dtype=dtype)
        bias = numpy.array([10, -10], dtype=dtype)

        c = conv.Conv(n_kernels=2, kx=3, ky=3, device=device)
        c.input = inp

        c.initialize()

        c.weights.map_invalidate()  # rewrite weights
        c.weights.v[:] = weights.reshape(c.weights.v.shape)[:]
        c.bias.map_invalidate()  # rewrite bias
        c.bias.v[:] = bias[:]

        c.run()
        c.output.map_read()  # get results back

        y = c.output.v.ravel()
        t = numpy.array([9, 5.3, 15, 5.65, 9, -3.5,
                         12, 1.25, 3, -2.8, 12, -4.4,
                         4, -7.05, 15, -7.7, 4, -4.65], dtype=dtype)
        max_diff = numpy.fabs(t - y).max()
        self.assertLess(max_diff, 0.0001,
                        "Result differs by %.6f" % (max_diff))

        pp = []
        for v in inp.v:
            for w in weights:
                ww = w.copy()
                for i in range(w.shape[0]):
                    ww[-(i + 1)] = w[i]
                www = ww.copy()
                for i in range(w.shape[1]):
                    www[:, -(i + 1)] = ww[:, i]
                pp.append(scipy.signal.convolve2d(v, www, "valid"))
        offs = 0
        for vv in c.output.v:
            for i_kernel in range(len(weights)):
                p = pp[offs]
                v = vv[:, :, i_kernel].reshape(vv.shape[0], vv.shape[1]).copy()
                v -= bias[i_kernel]
                max_diff = numpy.fabs(v.ravel() - p.ravel()).max()
                if max_diff > 0.0001:
                    print(p)
                self.assertLess(max_diff, 0.0001,
                                "Result differs by %.6f" % (max_diff))
                offs += 1

        print("All Ok")
        units.pool.shutdown()

    def test_vs_python(self):
        print("Will test convolutional layer vs python on image")

        print("OpenCL")
        cl = opencl.DeviceList()
        device = cl.get_device()

        inp = formats.Vector()
        dtype = config.dtypes[config.dtype]
        inp.v = numpy.zeros([3, 512, 512], dtype=dtype)
        inp.v[0] = scipy.misc.imread(
            "%s/512.png" % (config.test_dataset_root), True).astype(dtype)
        formats.normalize(inp.v[0])
        inp.v[1] = scipy.misc.imread(
            "%s/512.1.png" % (config.test_dataset_root), True).astype(dtype)
        formats.normalize(inp.v[1])
        inp.v[2] = scipy.misc.imread(
            "%s/512.2.png" % (config.test_dataset_root), True).astype(dtype)
        formats.normalize(inp.v[2])

        c = conv.Conv(n_kernels=7, kx=3, ky=3, device=device)
        c.input = inp

        c.initialize()

        weights = c.weights.v.reshape(c.n_kernels, c.ky, c.kx)

        c.bias.map_invalidate()  # rewrite bias
        c.bias.v[:] = 0

        t0 = time.time()
        c.run()
        dt0 = time.time() - t0
        print("OpenCL convolved in %.2f seconds" % (dt0))

        c.output.map_read()  # get results back

        print("Numpy")
        t0 = time.time()
        pp = []
        for v in inp.v:
            for w in weights:
                ww = w.copy()
                for i in range(w.shape[0]):
                    ww[-(i + 1)] = w[i]
                www = ww.copy()
                for i in range(w.shape[1]):
                    www[:, -(i + 1)] = ww[:, i]
                pp.append(scipy.signal.convolve2d(v, www, "valid"))
        dt1 = time.time() - t0
        print("Numpy convolved in %.2f seconds" % (dt1))
        print("OpenCL was %.2f times faster than Numpy" % (dt1 / dt0))
        print("Will compare results")
        offs = 0
        for vv in c.output.v:
            for i_kernel in range(len(weights)):
                p = pp[offs]
                v = vv[:, :, i_kernel].reshape(vv.shape[0], vv.shape[1])
                max_diff = numpy.fabs(v.ravel() - p.ravel()).max()
                self.assertLess(max_diff, 0.0001,
                                "Result differs by %.6f" % (max_diff))
                offs += 1

        print("All Ok")
        units.pool.shutdown()

    def test_vs_python_rgb(self):
        print("Will test convolutional layer vs python on color image")

        print("OpenCL")
        cl = opencl.DeviceList()
        device = cl.get_device()

        inp = formats.Vector()
        dtype = config.dtypes[config.dtype]
        inp.v = numpy.zeros([3, 512, 512, 3], dtype=dtype)
        inp.v[0] = scipy.misc.imread(
            "%s/512.png" % (config.test_dataset_root)).astype(dtype)
        formats.normalize(inp.v[0])
        inp.v[1] = scipy.misc.imread(
            "%s/512.1.png" % (config.test_dataset_root)).astype(dtype)
        formats.normalize(inp.v[1])
        inp.v[2] = scipy.misc.imread(
            "%s/512.2.png" % (config.test_dataset_root)).astype(dtype)
        formats.normalize(inp.v[2])

        c = conv.Conv(n_kernels=4, kx=3, ky=3, device=device)
        c.input = inp

        c.initialize()

        c.bias.map_invalidate()  # rewrite bias
        c.bias.v[:] = 0

        t0 = time.time()
        c.run()
        dt0 = time.time() - t0
        print("OpenCL convolved in %.2f seconds" % (dt0))

        c.output.map_read()  # get results back

        print("Numpy with FFT")
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
        print("Numpy convolved in %.2f seconds" % (dt1))

        print("OpenCL was %.2f times faster than Numpy" % (dt1 / dt0))

        print("Will compare results")

        offs = 0
        for vv in c.output.v:
            for i_kernel in range(len(c.weights.v)):
                p = pp[offs]
                v = vv[:, :, i_kernel].reshape(vv.shape[0], vv.shape[1])
                max_diff = numpy.fabs(v.ravel() - p.ravel()).max()
                self.assertLess(max_diff, 0.0001,
                                "Result differs by %.6f" % (max_diff))
                offs += 1

        print("All Ok")
        units.pool.shutdown()


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
