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


class TestConvBase(unittest.TestCase):
    def setUp(self):
        root.common.unit_test = True
        root.common.plotters_disabled = True
        self._dtype = opencl_types.dtypes[root.common.dtype]

    def tearDown(self):
        pass

    def _run_test(self, unit, device, input_data, weights, bias,
                 gold_output):
        """Run test for specified unit with specified device.

        Tested unit should be an instance of conv.Conv class.
        input_data.shape = [batch_size, y_size, x_size, channels_num]
        weights.shape = [kernels_num, ky, kx]
        bias.shape = [kernels_num]
        gold_output = [batch_size, out_y, out_x, kernels_num]

        Args:
            unit: Veles unit which is tested.
            device: OpenCL device instance (if equals to None - CPU version of
                algorithm should be run).
            input_data: numpy array which is passed to unit as its input.
            weights: numpy array which is passed to unit as NN weights.
            bias: numpy array which is passed to unit as NN bias.

        Raises:
            LessError: if unit output is wrong.
        """
        assert(unit.__class__ == conv.Conv)
        # set unit input and start initialization
        input_vector = formats.Vector()
        input_vector.mem = input_data
        unit.input = input_vector
        unit.initialize(device=device)

        # set weights and bias using allocated memory during the initialization
        unit.weights.map_invalidate()
        unit.weights.mem[:] = weights.reshape(unit.weights.mem.shape)
        unit.bias.map_invalidate()
        unit.bias.mem[:] = bias.reshape(unit.bias.mem.shape)

        unit.run()
        unit.output.map_read()
        max_diff = numpy.fabs(unit.output.mem.ravel() -
                              gold_output.ravel()).max()
        self.assertLess(max_diff, 1E-06, "Result differs by %.2e" % (max_diff))


class TestConv(TestConvBase):
    """Tests convolutional layer forward propagation without padding and with
    sliding = (1, 1).
    """
    def _simple_test(self, device):
        """ Run simple test, which checks trivial cases.

        Args:
            device: OpenCL device instance (if value is equal to None, CPU
                version of algorithm should be run. In other case -
                OpenCL version).
        """
        # set data size
        batch_size = 3
        y_size = x_size = 5
        channels_num = 3
        kx = ky = 3
        kernels_num = 2
        sliding = (1, 1)
        padding = (0, 0, 0, 0)  # left, top, right, bottom

        # calculate x and y size of unit output
        out_y = (y_size + padding[1] + padding[3] - ky) // sliding[1] + 1
        out_x = (x_size + padding[0] + padding[2] - kx) // sliding[0] + 1

        unit = conv.Conv(DummyWorkflow(), n_kernels=kernels_num, kx=kx, ky=ky,
                         sliding=sliding)

        logging.info("run conv with input = 0, random weights, random bias...")
        input_data = numpy.zeros((batch_size, y_size, x_size, channels_num))
        weights = numpy.random.uniform(size=(kernels_num * kx * ky *
            channels_num)).reshape(kernels_num, kx, ky, channels_num)
        bias = numpy.random.uniform(size=kernels_num)
        gold_output = numpy.empty((batch_size, out_y, out_x, kernels_num))
        for batch, i, j in ((batch, i, j) for batch in range(batch_size)
                            for i in range(out_y) for j in range(out_x)):
            gold_output[batch, i, j, :] = bias[:]
        self._run_test(unit, device, input_data, weights, bias, gold_output)

        logging.info("run conv with random input, weights = 0, random bias...")
        input_data = numpy.random.uniform(size=(batch_size * y_size * x_size *
            channels_num)).reshape(batch_size, y_size, x_size, channels_num)
        weights = numpy.zeros((kernels_num, kx, ky, channels_num))
        bias = numpy.random.uniform(size=kernels_num)
        gold_output = numpy.empty((batch_size, out_y, out_x, kernels_num))
        for batch, i, j in ((batch, i, j) for batch in range(batch_size)
                            for i in range(out_y) for j in range(out_x)):
            gold_output[batch, i, j, :] = bias[:]
        self._run_test(unit, device, input_data, weights, bias, gold_output)

        logging.info("run conv with input = 1, weights = 1, bias = 0...")
        input_data = numpy.empty((batch_size, y_size, x_size, channels_num))
        input_data.fill(1)
        weights = numpy.empty((kernels_num, kx, ky, channels_num))
        weights.fill(1)
        bias = numpy.zeros(kernels_num)
        gold_output = numpy.empty((batch_size, out_y, out_x, kernels_num))
        gold_output.fill(kx * ky * channels_num)
        self._run_test(unit, device, input_data, weights, bias, gold_output)

    def test_simple_ocl(self):
        logging.info("start simple OpenCL test...")
        self._simple_test(opencl.Device())
        logging.info("TEST PASSED")

    def test_simple_cpu(self):
        logging.info("start simple CPU test...")
        self._simple_test(None)
        logging.info("TEST PASSED")

    def _1_channel_test(self, device):
        """ Run test with 1 channel input without padding and with
        sliding = (1, 1).

        Args:
            device: OpenCL device instance (if value is equal to None, CPU
                version of algorithm should be run. In other case -
                OpenCL version).
        """
        input_data = numpy.array([[[[1], [2], [3], [2], [1]],
                                   [[0], [1], [2], [1], [0]],
                                   [[0], [1], [0], [1], [0]],
                                   [[2], [0], [1], [0], [2]],
                                   [[1], [0], [1], [0], [1]]]],
                                 dtype=self._dtype)
        weights = numpy.array([[[-1, -1, -1],
                                [-1, 8, -1],
                                [-1, -1, -1]],
                               [[1.1, 2.1, 3.1],
                                [-1.1, -0.5, 1.3],
                                [1.7, -1.4, 0.05]]], dtype=self._dtype)
        bias = numpy.array([10, -10], dtype=self._dtype)
        gold_output = numpy.array([[[[9, 5.3], [15, 5.65], [9, -3.5]],
                                    [[12, 1.25], [3, -2.8], [12, -4.4]],
                                    [[4, -7.05], [15, -7.7], [4, -4.65]]]],
                                  dtype=self._dtype)

        unit = conv.Conv(DummyWorkflow(), n_kernels=weights.shape[0], kx=3,
                         ky=3)
        self._run_test(unit, device, input_data, weights, bias, gold_output)

    def test_1_channel_ocl(self):
        logging.info("start OpenCL conv. 1 channel layer forward"
                     "propagation...")
        self._1_channel_test(opencl.Device())
        logging.info("TEST PASSED")

    def test_1_channel_cpu(self):
        logging.info("start CPU conv. 1 channel layer forward propagation...")
        self._1_channel_test(None)
        logging.info("TEST PASSED")

    def _test_fixed_cpu(self):
        logging.info("Will test CPU convolutional layer forward propagation")

        inp = formats.Vector()
        dtype = opencl_types.dtypes[root.common.dtype]
        inp.mem = numpy.array([[[1, 2, 3, 2, 1],
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
        c.weights.mem[:] = weights.reshape(c.weights.mem.shape)[:]
        c.bias.map_invalidate()  # rewrite bias
        c.bias.mem[:] = bias[:]

        c.run()
        nz = numpy.count_nonzero(c.output.vv[c.output.mem.shape[0]:].ravel())
        self.assertEqual(nz, 0, "Overflow occured")

        y = c.output.mem.ravel()
        t = numpy.array([9, 5.3, 15, 5.65, 9, -3.5,
                         12, 1.25, 3, -2.8, 12, -4.4,
                         4, -7.05, 15, -7.7, 4, -4.65], dtype=dtype)
        max_diff = numpy.fabs(t - y).max()
        self.assertLess(max_diff, 0.0001,
                        "Result differs by %.6f" % (max_diff))

        logging.info("All Ok")

    def _test_padding_sliding(self):
        logging.info("Will test convolutional layer forward propagation")

        inp = formats.Vector()
        dtype = opencl_types.dtypes[root.common.dtype]
        inp.mem = numpy.array([[[1, 2, 3, 2, 1],
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
        c.weights.mem[:] = weights.reshape(c.weights.mem.shape)[:]
        c.bias.map_invalidate()  # rewrite bias
        c.bias.mem[:] = bias[:]

        c.run()
        c.output.map_read()  # get results back
        nz = numpy.count_nonzero(c.output.vv[c.output.mem.shape[0]:].ravel())
        self.assertEqual(nz, 0, "Overflow occured")

        y = c.output.mem.ravel()
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
        inp.mem = numpy.zeros([27, 28, 28], dtype=dtype)
        rnd.get().fill(inp.mem)

        c = Unit(DummyWorkflow(), n_kernels=25, kx=9, ky=9)
        c.input = inp

        c.initialize(device=self.device)

        weights = c.weights.mem.reshape(c.n_kernels, c.ky, c.kx)
        bias = c.bias.mem

        t0 = time.time()
        c.run()
        dt0 = time.time() - t0
        logging.info("OpenCL convolved in %.2f seconds" % (dt0))

        c.output.map_read()  # get results back
        nz = numpy.count_nonzero(c.output.vv[c.output.mem.shape[0]:].ravel())
        self.assertEqual(nz, 0, "Overflow occured")

        logging.info("Numpy")
        t0 = time.time()
        pp = []
        for mem in inp.mem:
            for j, w in enumerate(weights):
                ww = w.copy()
                for i in range(w.shape[0]):
                    ww[-(i + 1)] = w[i]
                www = ww.copy()
                for i in range(w.shape[1]):
                    www[:, -(i + 1)] = ww[:, i]
                out = scipy.signal.convolve2d(mem, www, "valid")
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
        for vv in c.output.mem:
            for i_kernel in range(len(weights)):
                p = pp[offs]
                mem = vv[:, :, i_kernel].reshape(vv.shape[0], vv.shape[1])
                max_diff = numpy.fabs(mem.ravel() - p.ravel()).max()
                self.assertLess(max_diff, 0.0001,
                                "Result differs by %.6f" % (max_diff))
                offs += 1

        logging.info("All Ok")

    def _do_test_vs_python_rgb(self, Unit):

        logging.info("OpenCL")

        inp = formats.Vector()
        dtype = opencl_types.dtypes[root.common.dtype]
        inp.mem = numpy.zeros([3, 128, 128, 3], dtype=dtype)
        rnd.get().fill(inp.mem)

        c = Unit(DummyWorkflow(), n_kernels=4, kx=3, ky=3)
        c.input = inp

        c.initialize(device=self.device)

        c.bias.map_invalidate()  # rewrite bias
        c.bias.mem[:] = 0

        t0 = time.time()
        c.run()
        dt0 = time.time() - t0
        logging.info("OpenCL convolved in %.2f seconds" % (dt0))

        c.output.map_read()  # get results back
        nz = numpy.count_nonzero(c.output.vv[c.output.mem.shape[0]:].ravel())
        self.assertEqual(nz, 0, "Overflow occured")

        logging.info("Numpy with FFT")
        t0 = time.time()
        pp = []
        for mem in inp.mem:
            for w_ in c.weights.mem:
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
                pp.append(scipy.signal.fftconvolve(mem, wwww, "valid"))
        dt1 = time.time() - t0
        logging.info("Numpy convolved in %.2f seconds" % (dt1))

        logging.info("OpenCL was %.2f times faster than Numpy" % (dt1 / dt0))

        logging.info("Will compare results")

        offs = 0
        for vv in c.output.mem:
            for i_kernel in range(len(c.weights.mem)):
                p = pp[offs]
                mem = vv[:, :, i_kernel].reshape(vv.shape[0], vv.shape[1])
                max_diff = numpy.fabs(mem.ravel() - p.ravel()).max()
                self.assertLess(max_diff, 0.0001,
                                "Result differs by %.6f" % (max_diff))
                offs += 1

        logging.info("All Ok")

    def _test_vs_python_rgb(self):
        logging.info("Will test linear convolutional"
                     " layer vs python on color image")
        self._do_test_vs_python_rgb(conv.Conv)

    def _test_vs_python(self):
        logging.info("Will test linear convolutional layer vs python on image")
        self._do_test_vs_python(conv.ConvTanh)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
