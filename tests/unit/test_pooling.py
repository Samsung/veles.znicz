"""
Created on Dec 4, 2013

Unit test for pooling layer forward propagation.

Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""


import logging
import numpy
import unittest

from veles.config import root
import veles.formats as formats
import veles.opencl as opencl
import veles.opencl_types as opencl_types
import veles.prng as prng
import veles.znicz.gd_pooling as gd_pooling
import veles.znicz.pooling as pooling
import veles.znicz.depooling as depooling
from veles.dummy import DummyWorkflow
from veles.znicz.tests.unit.gd_numdiff import GDNumDiff


class TestMaxPooling(unittest.TestCase):
    def setUp(self):
        root.common.unit_test = True
        root.common.plotters_disabled = True

        self._input = formats.Vector()
        self._dtype = opencl_types.dtypes[root.common.precision_type]
        self._input.mem = numpy.array(
            [3, 4, 3, 1, -1, -2, 1, 3, 2, 3, 3, 0, 4, 1,
             (-2), 0, 4, 4, -2, 1, 3, -3, -3, 4, 1, -3, -2, -4,
             (-3), 2, -1, 4, 2, 0, -3, 3, 1, -3, -4, -3, 0, -3,
             (-1), 0, -2, 2, 2, -4, -1, -1, 0, -2, 1, 3, 1, 2,
             2, -2, 4, 0, -1, 0, 1, 0, 0, 3, -3, 3, -1, 1,
             4, 0, -1, -2, 3, 4, -4, -2, -4, 3, -2, -3, -1, -1,
             (-1), -3, 3, 3, -2, -1, 3, 2, -1, -2, 4, -1, 2, 4,
             (-2), -1, 1, 3, -2, -2, 0, -2, 0, 4, -1, -2, -2, -3,
             3, 2, -2, 3, 1, -3, -2, -1, 4, -2, 0, -3, -1, 2,
             2, -3, -1, -1, -3, -2, 2, 3, 0, -2, 1, 2, 0, -3,
             (-4), 1, -1, 2, -1, 0, 3, -2, 4, -3, 4, 4, 1, -4,
             0, -1, 1, 3, 0, 1, 3, 4, -3, 2, 4, 3, -1, 0,
             (-1), 0, 1, -2, -4, 0, -4, -4, 2, 3, 2, -3, 1, 1,
             1, -1, -4, 3, 1, -1, -3, -4, -4, 3, -1, -4, -1, 0,
             (-1), -3, 4, 1, 2, -1, -2, -3, 3, 1, 3, -3, 4, -2],
            dtype=self._dtype).reshape(3, 5, 7, 2)

        self._gold_output = numpy.array(
            [[[[4, 4], [3, 3], [3, 4], [4, -4]],
              [[-3, 4], [-3, -4], [-4, -3], [1, -3]],
              [[4, -2], [-1, 0], [-3, 3], [-1, 1]]],
             [[[4, -3], [-4, 4], [-4, 3], [2, 4]],
              [[3, 3], [-2, -3], [4, 4], [-2, -3]],
              [[2, -3], [-3, 3], [1, -2], [0, -3]]],
             [[[-4, 3], [3, 4], [4, 4], [1, -4]],
              [[-4, 3], [-4, -4], [-4, -4], [1, 1]],
              [[4, -3], [2, -3], [3, -3], [4, -2]]]], dtype=self._dtype)

        self._gold_offs = numpy.array(
            [[[[16, 1], [20, 7], [10, 23], [12, 27]],
              [[28, 31], [34, 47], [38, 37], [54, 41]],
              [[58, 57], [60, 61], [66, 65], [68, 69]]],
             [[[70, 85], [76, 75], [78, 79], [96, 97]],
              [[112, 101], [102, 117], [120, 107], [110, 111]],
              [[126, 127], [130, 133], [136, 135], [138, 139]]],
             [[[140, 157], [146, 161], [148, 151], [152, 153]],
              [[184, 185], [172, 175], [190, 193], [180, 181]],
              [[198, 197], [200, 203], [204, 207], [208, 209]]]],
            dtype=numpy.int32)

    def tearDown(self):
        pass

    def test_ocl(self):
        self._do_test(opencl.Device())

    def test_cpu(self):
        self._do_test(None)

    def _do_test(self, device):
        c = pooling.MaxAbsPooling(DummyWorkflow(), kx=2, ky=2)
        c.input = self._input
        c.initialize(device=device)
        c.run()
        c.output.map_read()

        max_diff = numpy.fabs(self._gold_output.ravel() -
                              c.output.mem.ravel()).max()
        self.assertLess(max_diff, 0.0001, "max difference in output matrix"
                        " is %.6f" % (max_diff))
        c.input_offset.map_read()
        self.assertTrue((c.input_offset.mem == self._gold_offs).all())


class TestStochasticPooling(unittest.TestCase):
    def setUp(self):
        root.common.unit_test = True
        root.common.plotters_disabled = True

        self.dtype = opencl_types.dtypes[root.common.precision_type]
        self.input = numpy.zeros([3, 17, 17, 7], dtype=self.dtype)
        prng.get().fill(self.input)

        self.random_state = prng.get().state

        self.device = opencl.Device()

    def _do_test(self, device, Unit):
        prng.get().state = self.random_state
        uniform = prng.Uniform(DummyWorkflow(), output_bytes=315)
        unit = Unit(DummyWorkflow(), kx=3, ky=3, sliding=(3, 3),
                    uniform=uniform)
        unit.input = formats.Vector(self.input.copy())
        unit.initialize(device=device)
        unit.run()
        unit.output.map_read()
        unit.input_offset.map_read()
        return unit.output.mem.copy(), unit.input_offset.mem.copy()

    def _test_gpu_cpu(self, Unit):
        c, d = self._do_test(None, Unit)
        a, b = self._do_test(self.device, Unit)
        a -= c
        b -= d
        self.assertEqual(numpy.count_nonzero(a), 0)
        self.assertEqual(numpy.count_nonzero(b), 0)

    def test_max(self):
        self._test_gpu_cpu(pooling.StochasticPooling)

    def test_maxabs(self):
        self._test_gpu_cpu(pooling.StochasticAbsPooling)


class TestGDMaxPooling(unittest.TestCase):
    def setUp(self):
        root.common.unit_test = True
        root.common.plotters_disabled = True

        self._dtype = opencl_types.dtypes[root.common.precision_type]
        self._input = numpy.array(
            [[[3, 3, -1, 1, 2, 3, 4],
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
              [-1, 4, 2, -2, 3, 3, 4]]], dtype=self._dtype)
        self._input.shape = (3, 5, 7, 1)
        self._input_offset = numpy.array(
            [8, 10, 5, 6, 14, 17, 19, 27, 29, 30, 33, 34,
             35, 38, 39, 48, 56, 51, 60, 55, 63, 65, 68, 69,
             70, 73, 74, 76, 92, 86, 95, 90, 99, 100,
             102, 104], dtype=numpy.int32)
        self._err_output = numpy.array(
            [1, 3, 0.5, -4, 1, -2, -3, -1, -1, 3, -3, -0.5,
             4, -4, -0.3, -3, -1, -3, 2, -2, -4, 2, -1, -3,
             (-4), 2, 3, 2, -1, -1, -3, 4, -2, 2, 0.3, -4], dtype=self._dtype)
        self._gold_err_input = numpy.array(
            [[[0, 0, 0, 0, 0, 0.5, -4],
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
              [0, -2, 2, 0, 0.3, 0, -4]]], dtype=self._dtype)

        if not hasattr(self, "device"):
            self.device = opencl.Device()

    def test_fixed_gpu(self):
        return self._test_fixed(self.device)

    def test_fixed_cpu(self):
        return self._test_fixed(None)

    def _test_fixed(self, device):
        logging.info('starting OpenCL max pooling layer gradient descent '
                     'test...')
        c = gd_pooling.GDMaxPooling(DummyWorkflow(), kx=2, ky=2,
                                    sliding=(2, 2))
        c.input = formats.Vector()
        c.input.mem = self._input.copy()
        c.input_offset = formats.Vector()
        c.input_offset.mem = self._input_offset.copy()
        c.err_output = formats.Vector()
        c.err_output.mem = self._err_output.copy()
        c.initialize(device=device)
        c.err_input.map_invalidate()
        c.err_input.mem[:] = 1.0e30
        c.run()
        c.err_input.map_read()  # get results back
        max_diff = numpy.fabs(self._gold_err_input.ravel() -
                              c.err_input.mem.ravel()).max()
        self.assertLess(max_diff, 0.0001,
                        "max difference in err_input is %.6f" % (max_diff))
        logging.info("test passed")

        # We cannot check by numeric differentiation here
        # 'cause of the non-differentiable function "max".


class TestAvgPooling(unittest.TestCase):
    def setUp(self):
        root.common.unit_test = True
        root.common.plotters_disabled = True

        self._input = formats.Vector()
        self._dtype = opencl_types.dtypes[root.common.precision_type]
        self._input.mem = numpy.array(
            [3, 4, 3, 1, -1, -2, 1, 3, 2, 3, 3, 0, 4, 1,
             (-2), 0, 4, 4, -2, 1, 3, -3, -3, 4, 1, -3, -2, -4,
             (-3), 2, -1, 4, 2, 0, -3, 3, 1, -3, -4, -3, 0, -3,
             (-1), 0, -2, 2, 2, -4, -1, -1, 0, -2, 1, 3, 1, 2,
             2, -2, 4, 0, -1, 0, 1, 0, 0, 3, -3, 3, -1, 1,
             4, 0, -1, -2, 3, 4, -4, -2, -4, 3, -2, -3, -1, -1,
             (-1), -3, 3, 3, -2, -1, 3, 2, -1, -2, 4, -1, 2, 4,
             (-2), -1, 1, 3, -2, -2, 0, -2, 0, 4, -1, -2, -2, -3,
             3, 2, -2, 3, 1, -3, -2, -1, 4, -2, 0, -3, -1, 2,
             2, -3, -1, -1, -3, -2, 2, 3, 0, -2, 1, 2, 0, -3,
             (-4), 1, -1, 2, -1, 0, 3, -2, 4, -3, 4, 4, 1, -4,
             0, -1, 1, 3, 0, 1, 3, 4, -3, 2, 4, 3, -1, 0,
             (-1), 0, 1, -2, -4, 0, -4, -4, 2, 3, 2, -3, 1, 1,
             1, -1, -4, 3, 1, -1, -3, -4, -4, 3, -1, -4, -1, 0,
             (-1), -3, 4, 1, 2, -1, -2, -3, 3, 1, 3, -3, 4, -2],
            dtype=self._dtype).reshape(3, 5, 7, 2)

        self._gold_output = numpy.array(
            [[[[2, 2.25], [0.25, -0.25], [0.75, 1], [1, -1.5]],
              [[-1.75, 2], [0, -0.5], [-0.5, -1.25], [0.5, -0.5]],
              [[3, -1], [0, 0], [-1.5, 3], [-1, 1]]],
             [[[1.25, -0.5], [0, 0.75], [-0.75, -0.75], [0.5, 1.5]],
              [[0, 1.75], [-0.75, -2], [0.75, -0.75], [-1.5, -0.5]],
              [[0.5, -2], [-0.5, 0.5], [0.5, 0], [0, -3]]],
             [[[-1, 1.25], [1.25, 0.75], [2.25, 1.5], [0, -2]],
              [[-0.75, 0], [-2.5, -2.25], [-0.25, -0.25], [0, 0.5]],
              [[1.5, -1], [0, -2], [3, -1], [4, -2]]]], dtype=self._dtype)

    def tearDown(self):
        pass

    def test_ocl(self):
        self._do_test(opencl.Device())

    def test_cpu(self):
        self._do_test(None)

    def _do_test(self, device):
        c = pooling.AvgPooling(DummyWorkflow(), kx=2, ky=2)
        c.input = self._input
        cur_device = opencl.Device()
        c.initialize(device=cur_device)
        c.run()
        c.output.map_read()
        max_diff = numpy.fabs(self._gold_output.ravel() -
                              c.output.mem.ravel()).max()
        self.assertLess(max_diff, 0.0001, "max difference in output matrix"
                        " is %.6f" % (max_diff))


class TestGDAvgPooling(unittest.TestCase, GDNumDiff):
    def setUp(self):
        root.common.unit_test = True
        root.common.plotters_disabled = True

        self._dtype = opencl_types.dtypes[root.common.precision_type]
        self._input = numpy.array([
            [[[3, 6], [3, 6], [-1, -2], [1, 2], [2, 4], [3, 6], [4, 8]],
             [[-2, -4], [4, 8], [-2, -4], [3, 6], [-3, -6], [1, 2], [-2, -4]],
             [[-3, -6], [-1, -2], [2, 4], [-3, -6], [1, 2], [-4, -8], [0, 0]],
             [[-1, -2], [-2, -4], [2, 4], [-1, -2], [0, 0], [1, 2], [1, 2]],
             [[2, 4], [4, 8], [-1, -2], [1, 2], [0, 0], [-3, -6], [-1, -2]]],
            [[[4, 8], [-1, -2], [3, 6], [-4, -8], [-4, -8], [-2, -4],
              [-1, -2]],
             [[-1, -2], [3, 6], [-2, -4], [3, 6], [-1, -2], [4, 8], [2, 4]],
             [[-2, -4], [1, 2], [-2, -4], [0, 0], [0, 0], [-1, -2], [-2, -4]],
             [[3, 6], [-2, -4], [1, 2], [-2, -4], [4, 8], [0, 0], [-1, -2]],
             [[2, 4], [-1, -2], [-3, -6], [2, 4], [0, 0], [1, 2], [0, 0]]],
            [[[-4, -8], [-1, -2], [-1, -2], [3, 6], [4, 8], [4, 8], [1, 2]],
             [[0, 0], [1, 2], [0, 0], [3, 6], [-3, -6], [4, 8], [-1, -2]],
             [[-1, -2], [1, 2], [-4, -8], [-4, -8], [2, 4], [2, 4], [1, 2]],
             [[1, 2], [-4, -8], [1, 2], [-3, -6], [-4, -8], [-1, -2],
              [-1, -2]],
             [[-1, -2], [4, 8], [2, 2], [-2, -4],
              [3, 6], [3, 6], [4, 8]]]], dtype=self._dtype)
        self._err_output = numpy.array(
            [[[[1, 2], [3, 6], [0.5, 1], [-4, -8]],
              [[1, 2], [-2, -4], [-3, -6], [-1, -2]],
              [[-1, -2], [3, 6], [-3, -6], [-0.5, -1]]],
             [[[4, 8], [-4, -8], [-0.3, -0.6], [-3, -6]],
              [[-1, -2], [-3, -6], [2, 4], [-2, -4]],
              [[-4, -8], [2, 4], [-1, -2], [-3, -6]]],
             [[[-4, -8], [2, 4], [3, 6], [2, 4]],
              [[-1, -2], [-1, -2], [-3, -6], [4, 8]],
              [[-2, -4], [2, 4], [0.3, 0.6], [-4, -8]]]], dtype=self._dtype)
        self._gold_err_input = numpy.array([
            [[[0.25, 0.5], [0.25, 0.5], [0.75, 1.5], [0.75, 1.5],
              [0.125, 0.25], [0.125, 0.25], [-2, -4]],
             [[0.25, 0.5], [0.25, 0.5], [0.75, 1.5], [0.75, 1.5],
              [0.125, 0.25], [0.125, 0.25], [-2, -4]],
             [[0.25, 0.5], [0.25, 0.5], [-0.5, -1], [-0.5, -1],
              [-0.75, -1.5], [-0.75, -1.5], [-0.5, -1]],
             [[0.25, 0.5], [0.25, 0.5], [-0.5, -1], [-0.5, -1],
              [-0.75, -1.5], [-0.75, -1.5], [-0.5, -1]],
             [[-0.5, -1], [-0.5, -1], [1.5, 3], [1.5, 3],
              [-1.5, -3], [-1.5, -3], [-0.5, -1]]],
            [[[1, 2], [1, 2], [-1, -2], [-1, -2],
              [-0.075, -0.15], [-0.075, -0.15], [-1.5, -3]],
             [[1, 2], [1, 2], [-1, -2], [-1, -2],
              [-0.075, -0.15], [-0.075, -0.15], [-1.5, -3]],
             [[-0.25, -0.5], [-0.25, -0.5], [-0.75, -1.5], [-0.75, -1.5],
              [0.5, 1], [0.5, 1], [-1, -2]],
             [[-0.25, -0.5], [-0.25, -0.5], [-0.75, -1.5], [-0.75, -1.5],
              [0.5, 1], [0.5, 1], [-1, -2]],
             [[-2, -4], [-2, -4], [1, 2], [1, 2],
              [-0.5, -1], [-0.5, -1], [-3, -6]]],
            [[[-1, -2], [-1, -2], [0.5, 1], [0.5, 1],
              [0.75, 1.5], [0.75, 1.5], [1, 2]],
             [[-1, -2], [-1, -2], [0.5, 1], [0.5, 1],
              [0.75, 1.5], [0.75, 1.5], [1, 2]],
             [[-0.25, -0.5], [-0.25, -0.5], [-0.25, -0.5], [-0.25, -0.5],
              [-0.75, -1.5], [-0.75, -1.5], [2, 4]],
             [[-0.25, -0.5], [-0.25, -0.5], [-0.25, -0.5], [-0.25, -0.5],
              [-0.75, -1.5], [-0.75, -1.5], [2, 4]],
             [[-1, -2], [-1, -2], [1, 2], [1, 2],
              [0.15, 0.3], [0.15, 0.3], [-4, -8]]]], dtype=self._dtype)

        if not hasattr(self, "device"):
            self.device = opencl.Device()

    def _test_fixed_gpu(self):
        return self._test_fixed(self.device)

    def _test_fixed_cpu(self):
        return self._test_fixed(None)

    def _test_fixed(self, device):
        logging.info('starting OpenCL avg pooling layer gradient descent '
                     'test...')
        c = gd_pooling.GDAvgPooling(DummyWorkflow(), kx=2, ky=2)
        c.input = formats.Vector()
        c.input.mem = self._input.copy()
        c.err_output = formats.Vector()
        c.err_output.mem = self._err_output.copy()
        c.initialize(device=device)
        c.err_input.map_invalidate()
        c.err_input.mem[:] = 1.0e30
        c.run()
        c.err_input.map_read()  # get results back

        max_diff = numpy.fabs(self._gold_err_input.ravel() -
                              c.err_input.mem.ravel()).max()
        self.assertLess(max_diff, 0.0001, "max difference in err_input matrix"
                        " is %.6f" % (max_diff))
        logging.info("test passed")

        logging.info("reverifying by numeric differentiation...")

        forward = pooling.AvgPooling(DummyWorkflow(), kx=c.kx, ky=c.ky)
        forward.input = formats.Vector()
        forward.input.mem = self._input.copy()
        forward.initialize(device=self.device)
        forward.run()
        forward.output.map_read()
        target = forward.output.mem.ravel() - self._err_output.ravel()

        self.numdiff_check_gd(forward, self._input, None, None, target,
                              c.err_input.mem, None, None,
                              logging.info, self.assertLess)

        logging.info("test passed")

    def test_random_numeric_gpu(self):
        self._test_random_numeric(self.device, (3, 3))
        self._test_random_numeric(self.device, (1, 1))

    def test_random_numeric_cpu(self):
        self._test_random_numeric(None, (3, 3))
        self._test_random_numeric(None, (1, 1))

    def _test_random_numeric(self, device, sliding):
        logging.info("Will test AvgPooling layer forward-backward "
                     "via numeric differentiation")

        inp = numpy.zeros([2, 6, 6, 3], dtype=self._dtype)
        prng.get().fill(inp)
        forward = pooling.AvgPooling(DummyWorkflow(), kx=3, ky=3,
                                     sliding=sliding)
        forward.input = formats.Vector()
        forward.input.mem = inp.copy()
        forward.initialize(device=self.device)
        forward.run()

        forward.output.map_read()
        target = numpy.zeros_like(forward.output.mem)
        prng.get().fill(target)
        err_output = forward.output.mem - target

        c = gd_pooling.GDAvgPooling(
            DummyWorkflow(), kx=forward.kx, ky=forward.ky,
            sliding=forward.sliding)
        c.err_output = formats.Vector()
        c.err_output.mem = err_output.copy()
        c.input = formats.Vector()
        c.input.mem = inp.copy()
        c.output = formats.Vector()
        c.output.mem = c.err_output.mem.copy()
        c.initialize(device=device)
        c.run()
        c.err_input.map_read()

        err_input = c.err_input.mem.ravel()

        self.numdiff_check_gd(forward, inp, None, None, target,
                              err_input, None, None,
                              logging.info, self.assertLess,
                              error_function_averaged=False)


class TestStochasticPoolingDepooling(unittest.TestCase):
    def setUp(self):
        root.common.unit_test = True
        root.common.plotters_disabled = True

        self.dtype = opencl_types.dtypes[root.common.precision_type]
        self.input = numpy.zeros([3, 17, 17, 7], dtype=self.dtype)
        prng.get().fill(self.input)

        self.random_state = prng.get().state

        self.device = opencl.Device()

    def _do_test(self, device, Unit, Forward):
        prng.get().state = self.random_state
        uniform = prng.Uniform(DummyWorkflow(), output_bytes=315)
        unit = Unit(DummyWorkflow(), kx=3, ky=3, sliding=(3, 3),
                    uniform=uniform)
        unit.input = formats.Vector(self.input.copy())
        unit.initialize(device=device)
        unit.run()
        unit.input.map_read()

        prng.get().state = self.random_state
        uniform = prng.Uniform(DummyWorkflow(), output_bytes=315)
        forward = Forward(DummyWorkflow(), kx=3, ky=3, sliding=(3, 3),
                          uniform=uniform)
        forward.input = formats.Vector(self.input.copy())
        forward.initialize(device=device)
        forward.run()

        de = depooling.Depooling(DummyWorkflow())
        de.output_offset = forward.input_offset
        de.input = forward.output
        de.get_output_shape_from = forward.input
        de.initialize(device)
        de.run()
        de.output.map_read()

        diff = de.output.mem - unit.input.mem
        self.assertEqual(numpy.count_nonzero(diff), 0)

        return unit.input.mem.copy()

    def test_max(self):
        self._do_test(self.device,
                      pooling.StochasticPoolingDepooling,
                      pooling.StochasticPooling)

    def test_maxabs(self):
        self._do_test(self.device,
                      pooling.StochasticAbsPoolingDepooling,
                      pooling.StochasticAbsPooling)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    unittest.main()
