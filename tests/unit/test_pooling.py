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
import veles.znicz.gd_pooling as gd_pooling
import veles.znicz.pooling as pooling
from veles.tests.dummy_workflow import DummyWorkflow


class TestMaxPooling(unittest.TestCase):
    def setUp(self):
        root.common.unit_test = True
        root.common.plotters_disabled = True

        self._inp = formats.Vector()
        self._dtype = opencl_types.dtypes[root.common.dtype]
        self._inp.v = numpy.array(
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
        logging.info('starting OpenCL max pooling layer forward propagation '
                     'test...')
        c = pooling.MaxPooling(DummyWorkflow(), kx=2, ky=2)
        c.input = self._inp
        cur_device = opencl.Device()
        c.initialize(device=cur_device)
        c.run()
        c.output.map_read()

        max_diff = numpy.fabs(self._gold_output.ravel() -
                              c.output.v.ravel()).max()
        self.assertLess(max_diff, 0.0001, "max difference in output matrix"
                        " is %.6f" % (max_diff))
        c.input_offs.map_read()
        max_diff = numpy.fabs(self._gold_offs.ravel() -
                              c.input_offs.v.ravel()).max()
        self.assertLess(max_diff, 0.0001, "max difference in offs matrix"
                        " is %.6f" % (max_diff))
        logging.info("test passed")

    def test_cpu(self):
        logging.info('starting CPU max pooling layer forward propagation '
                     'test...')
        c = pooling.MaxPooling(DummyWorkflow(), kx=2, ky=2)
        c.input = self._inp
        c.initialize(device=None)
        c.run()

        max_diff = numpy.fabs(self._gold_output.ravel() -
                              c.output.v.ravel()).max()
        self.assertLess(max_diff, 0.0001, "max difference in output matrix"
                        " is %.6f" % (max_diff))
        max_diff = numpy.fabs(self._gold_offs.ravel() -
                              c.input_offs.v.ravel()).max()
        self.assertLess(max_diff, 0.0001, "max difference in offs matrix"
                        " is %.6f" % (max_diff))
        logging.info("test passed")


class TestGDMaxPooling(unittest.TestCase):
    def setUp(self):
        root.common.unit_test = True
        root.common.plotters_disabled = True

        self._inp = formats.Vector()
        self._dtype = opencl_types.dtypes[root.common.dtype]
        self._inp.v = numpy.array(
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
        self._inp.v = self._inp.v.reshape(3, 5, 7, 1)
        self._input_offs = formats.Vector()
        self._input_offs.v = numpy.array(
            [8, 10, 5, 6, 14, 17, 19, 27, 29, 30, 33, 34,
             35, 38, 39, 48, 56, 51, 60, 55, 63, 65, 68, 69,
             70, 73, 74, 76, 92, 86, 95, 90, 99, 100,
             102, 104], dtype=numpy.int32)
        self._err_output = formats.Vector()
        self._err_output.v = numpy.array(
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

    def tearDown(self):
        pass

    def test_ocl(self):
        logging.info('starting OpenCL max pooling layer gradient descent '
                     'test...')
        c = gd_pooling.GDMaxPooling(DummyWorkflow(), kx=2, ky=2)
        c.input = self._inp
        c.input_offs = self._input_offs
        c.err_output = self._err_output
        cur_device = opencl.Device()
        c.initialize(device=cur_device)
        c.err_input.map_write()
        c.err_input.v[:] = 1.0e30
        c.run()
        c.err_input.map_read()  # get results back
        max_diff = numpy.fabs(self._gold_err_input.ravel() -
                              c.err_input.v.ravel()).max()
        self.assertLess(max_diff, 0.0001,
                        "max difference in err_input is %.6f" % (max_diff))
        logging.info("test passed")

    def test_cpu(self):
        logging.info('starting CPU max pooling layer gradient descent '
                     'test...')
        c = gd_pooling.GDMaxPooling(DummyWorkflow(), kx=2, ky=2)
        c.input = self._inp
        c.input_offs = self._input_offs
        c.err_output = self._err_output
        c.initialize(device=None)
        c.err_input.v[:] = 1.0e30
        c.run()

        max_diff = numpy.fabs(self._gold_err_input.ravel() -
                              c.err_input.v.ravel()).max()
        self.assertLess(max_diff, 0.0001,
                        "max difference in err_input is %.6f" % (max_diff))
        logging.info("test passed")


class TestAvgPooling(unittest.TestCase):
    def setUp(self):
        root.common.unit_test = True
        root.common.plotters_disabled = True

        self._inp = formats.Vector()
        self._dtype = opencl_types.dtypes[root.common.dtype]
        self._inp.v = numpy.array(
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
        logging.info('starting OpenCL avg pooling layer forward propagation '
                     'test...')
        c = pooling.AvgPooling(DummyWorkflow(), kx=2, ky=2)
        c.input = self._inp
        cur_device = opencl.Device()
        c.initialize(device=cur_device)
        c.run()
        c.output.map_read()
        max_diff = numpy.fabs(self._gold_output.ravel() -
                              c.output.v.ravel()).max()
        self.assertLess(max_diff, 0.0001, "max difference in output matrix"
                        " is %.6f" % (max_diff))
        logging.info("test passed")

    def test_cpu(self):
        logging.info('starting CPU avg pooling layer forward propagation '
                     'test...')
        c = pooling.AvgPooling(DummyWorkflow(), kx=2, ky=2)
        c.input = self._inp
        c.initialize(device=None)

        c.run()
        max_diff = numpy.fabs(self._gold_output.ravel() -
                              c.output.v.ravel()).max()
        self.assertLess(max_diff, 0.0001, "max difference in output matrix"
                        " is %.6f" % (max_diff))
        logging.info("test passed")


class TestGDAvgPooling(unittest.TestCase):
    def setUp(self):
        root.common.unit_test = True
        root.common.plotters_disabled = True

        self._inp = formats.Vector()
        self._dtype = opencl_types.dtypes[root.common.dtype]
        self._inp.v = numpy.array([
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
        self._err_output = formats.Vector()
        self._err_output.v = numpy.array(
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

    def tearDown(self):
        pass

    def test_ocl(self):
        logging.info('starting OpenCL avg pooling layer gradient descent '
                     'test...')
        c = gd_pooling.GDAvgPooling(DummyWorkflow(), kx=2, ky=2)
        c.input = self._inp
        c.err_output = self._err_output
        cur_device = opencl.Device()
        c.initialize(device=cur_device)
        c.err_input.map_write()
        c.err_input.v[:] = 1.0e30
        c.run()
        c.err_input.map_read()  # get results back

        max_diff = numpy.fabs(self._gold_err_input.ravel() -
                              c.err_input.v.ravel()).max()
        self.assertLess(max_diff, 0.0001, "max difference in err_input matrix"
                        " is %.6f" % (max_diff))
        logging.info("test passed")

    def test_cpu(self):
        logging.info('starting CPU avg pooling layer gradient descent '
                     'test...')
        c = gd_pooling.GDAvgPooling(DummyWorkflow(), kx=2, ky=2)
        c.input = self._inp
        c.err_output = self._err_output
        c.initialize(device=None)
        c.err_input.v[:] = 1.0e30
        c.run()

        max_diff = numpy.fabs(self._gold_err_input.ravel() -
                              c.err_input.v.ravel()).max()
        self.assertLess(max_diff, 0.0001,
                        "max difference in err_input is %.6f" % (max_diff))
        logging.info("test passed")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    unittest.main()
