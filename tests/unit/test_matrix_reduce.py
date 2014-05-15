"""
Created on Nov 22, 2013

Unit test for OpenCL kernel which does reduce over matrix rows or columns.

Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""


import logging
import numpy
import unittest

from veles.config import root
import veles.formats as formats
import veles.opencl as opencl
import veles.opencl_types as opencl_types
from veles.opencl_units import OpenCLUnit
import veles.random_generator as rnd
from veles.tests.dummy_workflow import DummyWorkflow


class TestMatrixReduce(unittest.TestCase):
    def setUp(self):
        root.common.unit_test = True
        root.common.plotters_disabled = True
        self.device = opencl.Device()

    def tearDown(self):
        del self.device

    def _build_program(self, a, b, A_WIDTH, A_HEIGHT, A_COL, REDUCE_SIZE):
        defines = {"A_WIDTH": A_WIDTH,
                   "A_HEIGHT": A_HEIGHT,
                   "REDUCE_SIZE": REDUCE_SIZE}
        if A_COL:
            defines["A_COL"] = 1

        src = ("#include \"defines.cl\"\n"
               "__kernel __attribute__("
               "(reqd_work_group_size(REDUCE_SIZE, 1, 1)))\n"
               "void test(__global c_dtype *A, __global c_dtype *b) {\n"
               "#include \"matrix_reduce.cl\"\n"
               "if (!tx) {\n"
               "  sum += AS[0];\n"
               "  b[bx] = sum;\n"
               "}}")
        fnme = "%s/test.cl" % (root.common.cache_dir)
        fout = open(fnme, "w")
        fout.write(src)
        fout.close()

        tmp = OpenCLUnit(DummyWorkflow())
        tmp.initialize(device=self.device)
        tmp.cl_sources_[fnme] = {}
        tmp.build_program(defines, fnme)

        krn = tmp.get_kernel("test")
        krn.set_arg(0, a.v_)
        krn.set_arg(1, b.v_)
        return krn

    def test_fixed(self):
        """Test with fixed input.
        """
        dtype = opencl_types.dtypes[root.common.precision_type]

        a = formats.Vector()
        a.v = numpy.array([[1, 2, 3],
                           [-1, -2, -3],
                           [9, 8, 7],
                           [-3, -4, -5],
                           [-1, -2, -3],
                           [7, 7, 7]], dtype=dtype)
        b = formats.Vector()
        b.v = numpy.zeros(a.v.shape[1] * 2, dtype=dtype)

        t = numpy.array([12, 9, 6], dtype=dtype)

        a.initialize(self.device)
        b.initialize(self.device)

        for reduce_size in range(1, 11):
            krn = self._build_program(a, b, a.v.shape[1], a.v.shape[0], True,
                                      reduce_size)
            global_size = [a.v.shape[1] * reduce_size]
            local_size = [reduce_size]
            self.device.queue_.execute_kernel(global_size, local_size,
                                              krn).wait()
            b.map_write()
            max_diff = numpy.fabs(b[:a.v.shape[1]] - t).max()
            self.assertLess(max_diff, 0.0001,
                            "Result differs by %.6f" % (max_diff))
            self.assertEqual(
                numpy.count_nonzero(b.v[a.v.shape[1]:]), 0,
                "Written some values outside of the target array bounds")
            b.v[:] = 0
            b.unmap()

        logging.info("test_fixed() succeeded")

    def test_random(self):
        """Test with random input vs numpy.
        """
        dtype = opencl_types.dtypes[root.common.precision_type]

        a = formats.Vector()
        a.v = numpy.zeros([3337, 775], dtype=dtype)
        rnd.get().fill(a.v)

        t_col = numpy.sum(a.v, axis=0)
        t = numpy.sum(a.v, axis=1)

        b = formats.Vector()
        b.v = numpy.zeros(numpy.max(a.v.shape) * 2, dtype=dtype)

        a.initialize(self.device)
        b.initialize(self.device)

        for reduce_size in range(4, 65, 4):
            krn = self._build_program(a, b, a.v.shape[1], a.v.shape[0], True,
                                      reduce_size)
            global_size = [a.v.shape[1] * reduce_size]
            local_size = [reduce_size]
            self.device.queue_.execute_kernel(global_size, local_size,
                                              krn).wait()
            b.map_write()
            max_diff = numpy.fabs(b[:a.v.shape[1]] - t_col).max()
            self.assertLess(max_diff, 0.0003,  # in case of float
                            "Result differs by %.6f" % (max_diff))
            self.assertEqual(
                numpy.count_nonzero(b.v[a.v.shape[1]:]), 0,
                "Written some values outside of the target array bounds")
            b.v[:] = 0
            b.unmap()

            krn = self._build_program(a, b, a.v.shape[1], a.v.shape[0], False,
                                      reduce_size)
            global_size = [a.v.shape[0] * reduce_size]
            local_size = [reduce_size]
            self.device.queue_.execute_kernel(global_size, local_size,
                                              krn).wait()
            b.map_write()
            max_diff = numpy.fabs(b[:a.v.shape[0]] - t).max()
            self.assertLess(max_diff, 0.0003,  # in case of float
                            "Result differs by %.6f" % (max_diff))
            self.assertEqual(
                numpy.count_nonzero(b.v[a.v.shape[0]:]), 0,
                "Written some values outside of the target array bounds")
            b.v[:] = 0
            b.unmap()

        logging.info("test_random() succeeded")


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
