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
import veles.random_generator as rnd
from veles.tests.dummy_workflow import DummyWorkflow
from veles.znicz.tests.unit import TrivialOpenCLUnit


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

        tmp = TrivialOpenCLUnit(DummyWorkflow())
        tmp.initialize(device=self.device)
        tmp.cl_sources_[fnme] = {}
        tmp.build_program(defines, fnme, show_ocl_logs=False)

        krn = tmp.get_kernel("test")
        krn.set_arg(0, a.devmem)
        krn.set_arg(1, b.devmem)
        return krn

    def test_fixed(self):
        """Test with fixed input.
        """
        dtype = opencl_types.dtypes[root.common.precision_type]

        a = formats.Vector()
        a.mem = numpy.array([[1, 2, 3],
                             [-1, -2, -3],
                             [9, 8, 7],
                             [-3, -4, -5],
                             [-1, -2, -3],
                             [7, 7, 7]], dtype=dtype)
        b = formats.Vector()
        b.mem = numpy.ones(a.mem.shape[1] * 2, dtype=dtype)

        t = numpy.array([12, 9, 6], dtype=dtype)

        a.initialize(self.device)
        b.initialize(self.device)

        for reduce_size in range(1, 11):
            krn = self._build_program(a, b, a.mem.shape[1],
                                      a.mem.shape[0], True,
                                      reduce_size)
            global_size = [a.mem.shape[1] * reduce_size]
            local_size = [reduce_size]
            self.device.queue_.execute_kernel(
                krn, global_size, local_size).wait()
            b.map_write()
            max_diff = numpy.fabs(b[:a.mem.shape[1]] - t).max()
            self.assertLess(max_diff, 0.0001,
                            "Result differs by %.6f" % (max_diff))
            self.assertEqual(
                numpy.count_nonzero(b.mem[a.mem.shape[1]:] - 1), 0,
                "Written some values outside of the target array bounds")
            b.unmap()

        logging.info("test_fixed() succeeded")

    def test_random(self):
        """Test with random input vs numpy.
        """
        dtype = opencl_types.dtypes[root.common.precision_type]

        a = formats.Vector()
        a.mem = numpy.zeros([3131, 1001], dtype=dtype)
        rnd.get().fill(a.mem)

        t_col = numpy.sum(a.mem, axis=0)
        t = numpy.sum(a.mem, axis=1)

        b = formats.Vector()
        b.mem = numpy.ones(numpy.max(a.mem.shape) * 2, dtype=dtype)

        a.initialize(self.device)
        b.initialize(self.device)

        for reduce_size in range(4, 64, 1):
            logging.info("reduce_size = %d", reduce_size)
            krn = self._build_program(a, b, a.mem.shape[1],
                                      a.mem.shape[0], True,
                                      reduce_size)
            global_size = [a.mem.shape[1] * reduce_size]
            local_size = [reduce_size]
            self.device.queue_.execute_kernel(
                krn, global_size, local_size).wait()
            b.map_write()
            max_diff = numpy.fabs(b[:a.mem.shape[1]] - t_col).max()
            self.assertLess(max_diff, 0.00032,  # in case of float
                            "Result differs by %.6f" % (max_diff))
            self.assertEqual(
                numpy.count_nonzero(b.mem[a.mem.shape[1]:] - 1), 0,
                "Written some values outside of the target array bounds")
            b[:] = 1
            b.unmap()

            krn = self._build_program(a, b, a.mem.shape[1],
                                      a.mem.shape[0], False,
                                      reduce_size)
            global_size = [a.mem.shape[0] * reduce_size]
            local_size = [reduce_size]
            self.device.queue_.execute_kernel(
                krn, global_size, local_size).wait()
            b.map_write()
            max_diff = numpy.fabs(b[:a.mem.shape[0]] - t).max()
            self.assertLess(max_diff, 0.00032,  # in case of float
                            "Result differs by %.6f" % (max_diff))
            self.assertEqual(
                numpy.count_nonzero(b.mem[a.mem.shape[0]:] - 1), 0,
                "Written some values outside of the target array bounds")
            b[:] = 1
            b.unmap()

        logging.info("test_random() succeeded")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
