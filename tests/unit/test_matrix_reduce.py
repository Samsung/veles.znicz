"""
Created on Nov 22, 2013

Unit test for OpenCL kernel which does reduce over matrix rows or columns.

@author: Kazantsev Alexey <a.kazantsev@samsung.com>
"""
import unittest
import units
import formats
import opencl
import pyopencl
import config
import numpy
import rnd


class TestMatrixReduce(unittest.TestCase):
    def setUp(self):
        self.device = opencl.Device()

    def tearDown(self):
        del self.device
        units.pool.shutdown()

    def build_program(self, a, b, A_WIDTH, A_HEIGHT, A_COL, REDUCE_SIZE):
        defines = ("%s\n"
                   "#define A_WIDTH %d\n"
                   "#define A_HEIGHT %d\n"
                   "%s\n"
                   "#define REDUCE_SIZE %d\n" % (
                   config.cl_defines[config.c_dtype],
                   A_WIDTH, A_HEIGHT,
                   "#define A_COL" if A_COL else "",
                   REDUCE_SIZE))

        src = (
        "__kernel __attribute__((reqd_work_group_size(REDUCE_SIZE, 1, 1)))\n"
        "void test(__global c_dtype *A, __global c_dtype *b) {\n"
        "MX_REDUCE\n"
        "if (!tx) {\n"
        "  sum += AS[0];\n"
        "  b[bx] = sum;\n"
        "}}")
        fnme = "%s/test.cl" % (config.cache_dir)
        fout = open(fnme, "w")
        fout.write(src)
        fout.close()

        tmp = units.OpenCLUnit(self.device)
        tmp.cl_sources_[fnme] = ""
        tmp.build_program(defines, fnme)

        krn = pyopencl.Kernel(tmp.prg_, "test")
        krn.set_arg(0, a.v_)
        krn.set_arg(1, b.v_)
        return krn

    def test_fixed(self):
        """Test with fixed input.
        """
        dtype = config.dtypes[config.c_dtype]

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
            krn = self.build_program(a, b, a.v.shape[1], a.v.shape[0], True,
                                     reduce_size)
            global_size = [a.v.shape[1] * reduce_size]
            local_size = [reduce_size]
            ev = pyopencl.enqueue_nd_range_kernel(self.device.queue_,
                krn, global_size, local_size)
            ev.wait()
            b.map_write()
            max_diff = numpy.fabs(b[:a.v.shape[1]] - t).max()
            self.assertLess(max_diff, 0.0001,
                            "Result differs by %.6f" % (max_diff))
            self.assertEqual(numpy.count_nonzero(b.v[a.v.shape[1]:]), 0,
                "Written some values outside of the target array bounds")
            b.v[:] = 0
            b.unmap()

    def test_random(self):
        """Test with random input vs numpy.
        """
        dtype = config.dtypes[config.c_dtype]

        a = formats.Vector()
        a.v = numpy.zeros([3337, 775], dtype=dtype)
        rnd.default.fill(a.v)

        t_col = numpy.sum(a.v, axis=0)
        t = numpy.sum(a.v, axis=1)

        b = formats.Vector()
        b.v = numpy.zeros(numpy.max(a.v.shape) * 2, dtype=dtype)

        a.initialize(self.device)
        b.initialize(self.device)

        for reduce_size in range(4, 65, 4):
            krn = self.build_program(a, b, a.v.shape[1], a.v.shape[0], True,
                                     reduce_size)
            global_size = [a.v.shape[1] * reduce_size]
            local_size = [reduce_size]
            ev = pyopencl.enqueue_nd_range_kernel(self.device.queue_,
                krn, global_size, local_size)
            ev.wait()
            b.map_write()
            max_diff = numpy.fabs(b[:a.v.shape[1]] - t_col).max()
            self.assertLess(max_diff, 0.0001,
                            "Result differs by %.6f" % (max_diff))
            self.assertEqual(numpy.count_nonzero(b.v[a.v.shape[1]:]), 0,
                "Written some values outside of the target array bounds")
            b.v[:] = 0
            b.unmap()

            krn = self.build_program(a, b, a.v.shape[1], a.v.shape[0], False,
                                     reduce_size)
            global_size = [a.v.shape[0] * reduce_size]
            local_size = [reduce_size]
            ev = pyopencl.enqueue_nd_range_kernel(self.device.queue_,
                krn, global_size, local_size)
            ev.wait()
            b.map_write()
            max_diff = numpy.fabs(b[:a.v.shape[0]] - t).max()
            self.assertLess(max_diff, 0.0001,
                            "Result differs by %.6f" % (max_diff))
            self.assertEqual(numpy.count_nonzero(b.v[a.v.shape[0]:]), 0,
                "Written some values outside of the target array bounds")
            b.v[:] = 0
            b.unmap()


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
