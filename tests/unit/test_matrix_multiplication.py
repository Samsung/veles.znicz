"""
Created on Nov 8, 2013

Will test correctness of OpenCL matrix multiplication.

Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""


import logging
import numpy
import os
import unittest

from veles.config import root
import veles.formats as formats
import veles.opencl as opencl
import veles.random_generator as prng
from veles.tests.dummy_workflow import DummyWorkflow
from veles.znicz.tests.unit import TrivialOpenCLUnit


class TestMatrixMultiplication(unittest.TestCase):
    def setUp(self):
        root.common.unit_test = True
        root.common.plotters_disabled = True
        self.device = opencl.Device()

    def tearDown(self):
        del self.device

    def _do_cpu_tst(self):
        """Pure single core CPU test
        """
        dtype = (numpy.complex128 if self.a.mem.dtype in (
            numpy.complex64, numpy.complex128) else numpy.float64)
        a = numpy.empty(self.a.mem.shape, dtype=dtype)
        a[:] = self.a.mem[:]
        bt = self.b.mem.transpose()
        b = numpy.empty(bt.shape, dtype=dtype)
        b[:] = bt[:]
        bias = numpy.empty(self.bias.mem.shape, dtype=dtype)
        bias[:] = self.bias.mem[:]
        c = numpy.empty(self.c[0].shape, dtype=dtype)
        if self.a_col:
            a = a.transpose()
        if self.b_col:
            b = b.transpose()
        numpy.dot(a, b, c)
        c[:] += bias
        c *= 0.6666
        numpy.tanh(c, c)
        c *= 1.7159
        return c

    def _prepare_tsts(self, BLOCK_SIZE,
                      AB_WIDTH, B_HEIGHT, A_HEIGHT, a_col, b_col):
        self.AB_WIDTH = AB_WIDTH
        self.B_HEIGHT = B_HEIGHT
        self.A_HEIGHT = A_HEIGHT
        self.a_col = a_col
        self.b_col = b_col

        self.a = formats.Vector()
        self.a.mem = numpy.zeros([self.A_HEIGHT * self.AB_WIDTH],
                                 dtype=self.dtype)
        prng.get().fill(self.a.mem, -0.1, 0.1)
        if a_col:
            self.a.mem.shape = (self.AB_WIDTH, self.A_HEIGHT)
        else:
            self.a.mem.shape = (self.A_HEIGHT, self.AB_WIDTH)

        self.b = formats.Vector()
        self.b.mem = numpy.zeros([self.B_HEIGHT * self.AB_WIDTH],
                                 dtype=self.dtype)
        prng.get().fill(self.b.mem, -0.1, 0.1)
        if b_col:
            self.b.mem.shape = (self.AB_WIDTH, self.B_HEIGHT)
        else:
            self.b.mem.shape = (self.B_HEIGHT, self.AB_WIDTH)

        self.bias = formats.Vector()
        self.bias.mem = numpy.zeros([self.B_HEIGHT], dtype=self.dtype)
        prng.get().fill(self.bias.mem, -0.1, 0.1)

        self.c = formats.Vector()
        self.c.mem = numpy.ones([2, self.A_HEIGHT, self.B_HEIGHT],
                                dtype=self.dtype)

    def _cleanup_after_tsts(self):
        del self.c
        del self.bias
        del self.b
        del self.a
        del self.A_HEIGHT
        del self.B_HEIGHT
        del self.AB_WIDTH

    def _do_tst(self, device, BLOCK_SIZE):
        """Do test for specific context
        """
        self.a.initialize(device)
        self.b.initialize(device)
        self.c.initialize(device)
        self.bias.initialize(device)

        obj = TrivialOpenCLUnit(DummyWorkflow())
        obj.initialize(device=device)
        obj.cl_sources_["forward.cl"] = {}
        defines = {
            "ACTIVATION_TANH": 1,
            "BLOCK_SIZE": BLOCK_SIZE,
            "H": self.AB_WIDTH,
            "Y": self.B_HEIGHT,
            "BATCH": self.A_HEIGHT}
        if self.a_col:
            defines["A_COL"] = 1
        if self.b_col:
            defines["B_COL"] = 1
        obj.build_program(
            defines, os.path.join(root.common.cache_dir, "test.cl"),
            dtype=self.dtype, show_ocl_logs=False)

        krn = obj.get_kernel("feed_layer")
        krn.set_arg(0, self.a.devmem)
        krn.set_arg(1, self.b.devmem)
        krn.set_arg(2, self.c.devmem)
        krn.set_arg(3, self.bias.devmem)

        global_size = [formats.roundup(self.B_HEIGHT, BLOCK_SIZE),
                       formats.roundup(self.A_HEIGHT, BLOCK_SIZE)]
        local_size = [BLOCK_SIZE, BLOCK_SIZE]

        self.device.queue_.execute_kernel(krn, global_size, local_size,
                                          need_event=False)

        self.c.map_read()

    def _tst_matrix_multiplication(self, block_size):
        N = 500
        logging.info("Will test %d matrix multiplications "
                     "with BLOCK_SIZE = %d, dtype=%s" %
                     (N // 22, block_size, str(self.dtype)))
        j = 0
        for i in range(0, N, 22):
            AB_WIDTH = prng.get().randint(1, ((i // 10) + 1) * 100)
            B_HEIGHT = prng.get().randint(1, ((i // 10) + 1) * 10)
            A_HEIGHT = prng.get().randint(1, ((i // 10) + 1) * 10)
            if j % 2 == 0:
                AB_WIDTH = formats.roundup(AB_WIDTH, block_size)
                B_HEIGHT = formats.roundup(B_HEIGHT, block_size)
                A_HEIGHT = formats.roundup(A_HEIGHT, block_size)
            if j % 4 == 0:
                a_col = False
                b_col = False
            elif j % 4 == 1:
                a_col = True
                b_col = False
            elif j % 4 == 2:
                a_col = False
                b_col = True
            else:
                a_col = True
                b_col = True
            j += 1
            logging.info("%d: [%d, %d] * [%d, %d] = [%d, %d]" %
                         (i, AB_WIDTH, A_HEIGHT, B_HEIGHT, AB_WIDTH,
                          A_HEIGHT, B_HEIGHT))
            self._prepare_tsts(BLOCK_SIZE=block_size, AB_WIDTH=AB_WIDTH,
                               B_HEIGHT=B_HEIGHT, A_HEIGHT=A_HEIGHT,
                               a_col=a_col, b_col=b_col)
            c = self._do_cpu_tst()
            self._do_tst(self.device, block_size)
            max_diff = numpy.fabs(c.ravel() - self.c[0].ravel()).max()
            self.assertLess(max_diff, 0.0001,
                            "Result differs by %.6f" % (max_diff))
            num_nz = numpy.count_nonzero(self.c[1].ravel() - 1)
            self.assertEqual(
                num_nz, 0,
                "Written some values outside of the target array bounds")
            self._cleanup_after_tsts()

    def test(self):
    # opt_block_size = self.device.device_info.BLOCK_SIZE[root.common.dtype]
        for dtype in (numpy.float32, numpy.float64):
            self.dtype = dtype
            for block_size in range(8, 32):
                self._tst_matrix_multiplication(block_size)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
