"""
Created on Nov 8, 2013

Will test correctness of OpenCL matrix multiplication.

Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""


import gc
import logging
import numpy
import opencl4py
import os
import time
import unittest

from veles.config import root
import veles.memory as formats
import veles.backends as opencl
import veles.prng as prng
from veles import opencl_types
from veles.dummy import DummyWorkflow
from veles.accelerated_units import TrivialOpenCLUnit
import veles.znicz as znicz
znicz.nothing()


class TestMatrixMultiplication(unittest.TestCase):
    def setUp(self):
        root.common.unit_test = True
        root.common.plotters_disabled = True
        self.device = opencl.Device()

    def tearDown(self):
        gc.collect()
        del self.device

    def _do_cpu_tst(self):
        """Pure CPU test.
        """
        dtype = numpy.float64
        a = numpy.empty(self.a.mem.shape, dtype=dtype)
        a[:] = self.a.mem[:]
        bt = self.b.mem.transpose()
        b = numpy.empty(bt.shape, dtype=dtype)
        b[:] = bt[:]
        c = numpy.empty(self.c[0].shape, dtype=dtype)
        if self.a_col:
            a = a.transpose()
        if self.b_col:
            b = b.transpose()
        numpy.dot(a, b, c)
        return c

    def _prepare_tsts(self, AB_WIDTH, B_HEIGHT, A_HEIGHT, a_col, b_col):
        self.AB_WIDTH = AB_WIDTH
        self.B_HEIGHT = B_HEIGHT
        self.A_HEIGHT = A_HEIGHT
        self.a_col = a_col
        self.b_col = b_col

        self.a = formats.Vector()
        self.a.mem = numpy.zeros([self.A_HEIGHT * self.AB_WIDTH],
                                 dtype=self.dtype)
        prng.get().fill(self.a.mem)
        if a_col:
            self.a.mem.shape = (self.AB_WIDTH, self.A_HEIGHT)
        else:
            self.a.mem.shape = (self.A_HEIGHT, self.AB_WIDTH)

        self.b = formats.Vector()
        self.b.mem = numpy.zeros([self.B_HEIGHT * self.AB_WIDTH],
                                 dtype=self.dtype)
        prng.get().fill(self.b.mem)
        if b_col:
            self.b.mem.shape = (self.AB_WIDTH, self.B_HEIGHT)
        else:
            self.b.mem.shape = (self.B_HEIGHT, self.AB_WIDTH)

        self.c = formats.Vector()
        self.c.mem = numpy.ones([2, self.A_HEIGHT, self.B_HEIGHT],
                                dtype=self.dtype)

    def _cleanup_after_tsts(self):
        del self.c
        del self.b
        del self.a
        del self.A_HEIGHT
        del self.B_HEIGHT
        del self.AB_WIDTH
        gc.collect()

    def _do_tst(self, device, BLOCK_SIZE):
        """Do test for specific context
        """
        obj = TrivialOpenCLUnit(DummyWorkflow())
        obj.initialize(device=device)

        self.a.initialize(device)
        self.b.initialize(device)
        self.c.initialize(device)

        obj.cl_sources_["all2all/forward"] = {}
        defines = {
            "INCLUDE_BIAS": 0,
            "WEIGHTS_TRANSPOSED": 0,
            "PRECISION_LEVEL": 0,
            "ACTIVATION_LINEAR": 1,
            "BLOCK_SIZE": BLOCK_SIZE,
            "H": self.AB_WIDTH,
            "Y": self.B_HEIGHT,
            "BATCH": self.A_HEIGHT,
            "VECTOR_OPT": self.device.device_info.vector_opt}
        if self.a_col:
            defines["A_COL"] = 1
        if self.b_col:
            defines["B_COL"] = 1
        obj.build_program(
            defines, os.path.join(root.common.cache_dir, "test"),
            dtype=self.dtype)

        krn = obj.get_kernel("feed_layer")
        krn.set_arg(0, self.a.devmem)
        krn.set_arg(1, self.b.devmem)
        krn.set_arg(2, self.c.devmem)

        global_size = [formats.roundup(self.B_HEIGHT, BLOCK_SIZE),
                       formats.roundup(self.A_HEIGHT, BLOCK_SIZE)]
        local_size = [BLOCK_SIZE, BLOCK_SIZE]

        t0 = time.time()
        self.device.queue_.execute_kernel(krn, global_size, local_size,
                                          need_event=False)
        self.c.map_read()
        dt = time.time() - t0
        logging.info("%.6f sec", dt)

    def _tst_matrix_multiplication(self, block_size):
        # Iterate over the different matrix configurations
        logging.info("#" * 80)
        logging.info("BLOCK_SIZE = %d", block_size)
        for (a_col, b_col) in ((False, False),
                               (False, True),
                               (True, False),
                               (True, True)
                               ):
            for A_HEIGHT, B_HEIGHT, AB_WIDTH in (
                    (1024, 1024, 4096),  # aligned
                    (1024, 64, 4096),  # one side differs
                    (1023, 511, 4095),  # unaligned
                    (13, 11, 3),  # small
                    (2, 11, 65),  # large common
                    (1, 25, 1),  # very small
                    ):
                logging.info("[%d, %d] * [%d, %d] = [%d, %d] %s %s",
                             AB_WIDTH, A_HEIGHT, B_HEIGHT, AB_WIDTH,
                             A_HEIGHT, B_HEIGHT, str(a_col), str(b_col))
                self._prepare_tsts(AB_WIDTH=AB_WIDTH,
                                   B_HEIGHT=B_HEIGHT,
                                   A_HEIGHT=A_HEIGHT,
                                   a_col=a_col, b_col=b_col)
                c = self._do_cpu_tst()
                try:
                    self._do_tst(self.device, block_size)
                    max_diff = numpy.fabs(c.ravel() - self.c[0].ravel()).max()
                    self.assertLess(max_diff, 0.001,
                                    "Result differs by %.6f" % (max_diff))
                    num_nz = numpy.count_nonzero(self.c[1].ravel() - 1)
                    self.assertEqual(
                        num_nz, 0, "Written some values outside of the "
                        "target array bounds")
                finally:
                    self._cleanup_after_tsts()

    def test(self):
        if not isinstance(self.device, opencl.OpenCLDevice):
            return
        for dtype in (numpy.float32, numpy.float64):
            logging.info("~" * 80)
            logging.info(str(dtype))
            self.dtype = dtype
            for block_size in range(
                    8, self.device.device_info.get_max_block_size(
                        opencl_types.numpy_dtype_to_opencl(dtype)) + 1,
                    4 if self.device.device_info.vector_opt else 1):
                try:
                    self._tst_matrix_multiplication(block_size)
                except opencl4py.CLRuntimeError as e:
                    logging.warning("OpenCL error: %s", str(e))
                    if e.code == -5:
                        break


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
