# -*- coding: utf-8 -*-
"""
.. invisible:
     _   _ _____ _     _____ _____
    | | | |  ___| |   |  ___/  ___|
    | | | | |__ | |   | |__ \ `--.
    | | | |  __|| |   |  __| `--. \
    \ \_/ / |___| |___| |___/\__/ /
     \___/\____/\_____|____/\____/

Created on Nov 22, 2013

Unit test for OpenCL kernel which does reduce over matrix rows or columns.

███████████████████████████████████████████████████████████████████████████████

Licensed to the Apache Software Foundation (ASF) under one
or more contributor license agreements.  See the NOTICE file
distributed with this work for additional information
regarding copyright ownership.  The ASF licenses this file
to you under the Apache License, Version 2.0 (the
"License"); you may not use this file except in compliance
with the License.  You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing,
software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
KIND, either express or implied.  See the License for the
specific language governing permissions and limitations
under the License.

███████████████████████████████████████████████████████████████████████████████
"""


import numpy
import os
import time

from veles.config import root
from veles.memory import Array
import veles.opencl_types as opencl_types
import veles.prng as prng
from veles.accelerated_units import TrivialAcceleratedUnit
from veles.tests import AcceleratedTest, assign_backend


class TestMatrixReduce(AcceleratedTest):
    def setUp(self):
        super(TestMatrixReduce, self).setUp()
        thisdir = os.path.dirname(__file__)
        if not len(thisdir):
            thisdir = "."
        if thisdir not in root.common.engine.source_dirs:
            root.common.engine.source_dirs.append(thisdir)

    def _build_program(self, a, b, A_WIDTH, A_HEIGHT, A_COL, REDUCE_SIZE):
        defines = {"A_WIDTH": A_WIDTH,
                   "A_HEIGHT": A_HEIGHT,
                   "REDUCE_SIZE": REDUCE_SIZE}
        if A_COL:
            defines["A_COL"] = 1

        tmp = TrivialAcceleratedUnit(self.parent)
        tmp.initialize(device=self.device)
        tmp.sources_["test_matrix_reduce"] = {}

        tmp.build_program(defines, "test_matrix_reduce")

        krn = tmp.get_kernel("test")
        krn.set_arg(0, a.devmem)
        krn.set_arg(1, b.devmem)
        return krn

    def test_fixed(self):
        """Test with fixed input.
        """
        dtype = opencl_types.dtypes[root.common.precision_type]

        a = Array()
        a.mem = numpy.array([[1, 2, 3],
                             [-1, -2, -3],
                             [9, 8, 7],
                             [-3, -4, -5],
                             [-1, -2, -3],
                             [7, 7, 7]], dtype=dtype)
        b = Array()
        b.mem = numpy.ones(a.mem.shape[1] * 2, dtype=dtype)

        t = numpy.array([12, 9, 6], dtype=dtype)

        a.initialize(self.device)
        b.initialize(self.device)

        for reduce_size in range(1, 33, 1):
            self._do_test(reduce_size, True, a, b, t)

        self.info("test_fixed() succeeded")

    def _do_test(self, reduce_size, A_COL, a, b, t):
        krn = self._build_program(
            a, b, a.shape[1], a.shape[0], A_COL, reduce_size)

        if self.device.backend_name == "ocl":
            global_size = [a.shape[1 if A_COL else 0] * reduce_size]
            local_size = [reduce_size]
            self.device.sync()
            t0 = time.time()
            self.device.queue_.execute_kernel(
                krn, global_size, local_size, need_event=False)
            self.device.sync()
            dt = time.time() - t0
        elif self.device.backend_name == "cuda":
            global_size = (a.shape[1 if A_COL else 0], 1, 1)
            local_size = (reduce_size, 1, 1)
            self.device.sync()
            t0 = time.time()
            krn(global_size, local_size)
            self.device.sync()
            dt = time.time() - t0
        else:
            self.warning("Unsupported device backend name %s",
                         self.device.name)
            return
        self.info("Completed in %.6f sec", dt)

        b.map_write()
        max_diff = numpy.fabs(b.mem[:a.shape[1 if A_COL else 0]] - t).max()
        self.assertLess(max_diff, 0.00032,  # in case of float
                        "Result differs by %.6f" % (max_diff))
        self.assertEqual(
            numpy.count_nonzero(b.mem[a.shape[0]:] - 1), 0,
            "Written some values outside of the target array bounds")
        b[:] = 1
        b.unmap()

    def test_random(self):
        """Test with random input vs numpy.
        """
        dtype = opencl_types.dtypes[root.common.precision_type]

        a = Array()
        a.mem = numpy.zeros([3131, 1001], dtype=dtype)
        prng.get().fill(a.mem)

        t_col = numpy.sum(a.mem, axis=0)
        t = numpy.sum(a.mem, axis=1)

        b = Array()
        b.mem = numpy.ones(numpy.max(a.mem.shape) * 2, dtype=dtype)

        a.initialize(self.device)
        b.initialize(self.device)

        for reduce_size in range(4, 64, 1):
            self.info("reduce_size = %d", reduce_size)
            self._do_test(reduce_size, True, a, b, t_col)
            self._do_test(reduce_size, False, a, b, t)

        self.info("test_random() succeeded")


@assign_backend("ocl")
class OpenCLTestMatrixReduce(TestMatrixReduce):
    pass


@assign_backend("cuda")
class CUDATestMatrixReduce(TestMatrixReduce):
    pass


if __name__ == "__main__":
    AcceleratedTest.main()
