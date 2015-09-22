# -*- coding: utf-8 -*-
"""
.. invisible:
     _   _ _____ _     _____ _____
    | | | |  ___| |   |  ___/  ___|
    | | | | |__ | |   | |__ \ `--.
    | | | |  __|| |   |  __| `--. \
    \ \_/ / |___| |___| |___/\__/ /
     \___/\____/\_____|____/\____/

Created on Aug 3, 2015

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
from veles.backends import NumpyDevice

from veles.memory import Array
import veles.prng as prng
from veles.tests import AcceleratedTest, assign_backend
from veles.znicz.multiplier import Multiplier, GDMultiplier
from veles.znicz.tests.unit.gd_numdiff import GDNumDiff


class TestMultiplier(AcceleratedTest, GDNumDiff):
    ABSTRACT = True

    def setUp(self):
        super(TestMultiplier, self).setUp()
        self.precision_threshold = {
            numpy.float64: 1.0e-8,
            numpy.float32: 1.0e-4,
            numpy.float16: 1.0e-2}[self.dtype]
        self.x = numpy.zeros([5, 7, 4, 9], dtype=self._dtype)
        prng.get().fill(self.x)
        self.y = numpy.zeros_like(self.x)
        prng.get().fill(self.y)
        self.target = numpy.zeros_like(self.x)
        prng.get().fill(self.target)

    def _forward(self, device):
        mul = Multiplier(self.parent)
        mul.x = Array(self.x.copy())
        mul.y = Array(self.y.copy())
        mul.initialize(device)
        mul.run()
        mul.output.map_read()
        return mul

    def _do_test(self, device):
        return self._forward(device).output.mem.copy()

    def test_cpu_gpu(self):
        gpu = self._do_test(self.device)
        cpu = self._do_test(NumpyDevice())
        max_diff = numpy.fabs(gpu - cpu).max()
        self.assertLess(max_diff, self.precision_threshold)

    def _do_test_gd(self, device):
        mul = self._forward(device)

        mul.x.map_read()
        mul.y.map_read()
        vector_value_map = {mul.x: mul.x.mem.copy(),
                            mul.y: mul.y.mem.copy()}

        mul.output.map_read()
        err_output = Array(mul.output.mem - self.target)

        gd = GDMultiplier(self.parent)
        gd.x = mul.x
        gd.y = mul.y
        gd.err_output = err_output

        gd.initialize(device)
        gd.run()

        gd.err_x.map_read()
        err_x = gd.err_x.mem.copy()
        gd.err_y.map_read()
        err_y = gd.err_y.mem.copy()

        self.numdiff_check(
            mul, mul.x, vector_value_map,
            mul.output, self.target, err_x,
            self.info, self.assertLess, GDNumDiff.sse, mul.x.shape[0])
        self.numdiff_check(
            mul, mul.y, vector_value_map,
            mul.output, self.target, err_y,
            self.info, self.assertLess, GDNumDiff.sse, mul.y.shape[0])

        return err_x, err_y

    def test_cpu_gpu_gd(self):
        gpu = self._do_test_gd(self.device)
        cpu = self._do_test_gd(NumpyDevice())
        max_diff = max(numpy.fabs(gpu[i] - cpu[i]).max()
                       for i in range(len(gpu)))
        self.assertLess(max_diff, self.precision_threshold)


@assign_backend("ocl")
class OpenCLTestMultiplier(TestMultiplier):
    pass


@assign_backend("cuda")
class CUDATestMultiplier(TestMultiplier):
    pass


if __name__ == "__main__":
    AcceleratedTest.main()
