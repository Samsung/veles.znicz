# -*- coding: utf-8 -*-
"""
.. invisible:
     _   _ _____ _     _____ _____
    | | | |  ___| |   |  ___/  ___|
    | | | | |__ | |   | |__ \ `--.
    | | | |  __|| |   |  __| `--. \
    \ \_/ / |___| |___| |___/\__/ /
     \___/\____/\_____|____/\____/

Created on Aug 4, 2014

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

from veles.memory import Vector
from veles.znicz.cutter import Cutter, GDCutter
import veles.prng as prng
from veles.tests import AcceleratedTest, assign_backend


class TestCutter(AcceleratedTest):
    def setUp(self):
        super(TestCutter, self).setUp()
        self.input = numpy.zeros([25, 37, 43, 23], dtype=numpy.float32)
        prng.get().fill(self.input)

    def _do_test(self, device):
        cutter = Cutter(self.parent, padding=(2, 3, 4, 5))
        cutter.input = Vector(self.input.copy())
        cutter.initialize(device)
        cutter.run()
        cutter.output.map_read()
        return cutter.output.mem.copy()

    def test_cpu_gpu(self):
        gpu = self._do_test(self.device)
        cpu = self._do_test(NumpyDevice())
        max_diff = numpy.fabs(gpu - cpu).max()
        self.assertEqual(max_diff, 0)

    def _do_test_gd(self, device):
        cutter = Cutter(self.parent, padding=(2, 3, 4, 5))
        cutter.input = Vector(self.input.copy())
        cutter.initialize(device)
        cutter.run()
        cutter.output.map_read()
        cutter.output.mem.copy()

        gd = GDCutter(self.parent, padding=cutter.padding)
        gd.input = cutter.input
        gd.err_output = Vector(cutter.output.mem.copy())
        numpy.square(gd.err_output.mem, gd.err_output.mem)  # modify
        gd.initialize(device)
        gd.run()
        gd.err_input.map_read()
        return gd.err_input.mem.copy()

    def test_cpu_gpu_gd(self):
        gpu = self._do_test_gd(self.device)
        cpu = self._do_test_gd(NumpyDevice())
        max_diff = numpy.fabs(gpu - cpu).max()
        self.assertEqual(max_diff, 0)


@assign_backend("ocl")
class OpenCLTestCutter(TestCutter):
    pass


@assign_backend("cuda")
class CUDATestCutter(TestCutter):
    pass

if __name__ == "__main__":
    AcceleratedTest.main()
