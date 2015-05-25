# -*- coding: utf-8 -*-
"""
.. invisible:
     _   _ _____ _     _____ _____
    | | | |  ___| |   |  ___/  ___|
    | | | | |__ | |   | |__ \ `--.
    | | | |  __|| |   |  __| `--. \
    \ \_/ / |___| |___| |___/\__/ /
     \___/\____/\_____|____/\____/

Created on Mart 31, 2014

Unit test for RELU convolutional layer forward propagation.

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

from veles.config import root
from veles.memory import Array
import veles.opencl_types as opencl_types
from veles.tests import AcceleratedTest, assign_backend
from veles.znicz.conv import ConvRELU
from veles.znicz.tests.unit.test_conv import PatchedConv


class PatchedConvRELU(ConvRELU, PatchedConv):
    pass


class TestConvRelu(AcceleratedTest):
    def test_fixed(self):
        self.info("Will test RELU convolutional layer forward propagation")

        inp = Array()
        dtype = opencl_types.dtypes[root.common.precision_type]
        inp.mem = numpy.array([[[1, 2, 3, 2, 1],
                              [0, 1, 2, 1, 0],
                              [0, 1, 0, 1, 0],
                              [2, 0, 1, 0, 2],
                              [1, 0, 1, 0, 1]]], dtype=dtype)

        weights = numpy.array([[[-1, -1, -1],
                                [-1, 8, -1],
                                [-1, -1, -1]],
                               [[1.1, 2.1, 3.1],
                                [-1.1, -0.5, 1.3],
                                [1.7, -1.4, 0.05]]], dtype=dtype)
        bias = numpy.array([10, -10], dtype=dtype)

        c = PatchedConvRELU(self.parent, n_kernels=2, kx=3, ky=3)
        c.input = inp

        c.initialize(device=self.device)

        c.weights.map_invalidate()  # rewrite weights
        c.weights.mem[:] = weights.reshape(c.weights.mem.shape)[:]
        c.bias.map_invalidate()  # rewrite bias
        c.bias.mem[:] = bias[:]

        c.run()
        c.output.map_read()  # get results back
        nz = numpy.count_nonzero(
            numpy.isnan(
                c.output.unit_test_mem[c.output.mem.shape[0]:].ravel()))
        self.assertEqual(
            nz, c.output.unit_test_mem[c.output.mem.shape[0]:].size,
            "Overflow occured")

        y = c.output.mem.ravel()
        t = numpy.array([9, 5.3, 15, 5.65, 9, -3.5,
                         12, 1.25, 3, -2.8, 12, -4.4,
                         4, -7.05, 15, -7.7, 4, -4.65], dtype=dtype)
        t = numpy.where(t > 15, t, numpy.log(numpy.exp(t) + 1.0))
        max_diff = numpy.fabs(t - y).max()
        self.assertLess(max_diff, 0.0001,
                        "Result differs by %.6f" % max_diff)

        self.info("All Ok")


@assign_backend("ocl")
class OpenCLTestConvRelu(TestConvRelu):
    pass


@assign_backend("cuda")
class CUDATestConvRelu(TestConvRelu):
    pass


if __name__ == "__main__":
    AcceleratedTest.main()
