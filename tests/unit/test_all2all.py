#!/usr/bin/env python3
# -*-coding: utf-8 -*-
"""
.. invisible:
     _   _ _____ _     _____ _____
    | | | |  ___| |   |  ___/  ___|
    | | | | |__ | |   | |__ \ `--.
    | | | |  __|| |   |  __| `--. \
    \ \_/ / |___| |___| |___/\__/ /
     \___/\____/\_____|____/\____/

Created on November 18, 2013

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
import time
from veles.backends import NumpyDevice

from veles.config import root
from veles.memory import Vector
import veles.opencl_types as opencl_types
import veles.prng as prng
from veles.tests import AcceleratedTest, assign_backend
import veles.znicz.all2all as all2all


class TestAll2All(AcceleratedTest):
    ABSTRACT = True

    def setUp(self):
        super(TestAll2All, self).setUp()
        prng.get().seed(1234)

    def test_with_fixed_input(self):
        self.info("Will test all2all unit")
        inp = Vector()
        dtype = opencl_types.dtypes[root.common.precision_type]
        inp.mem = numpy.array([[1, 2, 3, 2, 1],
                               [0, 1, 2, 1, 0],
                               [0, 1, 0, 1, 0],
                               [2, 0, 1, 0, 2],
                               [1, 0, 1, 0, 1]], dtype=dtype)

        weights = numpy.array([[1, 0, 2, 1, -1],
                              [3, 1, 0, 2, 3],
                              [-1, 2, 0, 1, 3]], dtype=dtype)
        bias = numpy.array([10, -10, 5], dtype=dtype)

        c = all2all.All2All(self.parent, output_sample_shape=[3],
                            weights_stddev=0.05)
        c.input = inp

        c.initialize(device=self.device)

        c.weights.map_invalidate()  # rewrite weights
        c.weights.mem[:] = weights.reshape(c.weights.mem.shape)[:]
        c.bias.map_invalidate()  # rewrite bias
        c.bias.mem[:] = bias[:]

        c.run()
        c.output.map_read()  # get results back

        y = c.output.mem.ravel()
        t = numpy.array([18, 2, 13, 15, -7, 8,
                         11, -7, 8, 12, 2, 9,
                         12, -4, 7], dtype=dtype)

        max_diff = numpy.fabs(t.ravel() - y.ravel()).max()
        self.assertLess(max_diff, 0.0001,
                        "Result differs by %.6f" % max_diff)
        self.info("All Ok")

    def _do_test(self, device, Unit, weights_transposed):
        inp = Vector()
        dtype = opencl_types.dtypes[root.common.precision_type]
        inp.mem = numpy.empty((1999, 1777), dtype=dtype)
        prng.get().fill(inp.mem)

        if not isinstance(device, NumpyDevice):
            self.x = inp.mem.copy()
        else:
            inp.mem[:] = self.x[:]

        c = Unit(self.parent, output_sample_shape=[75, 75],
                 weights_transposed=weights_transposed)
        c.input = inp
        c.initialize(device=device)

        if not isinstance(device, NumpyDevice):
            self.W = c.weights.mem.copy()
            self.b = c.bias.mem.copy()
        else:
            c.weights.map_invalidate()
            c.bias.map_invalidate()
            c.weights.mem[:] = self.W[:]
            c.bias.mem[:] = self.b[:]

        if not isinstance(device, NumpyDevice):
            device.sync()
            t0 = time.time()
            for _ in range(3):
                c.run()
            device.sync()
            dt = time.time() - t0
            self.info("Completed in %.6f sec", dt)
        else:
            c.run()
        c.output.map_read()  # get results back

        if hasattr(c, "max_idx"):
            c.max_idx.map_read()
            return c.output.mem.copy(), c.max_idx.mem.copy()

        return c.output.mem.copy(),

    def _do_gpu_cpu(self, Unit):
        self._do_gpu_cpu_transposed(Unit, False)
        self._do_gpu_cpu_transposed(Unit, True)

    def _do_gpu_cpu_transposed(self, Unit, weights_transposed):
        y_gpus = self._do_test(self.device, Unit, weights_transposed)
        y_cpus = self._do_test(NumpyDevice(), Unit, weights_transposed)
        for i, y_gpu in enumerate(y_gpus):
            y_cpu = y_cpus[i]
            max_diff = numpy.fabs(y_gpu.ravel() - y_cpu.ravel()).max()
            self.info("Difference is %.12f (weights_transposed is %s)",
                      max_diff, str(weights_transposed))
            self.assertLess(max_diff, 0.0001,
                            "Result differs by %.6f" % max_diff)
        self.info("All Ok")

    def test_linear(self):
        self.info("Will test linear all2all unit for gpu/cpu correctness")
        self._do_gpu_cpu(all2all.All2All)

    def test_tanh(self):
        self.info("Will test Tanh all2all unit for gpu/cpu correctness")
        self._do_gpu_cpu(all2all.All2AllTanh)

    def test_sigmoid(self):
        self.info("Will test Sigmoid all2all unit for gpu/cpu correctness")
        self._do_gpu_cpu(all2all.All2AllSigmoid)

    def test_relu(self):
        self.info("Will test RELU all2all unit for gpu/cpu correctness")
        self._do_gpu_cpu(all2all.All2AllRELU)

    def test_softmax(self):
        self.info("Will test Softmax all2all unit for gpu/cpu correctness")
        self._do_gpu_cpu(all2all.All2AllSoftmax)

    def test_two_stage(self):
        dtype = opencl_types.dtypes[root.common.precision_type]
        a2a = all2all.All2All(self.parent, output_sample_shape=(75, 75),
                              output_samples_number=1999, output_dtype=dtype)
        a2a.input = Vector()
        a2a.initialize(self.device)
        self.assertTrue(a2a.output)
        self.assertEqual(a2a.output.shape, (1999, 75, 75))
        self.assertEqual(a2a.output.dtype, dtype)
        self.assertFalse(a2a.weights)
        self.assertFalse(a2a.bias)
        out = a2a.output.mem

        inp = Vector()
        dtype = opencl_types.dtypes[root.common.precision_type]
        inp.mem = numpy.empty((1999, 1777), dtype=dtype)
        prng.get().fill(inp.mem)
        a2a.input = inp
        a2a.initialize(self.device)
        self.assertTrue((out == a2a.output.mem).all())
        self.assertTrue(a2a.weights)
        self.assertTrue(a2a.bias)


@assign_backend("ocl")
class OpenCLTestAll2All(TestAll2All):
    pass


@assign_backend("cuda")
class CUDATestAll2All(TestAll2All):
    pass


if __name__ == "__main__":
    AcceleratedTest.main()
