#!/usr/bin/python3 -O
# encoding: utf-8
"""
  _   _ _____ _     _____ _____
 | | | |  ___| |   |  ___/  ___|
 | | | | |__ | |   | |__ \ `--.
 | | | |  __|| |   |  __| `--. \
 \ \_/ / |___| |___| |___/\__/ /
  \___/\____/\_____|____/\____/

Created on April 24, 2014

A unit test for local response normalization.

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

from veles.memory import Vector
from veles.tests import AcceleratedTest, assign_backend
from veles.znicz.normalization import LRNormalizerForward, LRNormalizerBackward


class TestNormalization(AcceleratedTest):
    ABSTRACT = True

    def test_normalization_forward(self):
        fwd_normalizer = LRNormalizerForward(self.parent)
        fwd_normalizer.input = Vector()
        in_vector = numpy.zeros(shape=(3, 2, 5, 19), dtype=self.dtype)

        for i in range(5):
            in_vector[0, 0, i, :] = numpy.linspace(10, 50, 19) * (i + 1)
            in_vector[0, 1, i, :] = numpy.linspace(10, 50, 19) * (i + 1) + 1
            in_vector[1, 0, i, :] = numpy.linspace(10, 50, 19) * (i + 1) + 2
            in_vector[1, 1, i, :] = numpy.linspace(10, 50, 19) * (i + 1) + 3
            in_vector[2, 0, i, :] = numpy.linspace(10, 50, 19) * (i + 1) + 4
            in_vector[2, 1, i, :] = numpy.linspace(10, 50, 19) * (i + 1) + 5

        fwd_normalizer.input.mem = in_vector
        fwd_normalizer.initialize(device=self.device)

        fwd_normalizer.ocl_run()
        fwd_normalizer.output.map_read()
        ocl_result = numpy.copy(fwd_normalizer.output.mem)

        fwd_normalizer.cpu_run()
        fwd_normalizer.output.map_read()
        cpu_result = numpy.copy(fwd_normalizer.output.mem)

        max_delta = numpy.fabs(cpu_result - ocl_result).max()

        self.info("FORWARD")
        self.assertLess(max_delta, 0.0001,
                        "Result differs by %.6f" % (max_delta))

        self.info("FwdProp done.")

    def test_normalization_backward(self):

        h = numpy.zeros(shape=(2, 1, 5, 5), dtype=self.dtype)
        for i in range(5):
            h[0, 0, i, :] = numpy.linspace(10, 50, 5) * (i + 1)
            h[1, 0, i, :] = numpy.linspace(10, 50, 5) * (i + 1) + 1

        err_y = numpy.zeros(shape=(2, 1, 5, 5), dtype=self.dtype)
        for i in range(5):
            err_y[0, 0, i, :] = numpy.linspace(2, 10, 5) * (i + 1)
            err_y[1, 0, i, :] = numpy.linspace(2, 10, 5) * (i + 1) + 1

        back_normalizer = LRNormalizerBackward(self.parent, n=3)
        back_normalizer.input = Vector()
        back_normalizer.err_output = Vector()

        back_normalizer.input.mem = h
        back_normalizer.err_output.mem = err_y

        back_normalizer.initialize(device=self.device)

        back_normalizer.cpu_run()
        cpu_result = back_normalizer.err_input.mem.copy()

        back_normalizer.err_input.map_invalidate()
        back_normalizer.err_input.mem[:] = 100

        back_normalizer.ocl_run()
        back_normalizer.err_input.map_read()
        ocl_result = back_normalizer.err_input.mem.copy()

        self.info("BACK")

        max_delta = numpy.fabs(cpu_result - ocl_result).max()
        self.assertLess(max_delta, 0.0001,
                        "Result differs by %.6f" % (max_delta))

        self.info("BackProp done.")


@assign_backend("ocl")
class OpenCLTestNormalization(TestNormalization):
    pass


@assign_backend("cuda")
class CUDATestNormalization(TestNormalization):
    pass


if __name__ == "__main__":
    AcceleratedTest.main()
