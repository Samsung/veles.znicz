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

Unit test for RELU convolutional layer back propagation

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
import veles.znicz.gd_conv as gd_conv


class TestGDRELUConv(AcceleratedTest):
    def test_fixed(self):
        self.info("Will test RELU convolutional layer back propagation")

        inp = Array()
        dtype = opencl_types.dtypes[root.common.precision_type]
        inp.mem = numpy.array([[[-1, 0, 2, 0, 3],
                              [0, 1, -2, 1, 2],
                              [2, 0, 1, 1, 0],
                              [-1, 1, 1, 0, 2],
                              [1, 0, 1, 0, 1]]], dtype=dtype)

        a = numpy.array([[-1, 0, 2, 0, 1, -2, 2, 0, 1],
                         [0, 2, 0, 1, -2, 1, 0, 1, 1],
                         [2, 0, 3, -2, 1, 2, 1, 1, 0],
                         [0, 1, -2, 2, 0, 1, -1, 1, 1],
                         [1, -2, 1, 0, 1, 1, 1, 1, 0],
                         [-2, 1, 2, 1, 1, 0, 1, 0, 2],
                         [2, 0, 1, -1, 1, 1, 1, 0, 1],
                         [0, 1, 1, 1, 1, 0, 0, 1, 0],
                         [1, 1, 0, 1, 0, 2, 1, 0, 1]], dtype=dtype)

        weights = numpy.array([[-1, -1, 4, 1, 8, -1, -1, 3, 2],
                               [2, 1, -1, 3, 0, 1, 4, 1, 3]], dtype=dtype)

        bias = numpy.array([10, -10], dtype=dtype)

        c = gd_conv.GDRELUConv(self.parent)
        c.n_kernels = 2
        c.kx = c.ky = 3
        c.padding = 0, 0, 0, 0
        c.sliding = 1, 1
        c.err_output = Array()
        c.err_output.mem = numpy.array(
            [[[-1, 3],
              [8, 2],
              [0, 1],
              [4, -1],
              [-1, 2],
              [0, 1],
              [-2, 3],
              [1, 2],
              [1, 1]]], dtype=dtype)
        c.input = inp
        c.weights = Array()
        c.weights.mem = weights
        c.bias = Array()
        c.bias.mem = bias
        c.output = Array()
        c.output.mem = c.err_output.mem.copy()
        c.unpack_size = 1
        c.unpack_data = Array()

        err_output = c.err_output.mem * (1.0 - numpy.exp(-c.output.mem))
        batch_size = err_output.shape[0]
        b = err_output.reshape(9 * batch_size, 2)
        gradient_weights = numpy.dot(b.transpose(), a)
        gradient_weights *= -c.learning_rate
        gradient_weights += weights * (-1) * (c.learning_rate *
                                              c.weights_decay)
        weights_new = weights + gradient_weights
        gradient_bias = b.sum(axis=0)
        gradient_bias *= -c.learning_rate_bias
        bias_new = bias + gradient_bias

        c.initialize(device=self.device)
        c.run()
        """
        c.err_input.map_read()
        """
        c.weights.map_read()
        c.bias.map_read()

        """
        t = numpy.array([[[7, 0, -11, 31, -1],
                          [2, 6, 93, -11, 0],
                          [22, 45, 18, 28, 7],
                          [-1, 11, 25, 14, 3],
                          [14, 4, 13, 12, 5]]], dtype=dtype)
        max_diff = numpy.fabs(t.ravel() - c.err_h.mem.ravel()).max()
        self.assertLess(max_diff, 0.0001,
                        "Result differs by %.6f" % (max_diff))
        self.info("Err_h is right")
        """

        max_diff = numpy.fabs(weights_new.ravel() -
                              c.weights.mem.ravel()).max()
        self.assertLess(max_diff, 0.0001,
                        "Result differs by %.6f" % (max_diff))
        self.info("Weights is right")

        max_diff = numpy.fabs(bias_new.ravel() - c.bias.mem.ravel()).max()
        self.assertLess(max_diff, 0.0001,
                        "Result differs by %.6f" % (max_diff))
        self.info("Bias is right")


@assign_backend("ocl")
class OpenCLTestGDRELUConv(TestGDRELUConv):
    pass


@assign_backend("cuda")
class CUDATestGDRELUConv(TestGDRELUConv):
    pass


if __name__ == "__main__":
    AcceleratedTest.main()
