# -*- coding: utf-8 -*-
"""
  _   _ _____ _     _____ _____
 | | | |  ___| |   |  ___/  ___|
 | | | | |__ | |   | |__ \ `--.
 | | | |  __|| |   |  __| `--. \
 \ \_/ / |___| |___| |___/\__/ /
  \___/\____/\_____|____/\____/

Created on May 28, 2014

Unit test for convolutional-ppoling-softmax 3-layer back propagation.

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

from veles.config import root
import veles.memory as formats
import veles.opencl_types as opencl_types
from veles.tests import assign_backend
from veles.znicz.nn_units import AcceleratedWorkflow
import veles.znicz.activation as activation
import veles.znicz.all2all as all2all
import veles.znicz.gd as gd
import veles.znicz.gd_conv as gd_conv
import veles.znicz.conv as conv
import veles.znicz.gd_pooling as gd_pooling
import veles.znicz.pooling as pooling
import veles.znicz.evaluator as evaluator
import veles.znicz.normalization as normalization
from veles.dummy import DummyLauncher
import veles.prng as rnd
from veles.znicz.tests.functional import StandardTest
from veles.znicz.tests.unit.gd_numdiff import GDNumDiff


class Workflow(AcceleratedWorkflow):
    def __init__(self, workflow, **kwargs):
        super(Workflow, self).__init__(workflow, **kwargs)

        ConvForward = kwargs["ConvForward"]
        ConvGD = kwargs["ConvGD"]
        dtype = opencl_types.dtypes[root.common.precision_type]
        self.batch_size = 2

        self.input = numpy.zeros([self.batch_size, 8, 8, 3], dtype=dtype)
        rnd.get().fill(self.input, -10, 10)

        self.labels = numpy.zeros(self.batch_size, dtype=numpy.int32)
        self.labels[:] = rnd.get().randint(2, size=self.labels.size)

        # First convolutional layer
        self.conv_forward = ConvForward(
            self, n_kernels=25, kx=3, ky=3,
            padding=(2, 2, 2, 2), sliding=(1, 1))
        self.conv_forward.link_from(self.start_point)
        self.conv_forward.input = formats.Vector()
        self.conv_forward.input.mem = self.input.copy()
        prev = self.conv_forward

        # First pooling layer
        self.pool_forward = pooling.MaxPooling(
            self, kx=3, ky=3, sliding=(2, 2))
        self.pool_forward.link_from(prev)
        self.pool_forward.link_attrs(prev, ("input", "output"))
        prev = self.pool_forward

        # First separate activation layer
        self.act_forward = activation.ForwardTanhLog(self)
        self.act_forward.link_from(prev)
        self.act_forward.link_attrs(prev, ("input", "output"))
        prev = self.act_forward

        # Second separate activation layer
        self.act_forward2 = activation.ForwardStrictRELU(self)
        self.act_forward2.link_from(prev)
        self.act_forward2.link_attrs(prev, ("input", "output"),
                                     "output")
        prev = self.act_forward2

        # Second convolutional layer
        self.conv_forward2 = ConvForward(
            self, n_kernels=50, kx=3, ky=3,
            padding=(2, 2, 2, 2), sliding=(1, 1))
        self.conv_forward2.link_from(prev)
        self.conv_forward2.link_attrs(prev, ("input", "output"))
        prev = self.conv_forward2

        # Second pooling layer
        self.pool_forward2 = pooling.AvgPooling(
            self, kx=3, ky=3, sliding=(2, 2))
        self.pool_forward2.link_from(prev)
        self.pool_forward2.link_attrs(prev, ("input", "output"))
        prev = self.pool_forward2

        # Normalization layer
        self.norm = normalization.LRNormalizerForward(self)
        self.norm.link_from(prev)
        self.norm.link_attrs(prev, ("input", "output"))
        prev = self.norm

        # Softmax layer
        self.sm_forward = all2all.All2AllSoftmax(
            self, output_sample_shape=[10])
        self.sm_forward.link_from(prev)
        self.sm_forward.link_attrs(prev, ("input", "output"))

        # Evaluator for softmax layer
        self.ev = evaluator.EvaluatorSoftmax(self)
        self.ev.link_from(self.sm_forward)
        self.ev.link_attrs(self.sm_forward, "output", "max_idx")
        self.ev.labels = formats.Vector()
        self.ev.labels.mem = self.labels.copy()
        self.ev.batch_size = self.batch_size

        # Gradient descent layer for softmax
        self.sm_gd = gd.GDSoftmax(
            self, gradient_moment=0, gradient_moment_bias=0,
            learning_rate=-1, weights_decay=0,
            learning_rate_bias=-1, weights_decay_bias=0)
        self.sm_gd.link_from(self.ev)
        self.sm_gd.link_attrs(self.ev, "err_output")
        self.sm_gd.link_attrs(self.sm_forward, "weights", "bias",
                              "input", "output")
        self.sm_gd.batch_size = self.batch_size
        prev = self.sm_gd

        # Gradient descent layer for normalization
        self.norm_gd = normalization.LRNormalizerBackward(self)
        self.norm_gd.link_from(prev)
        self.norm_gd.link_attrs(prev, ("err_output", "err_input"))
        self.norm_gd.link_attrs(self.norm, "input")
        prev = self.norm_gd

        # Gradient descent layer for second pooling
        self.pool_gd2 = gd_pooling.GDAvgPooling(self)
        self.pool_gd2.link_pool_attrs(self.pool_forward2)
        self.pool_gd2.link_from(prev)
        self.pool_gd2.link_attrs(prev, ("err_output", "err_input"))
        self.pool_gd2.link_attrs(self.pool_forward2, "input")
        prev = self.pool_gd2

        # Gradient descent layer for second convolutional layer
        self.conv_gd2 = ConvGD(
            self,
            gradient_moment=0, gradient_moment_bias=0,
            learning_rate=-1, weights_decay=0,
            learning_rate_bias=-1, weights_decay_bias=0)
        self.conv_gd2.link_conv_attrs(self.conv_forward2)
        self.conv_gd2.link_from(prev)
        self.conv_gd2.link_attrs(prev, ("err_output", "err_input"))
        self.conv_gd2.link_attrs(self.conv_forward2, "weights", "bias",
                                 "input", "output")
        self.conv_gd2.batch_size = self.batch_size
        prev = self.conv_gd2

        # Gradient descent for second separate activation layer
        self.act_backward2 = activation.BackwardStrictRELU(self)
        self.act_backward2.link_from(prev)
        self.act_backward2.link_attrs(prev, ("err_output", "err_input"))
        self.act_backward2.link_attrs(self.act_forward2, "input", "output")
        prev = self.act_backward2

        # Gradient descent for first separate activation layer
        self.act_backward = activation.BackwardTanhLog(self)
        self.act_backward.link_from(prev)
        self.act_backward.link_attrs(prev,
                                     ("err_output", "err_input"))
        self.act_backward.link_attrs(self.act_forward, "input", "output")
        prev = self.act_backward

        # Gradient descent layer for first pooling
        self.pool_gd = gd_pooling.GDMaxPooling(self)
        self.pool_gd.link_pool_attrs(self.pool_forward)
        self.pool_gd.link_from(prev)
        self.pool_gd.link_attrs(prev, ("err_output", "err_input"))
        self.pool_gd.link_attrs(self.pool_forward, "input", "input_offset")
        prev = self.pool_gd

        # Gradient descent layer for first convolutional layer
        self.conv_gd = ConvGD(
            self, gradient_moment=0, gradient_moment_bias=0,
            learning_rate=-1, weights_decay=0,
            learning_rate_bias=-1, weights_decay_bias=0)
        self.conv_gd.link_conv_attrs(self.conv_forward)
        self.conv_gd.link_from(prev)
        self.conv_gd.link_attrs(prev, ("err_output", "err_input"))
        self.conv_gd.link_attrs(self.conv_forward, "weights", "bias",
                                "input", "output")
        self.conv_gd.batch_size = self.batch_size
        prev = self.conv_gd

        self.end_point.link_from(prev)

    def run(self):
        self.stopped = False
        super(Workflow, self).run()


class TestGDWorkflow(StandardTest, GDNumDiff):
    ABSTRACT = True

    def test_random_numeric_gpu(self):
        self._test_random_numeric(self.device, conv.Conv,
                                  gd_conv.GradientDescentConv)

    def test_random_numeric_gpu_tanh(self):
        self._test_random_numeric(self.device, conv.ConvTanh,
                                  gd_conv.GDTanhConv)

    def test_random_numeric_gpu_relu(self):
        self._test_random_numeric(self.device, conv.ConvRELU,
                                  gd_conv.GDRELUConv)

    def test_random_numeric_cpu(self):
        self._test_random_numeric(None, conv.Conv,
                                  gd_conv.GradientDescentConv)

    def test_random_numeric_cpu_tanh(self):
        self._test_random_numeric(None, conv.ConvTanh,
                                  gd_conv.GDTanhConv)

    def test_random_numeric_cpu_relu(self):
        self._test_random_numeric(None, conv.ConvRELU,
                                  gd_conv.GDRELUConv)

    def _test_random_numeric(self, device, ConvForward, ConvGD):
        self.info("Will check %s <=> %s via numeric differentiation on %s",
                  ConvForward.__name__, ConvGD.__name__,
                  "CPU, limited to 2 checks" if isinstance(device, NumpyDevice)
                  else "GPU")
        launcher = DummyLauncher()
        w = Workflow(launcher, ConvForward=ConvForward, ConvGD=ConvGD)
        w.initialize(device=device, snapshot=False)

        w.conv_forward.weights.map_read()
        conv_weights = w.conv_forward.weights.mem.copy()
        w.conv_forward.bias.map_read()
        conv_bias = w.conv_forward.bias.mem.copy()

        w.conv_forward2.weights.map_read()
        conv_weights2 = w.conv_forward2.weights.mem.copy()
        w.conv_forward2.bias.map_read()
        conv_bias2 = w.conv_forward2.bias.mem.copy()

        w.sm_forward.weights.map_read()
        sm_weights = w.sm_forward.weights.mem.copy()
        w.sm_forward.bias.map_read()
        sm_bias = w.sm_forward.bias.mem.copy()

        target = numpy.zeros_like(w.sm_forward.output.mem)
        for i, sample in enumerate(target):
            sample[w.labels[i]] = 1

        w.run()

        w.conv_gd.err_input.map_read()
        w.conv_gd.weights.map_read()
        w.conv_gd.bias.map_read()

        err_input = w.conv_gd.err_input.mem.ravel()
        weights_derivative = w.conv_gd.weights.mem - conv_weights
        bias_derivative = w.conv_gd.bias.mem - conv_bias

        w.end_point.unlink_before()
        w.ev.unlink_before()
        w.end_point.link_from(w.sm_forward)

        vv_map = {w.conv_gd.input: w.input,
                  w.conv_gd.weights: conv_weights,
                  w.conv_gd.bias: conv_bias,
                  w.conv_gd2.weights: conv_weights2,
                  w.conv_gd2.bias: conv_bias2,
                  w.sm_forward.weights: sm_weights,
                  w.sm_forward.bias: sm_bias}
        for v2c, d2c, nme in (
                (w.conv_gd.input, err_input, "err_input"),
                (w.conv_gd.weights, weights_derivative, "weights"),
                (w.conv_gd.bias, bias_derivative, "bias")):
            self.info("Checking %s via numeric differentiation", nme)
            self.numdiff_check(w, v2c, vv_map, w.sm_forward.output, target,
                               d2c, self.info, self.assertLess,
                               GDNumDiff.cross_entropy_mean, w.batch_size,
                               limit=(2 if isinstance(device, NumpyDevice)
                                      else None))
        del launcher


@assign_backend("ocl")
class OpenCLTestGDWorkflow(TestGDWorkflow):
    pass


@assign_backend("cuda")
class CUDATestGDWorkflow(TestGDWorkflow):
    pass


if __name__ == "__main__":
    StandardTest.main()
