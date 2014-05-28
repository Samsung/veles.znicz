"""
Created on May 28, 2014

Unit test for convolutional-ppoling-softmax 3-layer back propagation.

Copyright (c) 2014 Samsung Electronics Co., Ltd.
"""


import logging
import numpy
import unittest

from veles.config import root
import veles.formats as formats
import veles.opencl_types as opencl_types
from veles.znicz.nn_units import OpenCLWorkflow
import veles.znicz.gd as gd
import veles.znicz.all2all as all2all
import veles.znicz.gd_conv as gd_conv
import veles.znicz.conv as conv
import veles.znicz.gd_pooling as gd_pooling
import veles.znicz.pooling as pooling
import veles.znicz.evaluator as evaluator
from veles.tests.dummy_workflow import DummyLauncher
import veles.random_generator as rnd
import veles.opencl as opencl
from veles.znicz.tests.unit.gd_numdiff import GDNumDiff


class Workflow(OpenCLWorkflow):
    def __init__(self, workflow, **kwargs):
        ConvForward = kwargs["ConvForward"]
        ConvGD = kwargs["ConvGD"]
        super(Workflow, self).__init__(workflow, **kwargs)

        dtype = opencl_types.dtypes[root.common.dtype]
        minibatch_size = 2

        self.input = numpy.zeros([minibatch_size, 8, 8, 3], dtype=dtype)
        rnd.get().fill(self.input)

        self.labels = numpy.zeros(minibatch_size, dtype=numpy.int32)
        self.labels[:] = rnd.get().randint(2, size=self.labels.size)

        self.conv_forward = ConvForward(
            self, n_kernels=5, kx=3, ky=3,
            padding=(2, 2, 2, 2), sliding=(1, 1))
        self.conv_forward.link_from(self.start_point)
        self.conv_forward.input = formats.Vector()
        self.conv_forward.input.mem = self.input.copy()

        self.pool_forward = pooling.MaxPooling(
            self, kx=3, ky=3, sliding=(2, 2))
        self.pool_forward.link_from(self.conv_forward)
        self.pool_forward.link_attrs(self.conv_forward, ("input", "output"))

        self.conv_forward2 = ConvForward(
            self, n_kernels=10, kx=3, ky=3,
            padding=(2, 2, 2, 2), sliding=(1, 1))
        self.conv_forward2.link_from(self.pool_forward)
        self.conv_forward2.link_attrs(self.pool_forward, ("input", "output"))

        self.pool_forward2 = pooling.AvgPooling(
            self, kx=3, ky=3, sliding=(2, 2))
        self.pool_forward2.link_from(self.conv_forward2)
        self.pool_forward2.link_attrs(self.conv_forward2, ("input", "output"))

        self.sm_forward = all2all.All2AllSoftmax(
            self, output_shape=[10])
        self.sm_forward.link_from(self.pool_forward2)
        self.sm_forward.link_attrs(self.pool_forward2, ("input", "output"))

        self.ev = evaluator.EvaluatorSoftmax(self)
        self.ev.link_from(self.sm_forward)
        self.ev.link_attrs(self.sm_forward, "output", "max_idx")
        self.ev.labels = formats.Vector()
        self.ev.labels.mem = self.labels.copy()
        self.ev.batch_size = minibatch_size

        self.sm_gd = gd.GDSM(
            self, gradient_moment=0, gradient_moment_bias=0,
            learning_rate=-1, weights_decay=0,
            learning_rate_bias=-1, weights_decay_bias=0)
        self.sm_gd.link_from(self.ev)
        self.sm_gd.link_attrs(self.ev, "err_output")
        self.sm_gd.link_attrs(self.sm_forward, "weights", "bias",
                              "input", "output")
        self.sm_gd.batch_size = minibatch_size

        self.pool_gd2 = gd_pooling.GDAvgPooling(
            self, kx=self.pool_forward2.kx, ky=self.pool_forward2.ky,
            sliding=self.pool_forward2.sliding)
        self.pool_gd2.link_from(self.sm_gd)
        self.pool_gd2.link_attrs(self.sm_gd, ("err_output", "err_input"))
        self.pool_gd2.link_attrs(self.pool_forward2, "input")

        self.conv_gd2 = ConvGD(
            self, n_kernels=self.conv_forward2.n_kernels,
            kx=self.conv_forward2.kx, ky=self.conv_forward2.ky,
            gradient_moment=0, gradient_moment_bias=0,
            learning_rate=-1, weights_decay=0,
            learning_rate_bias=-1, weights_decay_bias=0,
            padding=self.conv_forward2.padding,
            sliding=self.conv_forward2.sliding)
        self.conv_gd2.link_from(self.pool_gd2)
        self.conv_gd2.link_attrs(self.pool_gd2, ("err_output", "err_input"))
        self.conv_gd2.link_attrs(self.conv_forward2, "weights", "bias",
                                 "input", "output")
        self.conv_gd2.batch_size = minibatch_size

        self.pool_gd = gd_pooling.GDMaxPooling(
            self, kx=self.pool_forward.kx, ky=self.pool_forward.ky,
            sliding=self.pool_forward.sliding)
        self.pool_gd.link_from(self.conv_gd2)
        self.pool_gd.link_attrs(self.conv_gd2, ("err_output", "err_input"))
        self.pool_gd.link_attrs(self.pool_forward, "input", "input_offs")

        self.conv_gd = ConvGD(
            self, n_kernels=self.conv_forward.n_kernels,
            kx=self.conv_forward.kx, ky=self.conv_forward.ky,
            gradient_moment=0, gradient_moment_bias=0,
            learning_rate=-1, weights_decay=0,
            learning_rate_bias=-1, weights_decay_bias=0,
            padding=self.conv_forward.padding,
            sliding=self.conv_forward.sliding)
        self.conv_gd.link_from(self.pool_gd)
        self.conv_gd.link_attrs(self.pool_gd, ("err_output", "err_input"))
        self.conv_gd.link_attrs(self.conv_forward, "weights", "bias",
                                "input", "output")
        self.conv_gd.batch_size = minibatch_size

        self.end_point.link_from(self.conv_gd)


class TestGDConvPoolingSoftmax(unittest.TestCase, GDNumDiff):
    def setUp(self):
        root.common.unit_test = True
        root.common.plotters_disabled = True
        if not hasattr(self, "device"):
            self.device = opencl.Device()

    def test_random_numeric_gpu(self):
        self._test_random_numeric(self.device, conv.Conv,
                                  gd_conv.GradientDescentConv)

    def test_random_numeric_gpu_tanh(self):
        self._test_random_numeric(self.device, conv.ConvTanh,
                                  gd_conv.GDTanhConv)

    def test_random_numeric_gpu_relu(self):
        self._test_random_numeric(self.device, conv.ConvRELU,
                                  gd_conv.GDRELUConv)

    def _test_random_numeric_cpu(self):
        self._test_random_numeric(None, conv.Conv,
                                  gd_conv.GradientDescentConv)

    def _test_random_numeric_cpu_tanh(self):
        self._test_random_numeric(None, conv.ConvTanh,
                                  gd_conv.GDTanhConv)

    def _test_random_numeric_cpu_relu(self):
        self._test_random_numeric(None, conv.ConvRELU,
                                  gd_conv.GDRELUConv)

    def _test_random_numeric(self, device, ConvForward, ConvGD):
        logging.info("Will test forward-backward "
                     "via numeric differentiation")

        w = Workflow(DummyLauncher(), ConvForward=ConvForward, ConvGD=ConvGD)
        w.initialize(device=device)

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
        weights_derivative = ((w.conv_gd.weights.mem - conv_weights) *
                              w.input.shape[0])
        bias_derivative = ((w.conv_gd.bias.mem - conv_bias) *
                           w.input.shape[0])

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
        for v2c, d2c in ((w.conv_gd.input, err_input),
                         (w.conv_gd.weights, weights_derivative),
                         (w.conv_gd.bias, bias_derivative)):
            self.numdiff_check(w, v2c, vv_map, w.sm_forward.output, target,
                               d2c, logging.info, self.assertLess,
                               GDNumDiff.cross_entropy)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
