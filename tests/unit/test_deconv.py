"""
Created on May 28, 2014

Unit test for deconvolutional unit.

Copyright (c) 2014 Samsung Electronics Co., Ltd.
"""


import logging
import numpy
import os
import unittest

from veles.config import root
import veles.formats as formats
import veles.opencl_types as opencl_types
import veles.znicz.conv as conv
import veles.znicz.deconv as deconv
import veles.znicz.gd_conv as gd_conv
import veles.znicz.gd_deconv as gd_deconv
from veles.tests.dummy_workflow import DummyWorkflow
import veles.random_generator as rnd
import veles.opencl as opencl


class TestDeconv(unittest.TestCase):
    def setUp(self):
        root.common.unit_test = True
        root.common.plotters_disabled = True
        self.device = opencl.Device()
        self.this_dir = os.path.dirname(__file__)
        if not len(self.this_dir):
            self.this_dir = "."

    def test_gd_deconv(self):
        logging.info("GDDeconv test...")
        _, forward, backward = self._test_deconv(self.device, None)
        gd = gd_deconv.GDDeconv(forward.workflow, n_kernels=forward.n_kernels,
                                kx=forward.kx, ky=forward.ky,
                                learning_rate=1.0, weights_decay=0.0,
                                gradient_moment=0.9)
        gd.weights = forward.weights
        gd.input = backward.input
        gd.err_output = backward.output
        gd.initialize(self.device)
        self.assertEqual(gd.err_input.shape, forward.output.shape)
        gd.run()
        gd.err_input.map_read()
        # TODO(a.kazantsev): add correctness check.

    def test_deconv(self):
        logging.info("GPU test...")
        gpu, forward, _ = self._test_deconv(self.device, None)
        logging.info("CPU test...")
        cpu, forward, _ = self._test_deconv(None, forward)
        max_diff = numpy.fabs(cpu - gpu).max()
        logging.info("CPU-GPU difference is %.6f", max_diff)
        self.assertLess(max_diff, 0.0001)
        logging.info("GD test...")
        gd = self._test_deconv_via_gd(forward)
        max_diff = numpy.fabs(gpu - gd).max()
        logging.info("GPU-GD difference is %.6f", max_diff)
        self.assertLess(max_diff, 0.0001)

    def _test_deconv(self, device, forward):
        rnd.get().seed("%s/seed" % self.this_dir,
                       dtype=numpy.int32, count=1024)

        if forward is None:
            batch_size = 7
            dtype = opencl_types.dtypes[root.common.precision_type]

            workflow = DummyWorkflow()
            forward = conv.Conv(workflow, n_kernels=25, kx=5, ky=5,
                                include_bias=False)
            inp = numpy.zeros([batch_size, 32, 32, 3], dtype=dtype)
            rnd.get().fill(inp)
            forward.input = formats.Vector(inp)
            forward.initialize(device)
            forward.run()

        backward = deconv.Deconv(forward.workflow, n_kernels=forward.n_kernels,
                                 kx=forward.kx, ky=forward.ky)
        backward.weights = forward.weights
        backward.input = forward.output
        backward.initialize(device)

        self.assertEqual(backward.output.shape, forward.input.shape,
                         "Shape test failed")

        backward.run()

        backward.output.map_read()

        if hasattr(backward.output, "vv"):
            nz = numpy.count_nonzero(
                backward.output.vv[backward.output.shape[0]:] - 1.0e30)
            self.assertEqual(nz, 0,
                             "Written some values outside of the target array")

        return backward.output.mem.copy(), forward, backward

    def _test_deconv_via_gd(self, forward):
        gd = gd_conv.GradientDescentConv(
            forward.workflow, n_kernels=forward.n_kernels,
            kx=forward.kx, ky=forward.ky, include_bias=False)
        gd.err_output = forward.output
        gd.weights = forward.weights
        gd.output = forward.output
        gd.input = forward.input
        gd.initialize(self.device)
        gd.gpu_err_input_update()
        gd.err_input.map_read()
        return gd.err_input.mem.copy()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
