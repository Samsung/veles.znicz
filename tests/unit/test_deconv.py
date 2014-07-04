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
from veles.znicz.tests.unit.gd_numdiff import GDNumDiff


class TestDeconv(unittest.TestCase, GDNumDiff):
    def setUp(self):
        root.common.unit_test = True
        root.common.plotters_disabled = True
        self.device = opencl.Device()
        self.this_dir = os.path.dirname(__file__)
        if not len(self.this_dir):
            self.this_dir = "."

    def test_gd_deconv(self):
        logging.info("GDDeconv err_input overflow test...")
        _, first, forward = self._test_deconv(self.device, None)

        first.weights.map_read()
        weights = first.weights.mem.copy()
        first.input.map_read()
        target = first.input.mem.copy()
        forward.output.map_read()
        out = forward.output.mem.copy()
        err_output = out - target
        forward.input.map_read()
        inp = forward.input.mem.copy()

        gd = gd_deconv.GDDeconv(first.workflow, n_kernels=first.n_kernels,
                                kx=first.kx, ky=first.ky,
                                padding=first.padding, sliding=first.sliding,
                                learning_rate=-1.0, weights_decay=0.0,
                                gradient_moment=0.9)
        gd.weights = first.weights
        gd.input = forward.input
        gd.err_output = formats.Vector(err_output)
        gd.initialize(self.device)
        self.assertEqual(gd.err_input.shape, first.output.shape)
        gd.run()
        gd.err_input.map_read()
        nz = numpy.count_nonzero(numpy.isnan(gd.err_input.mem))
        self.assertEqual(nz, 0, "NaNs encountered in err_input")
        nz = numpy.count_nonzero(numpy.isnan(
            gd.err_input.vv[gd.err_input.shape[0]:]))
        self.assertEqual(nz, gd.err_input.size,
                         "Written some values outside of the target array")

        logging.info("GDDeconv numeric derivative test...")
        gd.weights.map_read()
        nz = numpy.count_nonzero(numpy.isnan(gd.weights.mem))
        self.assertEqual(nz, 0, "NaNs encountered in weights")
        err_input = gd.err_input.mem
        weights_derivative = gd.weights.mem - weights
        self.numdiff_check_gd(forward, inp, weights, None, target,
                              err_input, weights_derivative, None,
                              logging.info, self.assertLess,
                              error_function_averaged=False,
                              threshold=1.0e-3)

    def test_deconv(self):
        logging.info("GPU test...")
        gpu, forward, _ = self._test_deconv(self.device, None)
        logging.info("CPU test...")
        cpu, forward, _ = self._test_deconv(None, forward)
        max_diff = numpy.fabs(cpu - gpu).max()
        logging.info("CPU-GPU difference is %.6f (cpu_max=%.6f gpu_max=%.6f)",
                     max_diff, numpy.fabs(cpu).max(), numpy.fabs(gpu).max())
        self.assertLess(max_diff, 0.0001)
        logging.info("GD test...")
        gd = self._test_deconv_via_gd(forward)
        max_diff = numpy.fabs(gpu - gd).max()
        logging.info("GPU-GD difference is %.6f", max_diff)
        self.assertLess(max_diff, 0.0001)

    def _test_deconv(self, device, forward):
        rnd.get().seed("%s/seed" % self.this_dir,
                       dtype=numpy.int32, count=1024)

        dtype = opencl_types.dtypes[root.common.precision_type]

        if forward is None:
            batch_size = 3
            workflow = DummyWorkflow()
            forward = conv.Conv(workflow, n_kernels=9, kx=5, ky=5,
                                padding=(2, 2, 2, 2), sliding=(1, 1),
                                include_bias=False)
            inp = formats.Vector(numpy.zeros([batch_size * 2, 16, 16, 3],
                                             dtype=dtype))
            inp.initialize(device)
            inp.map_write()
            inp.vv = inp.mem
            inp.mem = inp.vv[:batch_size]
            formats.assert_addr(inp.vv, inp.mem)
            rnd.get().fill(inp.mem)
            inp.vv[batch_size:] = numpy.nan
            forward.input = inp
            forward.initialize(device)
            forward.run()

        forward.output.map_read()
        sh = list(forward.output.mem.shape)
        sh[0] <<= 1
        out = formats.Vector(numpy.zeros(sh, dtype=dtype))
        out.initialize(device)
        out.map_write()
        out.vv = out.mem
        sh[0] >>= 1
        out.mem = out.vv[:sh[0]]
        formats.assert_addr(out.mem, out.vv)
        out.mem[:] = forward.output.mem[:]
        out.vv[sh[0]:] = numpy.nan

        backward = deconv.Deconv(forward.workflow, n_kernels=forward.n_kernels,
                                 kx=forward.kx, ky=forward.ky,
                                 padding=forward.padding,
                                 sliding=forward.sliding)
        backward.weights = forward.weights
        backward.input = out
        backward.initialize(device)

        self.assertEqual(backward.output.shape, forward.input.shape,
                         "Shape test failed")

        backward.run()

        backward.output.map_read()

        nz = numpy.count_nonzero(numpy.isnan(backward.output.mem))
        self.assertEqual(nz, 0, "NaNs encountered")

        if device is not None:
            nz = numpy.count_nonzero(
                numpy.isnan(backward.output.vv[backward.output.shape[0]:]))
            self.assertEqual(nz, backward.output.vv.size >> 1,
                             "Written some values outside of the target array")

        return backward.output.mem.copy(), forward, backward

    def _test_deconv_via_gd(self, forward):
        gd = gd_conv.GradientDescentConv(
            forward.workflow, n_kernels=forward.n_kernels,
            kx=forward.kx, ky=forward.ky, include_bias=False,
            padding=forward.padding, sliding=forward.sliding)
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
