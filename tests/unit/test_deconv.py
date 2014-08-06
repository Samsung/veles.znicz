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
import veles.znicz.gd_deconv as gd_deconv
from veles.tests.dummy_workflow import DummyWorkflow
import veles.prng as rnd
import veles.opencl as opencl
from veles.znicz.tests.unit.gd_numdiff import GDNumDiff
from veles.formats import Vector


class TestDeconv(unittest.TestCase, GDNumDiff):
    def setUp(self):
        root.common.unit_test = True
        root.common.plotters_disabled = True
        self.device = opencl.Device()
        self.this_dir = os.path.dirname(__file__)
        if not len(self.this_dir):
            self.this_dir = "."
        self.dtype = opencl_types.dtypes[root.common.precision_type]

    def test_fixed(self):
        inp = numpy.ones([1, 4, 4, 1], dtype=self.dtype)
        forward = conv.Conv(DummyWorkflow(), kx=3, ky=3, n_kernels=1,
                            padding=(0, 0, 0, 0), sliding=(1, 1),
                            include_bias=False)
        forward.input = Vector(inp)
        forward.initialize(self.device)
        forward.weights.map_invalidate()
        forward.weights.mem[:] = 1.0
        forward.run()

        de = deconv.Deconv(DummyWorkflow(), kx=forward.kx, ky=forward.ky,
                           n_kernels=forward.n_kernels,
                           padding=forward.padding, sliding=forward.sliding,
                           unsafe_padding=True)
        de.input = forward.output
        de.get_output_shape_from = forward.input
        de.weights = forward.weights
        de.initialize(self.device)
        de.run()
        de.output.map_read()
        nz = numpy.count_nonzero(de.output.mem - inp * 9)
        self.assertEqual(nz, 0)

    def test_compute_padding(self):
        sx = 128
        for kx, slide in ((2, 1), (3, 1), (4, 1), (4, 2), (5, 1),
                          (6, 1), (6, 2), (6, 3), (7, 1),
                          (8, 1), (8, 2), (8, 4),
                          (9, 1), (9, 3),
                          (10, 1), (10, 2), (10, 5), (11, 1),
                          (12, 1), (12, 2), (12, 3), (12, 4), (12, 6),
                          (13, 1), (14, 1), (14, 2), (14, 7),
                          (15, 1), (15, 3), (15, 5),
                          (16, 1), (16, 2), (16, 4), (16, 8),
                          (17, 1),
                          (18, 1), (18, 2), (18, 3), (18, 6), (18, 9),
                          (19, 1),
                          (20, 1), (20, 2), (20, 4), (20, 5), (20, 10)):
            self._test_compute_padding(sx, sx, kx, kx, (slide, slide))

    def _test_compute_padding(self, sx, sy, kx, ky, sliding):
        padding = deconv.Deconv.compute_padding(sx, sy, kx, ky, sliding)
        a = numpy.zeros([sy + padding[1] + padding[3],
                         sx + padding[0] + padding[2]], dtype=numpy.int32)
        b = numpy.ones([ky, kx], dtype=numpy.int32)
        for y in range(0, a.shape[0] - ky + 1, sliding[1]):
            for x in range(0, a.shape[1] - kx + 1, sliding[0]):
                a[y:y + ky, x:x + kx] += b
        c = a[padding[1]:sy - padding[3], padding[0]:sx - padding[2]]
        self.assertEqual(c.min(), c.max(), "Unequal distribution")
        self.assertEqual(c.min(), (kx // sliding[1]) * (ky // sliding[0]),
                         "Wrong value")

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

    def _test_deconv(self, device, forward):
        rnd.get().seed("%s/seed" % self.this_dir,
                       dtype=numpy.int32, count=1024)

        dtype = opencl_types.dtypes[root.common.precision_type]

        if forward is None:
            batch_size = 3
            workflow = DummyWorkflow()
            forward = conv.Conv(workflow, n_kernels=9, kx=6, ky=6,
                                padding=(4, 4, 4, 4), sliding=(2, 2),
                                include_bias=False)
            inp = formats.Vector(numpy.zeros([batch_size * 2, 18, 18, 4],
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
        backward.get_output_shape_from = forward.input
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


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
