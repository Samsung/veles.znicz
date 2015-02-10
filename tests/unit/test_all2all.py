#!/usr/bin/python3 -O
"""
Created on November 18, 2013

Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""

import gc
import logging
import numpy
import time
import unittest

from veles.config import root
from veles.memory import Vector
import veles.backends as opencl
import veles.opencl_types as opencl_types
import veles.prng as prng
import veles.znicz.all2all as all2all
from veles.dummy import DummyWorkflow


class TestAll2All(unittest.TestCase):
    def setUp(self):
        prng.get().seed(1234)
        self.device = opencl.Device()

    def tearDown(self):
        gc.collect()
        del self.device

    def test_with_fixed_input(self):
        logging.info("Will test all2all unit")
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

        c = all2all.All2All(DummyWorkflow(), output_sample_shape=[3],
                            weights_amplitude=0.05)
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
                        "Result differs by %.6f" % (max_diff))
        logging.info("All Ok")

    def _do_test(self, device, Unit):
        inp = Vector()
        dtype = opencl_types.dtypes[root.common.precision_type]
        inp.mem = numpy.empty([1999, 1777], dtype=dtype)
        prng.get().fill(inp.mem)

        if device is not None:
            self.x = inp.mem.copy()
        else:
            inp.mem[:] = self.x[:]

        c = Unit(DummyWorkflow(), output_sample_shape=[75, 75])
        c.input = inp
        c.initialize(device=device)

        if device is not None:
            self.W = c.weights.mem.copy()
            self.b = c.bias.mem.copy()
        else:
            c.weights.map_invalidate()
            c.bias.map_invalidate()
            c.weights.mem[:] = self.W[:]
            c.bias.mem[:] = self.b[:]

        if device is not None:
            device.sync()
            t0 = time.time()
            for _ in range(3):
                c.run()
            device.sync()
            dt = time.time() - t0
            logging.info("Completed in %.6f sec", dt)
        else:
            c.run()
        c.output.map_read()  # get results back

        if hasattr(c, "max_idx"):
            c.max_idx.map_read()
            return (c.output.mem.copy(), c.max_idx.mem.copy())

        return (c.output.mem.copy(),)

    def _do_gpu_cpu(self, Unit):
        y_gpus = self._do_test(self.device, Unit)
        y_cpus = self._do_test(None, Unit)
        for i, y_gpu in enumerate(y_gpus):
            y_cpu = y_cpus[i]
            max_diff = numpy.fabs(y_gpu.ravel() - y_cpu.ravel()).max()
            logging.info("Difference is %.12f", max_diff)
            self.assertLess(max_diff, 0.0001,
                            "Result differs by %.6f" % (max_diff))
        logging.info("All Ok")

    def test_linear(self):
        logging.info("Will test linear all2all unit for gpu/cpu correctness")
        self._do_gpu_cpu(all2all.All2All)

    def test_tanh(self):
        logging.info("Will test Tanh all2all unit for gpu/cpu correctness")
        self._do_gpu_cpu(all2all.All2AllTanh)

    def test_relu(self):
        logging.info("Will test RELU all2all unit for gpu/cpu correctness")
        self._do_gpu_cpu(all2all.All2AllRELU)

    def test_softmax(self):
        logging.info("Will test Softmax all2all unit for gpu/cpu correctness")
        self._do_gpu_cpu(all2all.All2AllSoftmax)

    def test_two_stage(self):
        dtype = opencl_types.dtypes[root.common.precision_type]
        a2a = all2all.All2All(DummyWorkflow(), output_sample_shape=(75, 75),
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

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
