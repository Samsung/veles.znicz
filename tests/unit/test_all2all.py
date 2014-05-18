#!/usr/bin/python3.3 -O
"""
Created on November 18, 2013

Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""


import logging
import numpy
import unittest

from veles.config import root
import veles.formats as formats
import veles.opencl as opencl
import veles.opencl_types as opencl_types
import veles.random_generator as rnd
import veles.znicz.all2all as all2all
from veles.tests.dummy_workflow import DummyWorkflow


class TestAll2All(unittest.TestCase):
    def setUp(self):
        root.common.unit_test = True
        root.common.plotters_disabled = True
        self.device = opencl.Device()

    def tearDown(self):
        del self.device

    def test_with_fixed_input(self):
        logging.info("Will test all2all unit")
        inp = formats.Vector()
        dtype = opencl_types.dtypes[root.common.dtype]
        inp.v = numpy.array([[1, 2, 3, 2, 1],
                             [0, 1, 2, 1, 0],
                             [0, 1, 0, 1, 0],
                             [2, 0, 1, 0, 2],
                             [1, 0, 1, 0, 1]], dtype=dtype)

        weights = numpy.array([[1, 0, 2, 1, -1],
                              [3, 1, 0, 2, 3],
                              [-1, 2, 0, 1, 3]], dtype=dtype)
        bias = numpy.array([10, -10, 5], dtype=dtype)

        c = all2all.All2All(DummyWorkflow(), output_shape=[3],
                            weights_amplitude=0.05)
        c.input = inp

        c.initialize(device=self.device)

        c.weights.map_invalidate()  # rewrite weights
        c.weights.v[:] = weights.reshape(c.weights.v.shape)[:]
        c.bias.map_invalidate()  # rewrite bias
        c.bias.v[:] = bias[:]

        c.run()
        c.output.map_read()  # get results back

        y = c.output.v.ravel()
        t = numpy.array([18, 2, 13, 15, -7, 8,
                         11, -7, 8, 12, 2, 9,
                         12, -4, 7], dtype=dtype)

        max_diff = numpy.fabs(t.ravel() - y.ravel()).max()
        self.assertLess(max_diff, 0.0001,
                        "Result differs by %.6f" % (max_diff))
        logging.info("All Ok")

    def _do_test(self, device, Unit):
        inp = formats.Vector()
        dtype = opencl_types.dtypes[root.common.dtype]
        inp.v = numpy.empty([1999, 1777], dtype=dtype)
        rnd.get().fill(inp.v)

        if device is not None:
            self.x = inp.v.copy()
        else:
            inp.v[:] = self.x[:]

        c = Unit(DummyWorkflow(), output_shape=[1, 1])
        c.input = inp
        c.initialize(device=device)

        if device is not None:
            self.W = c.weights.v.copy()
            self.b = c.bias.v.copy()
        else:
            c.weights.map_invalidate()
            c.bias.map_invalidate()
            c.weights.v[:] = self.W[:]
            c.bias.v[:] = self.b[:]

        c.run()
        c.output.map_read()  # get results back

        if hasattr(c, "max_idx"):
            c.max_idx.map_read()
            return (c.output.v.copy(), c.max_idx.v.copy())

        return (c.output.v.copy(),)

    def _do_gpu_cpu(self, Unit):
        y_gpus = self._do_test(self.device, Unit)
        y_cpus = self._do_test(None, Unit)
        for i, y_gpu in enumerate(y_gpus):
            y_cpu = y_cpus[i]
            max_diff = numpy.fabs(y_gpu.ravel() - y_cpu.ravel()).max()
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


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
