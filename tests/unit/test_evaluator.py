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
import veles.random_generator as random_generator
from veles.tests.dummy_workflow import DummyWorkflow
import veles.znicz.evaluator as evaluator


class TestEvaluator(unittest.TestCase):
    def setUp(self):
        root.common.unit_test = True
        root.common.plotters_disabled = True
        if not hasattr(self, "device"):
            self.device = opencl.Device()

    def test_mse(self):
        batch_size = 25
        sample_size = 7500

        dtype = opencl_types.dtypes[root.common.dtype]
        output = numpy.empty([batch_size, sample_size], dtype=dtype)
        random_generator.get().fill(output)

        target = numpy.empty_like(output)
        random_generator.get().fill(target)

        ev = evaluator.EvaluatorMSE(DummyWorkflow())
        ev.output = formats.Vector()
        ev.output.mem = output.copy()
        ev.target = formats.Vector()
        ev.target.mem = target.copy()
        ev.batch_size = batch_size - 5
        gold_err_output = output - target
        gold_err_output[ev.batch_size:] = 0

        ev.initialize(device=self.device)
        ev.err_output.map_invalidate()
        ev.err_output.mem[:] = 1.0e30
        ev.run()

        ev.err_output.map_read()
        max_diff = numpy.fabs(ev.err_output.mem - gold_err_output).max()
        self.assertLess(max_diff, 1.0e-4)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
