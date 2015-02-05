#!/usr/bin/python3 -O
"""
Created on November 18, 2013

Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""


import gc
import logging
import numpy
import unittest

from veles.config import root
import veles.memory as formats
import veles.backends as opencl
import veles.opencl_types as opencl_types
import veles.prng as random_generator
from veles.dummy import DummyWorkflow
import veles.znicz.evaluator as evaluator


class TestEvaluator(unittest.TestCase):
    def setUp(self):
        self.device = opencl.Device()

    def tearDown(self):
        gc.collect()
        del self.device

    def test_mse(self):
        batch_size = 25
        sample_size = 7500

        dtype = opencl_types.dtypes[root.common.precision_type]
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
        gold_err_output = (output - target) / (batch_size - 5)
        gold_err_output[ev.batch_size:] = 0

        ev.initialize(device=self.device)
        ev.err_output.map_invalidate()
        ev.err_output.mem[:] = 1.0e30
        ev.run()

        ev.err_output.map_read()
        max_diff = numpy.fabs(ev.err_output.mem - gold_err_output).max()
        logging.info("Difference is %.12f", max_diff)
        self.assertLess(max_diff, 1.0e-4)

    def test_softmax(self):
        batch_size = 25
        n_classes = 75

        dtype = opencl_types.dtypes[root.common.precision_type]
        output = numpy.empty([batch_size, n_classes], dtype=dtype)
        random_generator.get().fill(output)
        max_idx = numpy.empty(batch_size, dtype=numpy.int32)
        for i, sample in enumerate(output):
            max_idx[i] = sample.argmax()
            sample -= sample[max_idx[i]]
            numpy.exp(sample, sample)
            sample /= sample.sum()

        labels = numpy.empty(batch_size, dtype=numpy.int32)
        labels[:] = random_generator.get().randint(0, n_classes, batch_size)

        target = numpy.empty_like(output)
        for i, sample in enumerate(target):
            sample[:] = 0
            sample[labels[i]] = 1

        ev = evaluator.EvaluatorSoftmax(DummyWorkflow())
        ev.output = formats.Vector()
        ev.output.mem = output.copy()
        ev.labels = formats.Vector()
        ev.labels.mem = labels.copy()
        ev.max_idx = formats.Vector()
        ev.max_idx.mem = max_idx
        ev.batch_size = batch_size - 5

        gold_err_output = (output - target) / (batch_size - 5)
        gold_err_output[ev.batch_size:] = 0

        ev.initialize(device=self.device)
        ev.err_output.map_invalidate()
        ev.err_output.mem[:] = 1.0e30
        ev.run()

        ev.err_output.map_read()
        max_diff = numpy.fabs(ev.err_output.mem - gold_err_output).max()
        logging.info("Difference is %.12f", max_diff)
        self.assertLess(max_diff, 1.0e-4)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
