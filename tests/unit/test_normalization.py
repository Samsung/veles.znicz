#!/usr/bin/python3.3 -O
# encoding: utf-8

"""
Created on April 24, 2014

A unit test for local response normalization.
"""

import unittest
import logging

import numpy as np
from veles import opencl

from veles.formats import Vector
from veles.znicz.normalization import LRNormalizerForward, LRNormalizerBackward
from veles.tests.dummy_workflow import DummyWorkflow


class TestNormalization(unittest.TestCase):
    def setUp(self):
        self.workflow = DummyWorkflow()
        self.device = opencl.Device()

    def tearDown(self):
        pass

    def test_normalization_forward(self):
        fwd_normalizer = LRNormalizerForward(self.workflow,
                                             device=self.device, n=3)
        fwd_normalizer.input = Vector()
        in_vector = np.zeros(shape=(1, 1, 5, 5), dtype=np.float64)

        for i in range(5):
            in_vector[0, 0, i, :] = np.linspace(10, 50, 5) * (i + 1)

        fwd_normalizer.input.v = in_vector
        fwd_normalizer.initialize(device=self.device)

        fwd_normalizer.ocl_run()
        fwd_normalizer.output.map_read()
        logging.info("FORWARD")
        logging.info(fwd_normalizer.output.v)

    def test_normalization_backward(self):

        h = np.zeros(shape=(1, 1, 5, 5), dtype=np.float64)
        for i in range(5):
            h[0, 0, i, :] = np.linspace(10, 50, 5) * (i + 1)

        err_y = np.zeros(shape=(1, 1, 5, 5), dtype=np.float64)
        for i in range(5):
            err_y[0, 0, i, :] = np.linspace(2, 10, 5) * (i + 1)

        back_normalizer = LRNormalizerBackward(self.workflow,
                                               device=self.device, n=3)

        back_normalizer.h = Vector()
        back_normalizer.err_y = Vector()

        back_normalizer.h.v = h
        back_normalizer.err_y.v = err_y

        back_normalizer.initialize(device=self.device)

        back_normalizer.ocl_run()
        back_normalizer.err_h.map_read()
        logging.info("BACK")
        logging.info(back_normalizer.err_h.v)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    logging.info("Running LR normalizer test!")
    unittest.main()
