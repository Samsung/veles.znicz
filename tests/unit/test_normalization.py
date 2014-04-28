#!/usr/bin/python3.3 -O
# encoding: utf-8

"""
Created on April 24, 2014

A unit test for local response normalization.
"""

import unittest
import logging

import numpy as np

from veles.formats import Vector
from veles.znicz.normalization import LRNormalizerForward, LRNormalizerBackward
from veles.tests.dummy_workflow import DummyWorkflow


class TestNormalization(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_normalization_forward(self):
        workflow = DummyWorkflow()
        fwd_normalizer = LRNormalizerForward(workflow, device=None, n=3)
        fwd_normalizer.input = Vector()
        in_vector = np.zeros(shape=(1, 1, 5, 5), dtype=np.float64)

        for i in range(5):
            in_vector[0, 0, i, :] = np.linspace(10, 50, 5) * (i + 1)

        fwd_normalizer.input.v = in_vector
        fwd_normalizer.initialize()

        fwd_normalizer.run()

        back_normalizer = LRNormalizerBackward(workflow, device=None, n=3)
        back_normalizer.h = fwd_normalizer.input
        back_normalizer.y = fwd_normalizer.output

        y_err_vector = np.zeros(shape=(1, 1, 5, 5), dtype=np.float64)
        for i in range(5):
            y_err_vector[0, 0, i, :] = np.linspace(2, 10, 5) * (i + 1)

        back_normalizer.err_y = Vector()
        back_normalizer.err_y.v = y_err_vector

        back_normalizer.run()


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    logging.info("Running LR normalizer test!")
    unittest.main()
