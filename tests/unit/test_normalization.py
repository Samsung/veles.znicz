#!/usr/bin/python3 -O
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
from veles.dummy import DummyWorkflow


class TestNormalization(unittest.TestCase):
    def setUp(self):
        self.workflow = DummyWorkflow()
        self.device = opencl.Device()

    def test_normalization_forward(self):
        fwd_normalizer = LRNormalizerForward(self.workflow,
                                             device=self.device, n=3)
        fwd_normalizer.input = Vector()
        in_vector = np.zeros(shape=(1, 2, 5, 5), dtype=np.float64)

        for i in range(5):
            in_vector[0, 0, i, :] = np.linspace(10, 50, 5) * (i + 1)
            in_vector[0, 1, i, :] = np.linspace(10, 50, 5) * (i + 1) + 1

        fwd_normalizer.input.mem = in_vector
        fwd_normalizer.initialize(device=self.device)

        fwd_normalizer.ocl_run()
        fwd_normalizer.output.map_read()
        ocl_result = np.copy(fwd_normalizer.output.mem)

        fwd_normalizer.cpu_run()
        fwd_normalizer.output.map_read()
        cpu_result = np.copy(fwd_normalizer.output.mem)

        max_delta = np.fabs(cpu_result - ocl_result).max()

        logging.info("FORWARD")
        self.assertLess(max_delta, 0.0001,
                        "Result differs by %.6f" % (max_delta))

        logging.info("FwdProp done.")

    def test_normalization_backward(self):

        h = np.zeros(shape=(2, 1, 5, 5), dtype=np.float64)
        for i in range(5):
            h[0, 0, i, :] = np.linspace(10, 50, 5) * (i + 1)
            h[1, 0, i, :] = np.linspace(10, 50, 5) * (i + 1) + 1

        err_y = np.zeros(shape=(2, 1, 5, 5), dtype=np.float64)
        for i in range(5):
            err_y[0, 0, i, :] = np.linspace(2, 10, 5) * (i + 1)
            err_y[1, 0, i, :] = np.linspace(2, 10, 5) * (i + 1) + 1

        back_normalizer = LRNormalizerBackward(self.workflow,
                                               device=self.device, n=3)
        back_normalizer.input = Vector()
        back_normalizer.err_output = Vector()

        back_normalizer.input.mem = h
        back_normalizer.err_output.mem = err_y

        back_normalizer.initialize(device=self.device)

        back_normalizer.cpu_run()
        cpu_result = back_normalizer.err_input.mem.copy()

        back_normalizer.err_input.map_invalidate()
        back_normalizer.err_input.mem[:] = 100

        back_normalizer.ocl_run()
        back_normalizer.err_input.map_read()
        ocl_result = back_normalizer.err_input.mem.copy()

        logging.info("BACK")

        max_delta = np.fabs(cpu_result - ocl_result).max()
        self.assertLess(max_delta, 0.0001,
                        "Result differs by %.6f" % (max_delta))
#        print(err_y)
#        print(cpu_result)
#        print(ocl_result)

        logging.info("BackProp done.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    logging.info("Running LR normalizer test!")
    unittest.main()
