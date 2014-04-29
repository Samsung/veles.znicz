#!/usr/bin/python3.3 -O
# encoding: utf-8

"""
Created on April 25, 2014

A unit test for drouput layer.
"""

import unittest
import logging
import numpy as np

from veles.formats import Vector
from veles.opencl import Device
from veles.znicz.dropout import DropoutForward, DropoutBackward
from veles.tests.dummy_workflow import DummyWorkflow


class TestNormalization(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_dropout(self):
        workflow = DummyWorkflow()
        fwd_dropout = DropoutForward(workflow, dropout_ratio=0.5)
        fwd_dropout.input = Vector()
        fwd_dropout.output = Vector()
        in_vector = np.zeros(shape=(1, 1, 5, 5), dtype=np.float64)

        for i in range(5):
            in_vector[0, 0, i, :] = np.linspace(10, 50, 5) * (i + 1)

        logging.info("IN_VECTOR")
        logging.info(in_vector)

        fwd_dropout.input.v = in_vector
        device = Device()
        fwd_dropout.initialize(device)

        fwd_dropout.cpu_run()

        logging.info("FWD")
        logging.info(fwd_dropout.output.v)

        back_drouput = DropoutBackward(workflow, drouput_ratio=0.5)
        back_drouput.h = fwd_dropout.input
        back_drouput.y = fwd_dropout.output
        back_drouput.weights = fwd_dropout.weights

        y_err_vector = np.zeros(shape=(1, 1, 5, 5), dtype=np.float64)
        for i in range(5):
            y_err_vector[0, 0, i, :] = np.linspace(2, 10, 5) * (i + 1)

        back_drouput.err_y = Vector()
        back_drouput.err_y.v = y_err_vector

        back_drouput.initialize(device)

        logging.info("Y_ERR")
        logging.info(y_err_vector)

        back_drouput.cpu_run()
        logging.info("BACK")
        logging.info(back_drouput.err_h.v)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    logging.info("Running dropout test!")
    unittest.main()
