#!/usr/bin/python3 -O
# encoding: utf-8

"""
Created on April 25, 2014

A unit test for dropout layer.
"""
import unittest
import logging
import numpy as np

from veles.memory import Vector
from veles.backends import Device
from veles.znicz.dropout import DropoutForward, DropoutBackward
from veles.dummy import DummyWorkflow


class TestType(object):
    CPU = 0
    OCL = 1

    def __init__(self):
        """
        Needed by PEP8.
        """
        pass


class TestDropout(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def _run_test(self, test_type):
        workflow = DummyWorkflow()
        fwd_dropout = DropoutForward(workflow, dropout_ratio=0.4)
        fwd_dropout.input = Vector()
        sz = 100
        in_matrix = np.zeros(shape=(1, 1, sz, sz), dtype=np.float64)

        for i in range(sz):
            in_matrix[0, 0, i, :] = np.linspace(0, sz * 10, sz) * (i + 1)
        fwd_dropout.input.mem = in_matrix
        logging.debug("[DropoutForward] input matrix:\n%s", in_matrix)

        device = Device()
        fwd_dropout.initialize(device)
        if test_type == TestType.OCL:
            fwd_dropout.ocl_run()
            fwd_dropout.output.map_read()
        else:
            fwd_dropout.cpu_run()

        logging.debug("[DropoutForward] output matrix:\n%s",
                      fwd_dropout.output.mem)
        ratio = 1.0 - float(np.count_nonzero(fwd_dropout.output.mem)) / \
            fwd_dropout.output.mem.size
        logging.debug("[DropoutForward] dropout ratio: %.4f", ratio)
        self.assertAlmostEqual(ratio, fwd_dropout.dropout_ratio,
                               delta=fwd_dropout.dropout_ratio / 10,
                               msg='error in DropoutForward results: ratio of '
                               'zero elements in output matrix is {0} '
                               '(target value is {1})'.format(
                                   ratio, fwd_dropout.dropout_ratio))

        back_dropout = DropoutBackward(workflow)
        back_dropout.mask = fwd_dropout.mask

        err_output = np.zeros(shape=(1, 1, sz, sz), dtype=np.float64)
        for i in range(sz):
            err_output[0, 0, i, :] = np.linspace(0, sz * 2, sz) * (i + 1)
        back_dropout.err_output = Vector()
        back_dropout.err_output.mem = err_output
        logging.debug("[DropoutBackward] err_y matrix:\n%s", err_output)

        back_dropout.initialize(device)
        if test_type == TestType.OCL:
            back_dropout.ocl_run()
            back_dropout.err_input.map_read()
        else:
            back_dropout.cpu_run()

        logging.debug("[DropoutBackward] err_input:")
        logging.debug(back_dropout.err_input.mem)
        ratio = 1.0 - float(np.count_nonzero(back_dropout.err_input.mem)) / \
            back_dropout.err_input.mem.size
        logging.debug("[DropoutBackward]  dropout ratio: %.4f", ratio)
        self.assertAlmostEqual(ratio, fwd_dropout.dropout_ratio,
                               delta=fwd_dropout.dropout_ratio / 10,
                               msg='error in DropoutBackward results: ratio of'
                               ' zero elements in err_input matrix is {0} '
                               '(target value is {1})'.format(
                                   ratio, fwd_dropout.dropout_ratio))

    def test_cpu(self):
        logging.info("start CPU test...")
        self._run_test(TestType.CPU)
        logging.info("TEST PASSED")

    def test_ocl(self):
        logging.info("start OpenCL test...")
        self._run_test(TestType.OCL)
        logging.info("TEST PASSED")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logging.info("Running dropout tests")
    unittest.main()
