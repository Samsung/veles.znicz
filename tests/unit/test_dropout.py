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


class TestType:
    CPU = 0
    OCL = 1


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
        fwd_dropout.input.v = in_matrix
        logging.info("[DropoutForward] input matrix:\n%s", in_matrix)

        device = Device()
        fwd_dropout.initialize(device)

        if test_type == TestType.OCL:
            fwd_dropout.ocl_run()
            fwd_dropout.output.map_read()
        else:
            fwd_dropout.cpu_run()

        logging.info("[DropoutForward] output matrix:\n%s",
                     fwd_dropout.output.v)
        ratio = 1.0 - np.count_nonzero(fwd_dropout.output.v) / \
            fwd_dropout.output.v.size
        logging.info("[DropoutForward] dropout ratio: %.4f", ratio)
        self.assertAlmostEqual(ratio, fwd_dropout.dropout_ratio,
                               delta=fwd_dropout.dropout_ratio / 10,
                               msg='error in DropoutForward results: ratio of '
                               'zero elements in output matrix is {0} '
                               '(target value is {1})'.format(ratio,
                               fwd_dropout.dropout_ratio))

        back_dropout = DropoutBackward(workflow)
        back_dropout.mask = fwd_dropout.mask

        err_y = np.zeros(shape=(1, 1, sz, sz), dtype=np.float64)
        for i in range(sz):
            err_y[0, 0, i, :] = np.linspace(0, sz * 2, sz) * (i + 1)
        back_dropout.err_y = Vector()
        back_dropout.err_y.v = err_y
        logging.info("[DropoutBackward] err_y matrix:\n%s", err_y)

        back_dropout.initialize(device)
        if test_type == TestType.OCL:
            back_dropout.ocl_run()
            back_dropout.err_y.map_read()
        else:
            back_dropout.cpu_run()

        logging.info("[DropoutBackward] modified err_y:")
        logging.info(back_dropout.err_y.v)
        ratio = 1.0 - np.count_nonzero(back_dropout.err_y.v) / \
            back_dropout.err_y.v.size
        logging.info("[DropoutBackward]  dropout ratio: %.4f", ratio)
        self.assertAlmostEqual(ratio, fwd_dropout.dropout_ratio,
                               delta=fwd_dropout.dropout_ratio / 10,
                               msg='error in DropoutBackward results: ratio of'
                               ' zero elements in err_y matrix is {0} '
                               '(target value is {1})'.format(ratio,
                               fwd_dropout.dropout_ratio))

    def test_cpu(self):
        logging.info("start CPU test...")
        self._run_test(TestType.CPU)

    def test_ocl(self):
        logging.info("start OpenCL test...")
        self._run_test(TestType.OCL)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logging.info("Running dropout test!")
    unittest.main()
