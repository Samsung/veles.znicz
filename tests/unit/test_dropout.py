#!/usr/bin/python3 -O
# encoding: utf-8

"""
Created on April 25, 2014

A unit test for dropout layer.
"""

import numpy as np

from veles.dummy import DummyWorkflow
from veles.memory import Vector
from veles.tests import AcceleratedTest, assign_backend
from veles.znicz.dropout import DropoutForward, DropoutBackward


class TestType(object):
    CPU = 0
    GPU = 1

    def __init__(self):
        """
        Needed by PEP8.
        """
        pass


class TestDropout(AcceleratedTest):
    ABSTRACT = True

    def _run_test(self, test_type):
        workflow = DummyWorkflow()
        fwd_dropout = DropoutForward(workflow, dropout_ratio=0.4)
        fwd_dropout.input = Vector()
        sz = 100
        in_matrix = np.zeros(shape=(1, 1, sz, sz), dtype=np.float64)

        for i in range(sz):
            in_matrix[0, 0, i, :] = np.linspace(0, sz * 10, sz) * (i + 1)
        fwd_dropout.input.mem = in_matrix
        self.debug("[DropoutForward] input matrix:\n%s", in_matrix)

        fwd_dropout.initialize(self.device)
        if test_type == TestType.GPU:
            fwd_dropout.run()
            fwd_dropout.output.map_read()
        else:
            fwd_dropout.cpu_run()

        self.debug("[DropoutForward] output matrix:\n%s",
                   fwd_dropout.output.mem)
        ratio = 1.0 - float(np.count_nonzero(fwd_dropout.output.mem)) / \
            fwd_dropout.output.mem.size
        self.debug("[DropoutForward] dropout ratio: %.4f", ratio)
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
        self.debug("[DropoutBackward] err_y matrix:\n%s", err_output)

        back_dropout.initialize(self.device)
        if test_type == TestType.GPU:
            back_dropout.run()
            back_dropout.err_input.map_read()
        else:
            back_dropout.cpu_run()

        self.debug("[DropoutBackward] err_input:")
        self.debug(back_dropout.err_input.mem)
        ratio = 1.0 - float(np.count_nonzero(back_dropout.err_input.mem)) / \
            back_dropout.err_input.mem.size
        self.debug("[DropoutBackward]  dropout ratio: %.4f", ratio)
        self.assertAlmostEqual(ratio, fwd_dropout.dropout_ratio,
                               delta=fwd_dropout.dropout_ratio / 10,
                               msg='error in DropoutBackward results: ratio of'
                               ' zero elements in err_input matrix is {0} '
                               '(target value is {1})'.format(
                                   ratio, fwd_dropout.dropout_ratio))

    def test_cpu(self):
        self.info("start CPU test...")
        self._run_test(TestType.CPU)
        self.info("TEST PASSED")

    def test_ocl(self):
        self.info("start GPU test...")
        self._run_test(TestType.GPU)
        self.info("TEST PASSED")


@assign_backend("ocl")
class OpenCLTestDropout(TestDropout):
    pass


@assign_backend("cuda")
class CUDATestDropout(TestDropout):
    pass


if __name__ == "__main__":
    AcceleratedTest.main()
