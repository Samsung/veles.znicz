#!/usr/bin/python3 -O
"""
Created on November 6, 2014

Copyright (c) 2014 Samsung Electronics Co., Ltd.
"""


import logging
import numpy
import os
import scipy.io
import unittest

from veles.dummy import DummyLauncher
from veles.znicz.samples.mnist_rbm import MnistRBMWorkflow


class TestRBMworkflow(unittest.TestCase):
    """Test RBM workflow for MNIST.
    """
    def setUp(self):
        pass

    def test_rbm(self):
        """This function creates RBM workflow for MNIST task
        and compares result with the output produced function RBM
        from  MATLAB (http://deeplearning.net/tutorial/rbm.html (25.11.14))
        Raises:
            AssertLess: if unit output is wrong.
        """
        device = None
        logging.basicConfig(level=logging.DEBUG)
        logging.info("MNIST RBM TEST")
        workflow = MnistRBMWorkflow(DummyLauncher(), 0)
        workflow.initialize(device=device, learning_rate=0,
                            weights_decay=0)
        test_data_path = os.path.abspath(os.path.join(
            os.path.dirname(__file__), '..', 'unit',
            'data', 'rbm', 'test_rbm.mat'))
        test_data = scipy.io.loadmat(test_data_path)
        workflow.fwds[0].weights.map_write()
        workflow.fwds[0].bias.map_write()
        workflow.fwds[0].vbias.map_write()
        workflow.fwds[0].weights.mem[:] = numpy.transpose(test_data["W"])[:]
        workflow.fwds[0].vbias.mem[:] = numpy.transpose(test_data["vbias"])[:]
        workflow.fwds[0].bias.mem = numpy.transpose(test_data["hbias"])[:]
        workflow.run()
        diff = numpy.sum(numpy.abs(test_data["errors"] -
                         workflow.evaluator.reconstruction_error))
        self.assertLess(diff, 1e-12, " total error  is %0.17f" % diff)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logging.info("Running RBM tests")
    unittest.main()
