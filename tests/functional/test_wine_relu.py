#!/usr/bin/python3 -O
"""
Created on April 2, 2014

Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""


import logging
import numpy
import unittest

from veles.config import root
import veles.opencl as opencl
import veles.prng as rnd
from veles.tests import timeout
import veles.znicz.tests.research.WineRelu.wine_relu as wine_relu
import veles.dummy as dummy_workflow


class TestWineRelu(unittest.TestCase):
    def setUp(self):
        root.common.unit_test = True
        root.common.plotters_disabled = True
        self.device = opencl.Device()

    @timeout(300)
    def test_wine_relu(self):
        logging.info("Will test wine relu workflow")
        rnd.get().seed(numpy.fromfile("%s/veles/znicz/tests/research/seed" %
                                      root.common.veles_dir,
                                      dtype=numpy.int32, count=1024))
        root.wine_relu.update({
            "decision": {"fail_iterations": 250},
            "snapshotter": {"prefix": "wine_relu"},
            "loader": {"minibatch_size": 10},
            "learning_rate": 0.03,
            "weights_decay": 0.0,
            "layers": [10, 3]})

        self.w = wine_relu.WineReluWorkflow(dummy_workflow.DummyWorkflow(),
                                            layers=root.wine_relu.layers)

        self.assertEqual(self.w.evaluator.labels,
                         self.w.loader.minibatch_labels)
        self.w.initialize(learning_rate=root.wine_relu.learning_rate,
                          weights_decay=root.wine_relu.weights_decay,
                          device=self.device)
        self.w.run()

        epoch = self.w.decision.epoch_number
        logging.info("Converged in %d epochs", epoch)
        self.assertEqual(epoch, 161)
        logging.info("All Ok")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
