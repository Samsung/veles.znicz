#!/usr/bin/python3 -O
"""
Created on April 3, 2014

Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""


import logging
import numpy
import unittest

from veles.config import root
import veles.opencl as opencl
import veles.prng as prng
from veles.snapshotter import Snapshotter
from veles.tests import timeout
import veles.znicz.tests.research.approximator as approximator
import veles.tests.dummy_workflow as dummy_workflow


class TestApproximator(unittest.TestCase):
    def setUp(self):
        root.common.unit_test = True
        root.common.plotters_disabled = True
        self.device = opencl.Device()

    def tearDown(self):
        del self.device

    @timeout(120)
    def test_approximator(self):
        logging.info("Will test approximator workflow")

        prng.get().seed(numpy.fromfile("%s/veles/znicz/tests/research/seed" %
                                       root.common.veles_dir,
                                       dtype=numpy.uint32, count=1024))
        root.approximator.update({
            "decision": {"fail_iterations": 1000,
                         "store_samples_mse": True},
            "snapshotter": {"prefix": "approximator_test"},
            "loader": {"minibatch_size": 100},
            "learning_rate": 0.0001,
            "weights_decay": 0.00005,
            "layers": [810, 9]})

        self.w = approximator.ApproximatorWorkflow(
            dummy_workflow.DummyWorkflow(),
            layers=root.approximator.layers, device=self.device)
        self.w.decision.max_epochs = 3
        self.w.snapshotter.interval = 3
        self.w.initialize(
            device=self.device,
            learning_rate=root.approximator.learning_rate,
            weights_decay=root.approximator.weights_decay,
            minibatch_size=root.approximator.loader.minibatch_size)
        self.w.run()
        file_name = self.w.snapshotter.file_name

        avg_mse = self.w.decision.epoch_metrics[2][0]
        self.assertAlmostEqual(avg_mse, 0.067443, places=5)
        self.assertEqual(3, self.w.loader.epoch_number)

        logging.info("Will load workflow from %s" % file_name)
        self.wf = Snapshotter.import_(file_name)
        self.assertTrue(self.wf.decision.epoch_ended)
        self.wf.decision.max_epochs = 5
        self.wf.decision.complete <<= False
        self.wf.initialize(
            device=self.device,
            learning_rate=root.approximator.learning_rate,
            weights_decay=root.approximator.weights_decay,
            minibatch_size=root.approximator.loader.minibatch_size)
        self.wf.run()

        avg_mse = self.wf.decision.epoch_metrics[2][0]
        self.assertAlmostEqual(avg_mse, 0.067241, places=5)
        self.assertEqual(5, self.wf.loader.epoch_number)
        logging.info("All Ok")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
