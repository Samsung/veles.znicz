#!/usr/bin/python3 -O
"""
Created on April 3, 2014

Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""


import logging
import numpy
import os
import unittest

from veles.config import root
import veles.backends as opencl
import veles.prng as prng
from veles.snapshotter import Snapshotter
from veles.tests import timeout
import veles.znicz.tests.research.Approximator.approximator as approximator
import veles.dummy as dummy_workflow


class TestApproximator(unittest.TestCase):
    def setUp(self):
        self.device = opencl.Device()

    def tearDown(self):
        del self.device

    @timeout(120)
    def test_approximator(self):
        logging.info("Will test approximator workflow")

        prng.get().seed(numpy.fromfile("%s/veles/znicz/tests/research/seed" %
                                       root.common.veles_dir,
                                       dtype=numpy.uint32, count=1024))
        root.common.update({
            "disable_plotting": True,
            "precision_level": 1,
            "precision_type": "double",
            "engine": {"backend": "ocl"}})

        target_dir = [os.path.join(root.common.test_dataset_root,
                                   "approximator/all_org_apertures.mat")]
        train_dir = [os.path.join(root.common.test_dataset_root,
                                  "approximator/all_dec_apertures.mat")]

        root.approximator.update({
            "decision": {"fail_iterations": 1000},
            "snapshotter": {"prefix": "approximator_test"},
            "loader": {"minibatch_size": 100, "train_paths": train_dir,
                       "target_paths": target_dir,
                       "normalization_type": "mean_disp",
                       "target_normalization_type": "mean_disp"},
            "learning_rate": 0.0001,
            "weights_decay": 0.00005,
            "layers": [810, 9]})

        self.w = approximator.ApproximatorWorkflow(
            dummy_workflow.DummyLauncher(),
            layers=root.approximator.layers)
        self.w.decision.max_epochs = 3
        self.w.snapshotter.time_interval = 0
        self.w.snapshotter.interval = 3
        self.w.initialize(
            device=self.device,
            learning_rate=root.approximator.learning_rate,
            weights_decay=root.approximator.weights_decay,
            minibatch_size=root.approximator.loader.minibatch_size)
        self.w.run()
        file_name = self.w.snapshotter.file_name

        avg_mse = self.w.decision.epoch_metrics[2][0]
        self.assertAlmostEqual(avg_mse, 0.1669484573, places=5)
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
        self.assertAlmostEqual(avg_mse, 0.15975260, places=5)
        self.assertEqual(5, self.wf.loader.epoch_number)
        logging.info("All Ok")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
