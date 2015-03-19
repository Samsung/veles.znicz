#!/usr/bin/python3 -O
"""
Created on April 3, 2014

Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""


import os

from veles.config import root
from veles.snapshotter import Snapshotter
from veles.tests import timeout, multi_device
from veles.znicz.tests.functional import StandardTest
import veles.znicz.tests.research.Approximator.approximator as approximator


class TestApproximator(StandardTest):
    @classmethod
    def setUpClass(cls):
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

    mse = {"ocl": [0.1669484573, 0.15975260],
           "cuda": [0.1669484573, 0.15975260]}

    @timeout(240)
    @multi_device
    def test_approximator(self):
        self.info("Will test approximator workflow")

        self.w = approximator.ApproximatorWorkflow(
            self.parent, layers=root.approximator.layers)
        self.w.decision.max_epochs = 3
        self.w.snapshotter.time_interval = 0
        self.w.snapshotter.interval = 3
        self.w.initialize(
            device=self.device,
            learning_rate=root.approximator.learning_rate,
            weights_decay=root.approximator.weights_decay,
            minibatch_size=root.approximator.loader.minibatch_size,
            snapshot=False)
        self.w.run()
        file_name = self.w.snapshotter.file_name

        avg_mse = self.w.decision.epoch_metrics[2][0]
        self.assertAlmostEqual(avg_mse, self.mse[self.device.backend_name][0],
                               places=5)
        self.assertEqual(3, self.w.loader.epoch_number)

        self.info("Will load workflow from %s", file_name)
        self.wf = Snapshotter.import_(file_name)
        self.assertTrue(self.wf.decision.epoch_ended)
        self.wf.decision.max_epochs = 5
        self.wf.decision.complete <<= False
        self.wf.initialize(
            device=self.device,
            learning_rate=root.approximator.learning_rate,
            weights_decay=root.approximator.weights_decay,
            minibatch_size=root.approximator.loader.minibatch_size,
            snapshot=True)
        self.wf.run()

        avg_mse = self.wf.decision.epoch_metrics[2][0]
        self.assertAlmostEqual(avg_mse, self.mse[self.device.backend_name][1],
                               places=5)
        self.assertEqual(5, self.wf.loader.epoch_number)
        self.info("All Ok")

if __name__ == "__main__":
    StandardTest.main()
