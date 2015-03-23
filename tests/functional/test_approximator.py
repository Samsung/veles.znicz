#!/usr/bin/python3 -O
"""
Created on April 3, 2014

Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""


import os

from veles.znicz.tests.functional import StandardTest
from veles.config import root
from veles.snapshotter import Snapshotter
from veles.tests import timeout, multi_device
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
    @multi_device()
    def test_approximator(self):
        self.info("Will test approximator workflow")

        workflow = approximator.ApproximatorWorkflow(
            self.parent, layers=root.approximator.layers)
        workflow.decision.max_epochs = 3
        workflow.snapshotter.time_interval = 0
        workflow.snapshotter.interval = 3
        workflow.initialize(
            device=self.device,
            learning_rate=root.approximator.learning_rate,
            weights_decay=root.approximator.weights_decay,
            minibatch_size=root.approximator.loader.minibatch_size,
            snapshot=False)
        workflow.run()
        file_name = workflow.snapshotter.file_name

        avg_mse = workflow.decision.epoch_metrics[2][0]
        self.assertAlmostEqual(avg_mse, self.mse[self.device.backend_name][0],
                               places=5)
        self.assertEqual(3, workflow.loader.epoch_number)

        self.info("Will load workflow from %s", file_name)
        workflow_from_snapshot = Snapshotter.import_(file_name)
        self.assertTrue(workflow_from_snapshot.decision.epoch_ended)
        workflow_from_snapshot.decision.max_epochs = 5
        workflow_from_snapshot.decision.complete <<= False
        workflow_from_snapshot.initialize(
            device=self.device,
            learning_rate=root.approximator.learning_rate,
            weights_decay=root.approximator.weights_decay,
            minibatch_size=root.approximator.loader.minibatch_size,
            snapshot=True)
        workflow_from_snapshot.run()

        avg_mse = workflow_from_snapshot.decision.epoch_metrics[2][0]
        self.assertAlmostEqual(avg_mse, self.mse[self.device.backend_name][1],
                               places=5)
        self.assertEqual(5, workflow_from_snapshot.loader.epoch_number)
        self.info("All Ok")

if __name__ == "__main__":
    StandardTest.main()
