#!/usr/bin/python3 -O
"""
Created on April 2, 2014

Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""


import os

from veles.config import root
from veles.snapshotter import Snapshotter
from veles.tests import timeout, multi_device
from veles.znicz.tests.functional import StandardTest
import veles.znicz.tests.research.Hands.hands as hands


class TestHands(StandardTest):
    @classmethod
    def setUpClass(cls):
        train_dir = [
            os.path.join(root.common.test_dataset_root, "hands/Training")]
        validation_dir = [
            os.path.join(root.common.test_dataset_root, "hands/Testing")]

        root.hands.update({
            "decision": {"fail_iterations": 100, "max_epochs": 2},
            "loss_function": "softmax",
            "loader_name": "hands_loader",
            "snapshotter": {"prefix": "hands", "interval": 2,
                            "time_interval": 0},
            "loader": {"minibatch_size": 40, "train_paths": train_dir,
                       "force_cpu": False, "color_space": "GRAY",
                       "background_color": (0,),
                       "normalization_type": "linear",
                       "validation_paths": validation_dir},
            "layers": [{"type": "all2all_tanh",
                        "->": {"output_sample_shape": 30},
                        "<-": {"learning_rate": 0.008, "weights_decay": 0.0}},
                       {"type": "softmax",
                        "<-": {"learning_rate": 0.008,
                               "weights_decay": 0.0}}]})

    @timeout(500)
    @multi_device()
    def test_hands(self):
        self.info("Will test hands workflow")

        workflow = hands.HandsWorkflow(
            self.parent,
            layers=root.hands.layers,
            decision_config=root.hands.decision,
            snapshotter_config=root.hands.snapshotter,
            loader_config=root.hands.loader,
            loss_function=root.hands.loss_function,
            loader_name=root.hands.loader_name)
        self.assertEqual(workflow.evaluator.labels,
                         workflow.loader.minibatch_labels)
        workflow.initialize(device=self.device, snapshot=False)
        self.assertEqual(workflow.evaluator.labels,
                         workflow.loader.minibatch_labels)
        workflow.run()
        self.assertIsNone(workflow.thread_pool.failure)
        file_name = workflow.snapshotter.file_name

        err = workflow.decision.epoch_n_err[1]
        self.assertEqual(err, 577)
        self.assertEqual(2, workflow.loader.epoch_number)

        self.info("Will load workflow from %s", file_name)
        workflow_from_snapshot = Snapshotter.import_(file_name)
        workflow_from_snapshot.workflow = self.parent
        self.assertTrue(workflow_from_snapshot.decision.epoch_ended)
        workflow_from_snapshot.decision.max_epochs = 9
        workflow_from_snapshot.decision.complete <<= False
        self.assertEqual(workflow_from_snapshot.evaluator.labels,
                         workflow_from_snapshot.loader.minibatch_labels)
        workflow_from_snapshot.initialize(device=self.device, snapshot=True)
        self.assertEqual(workflow_from_snapshot.evaluator.labels,
                         workflow_from_snapshot.loader.minibatch_labels)
        workflow_from_snapshot.run()
        self.assertIsNone(workflow_from_snapshot.thread_pool.failure)

        err = workflow_from_snapshot.decision.epoch_n_err[1]
        self.assertEqual(err, 593)
        self.assertEqual(9, workflow_from_snapshot.loader.epoch_number)
        self.info("All Ok")

if __name__ == "__main__":
    StandardTest.main()
