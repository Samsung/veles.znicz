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

        self.w = hands.HandsWorkflow(
            self.parent,
            layers=root.hands.layers,
            decision_config=root.hands.decision,
            snapshotter_config=root.hands.snapshotter,
            loader_config=root.hands.loader,
            loss_function=root.hands.loss_function,
            loader_name=root.hands.loader_name)
        self.assertEqual(self.w.evaluator.labels,
                         self.w.loader.minibatch_labels)
        self.w.initialize(device=self.device, snapshot=False)
        self.assertEqual(self.w.evaluator.labels,
                         self.w.loader.minibatch_labels)
        self.w.run()
        file_name = self.w.snapshotter.file_name

        err = self.w.decision.epoch_n_err[1]
        self.assertEqual(err, 577)
        self.assertEqual(2, self.w.loader.epoch_number)

        self.info("Will load workflow from %s", file_name)
        self.wf = Snapshotter.import_(file_name)
        self.assertTrue(self.wf.decision.epoch_ended)
        self.wf.decision.max_epochs = 9
        self.wf.decision.complete <<= False
        self.assertEqual(self.wf.evaluator.labels,
                         self.wf.loader.minibatch_labels)
        self.wf.initialize(device=self.device, snapshot=True)
        self.assertEqual(self.wf.evaluator.labels,
                         self.wf.loader.minibatch_labels)
        self.wf.run()

        err = self.wf.decision.epoch_n_err[1]
        self.assertEqual(err, 593)
        self.assertEqual(9, self.wf.loader.epoch_number)
        self.info("All Ok")

if __name__ == "__main__":
    StandardTest.main()
