#!/usr/bin/env python3
# -*-coding: utf-8 -*-
"""
.. invisible:
     _   _ _____ _     _____ _____
    | | | |  ___| |   |  ___/  ___|
    | | | | |__ | |   | |__ \ `--.
    | | | |  __|| |   |  __| `--. \
    \ \_/ / |___| |___| |___/\__/ /
     \___/\____/\_____|____/\____/

Created on April 2, 2014

███████████████████████████████████████████████████████████████████████████████

Licensed to the Apache Software Foundation (ASF) under one
or more contributor license agreements.  See the NOTICE file
distributed with this work for additional information
regarding copyright ownership.  The ASF licenses this file
to you under the Apache License, Version 2.0 (the
"License"); you may not use this file except in compliance
with the License.  You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing,
software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
KIND, either express or implied.  See the License for the
specific language governing permissions and limitations
under the License.

███████████████████████████████████████████████████████████████████████████████
"""


import os

from veles.config import root
from veles.snapshotter import SnapshotterToFile
from veles.tests import timeout, multi_device
from veles.znicz.tests.functional import StandardTest
import veles.znicz.tests.research.Hands.hands as hands


class TestHands(StandardTest):
    @classmethod
    def setUpClass(cls):
        train_dir = [
            os.path.join(root.common.dirs.datasets, "hands/Training")]
        validation_dir = [
            os.path.join(root.common.dirs.datasets, "hands/Testing")]

        root.hands.update({
            "decision": {"fail_iterations": 100, "max_epochs": 2},
            "downloader": {
                "url":
                "https://s3-eu-west-1.amazonaws.com/veles.forge/Hands/"
                "hands.tar",
                "directory": root.common.dirs.datasets,
                "files": ["hands"]},
            "loss_function": "softmax",
            "loader_name": "hands_loader",
            "snapshotter": {"prefix": "hands", "interval": 2,
                            "time_interval": 0},
            "loader": {"minibatch_size": 40, "train_paths": train_dir,
                       "force_numpy": False, "color_space": "GRAY",
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
            downloader_config=root.hands.downloader,
            loss_function=root.hands.loss_function,
            loader_name=root.hands.loader_name)
        self.assertEqual(workflow.evaluator.labels,
                         workflow.loader.minibatch_labels)
        workflow.initialize(device=self.device)
        self.assertEqual(workflow.evaluator.labels,
                         workflow.loader.minibatch_labels)
        workflow.run()
        self.assertIsNone(workflow.thread_pool.failure)
        file_name = workflow.snapshotter.destination

        err = workflow.decision.epoch_n_err[1]
        self.assertEqual(err, 577)
        self.assertEqual(2, workflow.loader.epoch_number)

        self.info("Will load workflow from %s", file_name)
        workflow_from_snapshot = SnapshotterToFile.import_(file_name)
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
