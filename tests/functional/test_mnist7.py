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

Created on October 13, 2014

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

from veles.config import root
from veles.snapshotter import Snapshotter
from veles.tests import timeout, multi_device
from veles.znicz.tests.functional import StandardTest
import veles.znicz.tests.research.Mnist7.mnist7 as mnist7


class TestMnist7(StandardTest):
    @classmethod
    def setUpClass(cls):
        root.mnist7.update({
            "decision": {"fail_iterations": 25, "max_epochs": 2},
            "snapshotter": {"prefix": "mnist7_test", "interval": 3,
                            "time_interval": 0},
            "loader": {"minibatch_size": 60, "force_numpy": False,
                       "normalization_type": "linear"},
            "learning_rate": 0.0001,
            "weights_decay": 0.00005,
            "layers": [100, 100, 7]})

    @timeout(400)
    @multi_device()
    def test_mnist7(self):
        self.info("Will test mnist7 workflow")

        workflow = mnist7.Mnist7Workflow(
            self.parent, layers=root.mnist7.layers)
        self.assertEqual(workflow.evaluator.labels,
                         workflow.loader.minibatch_labels)
        workflow.initialize(
            device=self.device,
            learning_rate=root.mnist7.learning_rate,
            weights_decay=root.mnist7.weights_decay,
            snapshot=False)
        self.assertEqual(workflow.evaluator.labels,
                         workflow.loader.minibatch_labels)
        workflow.run()
        file_name = workflow.snapshotter.file_name

        err = workflow.decision.epoch_n_err[1]
        self.assertEqual(err, 8990)
        avg_mse = workflow.decision.epoch_metrics[1][0]
        self.assertAlmostEqual(avg_mse, 0.821236, places=5)
        self.assertEqual(2, workflow.loader.epoch_number)

        self.info("Will load workflow from %s", file_name)
        workflow_from_snapshot = Snapshotter.import_file(file_name)
        workflow_from_snapshot.workflow = self.parent
        self.assertTrue(workflow_from_snapshot.decision.epoch_ended)
        workflow_from_snapshot.decision.max_epochs = 5
        workflow_from_snapshot.decision.complete <<= False
        self.assertEqual(workflow_from_snapshot.evaluator.labels,
                         workflow_from_snapshot.loader.minibatch_labels)
        workflow_from_snapshot.initialize(
            device=self.device,
            learning_rate=root.mnist7.learning_rate,
            weights_decay=root.mnist7.weights_decay,
            snapshot=True)
        self.assertEqual(workflow_from_snapshot.evaluator.labels,
                         workflow_from_snapshot.loader.minibatch_labels)
        workflow_from_snapshot.run()

        err = workflow_from_snapshot.decision.epoch_n_err[1]
        self.assertEqual(err, 8774)
        avg_mse = workflow_from_snapshot.decision.epoch_metrics[1][0]
        self.assertAlmostEqual(avg_mse, 0.759152, places=4)
        self.assertEqual(5, workflow_from_snapshot.loader.epoch_number)
        self.info("All Ok")


if __name__ == "__main__":
    StandardTest.main()
