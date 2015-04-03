#!/usr/bin/env python3
# -*-coding: utf-8 -*-
"""
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
from veles.tests import timeout, multi_device
import veles.znicz.samples.Wine.wine as wine
from veles.znicz.tests.functional import StandardTest


class TestWine(StandardTest):
    @classmethod
    def setUpClass(cls):
        # We must test how snapshotting works, at least one-way
        root.wine.update({
            "decision": {"fail_iterations": 200,
                         "snapshot_prefix": "wine"},
            "snapshotter": {"interval": 13, "time_interval": 0},
            "loader": {"minibatch_size": 10,
                       "force_cpu": False},
            "learning_rate": 0.3,
            "weights_decay": 0.0,
            "layers": [8, 3],
            "data_paths": os.path.join(root.common.veles_dir,
                                       "veles/znicz/samples/wine/wine.data")})

    @timeout(300)
    @multi_device(True)
    def test_wine(self):
        self.info("Will test wine workflow")

        workflow = wine.WineWorkflow(self.parent, layers=root.wine.layers)

        self.assertEqual(workflow.evaluator.labels,
                         workflow.loader.minibatch_labels)
        workflow.initialize(
            learning_rate=root.wine.learning_rate,
            weights_decay=root.wine.weights_decay,
            device=self.device, snapshot=False)
        workflow.run()
        self.assertIsNone(workflow.thread_pool.failure)

        epoch = workflow.decision.epoch_number
        self.info("Converged in %d epochs", epoch)
        self.assertEqual(epoch, 12)
        self.info("All Ok")


if __name__ == "__main__":
    StandardTest.main()
