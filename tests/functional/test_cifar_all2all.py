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
from veles.snapshotter import Snapshotter
from veles.tests import timeout, multi_device
from veles.znicz.tests.functional import StandardTest
import veles.znicz.tests.research.CIFAR10.cifar as cifar


class TestCifarAll2All(StandardTest):
    @classmethod
    def setUpClass(cls):
        train_dir = os.path.join(root.common.test_dataset_root, "cifar/10")
        validation_dir = os.path.join(root.common.test_dataset_root,
                                      "cifar/10/test_batch")
        root.cifar.update({
            "decision": {"fail_iterations": 1000, "max_epochs": 2},
            "loader_name": "cifar_loader",
            "lr_adjuster": {"do": False},
            "loss_function": "softmax",
            "add_plotters": False,
            "image_saver": {"do": False,
                            "out_dirs":
                            [os.path.join(root.common.cache_dir, "tmp/test"),
                             os.path.join(root.common.cache_dir,
                                          "tmp/validation"),
                             os.path.join(root.common.cache_dir,
                                          "tmp/train")]},
            "loader": {"minibatch_size": 81, "force_cpu": False,
                       "normalization_type": "linear"},
            "accumulator": {"n_bars": 30},
            "weights_plotter": {"limit": 25},
            "layers": [{"type": "all2all",
                        "->": {"output_sample_shape": 486},
                        "<-": {"learning_rate": 0.0005, "weights_decay": 0.0}},
                       {"type": "activation_sincos"},
                       {"type": "all2all",
                        "->": {"output_sample_shape": 486},
                        "<-": {"learning_rate": 0.0005, "weights_decay": 0.0}},
                       {"type": "activation_sincos"},
                       {"type": "softmax",
                        "->": {"output_sample_shape": 10},
                        "<-": {"learning_rate": 0.0005,
                               "weights_decay": 0.0}}],
            "snapshotter": {"prefix": "cifar_test", "time_interval": 0,
                            "interval": 3},
            "data_paths": {"train": train_dir, "validation": validation_dir}})

    @timeout(1200)
    @multi_device()
    def test_cifar_all2all(self):
        self.info("Will test cifar fully connected workflow")
        workflow = cifar.CifarWorkflow(
            self.parent,
            decision_config=root.cifar.decision,
            snapshotter_config=root.cifar.snapshotter,
            image_saver_config=root.cifar.image_saver,
            layers=root.cifar.layers,
            loss_function=root.cifar.loss_function,
            loader_name=root.cifar.loader_name,
            loader_config=root.cifar.loader)
        self.assertEqual(workflow.evaluator.labels,
                         workflow.loader.minibatch_labels)
        workflow.initialize(
            device=self.device,
            minibatch_size=root.cifar.loader.minibatch_size,
            snapshot=False)
        self.assertEqual(workflow.evaluator.labels,
                         workflow.loader.minibatch_labels)
        workflow.run()
        file_name = workflow.snapshotter.file_name

        err = workflow.decision.epoch_n_err[1]
        self.assertEqual(err, 7373)
        self.assertEqual(2, workflow.loader.epoch_number)

        self.info("Will load workflow from %s", file_name)
        workflow_from_snapshot = Snapshotter.import_(file_name)
        self.assertTrue(workflow_from_snapshot.decision.epoch_ended)
        workflow_from_snapshot.workflow = self.parent
        workflow_from_snapshot.decision.max_epochs = 5
        workflow_from_snapshot.decision.complete <<= False
        self.assertEqual(workflow_from_snapshot.evaluator.labels,
                         workflow_from_snapshot.loader.minibatch_labels)
        workflow_from_snapshot.initialize(
            device=self.device,
            minibatch_size=root.cifar.loader.minibatch_size,
            snapshot=True)
        self.assertEqual(workflow_from_snapshot.evaluator.labels,
                         workflow_from_snapshot.loader.minibatch_labels)
        workflow_from_snapshot.run()

        err = workflow_from_snapshot.decision.epoch_n_err[1]
        self.assertEqual(err, 7046)
        self.assertEqual(5, workflow_from_snapshot.loader.epoch_number)
        self.info("All Ok")

if __name__ == "__main__":
    StandardTest.main()
