# -*- coding: utf-8 -*-
# !/usr/bin/python3 -O
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

import gc
import os

from veles.config import root
from veles.snapshotter import Snapshotter
from veles.tests import timeout, multi_device
import veles.znicz.samples.Lines.lines as lines
from veles.znicz.tests.functional import StandardTest


class TestLines(StandardTest):
    @classmethod
    def setUpClass(cls):
        train = os.path.join(root.common.test_dataset_root,
                             "Lines/lines_min/learn")

        valid = os.path.join(root.common.test_dataset_root,
                             "Lines/lines_min/test")

        root.lines.update({
            "loader_name": "full_batch_auto_label_file_image",
            "loss_function": "softmax",
            "decision": {"fail_iterations": 100,
                         "max_epochs": 9},
            "snapshotter": {"prefix": "lines",
                            "time_interval": 0, "interval": 9 + 1},
            "image_saver": {"out_dirs": [
                os.path.join(root.common.cache_dir, "tmp/test"),
                os.path.join(
                    root.common.cache_dir, "tmp/validation"),
                os.path.join(
                    root.common.cache_dir, "tmp/train")]},
            "loader": {"minibatch_size": 12, "force_cpu": False,
                       "normalization_type": "mean_disp",
                       "color_space": "RGB", "file_subtypes": ["jpeg"],
                       "train_paths": [train], "validation_paths": [valid]},
            "weights_plotter": {"limit": 32},
            "layers": [
                {"type": "conv_relu",
                 "->": {"n_kernels": 32, "kx": 11, "ky": 11,
                        "sliding": (4, 4), "weights_filling": "gaussian",
                        "weights_stddev": 0.001,
                        "bias_filling": "gaussian", "bias_stddev": 0.001},
                 "<-": {"learning_rate": 0.003,
                        "weights_decay": 0.0, "gradient_moment": 0.9}},
                {"type": "max_pooling",
                 "->": {"kx": 3, "ky": 3, "sliding": (2, 2)}},
                {"type": "all2all_relu",
                 "->": {"weights_filling": "uniform",
                        "bias_filling": "uniform",
                        "output_sample_shape": 32,
                        "weights_stddev": 0.05,
                        "bias_stddev": 0.05},
                 "<-": {"learning_rate": 0.001, "weights_decay": 0.0,
                        "gradient_moment": 0.9}},
                {"type": "softmax",
                 "->": {"output_sample_shape": 4, "weights_filling": "uniform",
                        "weights_stddev": 0.05, "bias_filling": "uniform",
                        "bias_stddev": 0.05},
                 "<-": {"learning_rate": 0.001, "weights_decay": 0.0,
                        "gradient_moment": 0.9}}]})

    def init_wf(self, workflow, snapshot):
        self.assertEqual(workflow.evaluator.labels,
                         workflow.loader.minibatch_labels)

        workflow.initialize(device=self.device, snapshot=snapshot)
        self.assertEqual(workflow.evaluator.labels,
                         workflow.loader.minibatch_labels)

    def check_write_error_rate(self, workflow, error):
        err = workflow.decision.epoch_n_err[1]
        self.assertEqual(err, error)
        self.assertEqual(
            workflow.decision.max_epochs, workflow.loader.epoch_number)

    @timeout(600)
    @multi_device()
    def test_lines(self):
        self.info("Will test lines workflow with one convolutional relu"
                  " layer and one fully connected relu layer")

        workflow = lines.LinesWorkflow(
            self.parent,
            decision_config=root.lines.decision,
            snapshotter_config=root.lines.snapshotter,
            image_saver_config=root.lines.image_saver,
            loader_config=root.lines.loader,
            layers=root.lines.layers,
            loader_name=root.lines.loader_name,
            loss_function=root.lines.loss_function)
        # Test workflow
        self.init_wf(workflow, False)
        workflow.run()
        self.check_write_error_rate(workflow, 54)

        file_name = workflow.snapshotter.file_name
        del workflow
        gc.collect()

        # Test loading from snapshot
        self.info("Will load workflow from %s", file_name)

        workflow_from_snapshot = Snapshotter.import_(file_name)
        workflow_from_snapshot.workflow = self.parent
        self.assertTrue(workflow_from_snapshot.decision.epoch_ended)
        workflow_from_snapshot.decision.max_epochs = 24
        workflow_from_snapshot.decision.complete <<= False

        self.init_wf(workflow_from_snapshot, True)
        workflow_from_snapshot.run()
        self.check_write_error_rate(workflow_from_snapshot, 46)

        self.info("All Ok")

if __name__ == "__main__":
    StandardTest.main()
