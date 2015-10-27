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

Created on May 12, 2015

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
from veles.snapshotter import SnapshotterToFile
from veles.tests import timeout, multi_device
import veles.znicz.tests.research.Stl10.stl10 as stl10
from veles.znicz.tests.functional import StandardTest


class TestSTL10(StandardTest):
    @classmethod
    def setUpClass(cls):
        root.stl.publisher.backends = {}

        root.stl.update({
            "loader_name": "full_batch_stl_10",
            "loss_function": "softmax",
            "decision": {"fail_iterations": 20, "max_epochs": 3},
            "loader": {"directory": "/data/veles/datasets/stl10_binary",
                       "minibatch_size": 50,
                       "scale": (32, 32),
                       "normalization_type": "internal_mean"},
            "downloader": {
                "url":
                "http://ai.stanford.edu/~acoates/stl10/stl10_binary.tar.gz",
                "directory": root.common.dirs.datasets,
                "files": ["stl10_binary"]},
            "weights_plotter": {"limit": 256, "split_channels": False},
            "layers": [{"name": "conv1",
                        "type": "conv",
                        "->": {"n_kernels": 32, "kx": 5, "ky": 5,
                               "padding": (2, 2, 2, 2), "sliding": (1, 1),
                               "weights_filling": "gaussian",
                               "weights_stddev": 0.0001,
                               "bias_filling": "constant", "bias_stddev": 0},
                        "<-": {"learning_rate": 0.001,
                               "learning_rate_bias": 0.002,
                               "weights_decay": 0.0005,
                               "weights_decay_bias": 0.0005,
                               "factor_ortho": 0.001, "gradient_moment": 0.9,
                               "gradient_moment_bias": 0.9},
                        },
                       {"name": "pool1", "type": "max_pooling",
                        "->": {"kx": 3, "ky": 3, "sliding": (2, 2)}},

                       {"name": "relu1", "type": "activation_str"},

                       {"name": "norm1", "type": "norm", "alpha": 0.00005,
                        "beta": 0.75, "n": 3, "k": 1},

                       {"name": "conv2", "type": "conv",
                        "->": {"n_kernels": 32, "kx": 5, "ky": 5,
                               "padding": (2, 2, 2, 2), "sliding": (1, 1),
                               "weights_filling": "gaussian",
                               "weights_stddev": 0.01,
                               "bias_filling": "constant", "bias_stddev": 0},
                        "<-": {"learning_rate": 0.001,
                               "learning_rate_bias": 0.002,
                               "weights_decay": 0.0005,
                               "weights_decay_bias": 0.0005,
                               "factor_ortho": 0.001, "gradient_moment": 0.9,
                               "gradient_moment_bias": 0.9}
                        },
                       {"name": "relu2", "type": "activation_str"},

                       {"name": "pool2", "type": "avg_pooling",
                        "->": {"kx": 3, "ky": 3, "sliding": (2, 2)}},

                       {"name": "norm2", "type": "norm",
                        "alpha": 0.00005, "beta": 0.75, "n": 3, "k": 1},

                       {"name": "conv3", "type": "conv",
                        "->": {"n_kernels": 64, "kx": 5, "ky": 5,
                               "padding": (2, 2, 2, 2), "bias_stddev": 0,
                               "sliding": (1, 1),
                               "weights_filling": "gaussian",
                               "weights_stddev": 0.01,
                               "bias_filling": "constant"},
                        "<-": {"learning_rate": 0.001,
                               "learning_rate_bias": 0.001,
                               "weights_decay": 0.0005,
                               "weights_decay_bias": 0.0005,
                               "factor_ortho": 0.001,
                               "gradient_moment": 0.9,
                               "gradient_moment_bias": 0.9},
                        },
                       {"name": "relu3", "type": "activation_str"},

                       {"name": "pool3", "type": "avg_pooling",
                        "->": {"kx": 3, "ky": 3, "sliding": (2, 2)}},

                       {"name": "a2asm4", "type": "softmax",
                        "->": {"output_sample_shape": 10,
                               "weights_filling": "gaussian",
                               "weights_stddev": 0.01,
                               "bias_filling": "constant", "bias_stddev": 0},
                        "<-": {"learning_rate": 0.001,
                               "learning_rate_bias": 0.002,
                               "weights_decay": 1.0, "weights_decay_bias": 0,
                               "gradient_moment": 0.9,
                               "gradient_moment_bias": 0.9}}]})

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

    def init_and_run(self, snapshot):
        workflow = stl10.Stl10(
            self.parent,
            loader_name=root.stl.loader_name,
            loss_function=root.stl.loss_function,
            loader_config=root.stl.loader,
            layers=root.stl.layers,
            downloader_config=root.stl.downloader,
            decision_config=root.stl.decision,
            weights_plotter_config=root.stl.weights_plotter,
            publisher_config=root.stl.publisher)
        workflow.publisher.workflow_graphs = {
            "png": bytes([2, 3, 1]), "svg": bytes([2, 3, 1])}
        self.init_wf(workflow, snapshot)
        workflow.run()
        return workflow

    @timeout(1500)
    @multi_device()
    def test_stl10_gpu(self):
        self.info("Will test convolutional stl10 workflow")

        errors = (5088, 4836, 5335)
        self.info("Will run workflow with double")
        root.common.precision_type = "double"

        # Test workflow
        workflow = self.init_and_run(False)
        self.assertIsNone(workflow.thread_pool.failure)
        self.check_write_error_rate(workflow, errors[0])

        self.parent.workflow = None
        file_name = workflow.snapshotter.destination

        # Test loading from snapshot
        self.info("Will load workflow from snapshot: %s", file_name)

        workflow_from_snapshot = SnapshotterToFile.import_(file_name)
        workflow_from_snapshot.workflow = self.parent
        self.assertTrue(workflow_from_snapshot.decision.epoch_ended)
        workflow_from_snapshot.decision.max_epochs = 6
        workflow_from_snapshot.decision.complete <<= False

        self.init_wf(workflow_from_snapshot, True)
        workflow_from_snapshot.run()
        self.check_write_error_rate(workflow_from_snapshot, errors[1])

        self.info("Will run workflow with float")
        root.common.precision_type = "float"

        # Test workflow with ocl and float
        workflow = self.init_and_run(False)
        self.assertIsNone(workflow.thread_pool.failure)
        self.check_write_error_rate(workflow, errors[2])

if __name__ == "__main__":
    StandardTest.main()
