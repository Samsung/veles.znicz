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

Created on October 15, 2014

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
from veles.znicz.tests.functional import StandardTest
import veles.znicz.samples.MNIST.mnist as mnist_conv


class TestMnistConv(StandardTest):
    @classmethod
    def setUpClass(cls):
        root.mnistr.lr_adjuster.lr_parameters = {
            "base_lr": 0.01, "gamma": 0.0001, "pow_ratio": 0.75}
        root.mnistr.lr_adjuster.bias_lr_parameters = {
            "base_lr": 0.01, "gamma": 0.0001, "pow_ratio": 0.75}

        root.mnistr.update({
            "loss_function": "softmax",
            "loader_name": "mnist_loader",
            "lr_adjuster": {"do": True, "lr_policy_name": "inv",
                            "bias_lr_policy_name": "inv"},
            "decision": {"max_epochs": 2,
                         "fail_iterations": 100},
            "snapshotter": {"prefix": "test_mnist_conv", "time_interval": 0,
                            "interval": 2},
            "weights_plotter": {"limit": 64},
            "loader": {"minibatch_size": 6, "force_numpy": False,
                       "normalization_type": "linear"},
            "layers": [{"type": "conv",
                        "->": {"n_kernels": 64, "kx": 5, "ky": 5,
                               "sliding": (1, 1), "weights_filling": "uniform",
                               "weights_stddev": 0.0944569801138958,
                               "bias_filling": "constant",
                               "bias_stddev": 0.048000},
                        "<-": {"learning_rate": 0.03,
                               "learning_rate_bias": 0.358000,
                               "gradient_moment": 0.385000,
                               "gradient_moment_bias": 0.385000,
                               "weights_decay": 0.0005,
                               "weights_decay_bias": 0.1980997902551238,
                               "factor_ortho": 0.001}},

                       {"type": "max_pooling",
                        "->": {"kx": 2, "ky": 2, "sliding": (2, 2)}},

                       {"type": "conv",
                        "->": {"n_kernels": 87, "kx": 5, "ky": 5,
                               "sliding": (1, 1),
                               "weights_filling": "uniform",
                               "weights_stddev": 0.067000,
                               "bias_filling": "constant",
                               "bias_stddev": 0.444000},
                        "<-": {"learning_rate": 0.03,
                               "learning_rate_bias": 0.381000,
                               "gradient_moment": 0.741000,
                               "gradient_moment_bias": 0.741000,
                               "weights_decay": 0.0005, "factor_ortho": 0.001,
                               "weights_decay_bias": 0.039000}},

                       {"type": "max_pooling",
                        "->": {"kx": 2, "ky": 2, "sliding": (2, 2)}},

                       {"type": "all2all_relu",
                        "->": {"output_sample_shape": 791,
                               "weights_stddev": 0.039000,
                               "bias_filling": "constant",
                               "weights_filling": "uniform",
                               "bias_stddev": 1.000000},

                        "<-": {"learning_rate": 0.03,
                               "learning_rate_bias": 0.196000,
                               "gradient_moment": 0.619000,
                               "gradient_moment_bias": 0.619000,
                               "weights_decay": 0.0005, "factor_ortho": 0.001,
                               "weights_decay_bias": 0.11487830567238211}},

                       {"type": "softmax",
                        "->": {"output_sample_shape": 10,
                               "weights_filling": "uniform",
                               "weights_stddev": 0.024000,
                               "bias_filling": "constant",
                               "bias_stddev": 0.255000},
                        "<-": {"learning_rate": 0.03,
                               "learning_rate_bias": 0.488000,
                               "gradient_moment": 0.8422143625658985,
                               "gradient_moment_bias": 0.8422143625658985,
                               "weights_decay": 0.0005,
                               "weights_decay_bias": 0.476000}}]})

    @timeout(1200)
    @multi_device()
    def test_mnist_conv(self):
        self.info("Will test mnist workflow with convolutional"
                  " (genetic generate) config")

        workflow = mnist_conv.MnistWorkflow(
            self.parent,
            decision_config=root.mnistr.decision,
            snapshotter_config=root.mnistr.snapshotter,
            loader_name=root.mnistr.loader_name,
            loader_config=root.mnistr.loader,
            layers=root.mnistr.layers,
            loss_function=root.mnistr.loss_function,
            lr_adjuster_config=root.mnistr.lr_adjuster)
        self.assertEqual(workflow.evaluator.labels,
                         workflow.loader.minibatch_labels)
        workflow.initialize(device=self.device, snapshot=False)
        self.assertEqual(workflow.evaluator.labels,
                         workflow.loader.minibatch_labels)
        workflow.run()
        self.assertIsNone(workflow.thread_pool.failure)
        file_name = workflow.snapshotter.destination

        err = workflow.decision.epoch_n_err[1]
        self.assertEqual(err, 125)
        self.assertEqual(2, workflow.loader.epoch_number)

        self.info("Will load workflow from %s", file_name)
        workflow_from_snapshot = SnapshotterToFile.import_(file_name)
        workflow_from_snapshot.workflow = self.parent
        self.assertTrue(workflow_from_snapshot.decision.epoch_ended)
        workflow_from_snapshot.decision.max_epochs = 3
        workflow_from_snapshot.decision.complete <<= False
        self.assertEqual(workflow_from_snapshot.evaluator.labels,
                         workflow_from_snapshot.loader.minibatch_labels)
        workflow_from_snapshot.initialize(device=self.device, snapshot=True)
        self.assertEqual(workflow_from_snapshot.evaluator.labels,
                         workflow_from_snapshot.loader.minibatch_labels)
        workflow_from_snapshot.run()
        self.assertIsNone(workflow_from_snapshot.thread_pool.failure)

        err = workflow_from_snapshot.decision.epoch_n_err[1]
        self.assertEqual(err, 105)
        self.assertEqual(3, workflow_from_snapshot.loader.epoch_number)
        self.info("All Ok")

if __name__ == "__main__":
    StandardTest.main()
