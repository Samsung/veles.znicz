#!/usr/bin/python3 -O
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


from veles.config import root
from veles.snapshotter import Snapshotter
from veles.tests import timeout, multi_device
from veles.znicz.tests.functional import StandardTest
import veles.znicz.tests.research.MNIST.mnist as mnist_relu


class TestMnistRelu(StandardTest):
    @classmethod
    def setUpClass(cls):
        root.mnistr.update({
            "loss_function": "softmax",
            "loader_name": "mnist_loader",
            "lr_adjuster": {"do": False},
            "all2all": {"weights_stddev": 0.05},
            "decision": {"fail_iterations": (0)},
            "snapshotter": {"prefix": "mnist_relu_test"},
            "loader": {"minibatch_size": 60, "normalization_type": "linear"},
            "layers": [{"type": "all2all_relu",
                        "->": {"output_sample_shape": 100,
                               "weights_filling": "uniform",
                               "weights_stddev": 0.05,
                               "bias_filling": "uniform", "bias_stddev": 0.05},
                        "<-": {"learning_rate": 0.03, "weights_decay": 0.0,
                               "learning_rate_bias": 0.03,
                               "weights_decay_bias": 0.0,
                               "gradient_moment": 0.0,
                               "gradient_moment_bias": 0.0,
                               "factor_ortho": 0.001}},
                       {"type": "softmax",
                        "->": {"output_sample_shape": 10,
                               "weights_filling": "uniform",
                               "weights_stddev": 0.05,
                               "bias_filling": "uniform", "bias_stddev": 0.05},
                        "<-": {"learning_rate": 0.03,
                               "learning_rate_bias": 0.03,
                               "weights_decay": 0.0,
                               "weights_decay_bias": 0.0,
                               "gradient_moment": 0.0,
                               "gradient_moment_bias": 0.0}}]})

    @timeout(300)
    @multi_device()
    def test_mnist_relu(self):
        self.info("Will test mnist workflow with relu config")

        workflow = mnist_relu.MnistWorkflow(
            self.parent,
            decision_config=root.mnistr.decision,
            snapshotter_config=root.mnistr.snapshotter,
            loader_name=root.mnistr.loader_name,
            loader_config=root.mnistr.loader,
            layers=root.mnistr.layers,
            loss_function=root.mnistr.loss_function)
        workflow.decision.max_epochs = 2
        workflow.snapshotter.time_interval = 0
        workflow.snapshotter.interval = 2 + 1
        self.assertEqual(workflow.evaluator.labels,
                         workflow.loader.minibatch_labels)
        workflow.initialize(device=self.device, snapshot=False)
        self.assertEqual(workflow.evaluator.labels,
                         workflow.loader.minibatch_labels)
        workflow.run()
        self.assertIsNone(workflow.thread_pool.failure)
        file_name = workflow.snapshotter.file_name

        err = workflow.decision.epoch_n_err[1]
        self.assertEqual(err, 840)
        self.assertEqual(2, workflow.loader.epoch_number)

        self.info("Will load workflow from %s", file_name)
        workflow_from_snapshot = Snapshotter.import_(file_name)
        workflow_from_snapshot.workflow = self.parent
        self.assertTrue(workflow_from_snapshot.decision.epoch_ended)
        workflow_from_snapshot.decision.max_epochs = 5
        workflow_from_snapshot.decision.complete <<= False
        self.assertEqual(workflow_from_snapshot.evaluator.labels,
                         workflow_from_snapshot.loader.minibatch_labels)
        workflow_from_snapshot.initialize(device=self.device, snapshot=True)
        self.assertEqual(workflow_from_snapshot.evaluator.labels,
                         workflow_from_snapshot.loader.minibatch_labels)
        workflow_from_snapshot.run()
        self.assertIsNone(workflow_from_snapshot.thread_pool.failure)

        err = workflow_from_snapshot.decision.epoch_n_err[1]
        self.assertEqual(err, 566)
        self.assertEqual(5, workflow_from_snapshot.loader.epoch_number)
        self.info("All Ok")

if __name__ == "__main__":
    StandardTest.main()
