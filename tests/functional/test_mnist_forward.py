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
import sys
import unittest

from veles.config import root
from veles.genetics import Range, fix_config
from veles.tests import timeout
from veles.znicz.tests.functional import StandardTest
import veles.znicz.samples.MNIST.mnist as mnist_all2all
from veles.znicz.samples.MNIST.mnist_forward import create_forward

import veles

root.common.engine.backend = "cuda"


class TestMnistAll2All(StandardTest):
    @classmethod
    def setUpClass(cls):
        root.mnistr.update({
            "loss_function": "softmax",
            "loader_name": "mnist_loader",
            "lr_adjuster": {"do": False},
            "decision": {"fail_iterations": 100,
                         "max_epochs": 3},
            "snapshotter": {"prefix": "mnist_all2all_test", "interval": 3},
            "weights_plotter": {"limit": 0},
            "loader": {"minibatch_size": Range(60, 1, 1000),
                       "force_numpy": False,
                       "normalization_type": "linear",
                       "data_path":
                       os.path.join(root.common.dirs.datasets, "MNIST")},
            "layers": [{"type": "all2all_tanh",
                        "->": {"output_sample_shape": Range(100, 10, 500),
                               "weights_filling": "uniform",
                               "weights_stddev": Range(0.05, 0.0001, 0.1),
                               "bias_filling": "uniform",
                               "bias_stddev": Range(0.05, 0.0001, 0.1)},
                        "<-": {"learning_rate": Range(0.03, 0.0001, 0.9),
                               "weights_decay": Range(0.0, 0.0, 0.9),
                               "learning_rate_bias": Range(0.03, 0.0001, 0.9),
                               "weights_decay_bias": Range(0.0, 0.0, 0.9),
                               "gradient_moment": Range(0.0, 0.0, 0.95),
                               "gradient_moment_bias": Range(0.0, 0.0, 0.95),
                               "factor_ortho": Range(0.001, 0.0, 0.1)}},
                       {"type": "softmax",
                        "->": {"output_sample_shape": 10,
                               "weights_filling": "uniform",
                               "weights_stddev": Range(0.05, 0.0001, 0.1),
                               "bias_filling": "uniform",
                               "bias_stddev": Range(0.05, 0.0001, 0.1)},
                        "<-": {"learning_rate": Range(0.03, 0.0001, 0.9),
                               "learning_rate_bias": Range(0.03, 0.0001, 0.9),
                               "weights_decay": Range(0.0, 0.0, 0.95),
                               "weights_decay_bias": Range(0.0, 0.0, 0.95),
                               "gradient_moment": Range(0.0, 0.0, 0.95),
                               "gradient_moment_bias": Range(0.0, .0, 0.95)}}]}
        )
        fix_config(root)

    @timeout(300)
    def test_forward_propagation(self):
        workflow = mnist_all2all.MnistWorkflow(
            self.parent,
            decision_config=root.mnistr.decision,
            snapshotter_config=root.mnistr.snapshotter,
            loader_name=root.mnistr.loader_name,
            loader_config=root.mnistr.loader,
            layers=root.mnistr.layers,
            loss_function=root.mnistr.loss_function)
        workflow.snapshotter.time_interval = 0
        workflow.snapshotter.interval = 3
        workflow.initialize(device=self.device)
        workflow.run()
        self.assertIsNone(workflow.thread_pool.failure)
        file_name = workflow.snapshotter.destination

        kwargs = {"dry_run": "init", "snapshot": file_name, "stealth": True}
        path_to_model = "veles/znicz/samples/MNIST/mnist.py"

        data_path = os.path.join(root.common.dirs.datasets, "mnist_test")

        try:
            launcher = veles(path_to_model, **kwargs)  # pylint: disable=E1102
        except TypeError:
            print("Failed to import veles package for %s"
                  % str(sys.version_info))
            return

        normalizer = launcher.workflow.loader.normalizer
        labels_mapping = launcher.workflow.loader.labels_mapping

        launcher.testing = True
        loader_config = {
            "minibatch_size": 10,
            "scale": (28, 28),
            "background_color": (0,),
            "color_space": "GRAY",
            "normalization_type": "linear",
            "base_directory": os.path.join(data_path, "pictures"),
            "path_to_test_text_file":
            [os.path.join(data_path, "mnist_test.txt")]}

        create_forward(
            launcher.workflow, normalizer, labels_mapping, loader_config)
        launcher.boot()

        results = launcher.workflow.gather_results()

        targets = {
            '2_2.png': 2,
            '9_3.png': 3,
            '9_1.png': 9,
            '1_2.png': 1,
            '5_2.png': 5,
            '3_1.png': 3,
            '1_3.png': 1,
            '1_1.png': 1,
            '6_1.png': 2,
            '8_1.png': 8,
            '5_1.png': 6,
            '9_2.png': 9,
            '2_1.png': 2,
            '0_2.png': 0,
            '4_3.png': 4,
            '7_1.png': 7,
            '7_2.png': 3,
            '4_1.png': 4,
            '0_1.png': 0,
            '4_2.png': 4}

        for key, value in targets.items():
            path_to_image = os.path.join(loader_config["base_directory"], key)
            assert results["Output"][path_to_image] == value

        self.info("All Ok")

    @staticmethod
    def main():
        unittest.main()

if __name__ == "__main__":
    TestMnistAll2All.main()
