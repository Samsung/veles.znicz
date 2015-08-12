# -*- coding: utf-8 -*-
# !/usr/bin/python3 -O
"""
.. invisible:
     _   _ _____ _     _____ _____
    | | | |  ___| |   |  ___/  ___|
    | | | | |__ | |   | |__ \ `--.
    | | | |  __|| |   |  __| `--. \
    \ \_/ / |___| |___| |___/\__/ /
     \___/\____/\_____|____/\____/

Created on September 26, 2014

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
from veles.tests import timeout
import veles.znicz.samples.DemoKohonen.kohonen as kohonen
from veles.znicz.tests.functional import StandardTest

# FIXME(v.markovtsev): remove this when Kohonen is ported to CUDA
root.common.engine.backend = "ocl"


class TestKohonen(StandardTest):
    @classmethod
    def setUpClass(cls):
        data_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "samples/DemoKohonen")
        root.kohonen.update({
            "forward": {"shape": (8, 8),
                        "weights_stddev": 0.05,
                        "weights_filling": "uniform"},
            "decision": {"snapshot_prefix": "kohonen",
                         "epochs": 160},
            "loader": {
                "minibatch_size": 10,
                "force_numpy": False,
                "dataset_file": os.path.join(data_path, "kohonen.txt.gz")},
            "train": {"gradient_decay": lambda t: 0.05 / (1.0 + t * 0.01),
                      "radius_decay": lambda t: 1.0 / (1.0 + t * 0.01)}})

    @timeout(700)
    def test_kohonen(self):
        self.info("Will test kohonen workflow")

        workflow = kohonen.KohonenWorkflow(self.parent)
        workflow.initialize(device=self.device)
        workflow.run()

        diff = workflow.decision.weights_diff
        self.assertAlmostEqual(diff, 0.00057525720324055766, places=7)
        self.assertEqual(160, workflow.loader.epoch_number)
        self.info("All Ok")

if __name__ == "__main__":
    StandardTest.main()
