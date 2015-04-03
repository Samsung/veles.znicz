#!/usr/bin/python3 -O
"""
  _   _ _____ _     _____ _____
 | | | |  ___| |   |  ___/  ___|
 | | | | |__ | |   | |__ \ `--.
 | | | |  __|| |   |  __| `--. \
 \ \_/ / |___| |___| |___/\__/ /
  \___/\____/\_____|____/\____/

Created on October 14, 2014

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
from veles.tests import timeout
from veles.znicz.tests.functional import StandardTest
import veles.znicz.tests.research.SpamKohonen.spam_kohonen as spam_kohonen

# FIXME(v.markovtsev): remove this when Kohonen is ported to CUDA
root.common.engine.backend = "ocl"


class TestSpamKohonen(StandardTest):
    @classmethod
    def setUpClass(cls):
        root.spam_kohonen.update({
            "forward": {"shape": (8, 8)},
            "decision": {"epochs": 5},
            "loader": {"minibatch_size": 80,
                       "force_cpu": True,
                       "ids": True,
                       "classes": False,
                       "file":
                       "/data/veles/VD/VDLogs/histogramConverter/data/hist"},
            "train": {"gradient_decay": lambda t: 0.002 / (1.0 + t * 0.00002),
                      "radius_decay": lambda t: 1.0 / (1.0 + t * 0.00002)},
            "exporter": {"file": "classified_fast4.txt"}})

    @timeout(700)
    def test_spamkohonen(self):
        self.info("Will test spam kohonen workflow")

        workflow = spam_kohonen.SpamKohonenWorkflow(self.parent)
        workflow.initialize(device=self.device, snapshot=False)
        workflow.run()
        self.assertIsNone(workflow.thread_pool.failure)

        diff = workflow.decision.weights_diff
        self.assertAlmostEqual(diff, 0.106724, places=6)
        self.assertEqual(5, workflow.loader.epoch_number)
        self.info("All Ok")

if __name__ == "__main__":
    StandardTest.main()
