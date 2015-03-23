#!/usr/bin/python3 -O
"""
Created on September 26, 2014

Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""


import os

from veles.config import root
from veles.tests import timeout, multi_device
import veles.znicz.samples.DemoKohonen.kohonen as kohonen
from veles.znicz.tests.functional import StandardTest


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
                "force_cpu": False,
                "dataset_file": os.path.join(data_path, "kohonen.txt.gz")},
            "train": {"gradient_decay": lambda t: 0.05 / (1.0 + t * 0.01),
                      "radius_decay": lambda t: 1.0 / (1.0 + t * 0.01)}})

    @timeout(700)
    @multi_device()
    def test_kohonen(self):
        self.info("Will test kohonen workflow")

        workflow = kohonen.KohonenWorkflow(self.parent)
        workflow.initialize(device=self.device, snapshot=False)
        workflow.run()

        diff = workflow.decision.weights_diff
        self.assertAlmostEqual(diff, 0.00057525720324055766, places=7)
        self.assertEqual(160, workflow.loader.epoch_number)
        self.info("All Ok")

if __name__ == "__main__":
    StandardTest.main()
