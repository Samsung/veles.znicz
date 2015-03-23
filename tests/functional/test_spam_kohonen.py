#!/usr/bin/python3 -O
"""
Created on October 14, 2014

Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""


from veles.config import root
from veles.tests import timeout
from veles.znicz.tests.functional import StandardTest
import veles.znicz.tests.research.SpamKohonen.spam_kohonen as spam_kohonen


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

        self.w = spam_kohonen.SpamKohonenWorkflow(
            self.parent)
        self.w.initialize(device=self.device, snapshot=False)
        self.w.run()

        diff = self.w.decision.weights_diff
        self.assertAlmostEqual(diff, 0.106724, places=6)
        self.assertEqual(5, self.w.loader.epoch_number)
        self.info("All Ok")

if __name__ == "__main__":
    StandardTest.main()
