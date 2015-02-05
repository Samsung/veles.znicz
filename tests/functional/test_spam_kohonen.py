#!/usr/bin/python3 -O
"""
Created on October 14, 2014

Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""


import logging
import numpy
import unittest

from veles.config import root
import veles.backends as opencl
import veles.prng as rnd
from veles.tests import timeout
import veles.znicz.tests.research.SpamKohonen.spam_kohonen as spam_kohonen
import veles.dummy as dummy_workflow


class TestSpamKohonen(unittest.TestCase):
    def setUp(self):
        self.device = opencl.Device()

    @timeout(700)
    def test_spamkohonen(self):
        logging.info("Will test spam kohonen workflow")
        rnd.get().seed(numpy.fromfile("%s/veles/znicz/tests/research/seed" %
                                      root.common.veles_dir,
                                      dtype=numpy.int32, count=1024))
        root.common.precision_level = 1
        root.spam_kohonen.update({
            "forward": {"shape": (8, 8)},
            "decision": {"epochs": 5},
            "loader": {"minibatch_size": 80,
                       "on_device": False,
                       "ids": True,
                       "classes": False,
                       "file":
                       "/data/veles/VD/VDLogs/histogramConverter/data/hist"},
            "train": {"gradient_decay": lambda t: 0.002 / (1.0 + t * 0.00002),
                      "radius_decay": lambda t: 1.0 / (1.0 + t * 0.00002)},
            "exporter": {"file": "classified_fast4.txt"}})

        self.w = spam_kohonen.SpamKohonenWorkflow(
            dummy_workflow.DummyLauncher(), device=self.device)
        self.w.initialize(device=self.device)
        self.w.run()

        diff = self.w.decision.weights_diff
        self.assertAlmostEqual(diff, 0.106724, places=6)
        self.assertEqual(5, self.w.loader.epoch_number)
        logging.info("All Ok")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
