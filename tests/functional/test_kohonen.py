#!/usr/bin/python3 -O
"""
Created on September 26, 2014

Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""


import logging
import numpy
import os
import unittest

from veles.config import root
import veles.opencl as opencl
import veles.prng as rnd
from veles.tests import timeout
import veles.znicz.samples.kohonen as kohonen
import veles.tests.dummy_workflow as dummy_workflow


class TestKohonen(unittest.TestCase):
    def setUp(self):
        root.common.unit_test = True
        root.common.plotters_disabled = True
        self.device = opencl.Device()

    @timeout(12000)
    def test_kohonen(self):
        logging.info("Will test kohonen workflow")
        rnd.get().seed(numpy.fromfile("%s/veles/znicz/tests/research/seed" %
                                      root.common.veles_dir,
                                      dtype=numpy.int32, count=1024))
        data_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "samples/kohonen")
        root.kohonen.update({
            "forward": {"shape": (8, 8),
                        "weights_stddev": 0.05,
                        "weights_filling": "uniform"},
            "decision": {"snapshot_prefix": "kohonen",
                         "epochs": 160},
            "loader": {"minibatch_size": 10,
                       "dataset_file": os.path.join(data_path, "kohonen.txt")},
            "train": {"gradient_decay": lambda t: 0.05 / (1.0 + t * 0.01),
                      "radius_decay": lambda t: 1.0 / (1.0 + t * 0.01)}})

        self.w = kohonen.KohonenWorkflow(dummy_workflow.DummyWorkflow(),
                                         device=self.device)
        self.w.initialize(device=self.device)
        self.w.run()

        diff = self.w.decision.weights_diff
        self.assertAlmostEqual(diff, 0.00057525720324055766, places=10)
        self.assertEqual(160, self.w.loader.epoch_number)
        logging.info("All Ok")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
