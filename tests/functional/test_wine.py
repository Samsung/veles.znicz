#!/usr/bin/python3 -O
"""
Created on April 2, 2014

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
import veles.znicz.samples.wine as wine
import veles.tests.dummy_workflow as dummy_workflow


class TestWine(unittest.TestCase):
    def setUp(self):
        root.common.unit_test = True
        root.common.plotters_disabled = True
        self.device = opencl.Device()

    @timeout()
    def test_wine(self):
        logging.info("Will test wine workflow")
        rnd.get().seed(numpy.fromfile("%s/veles/znicz/tests/research/seed" %
                                      root.common.veles_dir,
                                      dtype=numpy.int32, count=1024))
        root.update = {
            "decision": {"fail_iterations": 200,
                         "snapshot_prefix": "wine"},
            "loader": {"minibatch_size": 10},
            "wine_test": {"learning_rate": 0.3,
                          "weights_decay": 0.0,
                          "layers": [8, 3],
                          "data_paths":
                          os.path.join(root.common.veles_dir,
                                       "veles/znicz/samples/wine/wine.data")}}

        self.w = wine.Workflow(dummy_workflow.DummyWorkflow(),
                               layers=root.wine_test.layers)

        self.assertEqual(self.w.evaluator.labels,
                         self.w.loader.minibatch_labels)
        self.w.initialize(learning_rate=root.wine_test.learning_rate,
                          weights_decay=root.wine_test.weights_decay,
                          device=self.device)
        self.w.run()

        epoch = self.w.decision.epoch_number
        logging.info("Converged in %d epochs", epoch)
        self.assertEqual(epoch, 12)
        logging.info("All Ok")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
