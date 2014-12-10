#!/usr/bin/python3 -O
"""
Created on April 2, 2014

Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""


import logging
import numpy
import unittest

from veles.config import root
import veles.backends as opencl
import veles.prng as rnd
from veles.snapshotter import Snapshotter
from veles.tests import timeout
import veles.znicz.tests.research.Hands.hands as hands
import veles.dummy as dummy_workflow


class TestHands(unittest.TestCase):
    def setUp(self):
        root.common.unit_test = True
        root.common.plotters_disabled = True
        self.device = opencl.Device()

    @timeout(300)
    def test_hands(self):
        logging.info("Will test hands workflow")
        rnd.get().seed(numpy.fromfile("%s/veles/znicz/tests/research/seed" %
                                      root.common.veles_dir,
                                      dtype=numpy.int32, count=1024))
        root.hands.update({
            "decision": {"fail_iterations": 100},
            "snapshotter": {"prefix": "hands_test"},
            "loader": {"minibatch_size": 60},
            "learning_rate": 0.0008,
            "weights_decay": 0.0,
            "layers": [30, 2]})

        self.w = hands.HandsWorkflow(dummy_workflow.DummyWorkflow(),
                                     layers=root.hands.layers,
                                     device=self.device)
        self.w.decision.max_epochs = 2
        self.w.snapshotter.interval = 2
        self.w.snapshotter.time_interval = 0
        self.assertEqual(self.w.evaluator.labels,
                         self.w.loader.minibatch_labels)
        self.w.initialize(device=self.device,
                          learning_rate=root.hands.learning_rate,
                          weights_decay=root.hands.weights_decay)
        self.assertEqual(self.w.evaluator.labels,
                         self.w.loader.minibatch_labels)
        self.w.run()
        file_name = self.w.snapshotter.file_name

        err = self.w.decision.epoch_n_err[1]
        self.assertEqual(err, 1481)
        self.assertEqual(2, self.w.loader.epoch_number)

        logging.info("Will load workflow from %s" % file_name)
        self.wf = Snapshotter.import_(file_name)
        self.assertTrue(self.wf.decision.epoch_ended)
        self.wf.decision.max_epochs = 6
        self.wf.decision.complete <<= False
        self.assertEqual(self.wf.evaluator.labels,
                         self.wf.loader.minibatch_labels)
        self.wf.initialize(device=self.device,
                           learning_rate=root.hands.learning_rate,
                           weights_decay=root.hands.weights_decay)
        self.assertEqual(self.wf.evaluator.labels,
                         self.wf.loader.minibatch_labels)
        self.wf.run()

        err = self.wf.decision.epoch_n_err[1]
        self.assertEqual(err, 1375)
        self.assertEqual(6, self.wf.loader.epoch_number)
        logging.info("All Ok")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
