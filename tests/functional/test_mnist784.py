#!/usr/bin/python3 -O
"""
Created on April 3, 2014

Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""


import logging
import numpy
import unittest

from veles.config import root
import veles.opencl as opencl
import veles.prng as prng
from veles.snapshotter import Snapshotter
from veles.tests import timeout
import veles.znicz.tests.research.mnist784 as mnist784
import veles.tests.dummy_workflow as dummy_workflow


class TestMnist784(unittest.TestCase):
    def setUp(self):
        root.common.unit_test = True
        root.common.plotters_disabled = True
        self.device = opencl.Device()

    def tearDown(self):
        del self.device

    @timeout(12000)
    def test_mnist784(self):
        logging.info("Will test mnist784 workflow")

        prng.get().seed(numpy.fromfile("%s/veles/znicz/tests/research/seed" %
                                       root.common.veles_dir,
                                       dtype=numpy.uint32, count=1024))
        root.update = {
            "decision": {"fail_iterations": 100},
            "snapshotter": {"prefix": "mnist_784_test"},
            "loader": {"minibatch_size": 100},
            "weights_plotter": {"limit": 16},
            "mnist784_test": {"learning_rate": 0.00001,
                              "weights_decay": 0.00005,
                              "layers": [784, 784]}}

        self.w = mnist784.Workflow(dummy_workflow.DummyWorkflow(),
                                   layers=root.mnist784_test.layers,
                                   device=self.device)
        self.w.decision.max_epochs = 5
        #self.w.snapshotter.interval = 0
        self.assertEqual(self.w.evaluator.labels,
                         self.w.loader.minibatch_labels)
        self.w.initialize(device=self.device,
                          learning_rate=root.mnist784_test.learning_rate,
                          weights_decay=root.mnist784_test.weights_decay)
        self.assertEqual(self.w.evaluator.labels,
                         self.w.loader.minibatch_labels)
        self.w.run()
        file_name = self.w.snapshotter.file_name

        err = self.w.decision.epoch_n_err[1]
        self.assertEqual(err, 7428)
        avg_mse = self.w.decision.epoch_metrics[1][0]
        self.assertAlmostEqual(avg_mse, 0.39173925, places=7)
        self.assertEqual(5, self.w.loader.epoch_number)

        logging.info("Will load workflow from %s" % file_name)
        self.wf = Snapshotter.import_(file_name)
        self.assertTrue(self.wf.decision.epoch_ended)
        self.wf.decision.max_epochs = 11
        self.wf.decision.complete <<= False
        self.assertEqual(self.wf.evaluator.labels,
                         self.wf.loader.minibatch_labels)
        self.wf.initialize(device=self.device,
                           learning_rate=root.mnist784_test.learning_rate,
                           weights_decay=root.mnist784_test.weights_decay)
        self.assertEqual(self.wf.evaluator.labels,
                         self.wf.loader.minibatch_labels)
        self.wf.run()

        err = self.w.decision.epoch_n_err[1]
        self.assertEqual(err, 6156)
        avg_mse = self.w.decision.epoch_metrics[1][0]
        self.assertAlmostEqual(avg_mse, 0.357516, places=7)
        self.assertEqual(11, self.wf.loader.epoch_number)
        logging.info("All Ok")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()