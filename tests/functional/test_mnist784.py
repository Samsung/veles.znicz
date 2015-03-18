#!/usr/bin/python3 -O
"""
Created on April 3, 2014

Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""


import logging
import numpy
import unittest

from veles.config import root
import veles.backends as opencl
import veles.prng as prng
from veles.snapshotter import Snapshotter
from veles.tests import timeout
import veles.znicz.tests.research.Mnist784.mnist784 as mnist784
import veles.dummy as dummy_workflow


class TestMnist784(unittest.TestCase):
    def setUp(self):
        self.device = opencl.Device()

    def tearDown(self):
        del self.device

    def init_wf(self, workflow, device, snapshot):
        workflow.initialize(device=device,
                            learning_rate=root.mnist784.learning_rate,
                            weights_decay=root.mnist784.weights_decay,
                            snapshot=snapshot)

    def check_write_error_rate(self, workflow, mse, error):
        avg_mse = workflow.decision.epoch_metrics[1][0]
        err = workflow.decision.epoch_n_err[1]
        self.assertEqual(err, error)
        self.assertAlmostEqual(avg_mse, mse, places=6)
        self.assertEqual(
            workflow.decision.max_epochs, workflow.loader.epoch_number)

    def init_and_run(self, device, snapshot):
        self.w = mnist784.Mnist784Workflow(dummy_workflow.DummyLauncher(),
                                           layers=root.mnist784.layers)
        self.init_wf(self.w, device, snapshot)
        self.w.run()

    @timeout(1000)
    def test_mnist784(self):
        logging.info("Will test mnist784 workflow")

        prng.get().seed(numpy.fromfile("%s/veles/znicz/tests/research/seed" %
                                       root.common.veles_dir,
                                       dtype=numpy.uint32, count=1024))
        root.common.update({
            "plotters_disabled": True,
            "precision_level": 1,
            "precision_type": "double",
            "engine": {"backend": "ocl"}})

        root.mnist784.update({
            "decision": {"fail_iterations": 100, "max_epochs": 2},
            "snapshotter": {"prefix": "mnist_784_test", "time_interval": 0,
                            "interval": 2},
            "loader": {"minibatch_size": 100, "normalization_type": "linear",
                       "target_normalization_type": "linear"},
            "weights_plotter": {"limit": 16},
            "learning_rate": 0.00001,
            "weights_decay": 0.00005,
            "layers": [784, 784]})

        self._test_mnist784_gpu(self.device)
        self._test_mnist784_cpu(None)
        logging.info("All Ok")

    def _test_mnist784_gpu(self, device):
        logging.info("Will run workflow with double and ocl backend")

        root.common.update({
            "precision_level": 1,
            "precision_type": "double",
            "engine": {"backend": "ocl"}})

        # Test workflow
        self.init_and_run(device, False)
        self.check_write_error_rate(self.w, 0.409835, 8357)

        file_name = self.w.snapshotter.file_name

        # Test loading from snapshot
        logging.info("Will load workflow from snapshot: %s" % file_name)

        self.wf = Snapshotter.import_(file_name)
        self.assertTrue(self.wf.decision.epoch_ended)
        self.wf.decision.max_epochs = 5
        self.wf.decision.complete <<= False

        self.init_wf(self.wf, device, True)
        self.wf.run()
        self.check_write_error_rate(self.wf, 0.39173925, 7589)

        logging.info("Will run workflow with double and cuda backend")

        root.common.update({
            "precision_level": 1,
            "precision_type": "double",
            "engine": {"backend": "cuda"}})

        # Test workflow with cuda and double
        root.mnist784.decision.max_epochs = 3
        self.init_and_run(device, False)
        self.check_write_error_rate(self.w, 0.403975599, 7967)

        logging.info("Will run workflow with float and ocl backend")

    def _test_mnist784_cpu(self, device):
        logging.info("Will run workflow with --disable-acceleration")

        # Test workflow with --disable-acceleration
        root.mnist784.decision.max_epochs = 3
        self.init_and_run(device, False)
        self.check_write_error_rate(self.w, 0.40309872, 8143)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
