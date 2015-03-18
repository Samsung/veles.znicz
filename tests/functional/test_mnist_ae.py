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
import veles.znicz.tests.research.MnistAE.mnist_ae as mnist_ae
import veles.dummy as dummy_workflow


class TestMnistAE(unittest.TestCase):
    def setUp(self):
        self.device = opencl.Device()

    def tearDown(self):
        del self.device

    def init_wf(self, workflow, device):
        workflow.initialize(device=device)

    def check_write_error_rate(self, workflow, error):
        err = workflow.decision.epoch_metrics[1][0]
        self.assertLess(err, error)
        self.assertEqual(
            workflow.decision.max_epochs, workflow.loader.epoch_number)

    def init_and_run(self, device):
        self.w = mnist_ae.MnistAEWorkflow(dummy_workflow.DummyLauncher(),
                                          layers=root.mnist_ae.layers)
        self.init_wf(self.w, device)
        self.w.run()

    @timeout(1500)
    def test_mnist_ae(self):
        logging.info("Will test mnist ae workflow")

        prng.get().seed(numpy.fromfile("%s/veles/znicz/tests/research/seed" %
                                       root.common.veles_dir,
                                       dtype=numpy.uint32, count=1024))
        root.common.update({
            "precision_level": 1,
            "precision_type": "double",
            "engine": {"backend": "ocl"}})

        root.mnist_ae.update({
            "all2all": {"weights_stddev": 0.05},
            "decision": {"fail_iterations": 20,
                         "max_epochs": 3},
            "snapshotter": {"prefix": "mnist", "time_interval": 0,
                            "interval": 3, "compress": ""},
            "loader": {"minibatch_size": 100, "force_cpu": False,
                       "normalization_type": "linear"},
            "learning_rate": 0.000001,
            "weights_decay": 0.00005,
            "gradient_moment": 0.00001,
            "weights_plotter": {"limit": 16},
            "pooling": {"kx": 3, "ky": 3, "sliding": (2, 2)},
            "include_bias": False,
            "unsafe_padding": True,
            "n_kernels": 5,
            "kx": 5,
            "ky": 5})

        self._test_mnist_ae_gpu(self.device)

    def _test_mnist_ae_gpu(self, device):
        logging.info("Will run workflow with double and ocl backend")

        root.common.update({
            "precision_level": 1,
            "precision_type": "double",
            "engine": {"backend": "ocl"}})

        # Test workflow
        self.init_and_run(device)
        self.check_write_error_rate(self.w, 0.96093162)

        file_name = self.w.snapshotter.file_name

        # Test loading from snapshot
        logging.info("Will load workflow from snapshot: %s" % file_name)

        self.wf = Snapshotter.import_(file_name)
        self.assertTrue(self.wf.decision.epoch_ended)
        self.wf.decision.max_epochs = 6
        self.wf.decision.complete <<= False

        self.init_wf(self.wf, device)
        self.wf.run()
        self.check_write_error_rate(self.wf, 0.9606072)

        logging.info("Will run workflow with double and cuda backend")

        root.common.update({
            "precision_level": 1,
            "precision_type": "double",
            "engine": {"backend": "cuda"}})

        # Test workflow with cuda and double
        self.init_and_run(device)
        self.check_write_error_rate(self.w, 0.9612299373)

        logging.info("Will run workflow with float and ocl backend")

        root.common.update({
            "precision_level": 1,
            "precision_type": "float",
            "engine": {"backend": "ocl"}})

        # Test workflow with ocl and float
        self.init_and_run(device)
        self.check_write_error_rate(self.w, 0.96072854)

        logging.info("Will run workflow with float and cuda backend")

        root.common.update({
            "precision_level": 1,
            "precision_type": "float",
            "engine": {"backend": "cuda"}})

        # Test workflow with cuda and float
        self.init_and_run(device)
        self.check_write_error_rate(self.w, 0.96101219)

        logging.info("All Ok")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
