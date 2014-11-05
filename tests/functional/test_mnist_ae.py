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
import veles.znicz.tests.research.mnist_ae as mnist_ae
import veles.dummy as dummy_workflow


class TestMnistAE(unittest.TestCase):
    def setUp(self):
        root.common.unit_test = True
        root.common.plotters_disabled = True
        self.device = opencl.Device()

    def tearDown(self):
        del self.device

    @timeout(300)
    def test_mnist_ae(self):
        logging.info("Will test mnist ae workflow")

        prng.get().seed(numpy.fromfile("%s/veles/znicz/tests/research/seed" %
                                       root.common.veles_dir,
                                       dtype=numpy.uint32, count=1024))
        root.mnist_ae.update({
            "all2all": {"weights_stddev": 0.05},
            "decision": {"fail_iterations": 20,
                         "max_epochs": 3},
            "snapshotter": {"prefix": "mnist", "time_interval": 0,
                            "compress": ""},
            "loader": {"minibatch_size": 100, "on_device": True},
            "learning_rate": 0.000001,
            "weights_decay": 0.00005,
            "gradient_moment": 0.00001,
            "weights_plotter": {"limit": 16},
            "pooling": {"kx": 3, "ky": 3, "sliding": (2, 2)},
            "include_bias": False,
            "unsafe_padding": True,
            "n_kernels": 5,
            "kx": 5,
            "ky": 5,
            "layers": [100, 10]})

        self.w = mnist_ae.MnistAEWorkflow(dummy_workflow.DummyWorkflow(),
                                          layers=root.mnist_ae.layers,
                                          device=self.device)
        self.w.snapshotter.interval = 3
        self.w.initialize(device=self.device)
        self.w.run()
        file_name = self.w.snapshotter.file_name

        avg_mse = self.w.decision.epoch_metrics[1][0]
        self.assertAlmostEqual(avg_mse, 0.960932, places=5)
        self.assertEqual(3, self.w.loader.epoch_number)

        logging.info("Will load workflow from %s" % file_name)
        self.wf = Snapshotter.import_(file_name)
        self.assertTrue(self.wf.decision.epoch_ended)
        self.wf.decision.max_epochs = 6
        self.wf.decision.complete <<= False
        self.wf.initialize(device=self.device)
        self.wf.run()

        avg_mse = self.wf.decision.epoch_metrics[1][0]
        self.assertAlmostEqual(avg_mse, 0.960581, places=5)
        self.assertEqual(6, self.wf.loader.epoch_number)
        logging.info("All Ok")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
