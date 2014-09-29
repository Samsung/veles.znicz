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
import veles.tests.dummy_workflow as dummy_workflow


class TestMnist(unittest.TestCase):
    def setUp(self):
        root.common.unit_test = True
        root.common.plotters_disabled = True
        self.device = opencl.Device()

    def tearDown(self):
        del self.device

    @timeout(12000)
    def test_mnist_ae(self):
        logging.info("Will test mnist ae workflow")

        prng.get().seed(numpy.fromfile("%s/veles/znicz/tests/research/seed" %
                                       root.common.veles_dir,
                                       dtype=numpy.uint32, count=1024))
        prng.get(2).seed(numpy.fromfile("%s/veles/znicz/tests/research/seed2" %
                                        root.common.veles_dir,
                                        dtype=numpy.uint32, count=1024))
        root.update = {
            "all2all": {"weights_stddev": 0.05},
            "decision": {"fail_iterations": 20,
                         "store_samples_mse": True},
            "snapshotter": {"prefix": "mnist_ae_test"},
            "loader": {"minibatch_size": 100},
            "mnist_ae": {"learning_rate": 0.000001,
                         "weights_decay": 0.00005,
                         "gradient_moment": 0.00001,
                         "n_kernels": 5,
                         "kx": 5,
                         "ky": 5},
            "mnist_ae_test": {"layers": [100, 10]}}

        self.w = mnist_ae.Workflow(dummy_workflow.DummyWorkflow(),
                                   layers=root.mnist_ae_test.layers,
                                   device=self.device)
        self.w.decision.max_epochs = 5
        #self.w.snapshotter.interval = 0
        #self.assertEqual(self.w.evaluator.labels,
        #                 self.w.loader.minibatch_labels)
        self.w.initialize(device=self.device)
        #self.assertEqual(self.w.evaluator.labels,
        #                 self.w.loader.minibatch_labels)
        self.w.run()
        file_name = self.w.snapshotter.file_name

        avg_mse = self.w.decision.epoch_metrics[1][0]
        self.assertEqual(avg_mse, 0.959566)
        self.assertEqual(5, self.w.loader.epoch_number)

        logging.info("Will load workflow from %s" % file_name)
        self.wf = Snapshotter.import_(file_name)
        self.assertTrue(self.wf.decision.epoch_ended)
        self.wf.decision.max_epochs = 20
        self.wf.decision.complete <<= False
        #self.assertEqual(self.wf.evaluator.labels,
        #                 self.wf.loader.minibatch_labels)
        self.wf.initialize(device=self.device)
        #self.assertEqual(self.wf.evaluator.labels,
        #                 self.wf.loader.minibatch_labels)
        self.wf.run()

        avg_mse = self.w.decision.epoch_metrics[1][0]
        self.assertEqual(avg_mse, 0.873293)
        self.assertEqual(20, self.wf.loader.epoch_number)
        logging.info("All Ok")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
