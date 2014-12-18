#!/usr/bin/python3 -O
"""
Created on April 3, 2014

Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""


import logging
import numpy
import os
import unittest

from veles.config import root
import veles.backends as opencl
import veles.prng as prng
from veles.snapshotter import Snapshotter
from veles.tests import timeout
import veles.znicz.tests.research.VideoAE.video_ae as video_ae
import veles.dummy as dummy_workflow


class TestVideoAE(unittest.TestCase):
    def setUp(self):
        root.common.unit_test = True
        root.common.plotters_disabled = True
        self.device = opencl.Device()

    def tearDown(self):
        del self.device

    @timeout(1800)
    def test_video_ae(self):
        logging.info("Will test video_ae workflow")

        prng.get().seed(numpy.fromfile("%s/veles/znicz/tests/research/seed" %
                                       root.common.veles_dir,
                                       dtype=numpy.uint32, count=1024))
        prng.get(2).seed(numpy.fromfile("%s/veles/znicz/tests/research/seed2" %
                                        root.common.veles_dir,
                                        dtype=numpy.uint32, count=1024))
        root.common.precision_level = 1
        root.video_ae.update({
            "decision": {"fail_iterations": 100},
            "snapshotter": {"prefix": "video_ae_test"},
            "loader": {"minibatch_size": 50},
            "weights_plotter": {"limit": 16},
            "learning_rate": 0.01,
            "weights_decay": 0.00005,
            "layers": [9, [90, 160]],
            "data_paths":
            os.path.join(root.common.test_dataset_root, "video_ae/img")})

        self.w = video_ae.VideoAEWorkflow(dummy_workflow.DummyWorkflow(),
                                          layers=root.video_ae.layers,
                                          device=self.device)
        self.w.decision.max_epochs = 4
        self.w.snapshotter.time_interval = 0
        self.w.snapshotter.interval = 4
        self.w.initialize(device=self.device,
                          learning_rate=root.video_ae.learning_rate,
                          weights_decay=root.video_ae.weights_decay)
        self.w.run()
        file_name = self.w.snapshotter.file_name

        avg_mse = self.w.decision.epoch_metrics[2][0]
        self.assertLess(avg_mse, 0.383)
        self.assertEqual(4, self.w.loader.epoch_number)

        logging.info("Will load workflow from %s" % file_name)
        self.wf = Snapshotter.import_(file_name)
        self.assertTrue(self.wf.decision.epoch_ended)
        self.wf.decision.max_epochs = 7
        self.wf.decision.complete <<= False
        self.wf.initialize(device=self.device,
                           learning_rate=root.video_ae.learning_rate,
                           weights_decay=root.video_ae.weights_decay)
        self.wf.run()

        avg_mse = self.wf.decision.epoch_metrics[2][0]
        self.assertLess(avg_mse, 0.354)
        self.assertEqual(7, self.wf.loader.epoch_number)
        logging.info("All Ok")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
