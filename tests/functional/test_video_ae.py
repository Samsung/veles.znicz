#!/usr/bin/python3 -O
"""
Created on April 3, 2014

Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""


import numpy
import os

from veles.config import root
import veles.prng as prng
from veles.snapshotter import Snapshotter
from veles.tests import timeout, multi_device
from veles.znicz.tests.functional import StandardTest
import veles.znicz.tests.research.VideoAE.video_ae as video_ae


class TestVideoAE(StandardTest):
    @classmethod
    def setUpClass(cls):
        prng.get(2).seed(numpy.fromfile("%s/veles/znicz/tests/research/seed2" %
                                        root.common.veles_dir,
                                        dtype=numpy.uint32, count=1024))
        root.video_ae.update({
            "snapshotter": {"prefix": "video_ae_test"},
            "decision": {"fail_iterations": 100},
            "loader": {
                "minibatch_size": 50, "force_cpu": False,
                "train_paths": (os.path.join(root.common.test_dataset_root,
                                "video_ae/img"),),
                "color_space": "GRAY",
                "background_color": (0x80,),
                "normalization_type": "mean_disp"},
            "weights_plotter": {"limit": 16},
            "learning_rate": 0.01,
            "weights_decay": 0.00005,
            "layers": [9, [90, 160]]})

    @timeout(1800)
    @multi_device()
    def test_video_ae(self):
        self.info("Will test video_ae workflow")

        workflow = video_ae.VideoAEWorkflow(
            self.parent,
            layers=root.video_ae.layers)
        workflow.decision.max_epochs = 4
        workflow.snapshotter.time_interval = 0
        workflow.snapshotter.interval = 4
        workflow.initialize(
            device=self.device,
            learning_rate=root.video_ae.learning_rate,
            weights_decay=root.video_ae.weights_decay,
            snapshot=False)
        workflow.run()
        self.assertIsNone(workflow.thread_pool.failure)
        file_name = workflow.snapshotter.file_name

        avg_mse = workflow.decision.epoch_metrics[2][0]
        self.assertLess(avg_mse, 0.1957180928)
        self.assertEqual(4, workflow.loader.epoch_number)

        self.info("Will load workflow from %s", file_name)
        workflow_from_snapshot = Snapshotter.import_(file_name)
        self.assertTrue(workflow_from_snapshot.decision.epoch_ended)
        workflow_from_snapshot.decision.max_epochs = 7
        workflow_from_snapshot.decision.complete <<= False
        workflow_from_snapshot.initialize(
            device=self.device,
            learning_rate=root.video_ae.learning_rate,
            weights_decay=root.video_ae.weights_decay,
            snapshot=True)
        workflow_from_snapshot.run()
        self.assertIsNone(workflow_from_snapshot.thread_pool.failure)

        avg_mse = workflow_from_snapshot.decision.epoch_metrics[2][0]
        self.assertLess(avg_mse, 0.18736321)
        self.assertEqual(7, workflow_from_snapshot.loader.epoch_number)
        self.info("All Ok")


if __name__ == "__main__":
    StandardTest.main()
