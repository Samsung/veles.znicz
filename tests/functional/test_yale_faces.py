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
from veles.snapshotter import Snapshotter
from veles.tests import timeout
import veles.znicz.samples.YaleFaces.yale_faces as yale_faces
import veles.dummy as dummy_workflow


class TestYaleFaces(unittest.TestCase):
    def setUp(self):
        root.common.unit_test = True
        root.common.plotters_disabled = True
        self.device = opencl.Device()

    @timeout(300)
    def test_yale_faces(self):
        logging.info("Will test fully connectected yale_faces workflow")
        rnd.get().seed(numpy.fromfile("%s/veles/znicz/tests/research/seed" %
                                      root.common.veles_dir,
                                      dtype=numpy.int32, count=1024))

        root.yalefaces.update({
            "decision": {"fail_iterations": 50, "max_epochs": 3},
            "loss_function": "softmax",
            "snapshotter": {"prefix": "yalefaces_test"},
            "loader": {"minibatch_size": 40, "on_device": False,
                       "validation_ratio": 0.15,
                       "common_dir": root.common.test_dataset_root,
                       "url":
                       "http://vision.ucsd.edu/extyaleb/CroppedYaleBZip/"
                       "CroppedYale.zip"},
            "layers": [{"type": "all2all_tanh", "learning_rate": 0.01,
                        "weights_decay": 0.00005, "output_shape": 100},
                       {"type": "softmax", "output_shape": 39,
                        "learning_rate": 0.01, "weights_decay": 0.00005}]})

        root.yalefaces.loader.data_dir = os.path.join(
            root.yalefaces.loader.common_dir, "CroppedYale")

        self.w = yale_faces.YaleFacesWorkflow(
            dummy_workflow.DummyWorkflow(),
            fail_iterations=root.yalefaces.decision.fail_iterations,
            max_epochs=root.yalefaces.decision.max_epochs,
            prefix=root.yalefaces.snapshotter.prefix,
            snapshot_dir=root.common.snapshot_dir,
            layers=root.yalefaces.layers,
            loss_function=root.yalefaces.loss_function, device=self.device)
        self.assertEqual(self.w.evaluator.labels,
                         self.w.loader.minibatch_labels)
        self.w.snapshotter.interval = 3
        self.w.initialize(device=self.device)
        self.assertEqual(self.w.evaluator.labels,
                         self.w.loader.minibatch_labels)
        self.w.run()
        file_name = self.w.snapshotter.file_name

        err = self.w.decision.epoch_n_err[1]
        self.assertEqual(err, 276)
        self.assertEqual(3, self.w.loader.epoch_number)

        logging.info("Will load workflow from %s" % file_name)
        self.wf = Snapshotter.import_(file_name)
        self.assertTrue(self.wf.decision.epoch_ended)
        self.wf.decision.max_epochs = 6
        self.wf.decision.complete <<= False
        self.assertEqual(self.wf.evaluator.labels,
                         self.wf.loader.minibatch_labels)
        self.wf.initialize(device=self.device)
        self.assertEqual(self.wf.evaluator.labels,
                         self.wf.loader.minibatch_labels)
        self.wf.run()

        err = self.wf.decision.epoch_n_err[1]
        self.assertEqual(err, 135)
        self.assertEqual(6, self.wf.loader.epoch_number)
        logging.info("All Ok")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    unittest.main()
