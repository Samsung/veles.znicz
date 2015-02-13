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
import veles.backends as opencl
import veles.prng as rnd
from veles.snapshotter import Snapshotter
from veles.tests import timeout
import veles.znicz.tests.research.Hands.hands as hands
import veles.dummy as dummy_workflow


class TestHands(unittest.TestCase):
    def setUp(self):
        self.device = opencl.Device()

    @timeout(300)
    def test_hands(self):
        logging.info("Will test hands workflow")
        rnd.get().seed(numpy.fromfile("%s/veles/znicz/tests/research/seed" %
                                      root.common.veles_dir,
                                      dtype=numpy.int32, count=1024))
        root.common.update({
            "precision_level": 1,
            "plotters_disabled": True,
            "precision_type": "double",
            "engine": {"backend": "ocl"}})

        train_dir = [
            os.path.join(root.common.test_dataset_root, "hands/Training")]
        validation_dir = [
            os.path.join(root.common.test_dataset_root, "hands/Testing")]

        root.hands.update({
            "decision": {"fail_iterations": 100, "max_epochs": 2},
            "loss_function": "softmax",
            "loader_name": "hands_loader",
            "snapshotter": {"prefix": "hands", "interval": 2,
                            "time_interval": 0},
            "loader": {"minibatch_size": 40, "train_paths": train_dir,
                       "on_device": True, "color_space": "GRAY",
                       "background_color": (0,),
                       "normalization_type": "linear",
                       "validation_paths": validation_dir},
            "layers": [{"type": "all2all_tanh", "learning_rate": 0.008,
                        "weights_decay": 0.0, "output_sample_shape": 30},
                       {"type": "softmax", "output_sample_shape": 2,
                        "learning_rate": 0.008, "weights_decay": 0.0}]})

        self.w = hands.HandsWorkflow(
            dummy_workflow.DummyLauncher(),
            layers=root.hands.layers,
            decision_config=root.hands.decision,
            snapshotter_config=root.hands.snapshotter,
            loader_config=root.hands.loader,
            loss_function=root.hands.loss_function,
            loader_name=root.hands.loader_name,
            device=self.device)
        self.assertEqual(self.w.evaluator.labels,
                         self.w.loader.minibatch_labels)
        self.w.initialize(device=self.device)
        self.assertEqual(self.w.evaluator.labels,
                         self.w.loader.minibatch_labels)
        self.w.run()
        file_name = self.w.snapshotter.file_name

        err = self.w.decision.epoch_n_err[1]
        self.assertEqual(err, 673)
        self.assertEqual(2, self.w.loader.epoch_number)

        logging.info("Will load workflow from %s" % file_name)
        self.wf = Snapshotter.import_(file_name)
        self.assertTrue(self.wf.decision.epoch_ended)
        self.wf.decision.max_epochs = 3
        self.wf.decision.complete <<= False
        self.assertEqual(self.wf.evaluator.labels,
                         self.wf.loader.minibatch_labels)
        self.wf.initialize(device=self.device)
        self.assertEqual(self.wf.evaluator.labels,
                         self.wf.loader.minibatch_labels)
        self.wf.run()

        err = self.wf.decision.epoch_n_err[1]
        self.assertEqual(err, 629)
        self.assertEqual(3, self.wf.loader.epoch_number)
        logging.info("All Ok")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
