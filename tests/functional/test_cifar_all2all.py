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
import veles.znicz.tests.research.CIFAR10.cifar as cifar
import veles.dummy as dummy_workflow


class TestCifarAll2All(unittest.TestCase):
    def setUp(self):
        self.device = opencl.Device()

    @timeout(300)
    def test_cifar_all2all(self):
        logging.info("Will test cifar fully connected workflow")
        rnd.get().seed(numpy.fromfile("%s/veles/znicz/tests/research/seed" %
                                      root.common.veles_dir,
                                      dtype=numpy.int32, count=1024))
        root.common.precision_level = 1
        train_dir = os.path.join(root.common.test_dataset_root, "cifar/10")
        validation_dir = os.path.join(root.common.test_dataset_root,
                                      "cifar/10/test_batch")
        root.cifar.update({
            "decision": {"fail_iterations": 1000, "max_epochs": 2},
            "learning_rate_adjust": {"do": False},
            "loss_function": "softmax",
            "add_plotters": True,
            "image_saver": {"do": False,
                            "out_dirs":
                            [os.path.join(root.common.cache_dir, "tmp/test"),
                             os.path.join(root.common.cache_dir,
                                          "tmp/validation"),
                             os.path.join(root.common.cache_dir,
                                          "tmp/train")]},
            "loader": {"minibatch_size": 81, "on_device": True,
                       "normalization_type": (-1, 1)},
            "accumulator": {"n_bars": 30},
            "weights_plotter": {"limit": 25},
            "layers": [{"type": "all2all", "output_sample_shape": 486,
                        "learning_rate": 0.0005, "weights_decay": 0.0},
                       {"type": "activation_sincos"},
                       {"type": "all2all", "output_sample_shape": 486,
                        "learning_rate": 0.0005, "weights_decay": 0.0},
                       {"type": "activation_sincos"},
                       {"type": "softmax", "output_sample_shape": 10,
                        "learning_rate": 0.0005, "weights_decay": 0.0}],
            "snapshotter": {"prefix": "cifar_test"},
            "data_paths": {"train": train_dir, "validation": validation_dir}})
        self.w = cifar.CifarWorkflow(
            dummy_workflow.DummyLauncher(),
            decision_config=root.cifar.decision,
            snapshotter_config=root.cifar.snapshotter,
            image_saver_config=root.cifar.image_saver,
            layers=root.cifar.layers,
            loss_function=root.cifar.loss_function,
            device=self.device)
        self.w.snapshotter.time_interval = 0
        self.w.snapshotter.interval = 2
        self.assertEqual(self.w.evaluator.labels,
                         self.w.loader.minibatch_labels)
        self.w.initialize(device=self.device,
                          minibatch_size=root.cifar.loader.minibatch_size)
        self.assertEqual(self.w.evaluator.labels,
                         self.w.loader.minibatch_labels)
        self.w.run()
        file_name = self.w.snapshotter.file_name

        err = self.w.decision.epoch_n_err[1]
        self.assertEqual(err, 7457)
        self.assertEqual(2, self.w.loader.epoch_number)

        logging.info("Will load workflow from %s" % file_name)
        self.wf = Snapshotter.import_(file_name)
        self.assertTrue(self.wf.decision.epoch_ended)
        self.wf.decision.max_epochs = 5
        self.wf.decision.complete <<= False
        self.assertEqual(self.wf.evaluator.labels,
                         self.wf.loader.minibatch_labels)
        self.wf.initialize(device=self.device,
                           minibatch_size=root.cifar.loader.minibatch_size)
        self.assertEqual(self.wf.evaluator.labels,
                         self.wf.loader.minibatch_labels)
        self.wf.run()

        err = self.wf.decision.epoch_n_err[1]
        self.assertEqual(err, 7055)
        self.assertEqual(5, self.wf.loader.epoch_number)
        logging.info("All Ok")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
