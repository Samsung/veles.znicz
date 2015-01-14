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
        root.common.unit_test = True
        root.common.plotters_disabled = True
        self.device = opencl.Device()

    @timeout(300)
    def test_cifar_all2all(self):
        logging.info("Will test cifar fully connected workflow")
        rnd.get().seed(numpy.fromfile("%s/veles/znicz/tests/research/seed" %
                                      root.common.veles_dir,
                                      dtype=numpy.int32, count=1024))
        root.common.precision_level = 1
        root.cifar.update({
            "snapshotter": {"prefix": "cifar_test"},
            "image_saver": {"out_dirs":
                            [os.path.join(root.common.cache_dir, "tmp/test"),
                             os.path.join(root.common.cache_dir,
                                          "tmp/validation"),
                             os.path.join(root.common.cache_dir,
                                          "tmp/train")]},
            "loader": {"minibatch_size": 81},
            "accumulator": {"n_bars": 30},
            "layers": [{"type": "all2all", "output_shape": 486,
                        "learning_rate": 0.0005, "weights_decay": 0.0},
                       {"type": "activation_sincos"},
                       {"type": "all2all", "output_shape": 486,
                        "learning_rate": 0.0005, "weights_decay": 0.0},
                       {"type": "activation_sincos"},
                       {"type": "softmax", "output_shape": 10,
                        "learning_rate": 0.0005, "weights_decay": 0.0}]})
        self.w = cifar.CifarWorkflow(
            dummy_workflow.DummyLauncher(),
            fail_iterations=root.cifar.decision.fail_iterations,
            max_epochs=root.cifar.decision.max_epochs,
            prefix=root.cifar.snapshotter.prefix,
            snapshot_interval=root.cifar.snapshotter.interval,
            snapshot_dir=root.common.snapshot_dir,
            layers=root.cifar.layers,
            out_dirs=root.cifar.image_saver.out_dirs,
            loss_function=root.cifar.loss_function,
            device=self.device)
        self.w.decision.max_epochs = 2
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
