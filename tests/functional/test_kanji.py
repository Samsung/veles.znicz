#!/usr/bin/python3 -O
"""
Created on April 2, 2014

Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""


import logging
import numpy
import os
import six
import sys
import unittest

from veles.config import root
import veles.backends as opencl
import veles.prng as prng
from veles.snapshotter import Snapshotter
from veles.tests import timeout
from veles.znicz.samples.Kanji import kanji
import veles.dummy as dummy_workflow


class TestKanji(unittest.TestCase):
    def setUp(self):
        root.common.unit_test = True
        root.common.plotters_disabled = True
        self.device = opencl.Device()

    @timeout(1000)
    def test_kanji(self):
        logging.info("Will test kanji workflow")

        prng.get().seed(numpy.fromfile("%s/veles/znicz/tests/research/seed" %
                                       root.common.veles_dir,
                                       dtype=numpy.uint32, count=1024))
        prng.get(2).seed(numpy.fromfile("%s/veles/znicz/tests/research/seed2" %
                                        root.common.veles_dir,
                                        dtype=numpy.uint32, count=1024))
        train_path = os.path.join(root.common.test_dataset_root, "kanji/train")

        root.kanji.update({
            "index_map": os.path.join(train_path, "index_map.%d.pickle" %
                                      (sys.version_info[0]))})
        root.common.precision_level = 1

        root.kanji.update({
            "decision": {"fail_iterations": 1000,
                         "max_epochs": 2},
            "loss_function": "mse",
            "add_plotters": False,
            "loader": {"minibatch_size": 50,
                       "validation_ratio": 0.15},
            "snapshotter": {"prefix": "kanji_test"},
            "layers": [{"type": "all2all_tanh", "learning_rate": 0.00001,
                        "learning_rate_bias": 0.01,
                        "weights_decay": 0.00005, "output_shape": 250},
                       {"type": "all2all_tanh", "learning_rate": 0.00001,
                        "learning_rate_bias": 0.01,
                        "weights_decay": 0.00005, "output_shape": 250},
                       {"type": "all2all_tanh", "output_shape": 24 * 24,
                        "learning_rate_bias": 0.01,
                        "learning_rate": 0.00001,
                        "weights_decay": 0.00005}],
            "data_paths": {"target":
                           os.path.join(root.common.test_dataset_root,
                                        ("kanji/target/targets.%d.pickle" %
                                         (sys.version_info[0]))),
                           "train": train_path}})

        self.w = kanji.KanjiWorkflow(
            dummy_workflow.DummyLauncher(),
            fail_iterations=root.kanji.decision.fail_iterations,
            max_epochs=root.kanji.decision.max_epochs,
            prefix=root.kanji.snapshotter.prefix,
            snapshot_dir=root.common.snapshot_dir,
            layers=root.kanji.layers,
            loss_function=root.kanji.loss_function,
            device=self.device)
        self.w.decision.max_epochs = 2
        self.w.snapshotter.time_interval = 0
        self.w.snapshotter.interval = 2
        self.assertEqual(self.w.evaluator.labels,
                         self.w.loader.minibatch_labels)
        self.w.initialize(device=self.device, weights=None, bias=None)
        self.assertEqual(self.w.evaluator.labels,
                         self.w.loader.minibatch_labels)
        self.w.run()
        file_name = self.w.snapshotter.file_name

        err = self.w.decision.epoch_n_err[1]
        self.assertEqual(err, 12913 if six.PY3 else 7560)
        avg_mse = self.w.decision.epoch_metrics[1][0]
        self.assertAlmostEqual(avg_mse, 0.444085 if six.PY3 else 0.438381707,
                               places=5)
        self.assertEqual(2, self.w.loader.epoch_number)

        logging.info("Will load workflow from %s" % file_name)
        self.wf = Snapshotter.import_(file_name)
        self.assertTrue(self.wf.decision.epoch_ended)
        self.wf.decision.max_epochs = 7
        self.wf.decision.complete <<= False
        self.assertEqual(self.wf.evaluator.labels,
                         self.wf.loader.minibatch_labels)
        self.wf.initialize(device=self.device, weights=None, bias=None)
        self.assertEqual(self.wf.evaluator.labels,
                         self.wf.loader.minibatch_labels)
        self.wf.run()

        err = self.wf.decision.epoch_n_err[1]
        self.assertEqual(err, 12806 if six.PY3 else 7560)
        avg_mse = self.wf.decision.epoch_metrics[1][0]
        self.assertAlmostEqual(avg_mse, 0.442661 if six.PY3 else 0.437215,
                               places=5)
        self.assertEqual(7, self.wf.loader.epoch_number)
        logging.info("All Ok")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
