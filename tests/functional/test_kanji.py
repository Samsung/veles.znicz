#!/usr/bin/python3 -O
"""
Created on April 2, 2014

Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""


import logging
import numpy
import os
import six
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

        root.common.update({
            "precision_level": 1,
            "plotters_disabled": True,
            "precision_type": "double",
            "engine": {"backend": "ocl"}})

        train_path = os.path.join(root.common.test_dataset_root,
                                  "new_kanji/train")

        target_path = os.path.join(root.common.test_dataset_root,
                                   "new_kanji/target")

        root.kanji.update({
            "decision": {"fail_iterations": 1000,
                         "max_epochs": 2},
            "loss_function": "mse",
            "loader_name": "full_batch_auto_label_file_image_mse",
            "add_plotters": False,
            "loader": {"minibatch_size": 50,
                       "force_cpu": False,
                       "filename_types": ["png"],
                       "train_paths": [train_path],
                       "target_paths": [target_path],
                       "color_space": "GRAY",
                       "normalization_type": "linear",
                       "target_normalization_type": "linear",
                       "targets_shape": (24, 24),
                       "background_color": (0,),
                       "validation_ratio": 0.15},
            "snapshotter": {"prefix": "kanji_test"},
            "layers": [{"type": "all2all_tanh",
                        "->": {"output_sample_shape": 250},
                        "<-": {"learning_rate": 0.00001,
                               "learning_rate_bias": 0.01,
                               "weights_decay": 0.00005}},
                       {"type": "all2all_tanh",
                        "->": {"output_sample_shape": 250},
                        "<-": {"learning_rate": 0.00001,
                               "learning_rate_bias": 0.01,
                               "weights_decay": 0.00005}},
                       {"type": "all2all_tanh",
                        "->": {"output_sample_shape": 24 * 24},
                        "<-": {"learning_rate_bias": 0.01,
                               "learning_rate": 0.00001,
                               "weights_decay": 0.00005}}]})

        self.w = kanji.KanjiWorkflow(
            dummy_workflow.DummyLauncher(),
            decision_config=root.kanji.decision,
            loader_config=root.kanji.loader,
            loader_name=root.kanji.loader_name,
            snapshotter_config=root.kanji.snapshotter,
            layers=root.kanji.layers,
            loss_function=root.kanji.loss_function,
            device=self.device)
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
