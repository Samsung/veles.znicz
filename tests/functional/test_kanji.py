#!/usr/bin/python3 -O
"""
Created on April 2, 2014

Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""

import numpy
import os

from veles.config import root
import veles.prng as prng
from veles.snapshotter import Snapshotter
from veles.tests import timeout, multi_device
from veles.znicz.samples.Kanji import kanji
from veles.znicz.tests.functional import StandardTest


class TestKanji(StandardTest):
    @classmethod
    def setUpClass(cls):
        prng.get(2).seed(numpy.fromfile("%s/veles/znicz/tests/research/seed2" %
                                        root.common.veles_dir,
                                        dtype=numpy.uint32, count=1024))
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
            "snapshotter": {"prefix": "kanji_test", "interval": 3,
                            "time_interval": 0},
            "layers": [{"type": "all2all_tanh",
                        "->": {"output_sample_shape": 250},
                        "<-": {"learning_rate": 0.0001,
                               "learning_rate_bias": 0.01,
                               "weights_decay": 0.00005}},
                       {"type": "all2all_tanh",
                        "->": {"output_sample_shape": 250},
                        "<-": {"learning_rate": 0.0001,
                               "learning_rate_bias": 0.01,
                               "weights_decay": 0.00005}},
                       {"type": "all2all_tanh",
                        "->": {"output_sample_shape": (24, 24)},
                        "<-": {"learning_rate_bias": 0.01,
                               "learning_rate": 0.0001,
                               "weights_decay": 0.00005}}]})

    @timeout(1200)
    @multi_device()
    def test_kanji(self):
        self.info("Will test kanji workflow")

        workflow = kanji.KanjiWorkflow(
            self.parent,
            decision_config=root.kanji.decision,
            loader_config=root.kanji.loader,
            loader_name=root.kanji.loader_name,
            snapshotter_config=root.kanji.snapshotter,
            layers=root.kanji.layers,
            loss_function=root.kanji.loss_function)
        self.assertEqual(workflow.evaluator.labels,
                         workflow.loader.minibatch_labels)
        workflow.initialize(
            device=self.device, weights=None, bias=None,
            snapshot=False)
        self.assertEqual(workflow.evaluator.labels,
                         workflow.loader.minibatch_labels)
        workflow.run()
        self.assertIsNone(workflow.thread_pool.failure)
        file_name = workflow.snapshotter.file_name

        err = workflow.decision.epoch_n_err[1]
        self.assertEqual(err, 7526)
        avg_mse = workflow.decision.epoch_metrics[1][0]
        self.assertAlmostEqual(avg_mse, 0.592094, places=5)
        self.assertEqual(2, workflow.loader.epoch_number)

        self.info("Will load workflow from %s", file_name)
        workflow_from_snapshot = Snapshotter.import_(file_name)
        workflow_from_snapshot.workflow = self.parent
        self.assertTrue(workflow_from_snapshot.decision.epoch_ended)
        workflow_from_snapshot.decision.max_epochs = 5
        workflow_from_snapshot.decision.complete <<= False
        self.assertEqual(workflow_from_snapshot.evaluator.labels,
                         workflow_from_snapshot.loader.minibatch_labels)
        workflow_from_snapshot.initialize(
            device=self.device, weights=None, bias=None,
            snapshot=True)
        self.assertEqual(workflow_from_snapshot.evaluator.labels,
                         workflow_from_snapshot.loader.minibatch_labels)
        workflow_from_snapshot.run()
        self.assertIsNone(workflow_from_snapshot.thread_pool.failure)

        err = workflow_from_snapshot.decision.epoch_n_err[1]
        self.assertEqual(err, 5641)
        avg_mse = workflow_from_snapshot.decision.epoch_metrics[1][0]
        self.assertAlmostEqual(avg_mse, 0.548595, places=5)
        self.assertEqual(5, workflow_from_snapshot.loader.epoch_number)
        self.info("All Ok")

if __name__ == "__main__":
    StandardTest.main()
