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
import veles.znicz.samples.YaleFaces.yale_faces as yale_faces
import veles.dummy as dummy_workflow


class TestYaleFaces(unittest.TestCase):
    def setUp(self):
        self.device = opencl.Device()

    @timeout(300)
    def test_yale_faces(self):
        logging.info("Will test fully connectected yale_faces workflow")
        rnd.get().seed(numpy.fromfile("%s/veles/znicz/tests/research/seed" %
                                      root.common.veles_dir,
                                      dtype=numpy.int32, count=1024))
        root.common.precision_level = 1

        root.yalefaces.update({
            "decision": {"fail_iterations": 50, "max_epochs": 3},
            "loss_function": "softmax",
            "snapshotter": {"prefix": "yalefaces_test"},
            "loader_name": "full_batch_auto_label_file_image",
            "loader": {"minibatch_size": 40, "force_cpu": False,
                       "validation_ratio": 0.15,
                       "filename_types": ["x-portable-graymap"],
                       "ignored_files": [".*Ambient.*"],
                       "shuffle_limit": numpy.iinfo(numpy.uint32).max,
                       "add_sobel": False,
                       "mirror": False,
                       "color_space": "GRAY",
                       "background_color": (0,),
                       "normalization_type": "mean_disp",
                       "train_paths":
                       [os.path.join(root.common.test_dataset_root,
                                     "CroppedYale")]},
            "layers": [{"type": "all2all_tanh",
                        "->": {"output_sample_shape": 100},
                        "<-": {"learning_rate": 0.01,
                               "weights_decay": 0.00005}},
                       {"type": "softmax",
                        "<-": {"learning_rate": 0.01,
                               "weights_decay": 0.00005}}]})

        self.w = yale_faces.YaleFacesWorkflow(
            dummy_workflow.DummyLauncher(),
            loader_name=root.yalefaces.loader_name,
            loader_config=root.yalefaces.loader,
            decision_config=root.yalefaces.decision,
            snapshotter_config=root.yalefaces.snapshotter,
            layers=root.yalefaces.layers,
            loss_function=root.yalefaces.loss_function,
            device=self.device)
        self.assertEqual(self.w.evaluator.labels,
                         self.w.loader.minibatch_labels)
        self.w.snapshotter.time_interval = 0
        self.w.snapshotter.interval = 4
        self.w.initialize(device=self.device)
        self.assertEqual(self.w.evaluator.labels,
                         self.w.loader.minibatch_labels)
        self.w.run()
        file_name = self.w.snapshotter.file_name

        err = self.w.decision.epoch_n_err[1]
        self.assertEqual(err, 239)
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
        self.assertEqual(err, 167)
        self.assertEqual(6, self.wf.loader.epoch_number)
        logging.info("All Ok")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    unittest.main()
