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
import veles.prng as rnd
from veles.snapshotter import Snapshotter
from veles.tests import timeout
import veles.znicz.tests.research.TvChannels.channels as channels
import veles.dummy as dummy_workflow


class TestChannels(unittest.TestCase):
    def setUp(self):
        self.device = opencl.Device()

    @timeout(1200)
    def test_channels_all2all(self):
        logging.info("Will test channels fully connected workflow")
        rnd.get().seed(numpy.fromfile("%s/veles/znicz/tests/research/seed" %
                                      root.common.veles_dir,
                                      dtype=numpy.int32, count=1024))

        root.common.update({
            "disable_plotting": True,
            "precision_level": 1,
            "precision_type": "double",
            "engine": {"backend": "ocl"}})

        root.channels.update({
            "decision": {"fail_iterations": 50,
                         "max_epochs": 3},
            "snapshotter": {"prefix": "test_channels", "interval": 4,
                            "time_interval": 0},
            "image_saver": {"out_dirs":
                            [os.path.join(root.common.cache_dir, "tmp/test"),
                             os.path.join(root.common.cache_dir,
                                          "tmp/validation"),
                             os.path.join(root.common.cache_dir,
                                          "tmp/train")]},
            "loader": {"minibatch_size": 30,
                       "force_cpu":  True,
                       "validation_ratio": 0.15,
                       "shuffle_limit": numpy.iinfo(numpy.uint32).max,
                       "normalization_type": "mean_disp",
                       "add_sobel": True,
                       "filename_types": ["png"],
                       "background_image":
                       "/data/veles/VD/video/dataset/black_4ch.png",
                       "mirror": False,
                       "color_space": "HSV",
                       "background_color": (0, 0, 0, 0),
                       "scale": (256, 256),
                       "scale_maintain_aspect_ratio": True,
                       "train_paths": ["/data/veles/VD/video/dataset/train"]},
            "loss_function": "softmax",
            "loader_name": "full_batch_auto_label_file_image",
            "layers": [{"type": "all2all_tanh",
                        "<-": {"learning_rate": 0.01,
                               "weights_decay": 0.00005},
                        "->": {"output_sample_shape": 100}},
                       {"type": "softmax",
                        "->": {"output_sample_shape": 8},
                        "<-": {"learning_rate": 0.01,
                               "weights_decay": 0.00005}}]})

        self.w = channels.ChannelsWorkflow(
            dummy_workflow.DummyLauncher(),
            decision_config=root.channels.decision,
            snapshotter_config=root.channels.snapshotter,
            image_saver_config=root.channels.image_saver,
            loader_config=root.channels.loader,
            layers=root.channels.layers,
            loader_name=root.channels.loader_name,
            loss_function=root.channels.loss_function)

        self.assertEqual(self.w.evaluator.labels,
                         self.w.loader.minibatch_labels)
        self.w.initialize(device=self.device,
                          minibatch_size=root.channels.loader.minibatch_size,
                          snapshot=False)
        self.assertEqual(self.w.evaluator.labels,
                         self.w.loader.minibatch_labels)
        self.w.run()
        file_name = self.w.snapshotter.file_name

        err = self.w.decision.epoch_n_err[1]
        self.assertEqual(err, 19)
        self.assertEqual(3, self.w.loader.epoch_number)

        logging.info("Will load workflow from %s" % file_name)
        self.wf = Snapshotter.import_(file_name)
        self.assertTrue(self.wf.decision.epoch_ended)
        self.wf.decision.max_epochs = 4
        self.wf.decision.complete <<= False
        self.assertTrue(self.wf.loader.force_cpu)
        self.assertEqual(self.wf.evaluator.labels,
                         self.wf.loader.minibatch_labels)
        self.wf.initialize(device=self.device,
                           minibatch_size=root.channels.loader.minibatch_size,
                           snapshot=True)
        self.assertEqual(self.wf.evaluator.labels,
                         self.wf.loader.minibatch_labels)
        self.wf.run()

        err = self.wf.decision.epoch_n_err[1]
        # PIL Image for python2 and PIL for python3 returns different values
        self.assertEqual(err, 15 if six.PY3 else 14)
        self.assertEqual(4, self.wf.loader.epoch_number)
        logging.info("All Ok")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
