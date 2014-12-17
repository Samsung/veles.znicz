#!/usr/bin/python3 -O
"""
Created on April 2, 2014

Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""


import logging
import numpy
import os
import sys
import six
import unittest

from veles.config import root
import veles.backends as opencl
import veles.prng as rnd
from veles.snapshotter import Snapshotter
from veles.tests import timeout
import veles.znicz.samples.Channels.channels as channels
import veles.dummy as dummy_workflow


class TestChannels(unittest.TestCase):
    def setUp(self):
        root.common.unit_test = True
        root.common.plotters_disabled = True
        self.device = opencl.Device()

    @timeout(600)
    def test_channels(self):
        logging.info("Will test channels fully connected workflow")
        rnd.get().seed(numpy.fromfile("%s/veles/znicz/tests/research/seed" %
                                      root.common.veles_dir,
                                      dtype=numpy.int32, count=1024))
        root.channels.model = "test"

        root.common.cache_dir = os.path.join(root.common.test_dataset_root,
                                             "cache")
        root.common.precision_level = 1

        root.channels.update({
            "accumulator": {"bars": 30},
            "decision": {"fail_iterations": 100,
                         "max_epochs": 2,
                         "do_export_weights": True},
            "snapshotter": {"prefix":
                            "test_channels_%s" % root.channels.model},
            "image_saver": {"out_dirs":
                            [os.path.join(root.common.cache_dir,
                                          "tmp_%s/test" %
                                          root.channels.model),
                             os.path.join(root.common.cache_dir,
                                          "tmp_%s/validation" %
                                          root.channels.model),
                             os.path.join(root.common.cache_dir,
                                          "tmp_%s/train" %
                                          root.channels.model)]},
            "loader": {"cache_file_name":
                       os.path.join(root.common.test_dataset_root,
                                    "channels_test.%d.pickle" %
                                    sys.version_info[0]),
                       "grayscale": False,
                       "minibatch_size": 81,
                       "n_threads": 32,
                       "channels_dir":
                       "/data/veles/VD/channels/russian_small/train",
                       "rect": (264, 129),
                       "validation_ratio": 0.15},
            "export": False,
            "find_negative": 0,
            "learning_rate": 0.001,
            "weights_decay": 0.00005,
            "layers": [{"type": "all2all_tanh", "output_shape": 54},
                       {"type": "softmax", "output_shape": 11}],
            "snapshot": ""})

        self.w = channels.ChannelsWorkflow(dummy_workflow.DummyWorkflow(),
                                           layers=root.channels.layers,
                                           device=self.device)
        w_neg = None
        self.w.snapshotter.interval = 2
        self.assertEqual(self.w.evaluator.labels,
                         self.w.loader.minibatch_labels)
        self.w.initialize(device=self.device,
                          learning_rate=root.channels.learning_rate,
                          weights_decay=root.channels.weights_decay,
                          minibatch_size=root.channels.loader.minibatch_size,
                          w_neg=w_neg)
        self.assertEqual(self.w.evaluator.labels,
                         self.w.loader.minibatch_labels)
        self.w.run()
        file_name = self.w.snapshotter.file_name

        err = self.w.decision.epoch_n_err[1]
        self.assertEqual(err, 111 if six.PY3 else 323)
        self.assertEqual(2, self.w.loader.epoch_number)

        logging.info("Will load workflow from %s" % file_name)
        self.wf = Snapshotter.import_(file_name)
        self.wf.loader.cache_file_name = os.path.join(
            root.common.test_dataset_root,
            "channels_test.%d.pickle" % sys.version_info[0])

        self.wf.snapshotter.directory = os.path.join(
            root.common.test_dataset_root, "snapshots")
        self.assertTrue(self.wf.decision.epoch_ended)
        self.wf.decision.max_epochs = 5
        self.wf.decision.complete <<= False
        self.assertEqual(self.wf.evaluator.labels,
                         self.wf.loader.minibatch_labels)
        self.wf.initialize(device=self.device,
                           learning_rate=root.channels.learning_rate,
                           weights_decay=root.channels.weights_decay,
                           minibatch_size=root.channels.loader.minibatch_size,
                           w_neg=w_neg)
        self.assertEqual(self.wf.evaluator.labels,
                         self.wf.loader.minibatch_labels)
        self.wf.run()

        err = self.wf.decision.epoch_n_err[1]
        self.assertEqual(err, 75 if six.PY3 else 312)
        self.assertEqual(5, self.wf.loader.epoch_number)
        logging.info("All Ok")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
