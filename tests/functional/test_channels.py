#!/usr/bin/python3.3 -O
"""
Created on June 6, 2014

Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""


import logging
import numpy
import os
import sys
import time
import traceback
import unittest

from veles.config import root
import veles.opencl as opencl
import veles.random_generator as rnd
from veles.tests import timeout
import veles.znicz.samples.channels as channels
import veles.tests.dummy_workflow as dummy_workflow


class TestChannels(unittest.TestCase):
    def setUp(self):
        root.common.unit_test = True
        root.common.plotters_disabled = True
        self.device = opencl.Device()

    @timeout()
    def test_channels(self):
        logging.info("Will test channels workflow")
        rnd.get().seed(numpy.fromfile("%s/veles/znicz/tests/research/seed" %
                                      root.common.veles_dir,
                                      dtype=numpy.int32, count=1024))
        rnd.get(2).seed(numpy.fromfile("%s/veles/znicz/tests/research/seed2" %
                                       root.common.veles_dir,
                                       dtype=numpy.int32, count=1024))
        self._do_test_all2all_config()

        self._do_test_conv_config()

    def _do_test(self):
        self.w = channels.Workflow(dummy_workflow.DummyWorkflow(),
                                   layers=root.channels_test.layers,
                                   device=self.device)
        w_neg = None
        if root.channels_test.export:
            tm = time.localtime()
            s = "%d.%02d.%02d_%02d.%02d.%02d" % (
                tm.tm_year, tm.tm_mon, tm.tm_mday,
                tm.tm_hour, tm.tm_min, tm.tm_sec)
            fnme = os.path.join(root.common.snapshot_dir,
                                "channels_workflow_%s" % s)
            try:
                self.w.export(fnme)
                logging.info("Exported successfully to %s.tar.gz" % (fnme))
            except:
                a, b, c = sys.exc_info()
                traceback.print_exception(a, b, c)
                logging.error("Error while exporting.")
            return
        if root.channels_test.find_negative > 0:
            if type(self.w) != tuple or len(self.w) != 2:
                logging.error(
                    "Snapshot with weights and biases only "
                    "should be provided when find_negative is supplied. "
                    "Will now exit.")
                return
            w_neg = self.w
        self.w.initialize(learning_rate=root.channels_test.learning_rate,
                          weights_decay=root.channels_test.weights_decay,
                          device=self.device,
                          minibatch_size=root.loader.minibatch_size,
                          w_neg=w_neg)
        self.w.run()

    def _do_test_all2all_config(self):
        root.model = "tanh"
        root.update = {
            "accumulator": {"bars": 30},
            "decision": {"fail_iterations": 100,
                         "max_epochs": 0,
                         "use_dynamic_alpha": False,
                         "do_export_weights": True},
            "snapshotter": {"prefix": "channels %s" % root.model},
            "image_saver": {"out_dirs":
                            [os.path.join(root.common.cache_dir,
                                          "tmp %s/test" % root.model),
                             os.path.join(root.common.cache_dir,
                                          "tmp %s/validation" % root.model),
                             os.path.join(root.common.cache_dir,
                                          "tmp %s/train" % root.model)]},
            "loader": {"cache_fnme": os.path.join(root.common.cache_dir,
                                                  "channels_%s.pickle" %
                                                  root.model),
                       "grayscale": False,
                       "minibatch_size": 81,
                       "n_threads": 32,
                       "channels_dir":
                       "/data/veles/VD/channels/russian_small/train",
                       "rect": (264, 129),
                       "validation_ratio": 0.15},
            "weights_plotter": {"limit": 16},
            "channels_test": {"export": False,
                              "find_negative": 0,
                              "learning_rate": 0.001,
                              "weights_decay": 0.00005,
                              "layers": [{"type": "all2all_tanh",
                                          "output_shape": 54},
                                         {"type": "softmax",
                                          "output_shape": 11}],
                              "snapshot": ""}}
        self._do_test()
        err = self.w.decision.epoch_n_err[1]
        logging.info("err %s" % err)
        self.assertEqual(err, 322)
        logging.info("All Ok")

    def _do_test_conv_config(self):
        root.model = "conv"
        root.update = {
            "accumulator": {"bars": 30},
            "decision": {"fail_iterations": 100,
                         "max_epochs": 0,
                         "use_dynamic_alpha": False,
                         "do_export_weights": True},
            "snapshotter": {"prefix": "channels %s" % root.model},
            "image_saver": {"out_dirs":
                            [os.path.join(root.common.cache_dir,
                                          "tmp %s/test" % root.model),
                             os.path.join(root.common.cache_dir,
                                          "tmp %s/validation" % root.model),
                             os.path.join(root.common.cache_dir,
                                          "tmp %s/train" % root.model)]},
            "loader": {"cache_fnme": os.path.join(root.common.cache_dir,
                                                  "channels_%s.pickle" %
                                                  root.model),
                       "grayscale": False,
                       "minibatch_size": 81,
                       "n_threads": 32,
                       "channels_dir":
                       "/data/veles/VD/channels/russian_small/train",
                       "rect": (264, 129),
                       "validation_ratio": 0.15},
            "weights_plotter": {"limit": 64},
            "channels_test": {"export": False,
                              "find_negative": 0,
                              "learning_rate": 0.0001,
                              "weights_decay": 0.004,
                              "layers":
                              [{"type": "conv", "n_kernels": 32,
                                "kx": 11, "ky": 11, "padding": (2, 2, 2, 2)},
                               {"type": "max_pooling",
                                "kx": 5, "ky": 5, "sliding": (2, 2)},
                               {"type": "conv", "n_kernels": 20,
                                "kx": 7, "ky": 7, "padding": (2, 2, 2, 2)},
                               {"type": "avg_pooling",
                                "kx": 5, "ky": 5, "sliding": (2, 2)},
                               {"type": "all2all_tanh", "output_shape": 20},
                               {"type": "softmax", "output_shape": 11}],
                              "snapshot": ""}}
        self._do_test()
        err = self.w.decision.epoch_n_err[1]
        logging.info("err %s" % err)
        self.assertEqual(err, 295)
        logging.info("All Ok")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
