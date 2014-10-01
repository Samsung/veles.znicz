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
import veles.opencl as opencl
import veles.prng as rnd
from veles.snapshotter import Snapshotter
from veles.tests import timeout
import veles.znicz.tests.research.cifar as cifar
import veles.tests.dummy_workflow as dummy_workflow


LR = 0.005
LRB = LR * 2
WD = 0.0005
WDB = WD
GM = 0.9
GMB = GM
WDSM = WD * 2
WDSMB = 0.0


class TestCifarConv(unittest.TestCase):
    def setUp(self):
        root.common.unit_test = True
        root.common.plotters_disabled = True
        self.device = opencl.Device()

    @timeout(12000)
    def test_cifar_conv(self):
        logging.info("Will test cifar convolutional workflow")
        rnd.get().seed(numpy.fromfile("%s/veles/znicz/tests/research/seed" %
                                      root.common.veles_dir,
                                      dtype=numpy.int32, count=1024))

        root.update = {
            "decision": {"fail_iterations": 250},
            "snapshotter": {"prefix": "cifar_conv_test"},
            "image_saver": {"out_dirs":
                            [os.path.join(root.common.cache_dir, "tmp/test"),
                             os.path.join(root.common.cache_dir,
                                          "tmp/validation"),
                             os.path.join(root.common.cache_dir,
                                          "tmp/train")]},
            "loader": {"minibatch_size": 100, "shuffle_limit": 2000000000},
            "softmax": {"error_function_avr": True},
            "weights_plotter": {"limit": 64},
            "cifar_conv_test": {"layers":
                                [{"type": "conv", "n_kernels": 32,
                                  "kx": 5, "ky": 5, "padding": (2, 2, 2, 2),
                                  "sliding": (1, 1),
                                  "weights_filling": "uniform",
                                  "bias_filling": "uniform",
                                  "learning_rate": LR,
                                  "learning_rate_bias": LRB,
                                  "weights_decay": WD,
                                  "weights_decay_bias": WDB,
                                  "gradient_moment": GM,
                                  "gradient_moment_bias": GMB},
                                 {"type": "norm", "alpha": 0.00005,
                                  "beta": 0.75, "n": 3, "k": 1},
                                 {"type": "activation_tanhlog"},
                                 {"type": "maxabs_pooling",
                                  "kx": 3, "ky": 3, "sliding": (2, 2)},

                                 {"type": "conv", "n_kernels": 32,
                                  "kx": 5, "ky": 5, "padding": (2, 2, 2, 2),
                                  "sliding": (1, 1),
                                  "weights_filling": "uniform",
                                  "bias_filling": "uniform",
                                  "learning_rate": LR,
                                  "learning_rate_bias": LRB,
                                  "weights_decay": WD,
                                  "weights_decay_bias": WDB,
                                  "gradient_moment": GM,
                                  "gradient_moment_bias": GMB},
                                 {"type": "norm", "alpha": 0.00005,
                                  "beta": 0.75, "n": 3, "k": 1},
                                 {"type": "activation_tanhlog"},
                                 {"type": "avg_pooling",
                                  "kx": 3, "ky": 3, "sliding": (2, 2)},

                                 {"type": "conv", "n_kernels": 64,
                                  "kx": 5, "ky": 5, "padding": (2, 2, 2, 2),
                                  "sliding": (1, 1),
                                  "weights_filling": "uniform",
                                  "bias_filling": "uniform",
                                  "learning_rate": LR,
                                  "learning_rate_bias": LRB,
                                  "weights_decay": WD,
                                  "weights_decay_bias": WDB,
                                  "gradient_moment": GM,
                                  "gradient_moment_bias": GMB},
                                 {"type": "activation_tanhlog"},
                                 {"type": "avg_pooling",
                                  "kx": 3, "ky": 3, "sliding": (2, 2)},

                                 {"type": "softmax", "output_shape": 10,
                                  "weights_filling": "uniform",
                                  "bias_filling": "uniform",
                                  "learning_rate": LR,
                                  "learning_rate_bias": LRB,
                                  "weights_decay": WDSM,
                                  "weights_decay_bias": WDSMB,
                                  "gradient_moment": GM,
                                  "gradient_moment_bias": GMB}]}}

        self.w = cifar.CifarWorkflow(dummy_workflow.DummyWorkflow(),
                                     layers=root.cifar_conv_test.layers,
                                     device=self.device)
        self.w.decision.max_epochs = 5
        self.assertEqual(self.w.evaluator.labels,
                         self.w.loader.minibatch_labels)
        self.w.initialize(device=self.device,
                          minibatch_size=root.loader.minibatch_size)
        self.assertEqual(self.w.evaluator.labels,
                         self.w.loader.minibatch_labels)
        self.w.run()
        file_name = self.w.snapshotter.file_name

        err = self.w.decision.epoch_n_err[1]
        self.assertEqual(err, 4618)
        self.assertEqual(5, self.w.loader.epoch_number)

        logging.info("Will load workflow from %s" % file_name)
        self.wf = Snapshotter.import_(file_name)
        self.assertTrue(self.wf.decision.epoch_ended)
        self.wf.decision.max_epochs = 10
        self.wf.decision.complete <<= False
        self.assertEqual(self.wf.evaluator.labels,
                         self.wf.loader.minibatch_labels)
        self.wf.initialize(device=self.device,
                           minibatch_size=root.loader.minibatch_size)
        self.assertEqual(self.wf.evaluator.labels,
                         self.wf.loader.minibatch_labels)
        self.wf.run()

        err = self.wf.decision.epoch_n_err[1]
        self.assertEqual(err, 3841)
        self.assertEqual(10, self.wf.loader.epoch_number)
        logging.info("All Ok")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
