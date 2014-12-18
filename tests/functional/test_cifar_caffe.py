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


class TestCifarCaffe(unittest.TestCase):
    def setUp(self):
        root.common.unit_test = True
        root.common.plotters_disabled = True
        self.device = opencl.Device()

    @timeout(300)
    def test_cifar_caffe(self):
        logging.info("Will test cifar convolutional"
                     "workflow with caffe config")
        rnd.get().seed(numpy.fromfile("%s/veles/znicz/tests/research/seed" %
                                      root.common.veles_dir,
                                      dtype=numpy.int32, count=1024))
        root.common.precision_level = 1
        root.cifar.update({
            "decision": {"fail_iterations": 250},
            "snapshotter": {"prefix": "cifar_caffe_test"},
            "image_saver": {"out_dirs":
                            [os.path.join(root.common.cache_dir, "tmp/test"),
                             os.path.join(root.common.cache_dir,
                                          "tmp/validation"),
                             os.path.join(root.common.cache_dir,
                                          "tmp/train")]},
            "loader": {"minibatch_size": 100, "norm": "mean", "sobel": False,
                       "shuffle_limit": 2000000000},
            "softmax": {"error_function_avr": True},
            "layers": [{"name": "conv1", "type": "conv", "n_kernels": 32,
                        "kx": 5, "ky": 5, "padding": (2, 2, 2, 2),
                        "sliding": (1, 1), "weights_filling": "gaussian",
                        "weights_stddev": 0.0001, "bias_filling": "constant",
                        "bias_stddev": 0, "learning_rate": 0.001,
                        "learning_rate_bias": 0.002, "weights_decay": 0.0005,
                        "weights_decay_bias": 0.0005, "factor_ortho": 0.001,
                        "gradient_moment": 0.9, "gradient_moment_bias": 0.9},

                       {"name": "pool1", "type": "max_pooling",
                        "kx": 3, "ky": 3, "sliding": (2, 2)},

                       {"name": "relu1", "type": "activation_str"},

                       {"name": "norm1", "type": "norm", "alpha": 0.00005,
                        "beta": 0.75, "n": 3, "k": 1},

                       {"name": "conv2", "type": "conv", "n_kernels": 32,
                        "kx": 5, "ky": 5, "padding": (2, 2, 2, 2),
                        "sliding": (1, 1), "weights_filling": "gaussian",
                        "weights_stddev": 0.01, "bias_filling": "constant",
                        "bias_stddev": 0, "learning_rate": 0.001,
                        "learning_rate_bias": 0.002, "weights_decay": 0.0005,
                        "weights_decay_bias": 0.0005, "factor_ortho": 0.001,
                        "gradient_moment": 0.9, "gradient_moment_bias": 0.9},

                       {"name": "relu2", "type": "activation_str"},

                       {"name": "pool2", "type": "avg_pooling",
                        "kx": 3, "ky": 3, "sliding": (2, 2)},

                       {"name": "norm2", "type": "norm", "alpha": 0.00005,
                        "beta": 0.75, "n": 3, "k": 1},

                       {"name": "conv3", "type": "conv", "n_kernels": 64,
                        "kx": 5, "ky": 5, "padding": (2, 2, 2, 2),
                        "sliding": (1, 1), "weights_filling": "gaussian",
                        "weights_stddev": 0.01, "bias_filling": "constant",
                        "bias_stddev": 0, "learning_rate": 0.001,
                        "learning_rate_bias": 0.001, "weights_decay": 0.0005,
                        "weights_decay_bias": 0.0005, "factor_ortho": 0.001,
                        "gradient_moment": 0.9, "gradient_moment_bias": 0.9},

                       {"name": "relu3", "type": "activation_str"},

                       {"name": "pool3", "type": "avg_pooling",
                        "kx": 3, "ky": 3, "sliding": (2, 2)},

                       {"name": "a2asm4", "type": "softmax",
                        "output_shape": 10, "weights_filling": "gaussian",
                        "weights_stddev": 0.01, "bias_filling": "constant",
                        "bias_stddev": 0, "learning_rate": 0.001,
                        "learning_rate_bias": 0.002, "weights_decay": 1.0,
                        "weights_decay_bias": 0, "gradient_moment": 0.9,
                        "gradient_moment_bias": 0.9}]})

        self.w = cifar.CifarWorkflow(
            dummy_workflow.DummyWorkflow(),
            fail_iterations=root.cifar.decision.fail_iterations,
            max_epochs=root.cifar.decision.max_epochs,
            prefix=root.cifar.snapshotter.prefix,
            snapshot_interval=root.cifar.snapshotter.interval,
            snapshot_dir=root.common.snapshot_dir,
            layers=root.cifar.layers,
            out_dirs=root.cifar.image_saver.out_dirs,
            loss_function=root.cifar.loss_function,
            device=self.device)
        self.w.decision.max_epochs = 1
        self.w.snapshotter.time_interval = 0
        self.w.snapshotter.interval = 1
        self.assertEqual(self.w.evaluator.labels,
                         self.w.loader.minibatch_labels)
        self.w.initialize(device=self.device,
                          minibatch_size=root.cifar.loader.minibatch_size)
        self.assertEqual(self.w.evaluator.labels,
                         self.w.loader.minibatch_labels)
        self.w.run()
        file_name = self.w.snapshotter.file_name

        err = self.w.decision.epoch_n_err[1]
        self.assertEqual(err, 5661)
        self.assertEqual(1, self.w.loader.epoch_number)

        logging.info("Will load workflow from %s" % file_name)
        self.wf = Snapshotter.import_(file_name)
        self.assertTrue(self.wf.decision.epoch_ended)
        self.wf.decision.max_epochs = 3
        self.wf.decision.complete <<= False
        self.assertEqual(self.wf.evaluator.labels,
                         self.wf.loader.minibatch_labels)
        self.wf.initialize(device=self.device,
                           minibatch_size=root.cifar.loader.minibatch_size)
        self.assertEqual(self.wf.evaluator.labels,
                         self.wf.loader.minibatch_labels)
        self.wf.run()

        err = self.wf.decision.epoch_n_err[1]
        self.assertEqual(err, 4692)
        self.assertEqual(3, self.wf.loader.epoch_number)
        logging.info("All Ok")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
