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
        self.device = opencl.Device()

    def init_wf(self, workflow, snapshot):
        self.assertEqual(workflow.evaluator.labels,
                         workflow.loader.minibatch_labels)

        workflow.initialize(device=self.device, snapshot=snapshot)
        self.assertEqual(workflow.evaluator.labels,
                         workflow.loader.minibatch_labels)

    def check_write_error_rate(self, workflow, error):
        err = workflow.decision.epoch_n_err[1]
        self.assertEqual(err, error)
        self.assertEqual(
            workflow.decision.max_epochs, workflow.loader.epoch_number)

    def init_and_run(self, snapshot):
        self.w = cifar.CifarWorkflow(
            dummy_workflow.DummyLauncher(),
            decision_config=root.cifar.decision,
            snapshotter_config=root.cifar.snapshotter,
            image_saver_config=root.cifar.image_saver,
            loader_name=root.cifar.loader_name,
            loader_config=root.cifar.loader,
            layers=root.cifar.layers,
            loss_function=root.cifar.loss_function)
        self.init_wf(self.w, snapshot)
        self.w.run()

    @timeout(1000)
    def test_cifar_caffe(self):
        logging.info("Will test cifar convolutional "
                     "workflow with caffe config")
        rnd.get().seed(numpy.fromfile("%s/veles/znicz/tests/research/seed" %
                                      root.common.veles_dir,
                                      dtype=numpy.int32, count=1024))

        train_dir = os.path.join(root.common.test_dataset_root, "cifar/10")
        validation_dir = os.path.join(
            root.common.test_dataset_root, "cifar/10/test_batch")

        root.cifar.update({
            "loader_name": "cifar_loader",
            "decision": {"fail_iterations": 250, "max_epochs": 1},
            "learning_rate_adjust": {"do": True},
            "snapshotter": {"prefix": "cifar_caffe", "interval": 2,
                            "time_interval": 0},
            "loss_function": "softmax",
            "add_plotters": False,
            "image_saver": {"do": False,
                            "out_dirs":
                            [os.path.join(root.common.cache_dir, "tmp/test"),
                             os.path.join(root.common.cache_dir,
                                          "tmp/validation"),
                             os.path.join(root.common.cache_dir,
                                          "tmp/train")]},
            "loader": {"minibatch_size": 100,
                       "normalization_type": "internal_mean",
                       "add_sobel": False,
                       "shuffle_limit": 2000000000,
                       "force_cpu": False},
            "softmax": {"error_function_avr": True},
            "weights_plotter": {"limit": 64},
            "similar_weights_plotter": {
                "form_threshold": 1.1, "peak_threshold": 0.5,
                "magnitude_threshold": 0.65},
            "layers": [{"name": "conv1",
                        "type": "conv",
                        "->": {"n_kernels": 32, "kx": 5, "ky": 5,
                               "padding": (2, 2, 2, 2), "sliding": (1, 1),
                               "weights_filling": "gaussian",
                               "weights_stddev": 0.0001,
                               "bias_filling": "constant", "bias_stddev": 0},
                        "<-": {"learning_rate": 0.001,
                               "learning_rate_bias": 0.002,
                               "weights_decay": 0.0005,
                               "weights_decay_bias": 0.0005,
                               "factor_ortho": 0.001, "gradient_moment": 0.9,
                               "gradient_moment_bias": 0.9}},
                       {"name": "pool1", "type": "max_pooling",
                        "->": {"kx": 3, "ky": 3, "sliding": (2, 2)}},

                       {"name": "relu1", "type": "activation_str"},

                       {"name": "norm1", "type": "norm", "alpha": 0.00005,
                        "beta": 0.75, "n": 3, "k": 1},

                       {"name": "conv2", "type": "conv",
                        "->": {"n_kernels": 32, "kx": 5, "ky": 5,
                               "padding": (2, 2, 2, 2), "sliding": (1, 1),
                               "weights_filling": "gaussian",
                               "weights_stddev": 0.01,
                               "bias_filling": "constant", "bias_stddev": 0},
                        "<-": {"learning_rate": 0.001,
                               "learning_rate_bias": 0.002,
                               "weights_decay": 0.0005,
                               "weights_decay_bias": 0.0005,
                               "factor_ortho": 0.001, "gradient_moment": 0.9,
                               "gradient_moment_bias": 0.9}},
                       {"name": "relu2", "type": "activation_str"},
                       {"name": "pool2", "type": "avg_pooling",
                        "->": {"kx": 3, "ky": 3, "sliding": (2, 2)}},

                       {"name": "norm2", "type": "norm",
                        "alpha": 0.00005, "beta": 0.75, "n": 3, "k": 1},

                       {"name": "conv3", "type": "conv",
                        "->": {"n_kernels": 64, "kx": 5, "ky": 5,
                               "padding": (2, 2, 2, 2), "bias_stddev": 0,
                               "sliding": (1, 1),
                               "weights_filling": "gaussian",
                               "weights_stddev": 0.01,
                               "bias_filling": "constant"},
                        "<-": {"learning_rate": 0.001,
                               "learning_rate_bias": 0.001,
                               "weights_decay": 0.0005,
                               "weights_decay_bias": 0.0005,
                               "factor_ortho": 0.001,
                               "gradient_moment": 0.9,
                               "gradient_moment_bias": 0.9}},
                       {"name": "relu3", "type": "activation_str"},

                       {"name": "pool3", "type": "avg_pooling",
                        "->": {"kx": 3, "ky": 3, "sliding": (2, 2)}},

                       {"name": "a2asm4", "type": "softmax",
                        "->": {"output_sample_shape": 10,
                               "weights_filling": "gaussian",
                               "weights_stddev": 0.01,
                               "bias_filling": "constant", "bias_stddev": 0},
                        "<-": {"learning_rate": 0.001,
                               "learning_rate_bias": 0.002,
                               "weights_decay": 1.0, "weights_decay_bias": 0,
                               "gradient_moment": 0.9,
                               "gradient_moment_bias": 0.9}}],
            "data_paths": {"train": train_dir, "validation": validation_dir}})

        logging.info("Will run workflow with double and ocl backend")

        root.common.update({
            "precision_level": 1,
            "precision_type": "double",
            "engine": {"backend": "ocl"}})

        # Test workflow
        self.init_and_run(False)
        self.check_write_error_rate(self.w, 5667)

        logging.info("Will run workflow with double and cuda backend")

        root.common.update({
            "precision_level": 1,
            "precision_type": "double",
            "engine": {"backend": "cuda"}})

        # Test workflow with cuda and double
        self.init_and_run(False)
        self.check_write_error_rate(self.w, 5877)

        file_name = self.w.snapshotter.file_name

        # Test loading from snapshot
        logging.info("Will load workflow from snapshot: %s" % file_name)

        self.wf = Snapshotter.import_(file_name)
        self.assertTrue(self.wf.decision.epoch_ended)
        self.wf.decision.max_epochs = 3
        self.wf.decision.complete <<= False

        self.init_wf(self.wf, True)
        self.wf.run()
        self.check_write_error_rate(self.wf, 4252)

        logging.info("All Ok")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
