#!/usr/bin/python3 -O
"""
Created on October 15, 2014

Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""


import logging
import numpy
import unittest

from veles.config import root
import veles.opencl as opencl
import veles.prng as rnd
from veles.snapshotter import Snapshotter
from veles.tests import timeout
import veles.znicz.tests.research.mnist as mnist_conv
import veles.tests.dummy_workflow as dummy_workflow


class TestMnistConv(unittest.TestCase):
    def setUp(self):
        root.common.unit_test = True
        root.common.plotters_disabled = True
        self.device = opencl.Device()

    @timeout(300)
    def test_mnist_conv(self):
        logging.info("Will test mnist workflow with convolutional"
                     " (genetic generate) config")
        rnd.get().seed(numpy.fromfile("%s/veles/znicz/tests/research/seed" %
                                      root.common.veles_dir,
                                      dtype=numpy.int32, count=1024))
        root.mnistr.update({
            "learning_rate_adjust": {"do": True},
            "decision": {"max_epochs": 2,
                         "fail_iterations": 100},
            "snapshotter": {"prefix": "test_mnist_conv", "time_interval": 0,
                            "compress": ""},
            "weights_plotter": {"limit": 64},
            "loader": {"minibatch_size": 6, "on_device": True},
            "layers": [{"type": "conv",
                        "n_kernels": 64, "kx": 5, "ky": 5,
                        "sliding": (1, 1), "learning_rate": 0.03,
                        "learning_rate_bias": 0.358000,
                        "gradient_moment": 0.36508255921752014,
                        "gradient_moment_bias": 0.385000,
                        "weights_filling": "uniform",
                        "weights_stddev": 0.0944569801138958,
                        "bias_filling": "constant",
                        "bias_stddev": 0.048000,
                        "weights_decay": 0.0005,
                        "weights_decay_bias": 0.1980997902551238,
                        "factor_ortho": 0.001},

                       {"type": "max_pooling",
                        "kx": 2, "ky": 2, "sliding": (2, 2)},

                       {"type": "conv", "n_kernels": 87,
                        "kx": 5, "ky": 5, "sliding": (1, 1),
                        "learning_rate": 0.03,
                        "learning_rate_bias": 0.381000,
                        "gradient_moment": 0.115000,
                        "gradient_moment_bias": 0.741000,
                        "weights_filling": "uniform",
                        "weights_stddev": 0.067000,
                        "bias_filling": "constant", "bias_stddev": 0.444000,
                        "weights_decay": 0.0005, "factor_ortho": 0.001,
                        "weights_decay_bias": 0.039000},

                       {"type": "max_pooling",
                        "kx": 2, "ky": 2, "sliding": (2, 2)},

                       {"type": "all2all_relu",
                        "output_shape": 791,
                        "learning_rate": 0.03,
                        "learning_rate_bias": 0.196000,
                        "gradient_moment": 0.810000,
                        "gradient_moment_bias": 0.619000,
                        "weights_filling": "uniform",
                        "weights_stddev": 0.039000,
                        "bias_filling": "constant", "bias_stddev": 1.000000,
                        "weights_decay": 0.0005, "factor_ortho": 0.001,
                        "weights_decay_bias": 0.11487830567238211},

                       {"type": "softmax",
                        "output_shape": 10,
                        "learning_rate": 0.03,
                        "learning_rate_bias": 0.488000,
                        "gradient_moment": 0.133000,
                        "gradient_moment_bias": 0.8422143625658985,
                        "weights_filling": "uniform",
                        "weights_stddev": 0.024000,
                        "bias_filling": "constant", "bias_stddev": 0.255000,
                        "weights_decay": 0.0005,
                        "weights_decay_bias": 0.476000}]})
        self.w = mnist_conv.MnistWorkflow(dummy_workflow.DummyWorkflow(),
                                          layers=root.mnistr.layers,
                                          device=self.device)
        self.assertEqual(self.w.evaluator.labels,
                         self.w.loader.minibatch_labels)
        self.w.snapshotter.interval = 2
        self.w.initialize(device=self.device)
        self.assertEqual(self.w.evaluator.labels,
                         self.w.loader.minibatch_labels)
        self.w.run()
        file_name = self.w.snapshotter.file_name

        err = self.w.decision.epoch_n_err[1]
        self.assertEqual(err, 125)
        self.assertEqual(2, self.w.loader.epoch_number)

        logging.info("Will load workflow from %s" % file_name)
        self.wf = Snapshotter.import_(file_name)
        self.assertTrue(self.wf.decision.epoch_ended)
        self.wf.decision.max_epochs = 3
        self.wf.decision.complete <<= False
        self.assertEqual(self.wf.evaluator.labels,
                         self.wf.loader.minibatch_labels)
        self.wf.initialize(device=self.device)
        self.assertEqual(self.wf.evaluator.labels,
                         self.wf.loader.minibatch_labels)
        self.wf.run()

        err = self.wf.decision.epoch_n_err[1]
        self.assertEqual(err, 104)
        self.assertEqual(3, self.wf.loader.epoch_number)
        logging.info("All Ok")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
