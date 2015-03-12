#!/usr/bin/python3 -O
"""
Created on April 2, 2014

Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""


import logging
import numpy
import unittest

from veles.config import root
from veles.genetics import Tune, fix_config
import veles.backends as opencl
import veles.prng as rnd
from veles.snapshotter import Snapshotter
from veles.tests import timeout
import veles.znicz.tests.research.MNIST.mnist as mnist_all2all
import veles.dummy as dummy_workflow


class TestMnistAll2All(unittest.TestCase):
    def setUp(self):
        self.device = opencl.Device()

    @timeout(300)
    def test_mnist_all2all(self):
        logging.info("Will test fully connectected mnist workflow")
        rnd.get().seed(numpy.fromfile("%s/veles/znicz/tests/research/seed" %
                                      root.common.veles_dir,
                                      dtype=numpy.int32, count=1024))

        root.common.update({
            "precision_level": 1,
            "precision_type": "double",
            "engine": {"backend": "ocl"}})

        root.mnistr.update({
            "loss_function": "softmax",
            "loader_name": "mnist_loader",
            "learning_rate_adjust": {"do": False},
            "decision": {"fail_iterations": 100,
                         "max_epochs": 3},
            "snapshotter": {"prefix": "mnist_all2all_test", "interval": 3},
            "loader": {"minibatch_size": Tune(60, 1, 1000),
                       "force_cpu": False,
                       "normalization_type": "linear"},
            "layers": [{"type": "all2all_tanh",
                        "->": {"output_sample_shape": Tune(100, 10, 500),
                               "weights_filling": "uniform",
                               "weights_stddev": Tune(0.05, 0.0001, 0.1),
                               "bias_filling": "uniform",
                               "bias_stddev": Tune(0.05, 0.0001, 0.1)},
                        "<-": {"learning_rate": Tune(0.03, 0.0001, 0.9),
                               "weights_decay": Tune(0.0, 0.0, 0.9),
                               "learning_rate_bias": Tune(0.03, 0.0001, 0.9),
                               "weights_decay_bias": Tune(0.0, 0.0, 0.9),
                               "gradient_moment": Tune(0.0, 0.0, 0.95),
                               "gradient_moment_bias": Tune(0.0, 0.0, 0.95),
                               "factor_ortho": Tune(0.001, 0.0, 0.1)}},
                       {"type": "softmax",
                        "->": {"output_sample_shape": 10,
                               "weights_filling": "uniform",
                               "weights_stddev": Tune(0.05, 0.0001, 0.1),
                               "bias_filling": "uniform",
                               "bias_stddev": Tune(0.05, 0.0001, 0.1)},
                        "<-": {"learning_rate": Tune(0.03, 0.0001, 0.9),
                               "learning_rate_bias": Tune(0.03, 0.0001, 0.9),
                               "weights_decay": Tune(0.0, 0.0, 0.95),
                               "weights_decay_bias": Tune(0.0, 0.0, 0.95),
                               "gradient_moment": Tune(0.0, 0.0, 0.95),
                               "gradient_moment_bias":
                               Tune(0.0, 0.0, 0.95)}}]})

        fix_config(root)
        self.w = mnist_all2all.MnistWorkflow(
            dummy_workflow.DummyLauncher(),
            decision_config=root.mnistr.decision,
            snapshotter_config=root.mnistr.snapshotter,
            loader_name=root.mnistr.loader_name,
            loader_config=root.mnistr.loader,
            layers=root.mnistr.layers,
            loss_function=root.mnistr.loss_function,
            device=self.device)
        self.assertEqual(self.w.evaluator.labels,
                         self.w.loader.minibatch_labels)
        self.w.snapshotter.time_interval = 0
        self.w.snapshotter.interval = 3 + 1
        self.w.initialize(device=self.device)
        self.assertEqual(self.w.evaluator.labels,
                         self.w.loader.minibatch_labels)
        self.w.run()
        file_name = self.w.snapshotter.file_name

        err = self.w.decision.epoch_n_err[1]
        self.assertEqual(err, 634)
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
        self.assertEqual(err, 474)
        self.assertEqual(6, self.wf.loader.epoch_number)
        logging.info("All Ok")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
